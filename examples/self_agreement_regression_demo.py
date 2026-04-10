"""Minimal SAGE-Reg demo across Gaussian, quantile, and bar regression heads."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import GaussianNLLLoss, MultiQuantileLoss
from torchregress.prediction import PredictiveBatch
from torchregress.semi_supervised import SelfAgreementTrainer


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 7
    n_labeled: int = 64
    n_unlabeled: int = 192
    n_test: int = 128
    hidden: int = 32
    epochs: int = 24
    k_views: int = 4
    lr: float = 5e-3
    dropout: float = 0.15
    unlabeled_noise: float = 0.05
    tau: float = 0.15
    agreement_weight: float = 0.5
    n_bins: int = 18
    batch_size: int = 32


class _Backbone(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))


class GaussianModel(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.backbone = _Backbone(hidden, dropout)
        self.mean_head = nn.Linear(hidden, 1)
        self.log_var_head = nn.Linear(hidden, 1)
        nn.init.constant_(self.log_var_head.bias, -1.4)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.backbone(x)
        return self.mean_head(h), self.log_var_head(h).clamp(min=-4.0, max=2.0)


class QuantileModel(nn.Module):
    quantile_levels = [0.1, 0.5, 0.9]

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.backbone = _Backbone(hidden, dropout)
        self.head = nn.Linear(hidden, len(self.quantile_levels))

    def forward(self, x: Tensor) -> Tensor:
        return torch.sort(self.head(self.backbone(x)), dim=-1).values


class BarModel(nn.Module):
    def __init__(self, hidden: int, dropout: float, bin_edges: Tensor) -> None:
        super().__init__()
        self.backbone = _Backbone(hidden, dropout)
        self.head = nn.Linear(hidden, bin_edges.numel() - 1)
        self.register_buffer("bin_edges", bin_edges)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x))

    def point_estimate(self, logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        return probs @ centers.unsqueeze(-1)


def _make_data(cfg: DemoConfig) -> dict[str, Tensor]:
    torch.manual_seed(cfg.seed)
    n_total = cfg.n_labeled + cfg.n_unlabeled + cfg.n_test
    x = torch.linspace(-3.0, 3.0, n_total).unsqueeze(-1)
    x = x[torch.randperm(n_total)]
    mean = torch.sin(1.4 * x) + 0.25 * x
    noise_scale = 0.12 + 0.10 * torch.sigmoid(1.2 * x)
    y = mean + noise_scale * torch.randn_like(mean)

    x_train = x[: cfg.n_labeled + cfg.n_unlabeled]
    y_train = y[: cfg.n_labeled + cfg.n_unlabeled]
    x_test = x[cfg.n_labeled + cfg.n_unlabeled :]
    y_test = y[cfg.n_labeled + cfg.n_unlabeled :]

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    return {
        "x_labeled": x_train[: cfg.n_labeled],
        "y_labeled": y_train[: cfg.n_labeled],
        "x_unlabeled": x_train[cfg.n_labeled :],
        "x_test": x_test,
        "y_test": y_test,
        "y_train_full": y_train,
    }


def _build_loaders(
    data: dict[str, Tensor], batch_size: int
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor]]]:
    labeled = DataLoader(
        TensorDataset(data["x_labeled"], data["y_labeled"]),
        batch_size=batch_size,
        shuffle=True,
    )
    unlabeled = DataLoader(
        TensorDataset(data["x_unlabeled"]),
        batch_size=batch_size,
        shuffle=True,
    )
    return labeled, unlabeled


def _augment_fn(cfg: DemoConfig) -> Callable[[Tensor], Tensor]:
    return lambda x: x + cfg.unlabeled_noise * torch.randn_like(x)


def _train_gaussian(cfg: DemoConfig, data: dict[str, Tensor]) -> dict[str, float]:
    model = GaussianModel(cfg.hidden, cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader = _build_loaders(data, cfg.batch_size)
    loss_fn = GaussianNLLLoss()

    def supervised_loss_fn(model_: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        return loss_fn(cast(GaussianModel, model_)(x), y)

    def predictive_batch_fn(model_: nn.Module, x: Tensor) -> PredictiveBatch:
        mean, log_var = cast(GaussianModel, model_)(x)
        return PredictiveBatch(mean=mean, std=torch.exp(0.5 * log_var))

    trainer = SelfAgreementTrainer(
        optimizer=optimizer,
        supervised_loss_fn=supervised_loss_fn,
        predictive_batch_fn=predictive_batch_fn,
        augment_fn=_augment_fn(cfg),
        n_views=cfg.k_views,
        tau=cfg.tau,
        agreement_weight=cfg.agreement_weight,
        ema_decay=0.95,
    )
    history = trainer.fit(model, labeled_loader, unlabeled_loader, epochs=cfg.epochs)

    model.eval()
    with torch.no_grad():
        pred_mean, _ = model(data["x_test"])
        mse = (pred_mean - data["y_test"]).square().mean()
    return {
        "test_mse": float(mse.item()),
        "supervised_loss": history["supervised_loss"][-1],
        "agreement_loss": history["unsupervised_loss"][-1],
        "mean_weight": history["mean_weight"][-1],
    }


def _train_quantile(cfg: DemoConfig, data: dict[str, Tensor]) -> dict[str, float]:
    model = QuantileModel(cfg.hidden, cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader = _build_loaders(data, cfg.batch_size)
    loss_fn = MultiQuantileLoss(quantiles=QuantileModel.quantile_levels)

    def supervised_loss_fn(model_: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        return loss_fn(cast(QuantileModel, model_)(x), y)

    def predictive_batch_fn(model_: nn.Module, x: Tensor) -> PredictiveBatch:
        return PredictiveBatch(
            quantiles=cast(QuantileModel, model_)(x),
            quantile_levels=list(QuantileModel.quantile_levels),
        )

    trainer = SelfAgreementTrainer(
        optimizer=optimizer,
        supervised_loss_fn=supervised_loss_fn,
        predictive_batch_fn=predictive_batch_fn,
        augment_fn=_augment_fn(cfg),
        n_views=cfg.k_views,
        tau=cfg.tau,
        agreement_weight=cfg.agreement_weight,
        ema_decay=0.95,
    )
    history = trainer.fit(model, labeled_loader, unlabeled_loader, epochs=cfg.epochs)

    model.eval()
    with torch.no_grad():
        quantiles = model(data["x_test"])
        mse = (quantiles[:, 1:2] - data["y_test"]).square().mean()
    return {
        "test_mse": float(mse.item()),
        "supervised_loss": history["supervised_loss"][-1],
        "agreement_loss": history["unsupervised_loss"][-1],
        "mean_weight": history["mean_weight"][-1],
    }


def _bar_targets(target: Tensor, bin_edges: Tensor) -> Tensor:
    return torch.bucketize(target.view(-1), bin_edges[1:-1]).long()


def _train_bar(cfg: DemoConfig, data: dict[str, Tensor]) -> dict[str, float]:
    y_train = data["y_train_full"]
    margin = 0.20
    bin_edges = torch.linspace(
        float(y_train.min().item()) - margin,
        float(y_train.max().item()) + margin,
        cfg.n_bins + 1,
    )
    model = BarModel(cfg.hidden, cfg.dropout, bin_edges)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader = _build_loaders(data, cfg.batch_size)

    def supervised_loss_fn(model_: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        logits = cast(BarModel, model_)(x)
        return F.cross_entropy(logits, _bar_targets(y, bin_edges))

    def predictive_batch_fn(model_: nn.Module, x: Tensor) -> PredictiveBatch:
        return PredictiveBatch(bar_logits=cast(BarModel, model_)(x), bin_edges=bin_edges)

    trainer = SelfAgreementTrainer(
        optimizer=optimizer,
        supervised_loss_fn=supervised_loss_fn,
        predictive_batch_fn=predictive_batch_fn,
        augment_fn=_augment_fn(cfg),
        n_views=cfg.k_views,
        tau=cfg.tau,
        agreement_weight=cfg.agreement_weight,
        ema_decay=0.95,
    )
    history = trainer.fit(model, labeled_loader, unlabeled_loader, epochs=cfg.epochs)

    model.eval()
    with torch.no_grad():
        logits = model(data["x_test"])
        point = model.point_estimate(logits)
        mse = (point - data["y_test"]).square().mean()
    return {
        "test_mse": float(mse.item()),
        "supervised_loss": history["supervised_loss"][-1],
        "agreement_loss": history["unsupervised_loss"][-1],
        "mean_weight": history["mean_weight"][-1],
    }


def main(cfg: DemoConfig | None = None) -> dict[str, dict[str, float]]:
    resolved = DemoConfig() if cfg is None else cfg
    data = _make_data(resolved)
    results = {
        "gaussian": _train_gaussian(resolved, data),
        "quantile": _train_quantile(resolved, data),
        "bar": _train_bar(resolved, data),
    }
    for name, metrics in results.items():
        print(
            f"{name:>8s} | mse={metrics['test_mse']:.4f} | "
            f"sup={metrics['supervised_loss']:.4f} | "
            f"agree={metrics['agreement_loss']:.4f} | "
            f"w={metrics['mean_weight']:.4f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the minimal SAGE-Reg regression demo.")
    parser.add_argument("--epochs", type=int, default=DemoConfig.epochs)
    parser.add_argument("--k-views", type=int, default=DemoConfig.k_views)
    args = parser.parse_args()
    main(DemoConfig(epochs=args.epochs, k_views=args.k_views))
