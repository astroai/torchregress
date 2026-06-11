"""Cross-backbone synthetic comparison for SAGE-Reg."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from benchmarks.self_agreement_synthetic import (  # noqa: E402
    SyntheticRegressionGeneratorConfig,
    generate_self_agreement_regression_split,
)
from torchregress.comparison import (  # noqa: E402
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses import GaussianNLLLoss, MultiQuantileLoss  # noqa: E402
from torchregress.metrics import distribution_metrics_report  # noqa: E402
from torchregress.prediction import PredictiveBatch  # noqa: E402
from torchregress.semi_supervised import SelfAgreementTrainer  # noqa: E402


@dataclass(frozen=True)
class BackboneTrainingConfig:
    lr: float
    dropout: float
    warmup_epochs: int
    unlabeled_noise: float
    tau: float
    agreement_weight: float


@dataclass(frozen=True)
class BackboneComparisonConfig:
    data: SyntheticRegressionGeneratorConfig = SyntheticRegressionGeneratorConfig(
        n_labeled=64,
        n_unlabeled=192,
        n_test=160,
        multimodal_prob=0.25,
        imbalance_strength=0.3,
        input_noise_std=0.05,
    )
    hidden: int = 32
    epochs: int = 32
    batch_size: int = 32
    ema_decay: float = 0.95
    n_views: int = 4
    n_bins: int = 24
    unlabeled_fractions: tuple[float, ...] = (0.25, 0.5, 1.0)
    gaussian: BackboneTrainingConfig = field(
        default_factory=lambda: BackboneTrainingConfig(
            lr=5e-3,
            dropout=0.15,
            warmup_epochs=4,
            unlabeled_noise=0.05,
            tau=0.15,
            agreement_weight=0.5,
        )
    )
    quantile: BackboneTrainingConfig = field(
        default_factory=lambda: BackboneTrainingConfig(
            lr=3e-3,
            dropout=0.05,
            warmup_epochs=10,
            unlabeled_noise=0.03,
            tau=0.28,
            agreement_weight=0.18,
        )
    )
    bar: BackboneTrainingConfig = field(
        default_factory=lambda: BackboneTrainingConfig(
            lr=2.5e-3,
            dropout=0.05,
            warmup_epochs=10,
            unlabeled_noise=0.03,
            tau=0.30,
            agreement_weight=0.16,
        )
    )
    bar_label_smoothing: float = 0.05


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


class _StructuredQuantileHead(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.center_head = nn.Linear(hidden, 1)
        self.left_step_head = nn.Linear(hidden, 2)
        self.right_step_head = nn.Linear(hidden, 2)
        nn.init.constant_(self.left_step_head.bias, -0.8)
        nn.init.constant_(self.right_step_head.bias, -0.8)

    def forward(self, features: Tensor) -> Tensor:
        center = self.center_head(features)
        left_steps = F.softplus(self.left_step_head(features)) + 1.0e-3
        right_steps = F.softplus(self.right_step_head(features)) + 1.0e-3

        q10 = center - left_steps[:, 0:1]
        q05 = q10 - left_steps[:, 1:2]
        q90 = center + right_steps[:, 0:1]
        q95 = q90 + right_steps[:, 1:2]
        return torch.cat([q05, q10, center, q90, q95], dim=-1)


class QuantileModel(nn.Module):
    quantile_levels = [0.05, 0.1, 0.5, 0.9, 0.95]

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.backbone = _Backbone(hidden, dropout)
        self.head = _StructuredQuantileHead(hidden)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x))


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


def _write_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _build_loaders(
    split_x_labeled: Tensor,
    split_y_labeled: Tensor,
    split_x_unlabeled: Tensor,
    *,
    batch_size: int,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor]]]:
    labeled = DataLoader(
        TensorDataset(split_x_labeled, split_y_labeled),
        batch_size=batch_size,
        shuffle=True,
    )
    unlabeled = DataLoader(
        TensorDataset(split_x_unlabeled),
        batch_size=batch_size,
        shuffle=True,
    )
    return labeled, unlabeled


def _subset_unlabeled_split(split, fraction: float, *, seed: int):
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    total = split.x_unlabeled.shape[0]
    keep = min(total, max(1, int(round(total * fraction))))
    generator = torch.Generator().manual_seed(seed + int(round(1000 * fraction)))
    indices = torch.randperm(total, generator=generator)[:keep]
    indices = indices.sort().values
    return replace(
        split,
        x_unlabeled=split.x_unlabeled[indices],
        y_unlabeled_true=split.y_unlabeled_true[indices],
    )


def _augment_fn(scale: float):
    return lambda x: x + scale * torch.randn_like(x)


def _run_supervised_epochs(
    model: nn.Module,
    labeled_loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    loss_fn,
) -> None:
    if epochs <= 0:
        return
    model.train()
    for _ in range(epochs):
        for xb, yb in labeled_loader:
            optimizer.zero_grad()
            loss = loss_fn(model, xb, yb)
            loss.backward()
            optimizer.step()


def _gaussian_predictive_batch(model_: nn.Module, x: Tensor) -> PredictiveBatch:
    mean, log_var = cast(GaussianModel, model_)(x)
    return PredictiveBatch(mean=mean, std=torch.exp(0.5 * log_var))


def _quantile_predictive_batch(model_: nn.Module, x: Tensor) -> PredictiveBatch:
    return PredictiveBatch(
        quantiles=cast(QuantileModel, model_)(x),
        quantile_levels=list(QuantileModel.quantile_levels),
    )


def _bar_predictive_batch(model_: nn.Module, x: Tensor) -> PredictiveBatch:
    model = cast(BarModel, model_)
    return PredictiveBatch(bar_logits=model(x), bin_edges=model.bin_edges)


def _quantile_dict(values: Tensor) -> dict[float, Tensor]:
    return {
        level: values[:, idx : idx + 1] for idx, level in enumerate(QuantileModel.quantile_levels)
    }


def _pseudo_nll_from_batch(batch: PredictiveBatch, y_true: Tensor) -> float:
    if batch.support is None or batch.density is None:
        raise ValueError("batch must include support and density")
    support = torch.as_tensor(batch.support, device=y_true.device, dtype=y_true.dtype)
    density = torch.as_tensor(batch.density, device=y_true.device, dtype=y_true.dtype)
    targets = y_true.reshape(-1)
    if support.ndim == 1:
        support = support.unsqueeze(0).expand(targets.shape[0], -1)

    density_at_target: list[Tensor] = []
    for row_idx in range(targets.shape[0]):
        x = support[row_idx]
        y = density[row_idx]
        target = targets[row_idx].to(device=x.device, dtype=x.dtype)
        idx = torch.searchsorted(x, target, right=False).clamp(1, x.shape[0] - 1)
        x0 = x[idx - 1]
        x1 = x[idx]
        y0 = y[idx - 1]
        y1 = y[idx]
        weight = ((target - x0) / (x1 - x0).clamp_min(1e-8)).clamp(0.0, 1.0)
        density_at_target.append(y0 + weight * (y1 - y0))
    values = torch.stack(density_at_target).clamp_min(1e-8)
    return float((-torch.log(values)).mean().item())


def _quantile_dict_from_batch_density(
    batch: PredictiveBatch,
    probs: tuple[float, ...] = (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95),
) -> dict[float, Tensor]:
    if batch.support is None or batch.density is None:
        raise ValueError("batch must include support and density")
    support = torch.as_tensor(batch.support)
    density = torch.as_tensor(batch.density)
    if density.ndim != 2:
        raise ValueError("density must have shape [batch, support]")
    if support.ndim == 1:
        support = support.unsqueeze(0).expand(density.shape[0], -1)

    prob_tensor = torch.tensor(probs, device=density.device, dtype=density.dtype)
    quantiles = torch.empty(
        density.shape[0], len(probs), device=density.device, dtype=density.dtype
    )

    for row_idx in range(density.shape[0]):
        x = support[row_idx]
        y = density[row_idx]
        dx = torch.diff(x)
        midpoint_mass = 0.5 * (y[:-1] + y[1:]) * dx
        cdf = torch.cat(
            [torch.zeros(1, device=y.device, dtype=y.dtype), torch.cumsum(midpoint_mass, 0)]
        )
        cdf = cdf / cdf[-1].clamp_min(1e-8)
        for prob_idx, prob in enumerate(prob_tensor):
            idx = torch.searchsorted(cdf, prob, right=False).clamp(1, cdf.shape[0] - 1)
            c0 = cdf[idx - 1]
            c1 = cdf[idx]
            x0 = x[idx - 1]
            x1 = x[idx]
            weight = ((prob - c0) / (c1 - c0).clamp_min(1e-8)).clamp(0.0, 1.0)
            quantiles[row_idx, prob_idx] = x0 + weight * (x1 - x0)

    return {level: quantiles[:, idx : idx + 1] for idx, level in enumerate(probs)}


def _evaluate_gaussian(model: GaussianModel, x_test: Tensor, y_test: Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        mean, log_var = model(x_test)
        std = torch.exp(0.5 * log_var).clamp_min(1e-4)
        report = distribution_metrics_report(
            dist={"mean": mean, "std": std},
            y_true=y_test,
        )
    rmse = float(torch.sqrt(torch.mean((mean - y_test).square())).item())
    return {
        "RMSE": rmse,
        "NLL": -float(report["log_prob"]),
        "CRPS": float(report["crps"]),
        "Cov90": float(report["coverage_90"]),
        "CoverageGap90": abs(float(report["coverage_90"]) - 0.9),
        "Width90": float(report["interval_width_90"]),
        "PITChi2": float(report["pit_chi2"]),
    }


def _evaluate_quantile(model: QuantileModel, x_test: Tensor, y_test: Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        quantiles = model(x_test)
        batch = PredictiveBatch(
            quantiles=quantiles,
            quantile_levels=list(QuantileModel.quantile_levels),
        ).with_density(n_support=128)
        report = distribution_metrics_report(
            y_true=y_test,
            y_pred_quantiles=_quantile_dict(quantiles),
        )
    rmse = float(torch.sqrt(torch.mean((quantiles[:, 2:3] - y_test).square())).item())
    return {
        "RMSE": rmse,
        "NLL": _pseudo_nll_from_batch(batch, y_test),
        "CRPS": float(report["crps"]),
        "Cov90": float(report["coverage_90"]),
        "CoverageGap90": abs(float(report["coverage_90"]) - 0.9),
        "Width90": float(report["interval_width_90"]),
        "PITChi2": float(report["pit_chi2"]),
    }


def _evaluate_bar(model: BarModel, x_test: Tensor, y_test: Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        batch = PredictiveBatch(bar_logits=logits, bin_edges=model.bin_edges).with_density(
            n_support=128
        )
        quantiles = _quantile_dict_from_batch_density(batch)
        report = distribution_metrics_report(
            y_true=y_test,
            y_pred_quantiles=quantiles,
            support=batch.support,
            density=batch.density,
        )
        point = model.point_estimate(logits)
    rmse = float(torch.sqrt(torch.mean((point - y_test).square())).item())
    return {
        "RMSE": rmse,
        "NLL": -float(report["log_prob"]),
        "CRPS": float(report["crps"]),
        "Cov90": float(report["coverage_90"]),
        "CoverageGap90": abs(float(report["coverage_90"]) - 0.9),
        "Width90": float(report["interval_width_90"]),
        "PITChi2": float(report["pit_chi2"]),
    }


def _train_gaussian(
    cfg: BackboneComparisonConfig, split, *, use_sage: bool
) -> tuple[GaussianModel, dict[str, float]]:
    tune = cfg.gaussian
    model = GaussianModel(cfg.hidden, tune.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=tune.lr)
    labeled_loader, unlabeled_loader = _build_loaders(
        split.x_labeled,
        split.y_labeled,
        split.x_unlabeled,
        batch_size=cfg.batch_size,
    )
    loss_fn = GaussianNLLLoss()

    def supervised_loss(model_: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        return cast(Tensor, loss_fn(cast(GaussianModel, model_)(x), y))

    if use_sage:
        warmup_epochs = min(tune.warmup_epochs, max(cfg.epochs - 1, 0))
        _run_supervised_epochs(model, labeled_loader, optimizer, warmup_epochs, supervised_loss)
        trainer = SelfAgreementTrainer(
            optimizer=optimizer,
            supervised_loss_fn=supervised_loss,
            predictive_batch_fn=_gaussian_predictive_batch,
            augment_fn=_augment_fn(tune.unlabeled_noise),
            n_views=cfg.n_views,
            tau=tune.tau,
            agreement_weight=tune.agreement_weight,
            ema_decay=cfg.ema_decay,
        )
        history = trainer.fit(
            model,
            labeled_loader,
            unlabeled_loader,
            epochs=max(1, cfg.epochs - warmup_epochs),
        )
        return model.eval(), {
            "mean_weight": history["mean_weight"][-1],
            "mean_disagreement": history["mean_disagreement"][-1],
        }

    _run_supervised_epochs(model, labeled_loader, optimizer, cfg.epochs, supervised_loss)
    return model.eval(), {"mean_weight": 0.0, "mean_disagreement": 0.0}


def _train_quantile(
    cfg: BackboneComparisonConfig, split, *, use_sage: bool
) -> tuple[QuantileModel, dict[str, float]]:
    tune = cfg.quantile
    model = QuantileModel(cfg.hidden, tune.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=tune.lr)
    labeled_loader, unlabeled_loader = _build_loaders(
        split.x_labeled,
        split.y_labeled,
        split.x_unlabeled,
        batch_size=cfg.batch_size,
    )
    loss_fn = MultiQuantileLoss(quantiles=QuantileModel.quantile_levels)

    def supervised_loss(model_: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        return cast(Tensor, loss_fn(cast(QuantileModel, model_)(x), y))

    if use_sage:
        warmup_epochs = min(tune.warmup_epochs, max(cfg.epochs - 1, 0))
        _run_supervised_epochs(model, labeled_loader, optimizer, warmup_epochs, supervised_loss)
        trainer = SelfAgreementTrainer(
            optimizer=optimizer,
            supervised_loss_fn=supervised_loss,
            predictive_batch_fn=_quantile_predictive_batch,
            augment_fn=_augment_fn(tune.unlabeled_noise),
            n_views=cfg.n_views,
            tau=tune.tau,
            agreement_weight=tune.agreement_weight,
            ema_decay=cfg.ema_decay,
        )
        history = trainer.fit(
            model,
            labeled_loader,
            unlabeled_loader,
            epochs=max(1, cfg.epochs - warmup_epochs),
        )
        return model.eval(), {
            "mean_weight": history["mean_weight"][-1],
            "mean_disagreement": history["mean_disagreement"][-1],
        }

    _run_supervised_epochs(model, labeled_loader, optimizer, cfg.epochs, supervised_loss)
    return model.eval(), {"mean_weight": 0.0, "mean_disagreement": 0.0}


def _bar_targets(target: Tensor, bin_edges: Tensor) -> Tensor:
    return torch.bucketize(target.view(-1), bin_edges[1:-1]).long()


def _train_bar(
    cfg: BackboneComparisonConfig, split, *, use_sage: bool
) -> tuple[BarModel, dict[str, float]]:
    y_train = torch.cat([split.y_labeled, split.y_unlabeled_true], dim=0)
    margin = 0.20
    bin_edges = torch.linspace(
        float(y_train.min().item()) - margin,
        float(y_train.max().item()) + margin,
        cfg.n_bins + 1,
    )
    tune = cfg.bar
    model = BarModel(cfg.hidden, tune.dropout, bin_edges)
    optimizer = torch.optim.Adam(model.parameters(), lr=tune.lr)
    labeled_loader, unlabeled_loader = _build_loaders(
        split.x_labeled,
        split.y_labeled,
        split.x_unlabeled,
        batch_size=cfg.batch_size,
    )

    def supervised_loss(model_: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        return cast(
            Tensor,
            F.cross_entropy(
                cast(BarModel, model_)(x),
                _bar_targets(y, bin_edges),
                label_smoothing=cfg.bar_label_smoothing,
            ),
        )

    if use_sage:
        warmup_epochs = min(tune.warmup_epochs, max(cfg.epochs - 1, 0))
        _run_supervised_epochs(model, labeled_loader, optimizer, warmup_epochs, supervised_loss)
        trainer = SelfAgreementTrainer(
            optimizer=optimizer,
            supervised_loss_fn=supervised_loss,
            predictive_batch_fn=_bar_predictive_batch,
            augment_fn=_augment_fn(tune.unlabeled_noise),
            n_views=cfg.n_views,
            tau=tune.tau,
            agreement_weight=tune.agreement_weight,
            ema_decay=cfg.ema_decay,
        )
        history = trainer.fit(
            model,
            labeled_loader,
            unlabeled_loader,
            epochs=max(1, cfg.epochs - warmup_epochs),
        )
        return model.eval(), {
            "mean_weight": history["mean_weight"][-1],
            "mean_disagreement": history["mean_disagreement"][-1],
        }

    _run_supervised_epochs(model, labeled_loader, optimizer, cfg.epochs, supervised_loss)
    return model.eval(), {"mean_weight": 0.0, "mean_disagreement": 0.0}


def _run_fraction(cfg: BackboneComparisonConfig, split, fraction: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    gaussian_supervised, g_sup_s = timed_call(_train_gaussian, cfg, split, use_sage=False)
    gaussian_sage, g_ssl_s = timed_call(_train_gaussian, cfg, split, use_sage=True)
    quantile_supervised, q_sup_s = timed_call(_train_quantile, cfg, split, use_sage=False)
    quantile_sage, q_ssl_s = timed_call(_train_quantile, cfg, split, use_sage=True)
    bar_supervised, b_sup_s = timed_call(_train_bar, cfg, split, use_sage=False)
    bar_sage, b_ssl_s = timed_call(_train_bar, cfg, split, use_sage=True)

    specs = [
        (
            "Gaussian",
            "Supervised",
            gaussian_supervised[0],
            gaussian_supervised[1],
            g_sup_s,
            _evaluate_gaussian,
        ),
        ("Gaussian", "SAGE-Reg", gaussian_sage[0], gaussian_sage[1], g_ssl_s, _evaluate_gaussian),
        (
            "Quantile",
            "Supervised",
            quantile_supervised[0],
            quantile_supervised[1],
            q_sup_s,
            _evaluate_quantile,
        ),
        ("Quantile", "SAGE-Reg", quantile_sage[0], quantile_sage[1], q_ssl_s, _evaluate_quantile),
        ("Bar", "Supervised", bar_supervised[0], bar_supervised[1], b_sup_s, _evaluate_bar),
        ("Bar", "SAGE-Reg", bar_sage[0], bar_sage[1], b_ssl_s, _evaluate_bar),
    ]

    for backbone, regime, model, meta, train_s, evaluator in specs:
        metrics, eval_s = timed_call(evaluator, model, split.x_test, split.y_test)
        rows.append(
            {
                "Method": f"{backbone} {regime}",
                "Backbone": backbone,
                "Regime": regime,
                "UnlabeledFraction": float(fraction),
                **metrics,
                "MeanWeight": float(meta["mean_weight"]),
                "MeanDisagreement": float(meta["mean_disagreement"]),
                "train_s": float(train_s),
                "eval_s": float(eval_s),
            }
        )
    return rows


def run_comparison(cfg: BackboneComparisonConfig) -> list[dict[str, object]]:
    base_split = generate_self_agreement_regression_split(cfg.data)
    rows: list[dict[str, object]] = []
    for fraction in cfg.unlabeled_fractions:
        split = _subset_unlabeled_split(base_split, float(fraction), seed=cfg.data.seed)
        rows.extend(_run_fraction(cfg, split, float(fraction)))
    return rows


def _rows_by_method(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        method = f"{row['Backbone']} {row['Regime']}"
        grouped.setdefault(method, []).append(row)
    for values in grouped.values():
        values.sort(key=lambda item: float(item["UnlabeledFraction"]))
    return grouped


def _plot_performance(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = _rows_by_method(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method, values in grouped.items():
        x = [float(value["UnlabeledFraction"]) for value in values]
        axes[0].plot(x, [float(value["CRPS"]) for value in values], marker="o", label=method)
        axes[1].plot(x, [float(value["RMSE"]) for value in values], marker="o", label=method)
    axes[0].set_title("CRPS vs unlabeled fraction")
    axes[0].set_xlabel("Unlabeled fraction")
    axes[0].set_ylabel("CRPS")
    axes[1].set_title("RMSE vs unlabeled fraction")
    axes[1].set_xlabel("Unlabeled fraction")
    axes[1].set_ylabel("RMSE")
    axes[1].legend(loc="best", fontsize=7)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_calibration(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = _rows_by_method(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method, values in grouped.items():
        x = [float(value["UnlabeledFraction"]) for value in values]
        axes[0].plot(
            x,
            [float(value["CoverageGap90"]) for value in values],
            marker="o",
            label=method,
        )
        axes[1].plot(x, [float(value["PITChi2"]) for value in values], marker="o", label=method)
    axes[0].set_title("Coverage gap vs unlabeled fraction")
    axes[0].set_xlabel("Unlabeled fraction")
    axes[0].set_ylabel("|Cov90 - 0.90|")
    axes[1].set_title("PIT chi-square vs unlabeled fraction")
    axes[1].set_xlabel("Unlabeled fraction")
    axes[1].set_ylabel("PITChi2")
    axes[1].legend(loc="best", fontsize=7)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _runtime_summary(rows: list[dict[str, object]], wall_s: float) -> str:
    train_total = sum(float(row["train_s"]) for row in rows)
    eval_total = sum(float(row["eval_s"]) for row in rows)
    fits = max(1, len(rows))
    return (
        "\nRuntime summary\n"
        f"  wall_s: {wall_s:.4f}\n"
        f"  train_total_s: {train_total:.4f}\n"
        f"  eval_total_s: {eval_total:.4f}\n"
        f"  mean_train_s_per_fit: {train_total / fits:.4f}\n"
        f"  mean_eval_s_per_fit: {eval_total / fits:.4f}"
    )


def main(
    cfg: BackboneComparisonConfig | None = None,
    *,
    output_csv: str | None = None,
    figure_path: str | None = None,
    performance_figure_path: str | None = None,
    calibration_figure_path: str | None = None,
    summary_json_path: str | None = None,
) -> list[dict[str, object]]:
    resolved = BackboneComparisonConfig() if cfg is None else cfg
    start = time.perf_counter()
    rows = run_comparison(resolved)
    wall_s = time.perf_counter() - start
    perf_path = performance_figure_path or figure_path
    calib_path = calibration_figure_path
    if perf_path is not None and calib_path is None:
        perf = Path(perf_path)
        calib_path = str(perf.with_name(f"{perf.stem}_calibration{perf.suffix or '.png'}"))
    print_fairness_notes(
        title="SAGE-Reg Backbone Comparison",
        seed_policy=f"single fixed seed ({resolved.data.seed}) across all backbones",
        train_budget=(
            f"same hidden width and same {resolved.epochs}-epoch cap per backbone; "
            "quantile/bar use modest backbone-specific warmup and regularization for stability"
        ),
        metric_policy="RMSE + distribution metrics report (NLL/CRPS/Cov90/Width90/PITChi2)",
    )
    print_comparison_summary(
        "SAGE-Reg Backbone Comparison",
        rows,
        metric_order=[
            "UnlabeledFraction",
            "RMSE",
            "NLL",
            "CRPS",
            "Cov90",
            "CoverageGap90",
            "Width90",
            "PITChi2",
            "MeanWeight",
            "MeanDisagreement",
            "train_s",
        ],
    )
    print(_runtime_summary(rows, wall_s))
    if output_csv:
        out = _write_csv(output_csv, rows)
        print(f"\nWrote CSV: {out}")
    if perf_path:
        out = _plot_performance(perf_path, rows)
        print(f"Wrote performance figure: {out}")
    if calib_path:
        out = _plot_calibration(calib_path, rows)
        print(f"Wrote calibration figure: {out}")
    if summary_json_path:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/benchmarks/self_agreement_backbone_comparison.py",
            task="self-agreement backbone comparison sweep on shared synthetic split",
            config=resolved,
            rows=rows,
            notes=[
                "Shared synthetic split from the SAGE-Reg benchmark generator.",
                "Compares supervised vs SAGE-Reg for Gaussian, quantile, and bar predictive families.",
                "Default tuning uses small backbone-specific warmup and regularization changes to stabilize quantile and bar predictors.",
                "MeanDisagreement reports the final average pairwise predictive disagreement used to derive SAGE-Reg sample weights.",
            ],
        )
        print(f"\nWrote summary JSON: {out}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SAGE-Reg cross-backbone comparison.")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--figure-path", type=str, default="")
    parser.add_argument("--performance-figure-path", type=str, default="")
    parser.add_argument("--calibration-figure-path", type=str, default="")
    parser.add_argument("--summary-json-path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=BackboneComparisonConfig.epochs)
    parser.add_argument("--n-bins", type=int, default=BackboneComparisonConfig.n_bins)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="*",
        default=list(BackboneComparisonConfig.unlabeled_fractions),
    )
    args = parser.parse_args()
    main(
        BackboneComparisonConfig(
            epochs=args.epochs,
            n_bins=args.n_bins,
            unlabeled_fractions=tuple(args.fractions),
        ),
        output_csv=args.output_csv or None,
        figure_path=args.figure_path or None,
        performance_figure_path=args.performance_figure_path or None,
        calibration_figure_path=args.calibration_figure_path or None,
        summary_json_path=args.summary_json_path or None,
    )
