import argparse
import csv
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

from torchregress.losses import (
    BaseEIVLoss,
    CauchyLoss,
    CVaRLoss,
    DensityWeightedLoss,
    EnsembleEIVLoss,
    ExpectileLoss,
    FunctionalEIVLoss,
    GaussianNLLLoss,
    HuberLoss,
    MSELoss,
    QuantileLoss,
    TukeyBiweightLoss,
)


class RegressionDataset(Dataset):
    def __init__(self, x: torch.Tensor, y_obs: torch.Tensor, y_true: torch.Tensor) -> None:
        self.x = x
        self.y_obs = y_obs
        self.y_true = y_true

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_obs[idx], self.y_true[idx], idx


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_data(
    n_samples: int,
    noise_scale: float,
    feature_noise: float,
    label_noise: float,
    tail_quantile: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    x_true = torch.empty(n_samples, 1).uniform_(-2.0, 2.0, generator=rng)
    y_true = x_true**3 + 0.5 * x_true

    if feature_noise > 0:
        x_obs = x_true + torch.randn(x_true.shape, generator=rng) * feature_noise
    else:
        x_obs = x_true

    hetero = noise_scale * (1.0 + 0.5 * torch.abs(x_true))
    y_obs = y_true + torch.randn(y_true.shape, generator=rng) * hetero

    if label_noise > 0:
        threshold = torch.quantile(torch.abs(y_true), tail_quantile)
        tail_mask = torch.abs(y_true) >= threshold
        noisy_mask = (torch.rand(y_true.shape, generator=rng) < label_noise) & tail_mask
        heavy_noise = torch.distributions.StudentT(df=2).sample(y_true.shape) * noise_scale * 5.0
        y_obs = torch.where(noisy_mask, y_obs + heavy_noise, y_obs)
    else:
        tail_mask = torch.abs(y_true) >= torch.quantile(torch.abs(y_true), tail_quantile)

    return x_obs, y_obs, y_true, tail_mask, hetero.mean()


def split_data(
    train_size: int,
    test_size: int,
    noise_scale: float,
    feature_noise: float,
    label_noise: float,
    tail_quantile: float,
    seed: int,
):
    x_train, y_obs_train, y_true_train, _, y_sigma_mean = make_data(
        train_size, noise_scale, feature_noise, label_noise, tail_quantile, seed
    )
    x_test, y_obs_test, y_true_test, tail_mask, _ = make_data(
        test_size, noise_scale, feature_noise, label_noise, tail_quantile, seed + 1
    )
    return (
        (x_train, y_obs_train, y_true_train),
        (x_test, y_obs_test, y_true_test, tail_mask),
        y_sigma_mean,
    )


def train_model(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    use_indices: bool = False,
) -> None:
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        for x, y_obs, _, idx in train_loader:
            x = x.to(device)
            y_obs = y_obs.to(device)
            optimizer.zero_grad()
            if isinstance(loss_fn, BaseEIVLoss):
                loss = loss_fn(x, y_obs)
            else:
                preds = model(x)
                if use_indices:
                    loss = loss_fn(preds, y_obs, sample_indices=idx.to(device))
                else:
                    loss = loss_fn(preds, y_obs)
            loss.backward()
            optimizer.step()


def predict(model: nn.Module, x: torch.Tensor, is_gaussian: bool) -> torch.Tensor:
    preds = model(x)
    if is_gaussian:
        mean, _ = torch.chunk(preds, 2, dim=-1)
        return mean
    return preds


def compute_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, tail_mask: torch.Tensor
) -> Dict[str, float]:
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    tail_mask = tail_mask.view(-1)

    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
    mae = torch.mean(torch.abs(y_pred - y_true))

    tail_pred = y_pred[tail_mask]
    tail_true = y_true[tail_mask]
    tail_rmse = torch.sqrt(torch.mean((tail_pred - tail_true) ** 2))
    tail_mae = torch.mean(torch.abs(tail_pred - tail_true))

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "tail_rmse": float(tail_rmse),
        "tail_mae": float(tail_mae),
        "tail_rmse_ratio": float(tail_rmse / rmse),
    }


@dataclass
class Method:
    name: str
    build: Callable[[], Tuple[nn.Module, nn.Module, bool, bool]]


def build_methods(train_targets: torch.Tensor, sigma_x: float, sigma_y: float) -> List[Method]:
    density_loss = DensityWeightedLoss(base_loss="huber", reweight_factor=1.0)
    density_loss.fit_density(train_targets)

    def build_functional_eiv():
        model = MLP(1, 1)
        loss = FunctionalEIVLoss(
            model=model,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
        )
        return model, loss, False, False

    def build_ensemble_eiv():
        model = MLP(1, 1)
        loss = EnsembleEIVLoss(
            model=model,
            sigma_x=sigma_x,
            n_samples=20,
        )
        return model, loss, False, False

    return [
        Method(
            "MSE",
            lambda: (MLP(1, 1), MSELoss(reduction="mean"), False, False),
        ),
        Method(
            "Huber",
            lambda: (MLP(1, 1), HuberLoss(reduction="mean"), False, False),
        ),
        Method(
            "Cauchy",
            lambda: (MLP(1, 1), CauchyLoss(reduction="mean", c=0.5), False, False),
        ),
        Method(
            "Tukey",
            lambda: (MLP(1, 1), TukeyBiweightLoss(reduction="mean", c=4.685), False, False),
        ),
        Method(
            "Quantile(0.9)",
            lambda: (MLP(1, 1), QuantileLoss(quantile=0.9), False, False),
        ),
        Method(
            "Expectile(0.9)",
            lambda: (MLP(1, 1), ExpectileLoss(expectile=0.9), False, False),
        ),
        Method(
            "DensityWeightedHuber",
            lambda: (MLP(1, 1), density_loss, False, True),
        ),
        Method(
            "CVaR-Huber(0.1)",
            lambda: (MLP(1, 1), CVaRLoss(alpha=0.1, base_loss="huber"), False, False),
        ),
        Method(
            "GaussianNLL",
            lambda: (MLP(1, 2), GaussianNLLLoss(), True, False),
        ),
        Method("FunctionalEIV", build_functional_eiv),
        Method("EnsembleEIV", build_ensemble_eiv),
    ]


def format_results(rows: List[Dict[str, float]]) -> str:
    headers = ["method", "rmse", "mae", "tail_rmse", "tail_mae", "tail_rmse_ratio"]
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            if h == "method":
                col_width = len(row[h])
            else:
                col_width = len(f"{row[h]:.4f}")
            col_widths[h] = max(col_widths[h], col_width)

    lines = []
    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in rows:
        cells = []
        for h in headers:
            if h == "method":
                cells.append(row[h].ljust(col_widths[h]))
            else:
                cells.append(f"{row[h]:.4f}".ljust(col_widths[h]))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tail performance benchmark for regression losses."
    )
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--feature-noise", type=float, default=0.1)
    parser.add_argument("--label-noise", type=float, default=0.2)
    parser.add_argument("--tail-quantile", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-csv", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)

    (
        (x_train, y_obs_train, y_true_train),
        (x_test, _, y_true_test, tail_mask),
        y_sigma_mean,
    ) = split_data(
        args.train_size,
        args.test_size,
        args.noise_scale,
        args.feature_noise,
        args.label_noise,
        args.tail_quantile,
        args.seed,
    )

    train_ds = RegressionDataset(x_train, y_obs_train, y_true_train)
    test_x = x_test.to(device)
    test_y_true = y_true_test.to(device)
    tail_mask = tail_mask.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    rows: List[Dict[str, float]] = []
    for method in build_methods(y_obs_train, args.feature_noise, float(y_sigma_mean)):
        model, loss_fn, is_gaussian, needs_indices = method.build()
        if isinstance(loss_fn, BaseEIVLoss):
            model = loss_fn.model
        train_model(model, loss_fn, train_loader, args.epochs, device, use_indices=needs_indices)
        model.eval()
        with torch.no_grad():
            preds = predict(model.to(device), test_x, is_gaussian)
        metrics = compute_metrics(preds, test_y_true, tail_mask)
        metrics["method"] = method.name
        rows.append(metrics)

    rows = sorted(rows, key=lambda r: r["tail_rmse"])
    print(format_results(rows))

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "rmse",
                    "mae",
                    "tail_rmse",
                    "tail_mae",
                    "tail_rmse_ratio",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


if __name__ == "__main__":
    main()
