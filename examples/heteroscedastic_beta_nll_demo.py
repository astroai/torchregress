"""
Heteroscedastic synthetic regression: Gaussian NLL vs Beta-NLL.

Trains two identical heads from the same initial weights on noise whose scale
grows with |x|, then reports validation RMSE and validation Gaussian NLL (same
held-out metric for both runs).
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import BetaNLLLoss, GaussianNLLLoss


class GaussHeadMLP(nn.Module):
    """MLP with two outputs: mean and log-variance (scalar y)."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean, log_var = out[:, :1], out[:, 1:2]
        return mean, log_var


def make_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n, 1).uniform_(-2.0, 2.0, generator=g)
    noise_std = 0.15 + 0.45 * torch.abs(x[:, 0:1])
    eps = torch.randn(n, 1, generator=g) * noise_std
    y = 2.0 * x + eps
    return x, y


def train(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    epochs: int,
    lr: float,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            mean, log_var = model(xb)
            loss = loss_fn((mean, log_var), yb)
            loss.backward()
            opt.step()


@torch.no_grad()
def eval_gaussian_nll(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    mean, log_var = model(x)
    nll = GaussianNLLLoss(reduction="mean")((mean, log_var), y)
    return float(nll.item())


@torch.no_grad()
def rmse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    mean, _ = model(x)
    return float(torch.sqrt(torch.mean((mean - y) ** 2)).item())


def run_once(
    loss_name: str,
    *,
    epochs: int,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    init_state: dict[str, Any],
    lr: float,
) -> dict[str, float]:
    ds = TensorDataset(x_tr, y_tr)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = GaussHeadMLP()
    model.load_state_dict(init_state)
    if loss_name == "nll":
        crit: nn.Module = GaussianNLLLoss()
    elif loss_name == "beta_nll":
        crit = BetaNLLLoss(beta=0.5)
    else:
        raise ValueError(loss_name)
    train(model, crit, loader, epochs=epochs, lr=lr)
    return {"rmse": rmse(model, x_va, y_va), "val_nll": eval_gaussian_nll(model, x_va, y_va)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    n_total = 2000
    n_val = 500
    x, y = make_data(n_total, args.seed)
    x_tr, y_tr = x[:-n_val], y[:-n_val]
    x_va, y_va = x[-n_val:], y[-n_val:]

    init = GaussHeadMLP().state_dict()
    n_metrics = run_once(
        "nll",
        epochs=args.epochs,
        x_tr=x_tr,
        y_tr=y_tr,
        x_va=x_va,
        y_va=y_va,
        init_state=init,
        lr=args.lr,
    )
    b_metrics = run_once(
        "beta_nll",
        epochs=args.epochs,
        x_tr=x_tr,
        y_tr=y_tr,
        x_va=x_va,
        y_va=y_va,
        init_state=init,
        lr=args.lr,
    )

    print(
        "Gaussian NLL — val RMSE:",
        round(n_metrics["rmse"], 5),
        "  val NLL:",
        round(n_metrics["val_nll"], 5),
    )
    print(
        "Beta-NLL     — val RMSE:",
        round(b_metrics["rmse"], 5),
        "  val NLL:",
        round(b_metrics["val_nll"], 5),
    )


if __name__ == "__main__":
    main()
