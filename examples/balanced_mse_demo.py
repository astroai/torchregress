"""
Toy long-tailed regression: plain MSE vs BalancedMSELoss vs BMCLoss.

Training targets are mostly near 0 with a few large values; balanced losses
upweight rare bins so the linear fit does not ignore the tail entirely.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from torchregress.losses import BalancedMSELoss, BMCLoss, MSELoss


def make_skewed_targets(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(-1.0, 1.0, n).unsqueeze(1)
    # 90% of mass near 0, 10% large positive shift
    y = 0.5 * x + 0.1 * torch.randn(n, 1, generator=g)
    big = torch.rand(n, generator=g) < 0.1
    y = y + big.float().unsqueeze(1) * 2.5
    return x, y


def train_linear(
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    *,
    steps: int,
    lr: float,
) -> float:
    model = nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(x)
        return float(torch.mean((pred - y) ** 2).item())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.05)
    args = p.parse_args()

    x, y = make_skewed_targets(args.n, args.seed)
    lo, hi = float(y.min()), float(y.max())
    edges = torch.linspace(lo, hi, 11)

    bmc = BMCLoss(num_bins=10, noise_sigma=1.0, binning="equal").fit(y)
    bal = BalancedMSELoss(bin_edges=edges, count_smoothing=0.5).fit(y)
    mse = MSELoss()

    rmse_mse = train_linear(x, y, mse, steps=args.steps, lr=args.lr)
    rmse_bal = train_linear(x, y, bal, steps=args.steps, lr=args.lr)
    rmse_bmc = train_linear(x, y, bmc, steps=args.steps, lr=args.lr)

    print("Mean squared error on training set (lower is better):")
    print(f"  MSE loss:              {rmse_mse:.6f}")
    print(f"  BalancedMSELoss:       {rmse_bal:.6f}")
    print(f"  BMCLoss:               {rmse_bmc:.6f}")


if __name__ == "__main__":
    main()
