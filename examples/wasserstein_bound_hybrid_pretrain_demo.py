"""
Hybrid schedule: short ``GaussianWassersteinBoundLoss`` (diagonal) pretraining against
neighbourhood covariance pseudo-labels, then ``GaussianNLLLoss`` fine-tuning.

Uses a tiny ``nn.Linear(1, 2)`` head (mean + log-variance) on scalar synthetic data with
heteroscedastic noise. Demonstrates the recipe from ``docs/research/plans/Wasserstein_Supervision.md``
Phase 3 without a full trainer abstraction.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from torchregress.algorithms import NeighborhoodCovariancePseudoLabeler
from torchregress.losses import GaussianNLLLoss, GaussianWassersteinBoundLoss


class ScalarGaussianHead(nn.Module):
    """Maps ``x [*, 1]`` to mean and log-variance (scalar outputs)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.lin(x)
        return o[..., :1], o[..., 1:2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrain-steps", type=int, default=80)
    p.add_argument("--finetune-steps", type=int, default=120)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    x = torch.linspace(-2.0, 2.0, args.n).unsqueeze(-1)
    noise_scale = 0.15 * (1.0 + torch.abs(x))
    y = 0.8 * x + 0.3 * (x**2) + noise_scale * torch.randn_like(x)

    head = ScalarGaussianHead()
    labeler = NeighborhoodCovariancePseudoLabeler(n_neighbors=16, metric="euclidean", temperature=0.5)
    cov_pseudo = labeler.fit_predict(x, y)
    target_var = cov_pseudo.squeeze(-1).squeeze(-1).clamp(min=1e-4).unsqueeze(-1)

    gw = GaussianWassersteinBoundLoss(
        covariance_parameterization="diagonal",
        mean_weight=1.0,
        covariance_weight=1.0,
        reduction="mean",
    )
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    for _ in range(args.pretrain_steps):
        opt.zero_grad(set_to_none=True)
        mu, logv = head(x)
        v = torch.exp(logv).clamp(min=1e-4)
        loss = gw(mu, y, v, target_var)
        loss.backward()
        opt.step()

    with torch.no_grad():
        mu0, logv0 = head(x)
        nll0 = GaussianNLLLoss(reduction="mean")(torch.cat([mu0, logv0], dim=-1), y)

    nll_loss = GaussianNLLLoss(reduction="mean")
    for _ in range(args.finetune_steps):
        opt.zero_grad(set_to_none=True)
        mu, logv = head(x)
        loss = nll_loss(torch.cat([mu, logv], dim=-1), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        mu1, logv1 = head(x)
        nll1 = nll_loss(torch.cat([mu1, logv1], dim=-1), y)

    print("NLL after W2-bound pretrain:", round(float(nll0.item()), 4))
    print("NLL after GaussianNLL fine-tune:", round(float(nll1.item()), 4))


if __name__ == "__main__":
    main()
