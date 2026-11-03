"""
GaussianWassersteinBoundLoss demos: one loss step, and a W2-bound-then-NLL training schedule.

Modes (``--mode``):

``loss``
    Minimal batched mean + full-covariance supervision. Builds random SPD predicted
    covariances, compares them to a fixed spherical target covariance, and runs one
    backward step (``--batch``, ``--dim``).

``hybrid``
    Short ``GaussianWassersteinBoundLoss`` (diagonal) pretraining against neighbourhood
    covariance pseudo-labels, then ``GaussianNLLLoss`` fine-tuning. Uses a tiny
    ``nn.Linear(1, 2)`` head (mean + log-variance) on scalar synthetic data with
    heteroscedastic noise (``--pretrain-steps``, ``--finetune-steps``, ``--n``, ``--lr``).

``all`` (default)
    Runs ``loss`` then ``hybrid``. Each mode seeds itself, so its output matches a
    standalone run of that mode.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from torchregress.algorithms import (
    NeighborhoodCovarianceConfig,
    NeighborhoodCovariancePseudoLabeler,
)
from torchregress.losses import GaussianNLLLoss, GaussianWassersteinBoundLoss

MODES = ("loss", "hybrid")


def run_loss_step(*, batch: int, dim: int, seed: int) -> None:
    """One mean + full-covariance surrogate loss evaluation and backward step."""
    torch.manual_seed(seed)
    b, d = batch, dim
    mu_pred = torch.randn(b, d, requires_grad=True)
    mu_tgt = torch.randn(b, d)
    raw = torch.randn(b, d, d, requires_grad=True)
    # Pin ``torch.eye`` to ``raw`` so the demo fixture doesn't implicitly rely
    # on the loss module handling dtype/device of input fixtures internally.
    sig_pred = raw @ raw.transpose(-1, -2) + 0.25 * torch.eye(
        d, device=raw.device, dtype=raw.dtype
    ).expand(b, d, d)
    sig_tgt = torch.eye(d, device=raw.device, dtype=raw.dtype).expand(b, d, d) * 0.5

    fn = GaussianWassersteinBoundLoss(
        covariance_parameterization="covariance",
        mean_weight=1.0,
        covariance_weight=1.0,
        jitter=1e-5,
        reduction="mean",
    )
    loss = fn(mu_pred, mu_tgt, sig_pred, sig_tgt)
    loss.backward()
    assert mu_pred.grad is not None and raw.grad is not None
    print("mean+covariance surrogate loss:", float(loss.item()))
    print("grad norm (mu):", float(mu_pred.grad.norm().item()))
    print("grad norm (raw factor):", float(raw.grad.norm().item()))


class ScalarGaussianHead(nn.Module):
    """Maps ``x [*, 1]`` to mean and log-variance (scalar outputs)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.lin(x)
        return o[..., :1], o[..., 1:2]


def run_hybrid_pretrain(
    *, pretrain_steps: int, finetune_steps: int, n: int, lr: float, seed: int
) -> None:
    """W2-bound pretraining on pseudo-covariances, then Gaussian NLL fine-tuning."""
    torch.manual_seed(seed)
    x = torch.linspace(-2.0, 2.0, n).unsqueeze(-1)
    noise_scale = 0.15 * (1.0 + torch.abs(x))
    y = 0.8 * x + 0.3 * (x**2) + noise_scale * torch.randn_like(x)

    head = ScalarGaussianHead()
    labeler = NeighborhoodCovariancePseudoLabeler(
        NeighborhoodCovarianceConfig(n_neighbors=16, metric="euclidean", temperature=0.5)
    )
    cov_pseudo = labeler.fit_predict(x, y)
    target_var = cov_pseudo.squeeze(-1).squeeze(-1).clamp(min=1e-4).unsqueeze(-1)

    gw = GaussianWassersteinBoundLoss(
        covariance_parameterization="diagonal",
        mean_weight=1.0,
        covariance_weight=1.0,
        reduction="mean",
    )
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(pretrain_steps):
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
    for _ in range(finetune_steps):
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


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=(*MODES, "all"), default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=16, help="loss mode: batch size")
    p.add_argument("--dim", type=int, default=3, help="loss mode: Gaussian dimension")
    p.add_argument("--pretrain-steps", type=int, default=80, help="hybrid mode")
    p.add_argument("--finetune-steps", type=int, default=120, help="hybrid mode")
    p.add_argument("--n", type=int, default=256, help="hybrid mode: number of points")
    p.add_argument("--lr", type=float, default=0.05, help="hybrid mode: Adam learning rate")
    args = p.parse_args()

    modes = MODES if args.mode == "all" else (args.mode,)
    for mode in modes:
        if len(modes) > 1:
            print(f"=== mode: {mode} ===")
        if mode == "loss":
            run_loss_step(batch=args.batch, dim=args.dim, seed=args.seed)
        else:
            run_hybrid_pretrain(
                pretrain_steps=args.pretrain_steps,
                finetune_steps=args.finetune_steps,
                n=args.n,
                lr=args.lr,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()
