"""
Minimal demo: GaussianWassersteinBoundLoss for batched mean + full covariance supervision.

Builds random SPD predicted covariances, compares them to a fixed spherical target
covariance, and runs one backward step.
"""

from __future__ import annotations

import argparse

import torch

from torchregress.losses import GaussianWassersteinBoundLoss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    b, d = args.batch, args.dim
    mu_pred = torch.randn(b, d, requires_grad=True)
    mu_tgt = torch.randn(b, d)
    raw = torch.randn(b, d, d, requires_grad=True)
    sig_pred = raw @ raw.transpose(-1, -2) + 0.25 * torch.eye(d).expand(b, d, d)
    sig_tgt = torch.eye(d).expand(b, d, d) * 0.5

    fn = GaussianWassersteinBoundLoss(
        covariance_parameterization="covariance",
        mean_weight=1.0,
        covariance_weight=1.0,
        jitter=1e-5,
        reduction="mean",
    )
    loss = fn(mu_pred, mu_tgt, sig_pred, sig_tgt)
    loss.backward()
    print("mean+covariance surrogate loss:", float(loss.item()))
    print("grad norm (mu):", float(mu_pred.grad.norm().item()))
    print("grad norm (raw factor):", float(raw.grad.norm().item()))


if __name__ == "__main__":
    main()
