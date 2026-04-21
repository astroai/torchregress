"""Synthetic checks for Wasserstein-bound loss with pseudo-covariance targets (plan 9.3)."""

from __future__ import annotations

import torch

from torchregress.algorithms import NeighborhoodCovarianceConfig, NeighborhoodCovariancePseudoLabeler
from torchregress.losses import GaussianWassersteinBoundLoss


def test_diagonal_gw_zero_when_mean_and_variance_match_targets() -> None:
    y = torch.tensor([[0.5], [1.0], [-0.2]])
    v = torch.tensor([[0.25], [0.16], [0.09]])
    gw = GaussianWassersteinBoundLoss(
        covariance_parameterization="diagonal",
        reduction="mean",
    )
    out = gw(y, y, v, v)
    assert out.item() < 1e-5


def test_diagonal_gw_with_pseudo_covariance_targets_is_finite_and_differentiable() -> None:
    torch.manual_seed(0)
    x = torch.randn(24, 1, requires_grad=False)
    y = 0.5 * x + 0.2 * torch.randn(24, 1)
    cov = NeighborhoodCovariancePseudoLabeler(NeighborhoodCovarianceConfig(n_neighbors=6, metric="euclidean")).fit_predict(x, y)
    target_var = cov.squeeze(-1).squeeze(-1).clamp(min=1e-3).unsqueeze(-1).detach()

    mu = torch.randn(24, 1, requires_grad=True)
    logv = torch.zeros(24, 1, requires_grad=True)
    v = torch.exp(logv).clamp(min=1e-3)
    gw = GaussianWassersteinBoundLoss(
        covariance_parameterization="diagonal",
        reduction="mean",
    )
    loss = gw(mu, y.detach(), v, target_var)
    assert torch.isfinite(loss)
    loss.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
