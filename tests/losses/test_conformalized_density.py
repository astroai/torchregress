"""Tests for conformal level sets around conditional densities."""

from __future__ import annotations

import torch

from torchregress.losses import ConformalizedDensity, ContrastiveFlowConformal


def _normal_log_prob(mean: torch.Tensor, target: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    err = (target.reshape(-1, 1) - mean.reshape(-1, 1)) / sigma
    return (-0.5 * err**2 - torch.log(torch.tensor(sigma * (2 * torch.pi) ** 0.5))).reshape(-1)


def test_conformalized_density_scalar_coverage() -> None:
    torch.manual_seed(0)
    cal_ctx = torch.randn(600, 1)
    cal_y = cal_ctx + 0.5 * torch.randn(600, 1)
    cp = ConformalizedDensity(alpha=0.1, grid_size=200)
    cp.calibrate(lambda c, y: -_normal_log_prob(c, y), cal_ctx, cal_y)
    assert cp.threshold is not None
    test_ctx = torch.randn(400, 1)
    test_y = test_ctx + 0.5 * torch.randn(400, 1)
    contains = cp.contains(lambda c, y: -_normal_log_prob(c, y), test_ctx, test_y)
    assert 0.82 < float(contains.double().mean()) < 0.98
    lo, hi = cp.predict_interval(lambda c, y: -_normal_log_prob(c, y), test_ctx[:10], -4.0, 4.0)
    assert lo.shape == hi.shape == (10, 1)
    assert bool((lo < hi).all())


def test_contrastive_flow_conformal_level_set() -> None:
    torch.manual_seed(0)

    class _MockFlow:
        def log_prob(self, context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return _normal_log_prob(context, target)

    flow = _MockFlow()
    cal_ctx = torch.randn(300, 1)
    cal_y = cal_ctx + 0.5 * torch.randn(300, 1)
    cp = ContrastiveFlowConformal(flow, alpha=0.1, grid_size=100)
    cp.calibrate(cal_ctx, cal_y)
    assert cp.threshold is not None
    test_ctx = torch.randn(50, 1)
    test_y = test_ctx
    assert cp.contains(test_ctx, test_y).shape == (50,)
