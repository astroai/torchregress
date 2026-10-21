"""Tests for conformal level sets around conditional densities."""

from __future__ import annotations

import pytest
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


def test_mondrian_conformal_density() -> None:
    torch.manual_seed(0)
    ctx = torch.randn(300, 1)
    y = ctx + 0.5 * torch.randn(300, 1)
    groups = (ctx[:, 0] > 0).long()
    cp = ConformalizedDensity(alpha=0.1, grid_size=100)
    cp.calibrate(lambda c, v: -_normal_log_prob(c, v), ctx, y, groups=groups)
    test_ctx = torch.randn(60, 1)
    test_y = test_ctx + 0.5 * torch.randn(60, 1)
    test_groups = (test_ctx[:, 0] > 0).long()
    assert cp.contains(
        lambda c, v: -_normal_log_prob(c, v), test_ctx, test_y, groups=test_groups
    ).shape == (60,)
    lo, hi = cp.predict_interval(
        lambda c, v: -_normal_log_prob(c, v),
        test_ctx[:10],
        -4,
        4,
        groups=test_groups[:10],
    )
    assert lo.shape == hi.shape == (10, 1)


def test_contrastive_flow_interval_matches_negative_log_density() -> None:
    """The flow wrapper must preserve the base level-set score's sign."""
    torch.manual_seed(17)

    class _MockFlow:
        def log_prob(self, context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return _normal_log_prob(context, target)

    flow = _MockFlow()
    ctx = torch.zeros(1000, 1)
    y = 0.5 * torch.randn_like(ctx)
    wrapped = ContrastiveFlowConformal(flow, alpha=0.1, grid_size=501)
    base = ConformalizedDensity(alpha=0.1, grid_size=501)
    wrapped.calibrate(ctx, y)
    base.calibrate_log_prob(flow.log_prob, ctx, y)
    expected = base.predict_interval(lambda c, v: -flow.log_prob(c, v), ctx[:1], -4, 4)
    actual = wrapped.predict_interval(ctx[:1], -4, 4)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert -1.0 < float(actual[0].item()) < -0.5
    assert 0.5 < float(actual[1].item()) < 1.0


def test_mondrian_density_membership_requires_prediction_groups() -> None:
    ctx = torch.zeros(20, 1)
    y = torch.linspace(-1, 1, 20).reshape(-1, 1)
    groups = torch.arange(20) % 2
    cp = ConformalizedDensity(alpha=0.1)
    cp.calibrate_log_prob(_normal_log_prob, ctx, y, groups=groups)
    with pytest.raises(ValueError, match="groups must be provided"):
        cp.contains(lambda c, v: -_normal_log_prob(c, v), ctx, y)
    expected_q = cp.cti._per_sample_q(len(y), groups=groups, device=y.device, dtype=y.dtype)
    assert torch.equal(
        cp.contains(lambda c, v: -_normal_log_prob(c, v), ctx, y, groups=groups),
        -_normal_log_prob(ctx, y) <= expected_q,
    )
