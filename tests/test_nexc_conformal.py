"""Unit tests for the weighted NexCP split conformal regressor (ICLR MTTA)."""

from __future__ import annotations

import pytest
import torch

from torchregress.losses.conformal import (
    NonExchangeableConformalRegressor,
    finite_sample_quantile,
)


def test_uniform_weights_recover_finite_sample_quantile() -> None:
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        scores = torch.randn(300, dtype=dtype)
        alpha = 0.1
        nexc = NonExchangeableConformalRegressor(alpha).calibrate(
            scores, torch.ones(300, dtype=dtype)
        )
        assert float(nexc.threshold_) == float(finite_sample_quantile(scores, alpha))
        # None-weights path is identical by construction.
        nexc_none = NonExchangeableConformalRegressor(alpha).calibrate(scores)
        assert float(nexc_none.threshold_) == float(finite_sample_quantile(scores, alpha))
        assert nexc.weight_tv_gap_ == 0.0
        lo, hi = nexc.two_sided_coverage_bounds()
        assert lo == 1.0 - alpha
        assert hi == pytest.approx(1.0 - alpha + 1.0 / 300.0)


def test_two_sided_bounds_contain_empirical() -> None:
    """Covariate shift with known density ratio; bounds must contain coverage.

    Over 200 seeded runs of a shifted Gaussian location family (source x~N(0,1),
    target x~N(1,1), heteroscedastic residual noise sigma(x)=0.3+0.2|x|), empirical
    target-side coverage of the weighted interval must lie inside
    ``two_sided_coverage_bounds()`` for at least 95% of seeds.
    """
    alpha = 0.1
    shift = 1.0
    n_cal, n_test = 250, 4000
    inside = 0
    for seed in range(200):
        g = torch.Generator().manual_seed(seed)
        x_cal = torch.randn(n_cal, generator=g)
        # Known density ratio w(x) = exp(shift*x - shift^2/2).
        w_cal = torch.exp(shift * x_cal - 0.5 * shift**2)
        nexc = NonExchangeableConformalRegressor(alpha).calibrate(x_cal.abs(), w_cal)

        x_new = torch.randn(n_test, generator=g) + shift
        r_new = (0.3 + 0.2 * x_new.abs()) * torch.randn(n_test, generator=g)
        emp = float((r_new <= float(nexc.threshold_)).float().mean())
        lo, hi = nexc.two_sided_coverage_bounds()
        if lo <= emp <= hi:
            inside += 1
    assert inside >= 190  # >=95% of 200 seeds


def test_interval_from_model_matches_manual_scores() -> None:
    torch.manual_seed(1)
    X = torch.randn(120, 4)
    y = X.sum(dim=-1, keepdim=True) + 0.1 * torch.randn(120, 1)
    Xt = torch.randn(40, 4)

    class Const(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 1)

    model = Const().eval()
    nexc = NonExchangeableConformalRegressor(0.2)
    lo, hi = nexc.interval_from_model(model, X, y, Xt, alpha=0.2)
    ref = float(finite_sample_quantile(y.reshape(-1).abs(), 0.2))
    assert torch.allclose(hi - lo, torch.full_like(lo, 2.0 * ref))


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        NonExchangeableConformalRegressor(0.0)
    with pytest.raises(ValueError):
        NonExchangeableConformalRegressor(1.5)
    nexc = NonExchangeableConformalRegressor(0.1)
    with pytest.raises(RuntimeError):
        nexc.two_sided_coverage_bounds()
    with pytest.raises(ValueError):
        nexc.calibrate(torch.randn(10), torch.ones(11))
    with pytest.raises(ValueError):
        nexc.calibrate(torch.randn(10), -torch.ones(10))
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError):
        nexc.interval_from_model(model, torch.randn(5, 2), torch.randn(7, 1), torch.randn(3, 2))
