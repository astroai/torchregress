"""Tests for the WALDO Wald-statistic confidence region."""

from __future__ import annotations

import math

import torch

from torchregress.losses import WaldoConformal, waldo_statistic


def test_waldo_statistic_shapes() -> None:
    mean = torch.zeros(4, 2)
    target = torch.ones(4, 2)
    var = torch.ones(4, 2)
    assert torch.allclose(waldo_statistic(mean, target, var), torch.full((4,), 2.0))
    cov = torch.eye(2, dtype=mean.dtype, device=mean.device).expand(4, 2, 2)
    assert torch.allclose(
        waldo_statistic(mean, target, cov, covariance_type="full"), torch.full((4,), 2.0)
    )


def test_waldo_scalar_coverage_and_interval() -> None:
    torch.manual_seed(0)
    mean = torch.randn(800, 1)
    std = torch.full((800, 1), 0.5)
    target = mean + std * torch.randn(800, 1)
    cp = WaldoConformal(alpha=0.1)
    cp.calibrate(mean, target, std**2)
    assert cp.threshold is not None
    test_mean = torch.randn(500, 1)
    test_std = torch.full((500, 1), 0.5)
    test_target = test_mean + test_std * torch.randn(500, 1)
    cov = cp.contains(test_mean, test_target, test_std**2)
    assert 0.82 < float(cov.double().mean()) < 0.98
    lo, hi = cp.predict_interval(test_mean[:10], test_std[:10], y_min=-6, y_max=6)
    assert lo.shape == hi.shape == (10, 1)
    assert bool((lo < hi).all())


def test_waldo_multi_target_region() -> None:
    torch.manual_seed(0)
    mean = torch.randn(300, 2)
    var = torch.ones(300, 2)
    target = mean + var.sqrt() * torch.randn(300, 2)
    cp = WaldoConformal(alpha=0.1, critical_bins=5)
    cp.calibrate(mean, target, var)
    assert cp.threshold is not None
    grid = torch.randn(20, 2)
    region = cp.predict_region(mean[:5], grid)
    assert region.shape == (5, 20)
    assert region.dtype == torch.bool
    assert cp.contains(mean[:5], target[:5], var[:5]).shape == (5,)


def test_waldo_posterior_calibrator() -> None:
    torch.manual_seed(0)
    from torchregress.losses import WaldoPosteriorCalibrator

    n, s, d = 200, 128, 2
    samples = torch.randn(n, s, d) * 0.5 + torch.randn(n, 1, d)
    target = samples.mean(dim=1) + 0.5 * torch.randn(n, d)
    cp = WaldoPosteriorCalibrator(alpha=0.1, covariance_type="full")
    cp.calibrate(samples, target)
    assert cp.threshold is not None
    assert cp.contains(samples[:10], target[:10]).shape == (10,)
    grid = torch.randn(20, d)
    assert cp.predict_region(samples[:5], grid).shape == (5, 20)


def test_waldo_density_conformal() -> None:
    torch.manual_seed(0)
    from torchregress.losses import WaldoDensityConformal

    def sample_fn(context: torch.Tensor, n: int) -> torch.Tensor:
        return context[:, :1].unsqueeze(1) + 0.5 * torch.randn(context.shape[0], n, 1)

    ctx = torch.randn(300, 2)
    target = sample_fn(ctx, 1)[:, 0]
    cp = WaldoDensityConformal(alpha=0.1, covariance_type="full", n_samples=128)
    cp.calibrate(sample_fn, ctx, target)
    assert cp.threshold is not None
    test_ctx = torch.randn(50, 2)
    test_target = sample_fn(test_ctx, 1)[:, 0]
    assert cp.contains(sample_fn, test_ctx, test_target).shape == (50,)
    lo, hi = cp.predict_interval(sample_fn, test_ctx[:5], y_min=-4, y_max=4)
    assert lo.shape == hi.shape == (5, 1)


def test_waldo_critical_value_regressor_fit_predict() -> None:
    from torchregress.losses.waldo import WaldoCriticalValueRegressor

    torch.manual_seed(0)
    features = torch.linspace(-2, 2, 300).reshape(-1, 1)
    # Heteroscedastic score: critical value grows with |target|.
    scores = 1.0 + 0.5 * features[:, 0] + 0.1 * torch.randn(300)
    reg = WaldoCriticalValueRegressor(alpha=0.1, n_bins=8, min_samples_per_bin=5)
    reg.fit(features, scores)
    pred = reg.predict_critical(features)
    assert pred.shape == (300,)
    assert torch.isfinite(pred).all()
    # Local quantiles should track the heteroscedastic trend.
    assert pred[-10:].mean() > pred[:10].mean()


def test_waldo_conformal_with_regressor_coverage() -> None:
    from torchregress.losses.waldo import WaldoConformal, WaldoCriticalValueRegressor

    torch.manual_seed(0)
    n = 600
    target = torch.randn(n, 1) * 1.5
    mean = torch.zeros(n, 1)
    var = (0.4 + 0.2 * target.abs()).clamp_min(0.05)
    reg = WaldoCriticalValueRegressor(alpha=0.1, n_bins=6, min_samples_per_bin=10)
    cp = WaldoConformal(alpha=0.1, critical_value_regressor=reg, covariance_type="diagonal")
    cp.calibrate(mean[:300], target[:300], var[:300])
    inside = cp.contains(mean[300:], target[300:], var[300:])
    assert inside.shape == (300,)
    coverage = inside.float().mean().item()
    assert 0.75 <= coverage <= 1.0
    assert cp.critical_value(target[:5]).shape == (5,)


def test_waldo_threshold_is_finite_sample_order_statistic() -> None:
    """q_hat is the ceil((n+1)(1-alpha))-th smallest score, not an interpolated quantile."""
    torch.manual_seed(1)
    n, d, alpha = 50, 2, 0.1
    mean = torch.zeros(n, d, dtype=torch.float64)
    target = torch.randn(n, d, dtype=torch.float64)
    var = torch.ones(n, d, dtype=torch.float64)
    cp = WaldoConformal(alpha=alpha)
    cp.calibrate(mean, target, var)
    scores = waldo_statistic(mean, target, var)
    k = math.ceil((n + 1) * (1 - alpha))
    assert cp.q_hat is not None
    assert torch.equal(cp.q_hat, torch.sort(scores).values[k - 1])
    assert cp.q_hat > torch.quantile(scores, 1 - alpha)


def test_waldo_binned_critical_value_is_inf_not_nan_for_tiny_bins() -> None:
    """Bins too small for alpha give +inf thresholds; interpolation must not produce nan."""
    torch.manual_seed(2)
    n, d = 40, 1
    mean = torch.zeros(n, d, dtype=torch.float64)
    target = torch.randn(n, d, dtype=torch.float64)
    var = torch.ones(n, d, dtype=torch.float64)
    cp = WaldoConformal(alpha=0.01, critical_bins=4)  # ~10 points per bin < 1/alpha
    cp.calibrate(mean, target, var)
    crit = cp.critical_value(torch.linspace(-2, 2, 9, dtype=torch.float64).reshape(-1, 1))
    assert not torch.isnan(crit).any()
    assert torch.isinf(crit).all()
