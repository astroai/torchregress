"""Tests for wifi ensembles (frequentist uncertainties on learned density ratios)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.optimize as so
import torch
from torch import nn

from torchregress.inference import MixtureFractionEstimate, WifiRatioEnsemble

# Numerator p = N(MU, 1), denominator q = N(0, 1): log r(x) = MU x - MU^2 / 2.
MU = 1.0


def _true_log_ratio(x: torch.Tensor) -> torch.Tensor:
    return MU * x - 0.5 * MU**2


def _basis(x: torch.Tensor) -> torch.Tensor:
    """Members whose span (with the constant) contains the true log ratio."""
    return torch.stack([x, torch.tanh(x), x**2 / (1.0 + x**2)], dim=1).to(torch.float64)


def _samples(generator: torch.Generator, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_p = MU + torch.randn(n, generator=generator, dtype=torch.float64)
    x_q = torch.randn(n, generator=generator, dtype=torch.float64)
    return x_p, x_q


def test_newton_fit_matches_scipy_minimiser() -> None:
    generator = torch.Generator().manual_seed(0)
    x_p, x_q = _samples(generator, 800)
    f_p, f_q = _basis(x_p), _basis(x_q)
    wifi = WifiRatioEnsemble().fit(f_p, f_q)
    assert wifi.converged_

    design_p = np.column_stack([np.ones(len(x_p)), f_p.numpy()])
    design_q = np.column_stack([np.ones(len(x_q)), f_q.numpy()])

    def loss(w: np.ndarray) -> float:
        s_p, s_q = design_p @ w, design_q @ w
        return float(np.mean(np.exp(-s_p) - s_p) + np.mean(np.exp(s_q) + s_q))

    reference = so.minimize(loss, np.zeros(4), method="BFGS", options={"gtol": 1e-10})
    np.testing.assert_allclose(wifi.weights_.numpy(), reference.x, atol=1e-5)


def test_recovers_true_ratio_with_large_samples() -> None:
    generator = torch.Generator().manual_seed(1)
    x_p, x_q = _samples(generator, 200_000)
    wifi = WifiRatioEnsemble().fit(_basis(x_p), _basis(x_q))
    grid = torch.linspace(-2.0, 3.0, 11, dtype=torch.float64)
    error = (wifi.log_ratio(_basis(grid)) - _true_log_ratio(grid)).abs().max()
    assert float(error) < 0.03
    # Standard errors shrink like 1/sqrt(n).
    assert float(wifi.log_ratio_std(_basis(grid)).max()) < 0.03


def test_log_ratio_intervals_have_nominal_coverage() -> None:
    generator = torch.Generator().manual_seed(2)
    points = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    truth = _true_log_ratio(points)
    z_scores = []
    for _ in range(300):
        x_p, x_q = _samples(generator, 2000)
        wifi = WifiRatioEnsemble().fit(_basis(x_p), _basis(x_q))
        estimate = wifi.log_ratio(_basis(points))
        z_scores.append((estimate - truth) / wifi.log_ratio_std(_basis(points)))
    z = torch.stack(z_scores).abs()
    one_sigma = (z < 1.0).to(torch.float64).mean(dim=0)
    two_sigma = (z < 2.0).to(torch.float64).mean(dim=0)
    # Binomial sd over 300 trials is ~0.027 at 68% and ~0.012 at 95%.
    assert float((one_sigma - 0.6827).abs().max()) < 0.07
    assert float((two_sigma - 0.9545).abs().max()) < 0.04


def test_mixture_fraction_coverage_needs_the_ratio_term() -> None:
    # Few ratio-fit samples, many mixture samples: the ratio uncertainty
    # dominates, so statistical-only errors under-cover while the
    # Gong-Samaniego total stays nominal.
    generator = torch.Generator().manual_seed(3)
    kappa_true = 0.3
    total_hits, statistical_hits = [], []
    for _ in range(300):
        x_p, x_q = _samples(generator, 400)
        wifi = WifiRatioEnsemble().fit(_basis(x_p), _basis(x_q))
        n_mix = 8000
        from_p = torch.rand(n_mix, generator=generator) < kappa_true
        x_mix = torch.randn(n_mix, generator=generator, dtype=torch.float64) + MU * from_p
        estimate = wifi.mixture_fraction(_basis(x_mix))
        assert not estimate.at_boundary
        error = abs(estimate.fraction - kappa_true)
        total_hits.append(error < estimate.std)
        statistical_hits.append(error < estimate.std_statistical)
    total = float(np.mean(total_hits))
    statistical = float(np.mean(statistical_hits))
    assert abs(total - 0.6827) < 0.07
    assert statistical < total - 0.1


def test_mixture_fraction_matches_closed_form_for_known_ratio() -> None:
    # With f = true log ratio as the single member, w ~ (0, 1) and kappa-hat is
    # the root of sum (r - 1) / (1 + kappa (r - 1)).
    generator = torch.Generator().manual_seed(4)
    x_p, x_q = _samples(generator, 50_000)
    wifi = WifiRatioEnsemble().fit(_true_log_ratio(x_p), _true_log_ratio(x_q))
    from_p = torch.rand(20_000, generator=generator) < 0.6
    x_mix = torch.randn(20_000, generator=generator, dtype=torch.float64) + MU * from_p
    estimate = wifi.mixture_fraction(_true_log_ratio(x_mix))
    ratio = torch.exp(wifi.log_ratio(_true_log_ratio(x_mix))).numpy()
    root = so.brentq(lambda k: float(np.sum((ratio - 1) / (1 + k * (ratio - 1)))), 0.01, 0.99)
    assert estimate.fraction == pytest.approx(root, abs=1e-9)
    assert isinstance(estimate, MixtureFractionEstimate)
    assert estimate.std == pytest.approx(math.hypot(estimate.std_statistical, estimate.std_ratio))
    assert abs(estimate.fraction - 0.6) < 4 * estimate.std


def test_collinear_members_keep_predictions_well_defined() -> None:
    generator = torch.Generator().manual_seed(5)
    x_p, x_q = _samples(generator, 3000)
    base = WifiRatioEnsemble().fit(_basis(x_p), _basis(x_q))
    duplicated = WifiRatioEnsemble().fit(
        torch.cat([_basis(x_p), _basis(x_p)[:, :1]], dim=1),
        torch.cat([_basis(x_q), _basis(x_q)[:, :1]], dim=1),
    )
    grid = torch.linspace(-2.0, 3.0, 7, dtype=torch.float64)
    grid_features = _basis(grid)
    torch.testing.assert_close(
        duplicated.log_ratio(torch.cat([grid_features, grid_features[:, :1]], dim=1)),
        base.log_ratio(grid_features),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        duplicated.log_ratio_std(torch.cat([grid_features, grid_features[:, :1]], dim=1)),
        base.log_ratio_std(grid_features),
        rtol=1e-4,
        atol=1e-8,
    )


def test_member_features_stacks_callables() -> None:
    torch.manual_seed(6)
    members = [nn.Linear(2, 1), nn.Linear(2, 1), nn.Linear(2, 1)]
    inputs = torch.randn(10, 2)
    features = WifiRatioEnsemble.member_features(members, inputs)
    assert features.shape == (10, 3) and features.dtype == torch.float64
    torch.testing.assert_close(features[:, 1], members[1](inputs).reshape(-1).double())
    with pytest.raises(ValueError, match="at least one"):
        WifiRatioEnsemble.member_features([], inputs)
    with pytest.raises(ValueError, match="different numbers"):
        WifiRatioEnsemble.member_features([lambda x: x[:, 0], lambda x: x[:5, 0]], inputs)


def test_validation_errors() -> None:
    wifi = WifiRatioEnsemble()
    with pytest.raises(RuntimeError, match="not fitted"):
        wifi.log_ratio(torch.zeros(3, 2))
    with pytest.raises(ValueError, match="member columns"):
        wifi.fit(torch.randn(5, 2), torch.randn(5, 3))
    with pytest.raises(ValueError, match="finite"):
        wifi.fit(torch.tensor([[float("nan")]]), torch.zeros(3, 1))
    with pytest.raises(TypeError, match="torch.Tensor"):
        wifi.fit(np.zeros((3, 1)), torch.zeros(3, 1))  # type: ignore[arg-type]
    for kwargs in ({"max_iter": 0}, {"tol": 0.0}, {"rcond": 1.0}):
        with pytest.raises(ValueError):
            WifiRatioEnsemble(**kwargs)  # type: ignore[arg-type]
    fitted = WifiRatioEnsemble().fit(torch.randn(50, 1), torch.randn(50, 1))
    with pytest.raises(ValueError, match="member columns"):
        fitted.log_ratio(torch.zeros(3, 2))
    with pytest.raises(ValueError, match="constant"):
        constant = WifiRatioEnsemble().fit(torch.zeros(20, 1), torch.zeros(20, 1))
        constant.mixture_fraction(torch.zeros(10, 1))


def test_mixture_fraction_with_all_ratios_above_one_stays_finite() -> None:
    # Every mixture point has r > 1, so the likelihood increases in kappa
    # without bound: the search stops at its limit and flags the boundary
    # instead of overflowing to kappa = inf (which divided by zero).
    generator = torch.Generator().manual_seed(7)
    x_p, x_q = _samples(generator, 2000)
    wifi = WifiRatioEnsemble().fit(_basis(x_p), _basis(x_q))
    x_mix = torch.linspace(2.5, 4.0, 8, dtype=torch.float64)
    assert bool((wifi.log_ratio(_basis(x_mix)) > 0).all())
    estimate = wifi.mixture_fraction(_basis(x_mix))
    assert estimate.at_boundary
    assert math.isfinite(estimate.fraction) and estimate.fraction > 1.0
    assert not math.isnan(estimate.std)
