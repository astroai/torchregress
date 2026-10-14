"""Hypothesis property tests for the scoring rules in :mod:`torchregress.metrics`.

Properties checked (float64, ``max_examples=50``):

* CRPS (Gaussian closed form and fair ensemble estimator) is non-negative;
* CRPS is location-scale equivariant: ``CRPS(a + b F, a + b y) = b CRPS(F, y)``;
* ``crps_gaussian`` is proper: the expected score under ``N(mu0, sigma0)``,
  computed by Gauss-Hermite quadrature, is minimised by the true distribution
  over a small grid of mis-specified forecasts;
* the interval score is non-negative;
* the pinball-loss minimiser over the data is the lower empirical
  ``tau``-quantile and is monotone in ``tau``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as hst  # noqa: E402

from torchregress.metrics import (  # noqa: E402
    crps_from_samples,
    crps_gaussian,
    interval_score,
    pinball_loss,
)

SETTINGS = settings(max_examples=50, deadline=None)

finite = hst.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
positive = hst.floats(min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False)


def _t(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


@SETTINGS
@given(
    mu=hst.lists(finite, min_size=1, max_size=20),
    data=hst.data(),
)
def test_crps_gaussian_non_negative(mu, data):
    n = len(mu)
    sigma = data.draw(hst.lists(positive, min_size=n, max_size=n))
    y = data.draw(hst.lists(finite, min_size=n, max_size=n))
    crps = crps_gaussian(_t(mu), _t(y), _t(sigma), reduction="none")
    assert bool((crps >= -1e-12 * (1 + _t(sigma))).all())


@SETTINGS
@given(
    samples=hst.lists(hst.lists(finite, min_size=3, max_size=3), min_size=1, max_size=15),
    y=hst.lists(finite, min_size=3, max_size=3),
)
def test_crps_from_samples_non_negative(samples, y):
    # Fair estimator: |x_i - x_j| <= |x_i - y| + |x_j - y| makes it >= 0.
    crps = crps_from_samples(_t(samples), _t(y), reduction="none")
    scale = 1.0 + np.abs(np.asarray(samples)).max() + np.abs(np.asarray(y)).max()
    assert bool((crps >= -1e-12 * scale).all())


@SETTINGS
@given(
    mu=finite,
    sigma=positive,
    y=finite,
    a=hst.floats(min_value=-100, max_value=100),
    b=hst.floats(min_value=1e-2, max_value=100),
)
def test_crps_gaussian_location_scale_equivariance(mu, sigma, y, a, b):
    base = crps_gaussian(_t([mu]), _t([y]), _t([sigma]))
    moved = crps_gaussian(_t([a + b * mu]), _t([a + b * y]), _t([b * sigma]))
    assert moved == pytest.approx(b * base, rel=1e-7, abs=1e-9 * b * (1 + abs(mu) + abs(y)))


@SETTINGS
@given(
    samples=hst.lists(hst.lists(finite, min_size=2, max_size=2), min_size=2, max_size=12),
    y=hst.lists(finite, min_size=2, max_size=2),
    a=hst.floats(min_value=-100, max_value=100),
    b=hst.floats(min_value=1e-2, max_value=100),
)
def test_crps_from_samples_location_scale_equivariance(samples, y, a, b):
    x, yy = np.asarray(samples), np.asarray(y)
    base = crps_from_samples(_t(x), _t(yy))
    moved = crps_from_samples(_t(a + b * x), _t(a + b * yy))
    scale = b * (1 + np.abs(x).max() + np.abs(yy).max())
    assert moved == pytest.approx(b * base, rel=1e-7, abs=1e-10 * scale)


_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(80)


def _expected_crps(m: float, s: float, mu0: float, sigma0: float) -> float:
    """E_{Y ~ N(mu0, sigma0)} CRPS(N(m, s), Y) by Gauss-Hermite quadrature."""
    y = mu0 + np.sqrt(2.0) * sigma0 * _GH_NODES
    crps = crps_gaussian(_t(np.full_like(y, m)), _t(y), _t(np.full_like(y, s)), reduction="none")
    return float(np.sum(_GH_WEIGHTS * np.asarray(crps)) / np.sqrt(np.pi))


@SETTINGS
@given(
    mu0=hst.floats(min_value=-5, max_value=5),
    sigma0=hst.floats(min_value=0.1, max_value=5),
)
def test_crps_gaussian_is_proper(mu0, sigma0):
    truth = _expected_crps(mu0, sigma0, mu0, sigma0)
    for dm in (-0.5, -0.1, 0.0, 0.1, 0.5):
        for fs in (0.5, 0.9, 1.0, 1.1, 2.0):
            if dm == 0.0 and fs == 1.0:
                continue
            other = _expected_crps(mu0 + dm * sigma0, fs * sigma0, mu0, sigma0)
            assert other > truth - 1e-12 * sigma0


@SETTINGS
@given(
    lower=hst.lists(finite, min_size=1, max_size=20),
    data=hst.data(),
    alpha=hst.floats(min_value=1e-3, max_value=0.999),
)
def test_interval_score_non_negative(lower, data, alpha):
    n = len(lower)
    width = data.draw(hst.lists(hst.floats(min_value=0.0, max_value=1e3), min_size=n, max_size=n))
    y = data.draw(hst.lists(finite, min_size=n, max_size=n))
    lo = np.asarray(lower)
    score = interval_score(_t(lo), _t(lo + np.asarray(width)), _t(y), alpha=alpha, reduction="none")
    assert bool((torch.as_tensor(score) >= 0).all())


def _pinball_minimiser(tau: float, y: np.ndarray) -> float:
    candidates = np.unique(y)
    # Integer data and dyadic levels keep every pinball sum exact in float64.
    sums = [float(pinball_loss(tau, _t(np.full_like(y, q)), _t(y)).sum()) for q in candidates]
    return float(candidates[int(np.argmin(sums))])


@SETTINGS
@given(
    y=hst.lists(hst.integers(min_value=-50, max_value=50), min_size=1, max_size=40),
    k=hst.lists(hst.integers(min_value=1, max_value=63), min_size=2, max_size=2, unique=True),
)
def test_pinball_minimiser_monotone_in_level(y, k):
    data = np.asarray(y, dtype=np.float64)
    tau1, tau2 = sorted(kk / 64 for kk in k)
    q1, q2 = _pinball_minimiser(tau1, data), _pinball_minimiser(tau2, data)
    assert q1 <= q2
    assert q1 == np.quantile(data, tau1, method="inverted_cdf")
    assert q2 == np.quantile(data, tau2, method="inverted_cdf")
