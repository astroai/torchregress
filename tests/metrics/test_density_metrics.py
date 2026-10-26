"""Tests for generic PIT, binned-CDF and bootstrap diagnostics."""

from __future__ import annotations

import numpy as np

from torchregress.metrics import (
    binned_cdf,
    crps_from_binned_cdf,
    paired_bootstrap_ci,
    pit_cvm,
)


def test_pit_cvm_uniform_and_biased() -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(size=5000)
    cvm = pit_cvm(u)
    assert 0.0 <= cvm < 1.0 / (6.0 * 5000) * 4.0
    assert pit_cvm(u, normalized=True) < 4.0
    assert pit_cvm(np.clip(u**0.5, 0, 1)) > cvm


def test_pit_cvm_exact_step() -> None:
    assert pit_cvm([0.5] * 100) > 0.0


def test_binned_cdf_and_crps() -> None:
    p = np.array([[0.2, 0.5, 0.3], [0.0, 0.0, 1.0]])
    cdf = binned_cdf(p)
    assert cdf.shape == (2, 4)
    assert np.allclose(cdf[:, -1], 1.0)
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    target = np.array([1.2, 2.8])
    crps = crps_from_binned_cdf(edges, cdf, target)
    assert np.isfinite(crps) and crps >= 0.0
    grid = np.linspace(0.0, 3.0, 30001)
    f = np.array([np.interp(grid, edges, row) for row in cdf])
    step = (grid[None, :] >= target[:, None]).astype(float)
    approx = np.trapezoid((f - step) ** 2, grid, axis=1).mean()
    assert abs(crps - approx) < 1e-3


def test_paired_bootstrap_ci() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.2, 0.1, 1000)
    b = rng.normal(0.0, 0.1, 1000)
    diff, lower, upper = paired_bootstrap_ci(a, b, n_boot=500, seed=0)
    assert lower <= diff <= upper


def test_crps_uniform_outside_support_exact() -> None:
    # For Uniform(0, 1), CRPS is |y-.5| - 1/6 outside support.
    values = crps_from_binned_cdf(
        [0.0, 1.0], [[0.0, 1.0]] * 4, [-2.0, 0.0, 1.0, 3.0], reduction="none"
    )
    np.testing.assert_allclose(values, [7 / 3, 1 / 3, 1 / 3, 7 / 3], atol=1e-14)


def test_density_diagnostics_reject_invalid_inputs() -> None:
    import pytest

    for bad in [[float("nan")], [-0.01], [1.01]]:
        with pytest.raises(ValueError):
            pit_cvm(bad)
    for bad in [[[0.0, 0.0]], [[-0.1, 1.1]], [[float("nan"), 1.0]]]:
        with pytest.raises(ValueError):
            binned_cdf(bad)
    with pytest.raises(ValueError):
        crps_from_binned_cdf([1.0, 0.0], [[0.0, 1.0]], [0.5])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])
