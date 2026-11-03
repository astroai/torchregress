"""Tests for generic PIT, binned-CDF and bootstrap diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

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


def _crps_binned_reference_loop(bin_edges, cdf, target):
    """The original per-row, per-bin loop, kept as an exact reference."""
    import torch

    edges = torch.as_tensor(bin_edges, dtype=torch.float64)
    c = torch.as_tensor(cdf, dtype=torch.float64)
    y = torch.as_tensor(target, dtype=torch.float64)

    def sq(fa, fb, width):
        return width * (fa**2 + fa * fb + fb**2) / 3.0

    widths = edges[1:] - edges[:-1]
    fa, fb = c[:, :-1], c[:, 1:]
    values = torch.zeros(y.numel(), dtype=torch.float64)
    for i in range(y.numel()):
        yi = float(y[i])
        if yi <= float(edges[0]):
            values[i] = edges[0] - yi + sq(1.0 - fa[i], 1.0 - fb[i], widths).sum()
            continue
        if yi >= float(edges[-1]):
            values[i] = yi - edges[-1] + sq(fa[i], fb[i], widths).sum()
            continue
        total = torch.zeros((), dtype=torch.float64)
        for j in range(len(widths)):
            a, b = float(edges[j]), float(edges[j + 1])
            if b <= yi:
                total = total + sq(fa[i, j], fb[i, j], widths[j])
            elif a >= yi:
                total = total + sq(1.0 - fa[i, j], 1.0 - fb[i, j], widths[j])
            else:
                fy = fa[i, j] + (yi - a) / (b - a) * (fb[i, j] - fa[i, j])
                total = total + sq(fa[i, j], fy, torch.as_tensor(yi - a, dtype=torch.float64))
                total = total + sq(
                    1.0 - fy, 1.0 - fb[i, j], torch.as_tensor(b - yi, dtype=torch.float64)
                )
        values[i] = total
    return values.numpy()


def test_crps_binned_matches_reference_loop() -> None:
    rng = np.random.default_rng(0)
    n = 500
    edges = np.linspace(-3.0, 3.0, 65)
    cdf = binned_cdf(rng.dirichlet(np.ones(64), n))
    target = rng.normal(size=n) * 1.5
    target[:5] = [-3.0, 3.0, edges[10], -7.0, 9.0]  # on edges and outside the range
    got = crps_from_binned_cdf(edges, cdf, target, reduction="none")
    ref = _crps_binned_reference_loop(edges, cdf, target)
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-12)
    assert crps_from_binned_cdf(edges, cdf, target) == pytest.approx(ref.mean(), abs=1e-12)
    assert crps_from_binned_cdf(edges, cdf, target, reduction="sum") == pytest.approx(
        ref.sum(), abs=1e-9
    )


def test_crps_binned_matches_quadrature() -> None:
    from scipy.integrate import quad

    edges = np.array([-1.0, -0.7, 0.0, 0.05, 1.3, 2.0])  # unequal widths
    probs = np.array([[0.1, 0.0, 0.5, 0.0, 0.4], [0.0, 0.0, 1.0, 0.0, 0.0]])  # empty bins
    cdf = binned_cdf(probs)
    targets = [-2.5, -1.0, 0.0, 0.03, 1.3, 2.0, 4.0]
    for row in cdf:
        rows = np.repeat(row[None, :], len(targets), axis=0)
        got = crps_from_binned_cdf(edges, rows, targets, reduction="none")
        for y, value in zip(targets, got):
            lo, hi = min(edges[0], y), max(edges[-1], y)
            points = sorted(set(edges.tolist() + [y]))

            def integrand(z: float, y: float = y) -> float:
                f = float(np.interp(z, edges, row))
                return (f - float(z >= y)) ** 2

            ref, _ = quad(integrand, lo, hi, points=points, limit=200, epsabs=1e-13)
            assert value == pytest.approx(ref, abs=1e-10)


def test_crps_binned_accepts_torch_and_keeps_output_types() -> None:
    import torch

    rng = np.random.default_rng(3)
    edges = np.linspace(0.0, 1.0, 9)
    cdf = binned_cdf(rng.dirichlet(np.ones(8), 20))
    target = rng.uniform(-0.2, 1.2, size=20)
    ref = crps_from_binned_cdf(edges, cdf, target, reduction="none")
    got = crps_from_binned_cdf(
        torch.as_tensor(edges),
        torch.as_tensor(cdf, dtype=torch.float32),
        torch.as_tensor(target),
        reduction="none",
    )
    assert isinstance(got, np.ndarray) and got.dtype == np.float64
    np.testing.assert_allclose(
        got,
        crps_from_binned_cdf(
            edges,
            torch.as_tensor(cdf, dtype=torch.float32).double().numpy(),
            target,
            reduction="none",
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert isinstance(crps_from_binned_cdf(edges, cdf, target), float)
    np.testing.assert_allclose(ref, _crps_binned_reference_loop(edges, cdf, target), atol=1e-12)
    with pytest.raises(ValueError, match="reduction"):
        crps_from_binned_cdf(edges, cdf, target, reduction="median")
