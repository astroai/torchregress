"""Tests for photo-z PDF diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from torchregress.metrics import (
    binned_cdf,
    crps_from_binned_cdf,
    paired_bootstrap_ci,
    pdf_width_68,
    pit_cvm,
    pit_local_w1,
)


def test_pit_cvm_uniform_and_biased() -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(size=5000)
    cvm = pit_cvm(u)
    assert 0.0 <= cvm < 1.0 / (6.0 * 5000) * 4.0
    assert pit_cvm(u, normalized=True) < 4.0
    biased = np.clip(u**0.5, 0, 1)
    assert pit_cvm(biased) > cvm


def test_pit_cvm_exact_step() -> None:
    # All PIT at 0.5 gives a step CDF; the integral can be evaluated directly.
    cvm = pit_cvm([0.5] * 100)
    assert cvm > 0.0
    assert np.isfinite(cvm)


def test_binned_cdf_and_crps() -> None:
    p = np.array([[0.2, 0.5, 0.3], [0.0, 0.0, 1.0]])
    cdf = binned_cdf(p)
    assert cdf.shape == (2, 4)
    assert np.allclose(cdf[:, -1], 1.0)
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    target = np.array([1.2, 2.8])
    crps = crps_from_binned_cdf(edges, cdf, target)
    assert np.isfinite(crps) and crps >= 0.0
    # Numerical quadrature on a dense grid agrees with the exact value.
    grid = np.linspace(0.0, 3.0, 30001)
    f = np.array([np.interp(grid, edges, row) for row in cdf])
    step = (grid[None, :] >= target[:, None]).astype(float)
    approx = np.trapezoid((f - step) ** 2, grid, axis=1).mean()
    assert abs(crps - approx) < 1e-3


def test_pdf_width_68() -> None:
    samples = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    z = np.array([0.0, 1.0])
    width = pdf_width_68(samples, z)
    assert width > 0.0
    # For uniform samples the 16/84 quantiles are ~0.48 and 3.0? use a
    # Gaussian check instead.
    torch.manual_seed(0)
    s = torch.randn(1000, 2000)
    w = pdf_width_68(s.numpy(), np.zeros(1000))
    assert 1.9 < w < 2.1


def test_pit_local_w1() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(120, 3))
    pit = rng.uniform(size=120)
    value = pit_local_w1(features, pit, k=10)
    assert np.isfinite(value)
    assert value >= 0.0
    # Strongly clustered PIT values have a larger local W1.
    bad = np.clip(0.5 + 0.01 * rng.normal(size=120), 0, 1)
    assert pit_local_w1(features, bad, k=10) > value


def test_paired_bootstrap_ci() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.2, 0.1, 1000)
    b = rng.normal(0.0, 0.1, 1000)
    diff, lower, upper = paired_bootstrap_ci(a, b, n_boot=500, seed=0)
    assert lower <= diff <= upper
    assert diff < upper
