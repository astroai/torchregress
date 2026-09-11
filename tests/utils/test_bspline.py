"""Tests for the unit-integral B-spline (M-spline) density basis."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.interpolate import BSpline

from torchregress.utils import BSplineDensityBasis


@pytest.fixture
def basis() -> BSplineDensityBasis:
    g = torch.Generator().manual_seed(0)
    samples = torch.rand(2000, generator=g, dtype=torch.float64) ** 2 * 3.0
    return BSplineDensityBasis.from_quantiles(samples, n_interior=6, lo=0.0, hi=3.0)


def test_shapes_and_knot_vector(basis: BSplineDensityBasis) -> None:
    # 7 interior+boundary breakpoints -> 8 breakpoints -> n_basis = 7 + degree
    assert basis.degree == 3
    assert basis.n_basis == basis.breakpoints.numel() - 1 + 3
    assert basis.knots.numel() == basis.n_basis + basis.order
    # Clamped: lo and hi repeated order times.
    assert torch.all(basis.knots[:4] == 0.0)
    assert torch.all(basis.knots[-4:] == 3.0)
    z = torch.rand(5, 7, dtype=torch.float64) * 3.0
    assert basis.evaluate(z).shape == (5, 7, basis.n_basis)


def test_matches_scipy_bspline(basis: BSplineDensityBasis) -> None:
    z = torch.linspace(0.0, 3.0, 1001, dtype=torch.float64)
    raw = basis.evaluate(z, normalized=False).numpy()
    t = basis.knots.numpy()
    for m in range(basis.n_basis):
        c = np.zeros(basis.n_basis)
        c[m] = 1.0
        ref = np.nan_to_num(BSpline(t, c, 3, extrapolate=False)(z.numpy()))
        np.testing.assert_allclose(raw[:, m], ref, atol=1e-12)


def test_partition_of_unity_and_nonnegativity(basis: BSplineDensityBasis) -> None:
    z = torch.linspace(0.0, 3.0, 4001, dtype=torch.float64)
    raw = basis.evaluate(z, normalized=False)
    assert torch.allclose(raw.sum(-1), torch.ones_like(z), atol=1e-12)
    assert torch.all(basis.evaluate(z) >= 0)
    # Zero outside the support, closed at hi.
    outside = basis.evaluate(torch.tensor([-0.5, 3.5], dtype=torch.float64))
    assert torch.all(outside == 0)
    assert basis.evaluate(torch.tensor([3.0], dtype=torch.float64)).sum() > 0


def test_unit_integral_normalisation(basis: BSplineDensityBasis) -> None:
    T = basis.bin_integrals(torch.tensor([0.0, 3.0], dtype=torch.float64))
    assert T.shape == (1, basis.n_basis)
    assert torch.allclose(T, torch.ones_like(T), atol=1e-12)
    # Independent check by dense trapezoid quadrature.
    z = torch.linspace(0.0, 3.0, 30001, dtype=torch.float64)
    trap = torch.trapezoid(basis.evaluate(z), z, dim=0)
    assert torch.allclose(trap, torch.ones_like(trap), atol=1e-6)


def test_bin_integrals_are_exact_and_additive(basis: BSplineDensityBasis) -> None:
    edges = torch.tensor([0.0, 0.3, 0.7, 1.4, 2.2, 3.0], dtype=torch.float64)
    T = basis.bin_integrals(edges)
    assert T.shape == (5, basis.n_basis)
    assert torch.all(T >= 0)
    assert torch.allclose(T.sum(0), torch.ones(basis.n_basis, dtype=torch.float64), atol=1e-12)
    # Reference: dense trapezoid per bin.
    for k in range(5):
        z = torch.linspace(edges[k].item(), edges[k + 1].item(), 20001, dtype=torch.float64)
        ref = torch.trapezoid(basis.evaluate(z), z, dim=0)
        assert torch.allclose(T[k], ref, atol=1e-7)
    # Splitting a bin conserves mass.
    fine = basis.bin_integrals(
        torch.tensor([0.0, 0.3, 0.5, 0.7, 1.4, 2.2, 3.0], dtype=torch.float64)
    )
    assert torch.allclose(fine[1] + fine[2], T[1], atol=1e-12)


def test_bin_integrals_outside_support_are_zero(basis: BSplineDensityBasis) -> None:
    T = basis.bin_integrals(torch.tensor([-1.0, 0.0, 3.0, 4.0], dtype=torch.float64))
    assert torch.all(T[0] == 0) and torch.all(T[2] == 0)
    assert torch.allclose(T[1], torch.ones(basis.n_basis, dtype=torch.float64))


def test_cdf_monotone_and_matches_cumulative_integral(basis: BSplineDensityBasis) -> None:
    z = torch.linspace(0.0, 3.0, 30001, dtype=torch.float64)
    F = basis.cdf(z)
    assert torch.allclose(F[0], torch.zeros(basis.n_basis, dtype=torch.float64))
    assert torch.allclose(F[-1], torch.ones(basis.n_basis, dtype=torch.float64), atol=1e-12)
    assert torch.all(F[1:] - F[:-1] >= -1e-12)
    cum = torch.cumulative_trapezoid(basis.evaluate(z), z, dim=0)
    assert torch.allclose(cum, F[1:], atol=1e-5)
    # Unsorted input and clamping.
    F2 = basis.cdf(torch.tensor([2.0, -1.0, 0.5, 10.0], dtype=torch.float64))
    assert torch.allclose(F2[1], torch.zeros(basis.n_basis, dtype=torch.float64))
    assert torch.allclose(F2[3], torch.ones(basis.n_basis, dtype=torch.float64))
    assert torch.allclose(F2[0], basis.cdf(torch.tensor([2.0], dtype=torch.float64))[0])


def test_simplex_coefficients_give_normalised_density(basis: BSplineDensityBasis) -> None:
    g = torch.Generator().manual_seed(1)
    coeffs = torch.softmax(torch.randn(6, basis.n_basis, generator=g, dtype=torch.float64), dim=-1)
    z = torch.linspace(0.0, 3.0, 6001, dtype=torch.float64)
    p = basis.density(coeffs, z)
    assert p.shape == (6, z.numel())
    assert torch.all(p >= 0)
    assert torch.allclose(
        torch.trapezoid(p, z, dim=-1), torch.ones(6, dtype=torch.float64), atol=1e-5
    )
    # Bin masses via T agree with trapezoid over the same bins.
    edges = torch.linspace(0.0, 3.0, 7, dtype=torch.float64)
    masses = coeffs @ basis.bin_integrals(edges).T
    assert torch.allclose(masses.sum(-1), torch.ones(6, dtype=torch.float64), atol=1e-12)
    ref = torch.stack(
        [
            torch.trapezoid(
                basis.density(
                    coeffs,
                    torch.linspace(edges[k].item(), edges[k + 1].item(), 4001, dtype=torch.float64),
                ),
                torch.linspace(edges[k].item(), edges[k + 1].item(), 4001, dtype=torch.float64),
                dim=-1,
            )
            for k in range(6)
        ],
        dim=-1,
    )
    assert torch.allclose(masses, ref, atol=1e-5)


def test_basis_means_match_quadrature(basis: BSplineDensityBasis) -> None:
    z = torch.linspace(0.0, 3.0, 30001, dtype=torch.float64)
    ref = torch.trapezoid(basis.evaluate(z) * z[:, None], z, dim=0)
    assert torch.allclose(basis.basis_means(), ref, atol=1e-6)
    # First basis function on clamped knots is (1 - x/h)^3 * 4/h with mean h/5.
    h = basis.breakpoints[1].item()
    assert abs(basis.basis_means()[0].item() - h / 5.0) < 1e-10


def test_gradients_flow_to_coefficients(basis: BSplineDensityBasis) -> None:
    logits = torch.zeros(3, basis.n_basis, dtype=torch.float64, requires_grad=True)
    coeffs = torch.softmax(logits, dim=-1)
    z = torch.tensor([0.4, 1.2, 2.9], dtype=torch.float64)
    nll = -(coeffs * basis.evaluate(z)).sum(-1).log().sum()
    nll.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_from_uniform_and_validation() -> None:
    b = BSplineDensityBasis.from_uniform(0.0, 1.0, n_intervals=4, degree=2)
    assert b.n_basis == 6
    with pytest.raises(ValueError):
        BSplineDensityBasis([0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        BSplineDensityBasis([1.0])
    with pytest.raises(ValueError):
        BSplineDensityBasis.from_uniform(0.0, 1.0, n_intervals=0)
    # degree 0 = histogram: each basis is 1/width on its bin.
    hist = BSplineDensityBasis([0.0, 0.5, 2.0], degree=0)
    vals = hist.evaluate(torch.tensor([0.25, 1.0], dtype=torch.float64))
    assert torch.allclose(vals, torch.tensor([[2.0, 0.0], [0.0, 1.0 / 1.5]], dtype=torch.float64))


def test_float32_inputs_supported(basis: BSplineDensityBasis) -> None:
    z = torch.linspace(0.0, 3.0, 101, dtype=torch.float32)
    vals = basis.evaluate(z)
    assert vals.dtype == torch.float32
    assert torch.allclose(basis.evaluate(z, normalized=False).sum(-1), torch.ones(101), atol=1e-5)
