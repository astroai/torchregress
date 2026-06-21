"""Unit tests for torchregress.utils.distributions.normal_cdf."""

from __future__ import annotations

import pytest
import torch
from scipy import stats

from torchregress.utils.distributions import normal_cdf


def test_normal_cdf_basic_known_values() -> None:
    """CDF at standard quantile checkpoints matches scipy."""
    z = torch.tensor([-3.0, -1.96, -1.0, -0.5, 0.0, 0.5, 1.0, 1.96, 3.0])
    cdf = normal_cdf(z)
    expected = torch.from_numpy(stats.norm.cdf(z.numpy())).to(torch.float32)
    torch.testing.assert_close(cdf, expected)


def test_normal_cdf_zero_is_half() -> None:
    """CDF at zero is exactly 0.5."""
    z = torch.tensor(0.0)
    cdf = normal_cdf(z)
    assert cdf.item() == 0.5


def test_normal_cdf_asymptotic_large_positive() -> None:
    """Large positive z → CDF approaches 1.0."""
    for z_val in [5.0, 10.0, 30.0]:
        cdf = normal_cdf(torch.tensor(z_val))
        assert cdf.item() > 0.9999
        assert cdf.item() <= 1.0


def test_normal_cdf_asymptotic_large_negative() -> None:
    """Large negative z → CDF approaches 0.0."""
    for z_val in [-5.0, -10.0, -30.0]:
        cdf = normal_cdf(torch.tensor(z_val))
        assert cdf.item() >= 0.0
        assert cdf.item() < 0.0001


def test_normal_cdf_monotonic_increasing() -> None:
    """CDF is non-decreasing and changes over its domain."""
    z = torch.linspace(-5.0, 5.0, 1000)
    cdf = normal_cdf(z)
    diffs = torch.diff(cdf)
    # erf saturates at extreme tails → diffs may be exactly 0 in float32
    assert (diffs >= 0).all()
    assert diffs.max() > 0


def test_normal_cdf_symmetry() -> None:
    """cdf(-z) == 1 - cdf(z) for all finite z."""
    z = torch.randn(100)
    cdf_pos = normal_cdf(z)
    cdf_neg = normal_cdf(-z)
    torch.testing.assert_close(cdf_pos, 1.0 - cdf_neg)


def test_normal_cdf_gradients_flow() -> None:
    """Gradients are finite and non-zero."""
    z = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    cdf = normal_cdf(z)
    cdf.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    # CDF derivative = PDF = exp(-z²/2)/√(2π) > 0
    assert (z.grad > 0).all()


def test_normal_cdf_gradient_near_extremes() -> None:
    """Gradient is small but finite at extreme values (float64 avoids underflow)."""
    z = torch.tensor([-10.0, 10.0], dtype=torch.float64, requires_grad=True)
    cdf = normal_cdf(z)
    cdf.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert (z.grad > 0).all()
    # PDF at ±10 is ~7.7e-23 — grad should be tiny but > 0
    assert (z.grad < 1e-20).all()


def test_normal_cdf_scalar_input() -> None:
    """0-D tensor input returns 0-D tensor output."""
    z = torch.tensor(1.5)
    cdf = normal_cdf(z)
    assert cdf.ndim == 0
    torch.testing.assert_close(cdf, torch.tensor(stats.norm.cdf(1.5), dtype=torch.float32))


def test_normal_cdf_1d_input() -> None:
    """1-D tensor preserves batch shape."""
    z = torch.randn(50)
    cdf = normal_cdf(z)
    assert cdf.shape == (50,)
    assert (cdf >= 0.0).all() and (cdf <= 1.0).all()


def test_normal_cdf_2d_input() -> None:
    """2-D tensor preserves all dimensions."""
    z = torch.randn(4, 8)
    cdf = normal_cdf(z)
    assert cdf.shape == (4, 8)
    assert (cdf >= 0.0).all() and (cdf <= 1.0).all()


def test_normal_cdf_3d_input() -> None:
    """3-D tensor preserves all dimensions."""
    z = torch.randn(2, 3, 7)
    cdf = normal_cdf(z)
    assert cdf.shape == (2, 3, 7)


def test_normal_cdf_dtype_float32() -> None:
    """Output dtype matches input dtype float32."""
    z = torch.tensor([0.0, 1.0], dtype=torch.float32)
    cdf = normal_cdf(z)
    assert cdf.dtype == torch.float32


def test_normal_cdf_dtype_float64() -> None:
    """Output dtype matches input dtype float64."""
    z = torch.tensor([0.0, 1.0], dtype=torch.float64)
    cdf = normal_cdf(z)
    assert cdf.dtype == torch.float64


def test_normal_cdf_no_nan_for_finite_inputs() -> None:
    """CDF never produces NaN for any finite input."""
    z = torch.randn(1000) * 100  # wide range
    cdf = normal_cdf(z)
    assert not torch.isnan(cdf).any()
    assert not torch.isinf(cdf).any()
    assert (cdf >= 0.0).all()
    assert (cdf <= 1.0).all()


def test_normal_cdf_no_nan_for_extreme_inputs() -> None:
    """CDF handles extreme values without NaN."""
    z = torch.tensor([-1e6, -1e3, 0.0, 1e3, 1e6])
    cdf = normal_cdf(z)
    assert not torch.isnan(cdf).any()
    assert (cdf >= 0.0).all()
    assert (cdf <= 1.0).all()
    # Extreme negative → 0, extreme positive → 1
    assert cdf[0].item() == 0.0
    assert cdf[-1].item() == 1.0


def test_normal_cdf_formula_match_erf() -> None:
    """Output exactly matches the manual erf formula."""
    z = torch.randn(200)
    cdf_direct = normal_cdf(z)
    sqrt_two = torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype))
    cdf_manual = 0.5 * (1.0 + torch.erf(z / sqrt_two))
    torch.testing.assert_close(cdf_direct, cdf_manual)


def test_normal_cdf_matches_scipy_high_precision() -> None:
    """float64 CDF matches scipy at high precision."""
    z = torch.linspace(-6.0, 6.0, 50, dtype=torch.float64)
    cdf = normal_cdf(z)
    expected = torch.from_numpy(stats.norm.cdf(z.numpy()))
    # float64 should match to ~1e-12
    torch.testing.assert_close(cdf, expected)


def test_normal_cdf_inplace_safety() -> None:
    """Input tensor is not mutated."""
    z = torch.tensor([-1.0, 0.0, 1.0])
    z_original = z.clone()
    _ = normal_cdf(z)
    assert torch.equal(z, z_original)


def test_normal_cdf_device_cpu() -> None:
    """Works on CPU."""
    z = torch.randn(20)
    cdf = normal_cdf(z)
    assert cdf.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_normal_cdf_device_cuda() -> None:
    """Works on CUDA when available."""
    z = torch.randn(20, device="cuda")
    cdf = normal_cdf(z)
    assert cdf.device.type == "cuda"
    assert (cdf >= 0.0).all() and (cdf <= 1.0).all()


def test_normal_cdf_range_is_unit_interval() -> None:
    """All CDF values fall in [0, 1]."""
    z = torch.randn(500) * 50  # very wide to saturate tails
    cdf = normal_cdf(z)
    assert cdf.min().item() >= 0.0
    assert cdf.max().item() <= 1.0
