"""Deep unit tests for torchregress.calibration.posthoc calibrators.

Extends the basic happy-path coverage in test_constraints_calibration.py with
edge cases, error paths, and internal-algorithm invariants (PAVA, temperature
optimisation, PIT mapping).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchregress.calibration.posthoc import (
    IsotonicMeanCalibrator,
    PITCalibrator,
    VarianceTemperatureScaler,
)

# ── VarianceTemperatureScaler ────────────────────────────────────────


def test_vts_fit_transform_basic():
    torch.manual_seed(1)
    mean = torch.randn(200)
    target = mean + 0.3 * torch.randn(200)
    var = torch.full_like(mean, 0.04)

    scaler = VarianceTemperatureScaler(temperature=2.0)
    scaler.fit(mean, var, target, max_iter=100)
    var_scaled = scaler.transform(var)
    assert var_scaled.shape == var.shape
    assert torch.all(var_scaled > 0)
    # Temperature should have moved from initial 2.0
    assert scaler.temperature != pytest.approx(2.0)


def test_vts_transform_without_fit_uses_default():
    """transform() before fit() uses the default temperature (1.0)."""
    scaler = VarianceTemperatureScaler()
    var = torch.full((10,), 0.01)
    var_scaled = scaler.transform(var)
    assert torch.all(var_scaled > 0)
    # With temperature=1.0 and eps=1e-8, var_scaled ≈ var
    torch.testing.assert_close(var_scaled, var.clamp_min(1e-8))


def test_vts_fit_shape_mismatch_raises():
    scaler = VarianceTemperatureScaler()
    with pytest.raises(ValueError, match="share shape"):
        scaler.fit(torch.randn(10), torch.randn(10, 2), torch.randn(10))


def test_vts_fit_predictions_and_target_mismatch():
    scaler = VarianceTemperatureScaler()
    with pytest.raises(ValueError, match="share shape"):
        scaler.fit(torch.randn(10), torch.randn(10), torch.randn(15))


def test_vts_fit_with_large_noise_produces_finite_temperature():
    """Large noise + tiny variance → optimizer drives temperature high."""
    torch.manual_seed(0)
    mean = torch.randn(500)
    target = mean + 5.0 * torch.randn(500)  # large noise
    var = torch.full_like(mean, 1e-6)  # tiny variance

    scaler = VarianceTemperatureScaler()
    scaler.fit(mean, var, target, max_iter=500, lr=0.1)
    assert scaler.temperature > 0.0
    assert math.isfinite(scaler.temperature)


def test_vts_fit_reduces_nll():
    """After fitting, NLL with scaled variance should be ≤ NLL with raw variance."""
    torch.manual_seed(2)
    mean = torch.randn(200)
    target = mean + 0.5 * torch.randn(200)
    var = torch.full_like(mean, 0.001)  # overly confident

    scaler = VarianceTemperatureScaler(eps=1e-8)
    scaler.fit(mean, var, target, max_iter=200)

    def nll(v):
        vc = v.clamp_min(1e-8)
        return 0.5 * (torch.log(vc) + ((target - mean) ** 2) / vc + math.log(2 * math.pi))

    raw_nll = nll(var).mean().item()
    scaled_nll = nll(scaler.transform(var)).mean().item()
    assert scaled_nll < raw_nll - 0.001  # strict NLL improvement


def test_vts_refit_updates_temperature():
    scaler = VarianceTemperatureScaler(temperature=3.0)
    t_before = scaler.temperature

    mean = torch.randn(100)
    target = mean + 0.1 * torch.randn(100)
    scaler.fit(mean, torch.ones_like(mean), target, max_iter=50)
    assert scaler.temperature != pytest.approx(t_before)


# ── IsotonicMeanCalibrator (PAVA) ────────────────────────────────────


class TestPAVA:
    """Pool Adjacent Violators Algorithm edge cases."""

    def test_empty_input(self):
        x, y = IsotonicMeanCalibrator._pava(
            np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        )
        assert x.size == 0
        assert y.size == 0

    def test_single_element(self):
        x = np.array([3.0])
        y = np.array([7.0])
        rx, ry = IsotonicMeanCalibrator._pava(x, y)
        assert rx == pytest.approx(np.array([3.0]))
        assert ry == pytest.approx(np.array([7.0]))

    def test_already_monotonic(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        rx, ry = IsotonicMeanCalibrator._pava(x, y)
        assert rx == pytest.approx(np.array([0.0, 1.0, 2.0, 3.0]))
        assert ry == pytest.approx(np.array([0.0, 1.0, 2.0, 3.0]))

    def test_reverse_sorted(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([3.0, 2.0, 1.0, 0.0])
        rx, ry = IsotonicMeanCalibrator._pava(x, y)
        # All violations → everything merges into one block
        assert len(rx) == 1
        assert ry[0] == pytest.approx(1.5)  # mean of [3,2,1,0] = 1.5

    def test_partial_violations(self):
        """Only some blocks violate monotonic order."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 0.5, 3.0, 4.0])  # 0.5 < 2.0 at index 2 is a violation
        rx, ry = IsotonicMeanCalibrator._pava(x, y)
        # Block [1.0, 2.0] with means [2.0, 0.5] merge → mean = 1.25
        assert np.all(np.diff(ry) >= -1e-10)  # monotonic
        assert len(rx) < len(x)  # at least one merge

    def test_preserves_monotonicity_invariant(self):
        rng = np.random.RandomState(42)
        for _ in range(10):
            x = rng.randn(30)
            y = rng.randn(30)
            rx, ry = IsotonicMeanCalibrator._pava(x, y)
            assert np.all(np.diff(ry) >= -1e-12)


class TestIsotonicMeanCalibrator:
    """Fit / transform / validation tests."""

    def test_fit_transform_basic(self):
        cal = IsotonicMeanCalibrator()
        pred = torch.linspace(-3, 3, 100)
        target = pred + 0.2 * torch.randn(100)
        cal.fit(pred, target)
        out = cal.transform(pred)
        assert out.shape == pred.shape

    def test_transform_before_fit_raises(self):
        cal = IsotonicMeanCalibrator()
        with pytest.raises(ValueError, match="fitted before transform"):
            cal.transform(torch.randn(10))

    def test_fit_shape_mismatch_raises(self):
        cal = IsotonicMeanCalibrator()
        with pytest.raises(ValueError, match="sample dimension"):
            cal.fit(torch.randn(10), torch.randn(15))

    def test_transform_preserves_shape(self):
        cal = IsotonicMeanCalibrator().fit(torch.randn(50), torch.randn(50))
        for shape in [(10,), (10, 1), (3, 4)]:
            out = cal.transform(torch.randn(shape))
            assert out.shape == shape

    def test_out_of_bounds_clip(self):
        cal = IsotonicMeanCalibrator(out_of_bounds="clip")
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 2.0, 4.0])
        cal.fit(x, y)
        # Extrapolation clipped to nearest endpoint
        out = cal.transform(torch.tensor([-1.0, 3.0]))
        # -1.0 → clipped to y at x=0 → 0.0
        # 3.0 → clipped to y at x=2 → 4.0
        torch.testing.assert_close(out, torch.tensor([0.0, 4.0]))

    def test_out_of_bounds_noclip_extrapolates(self):
        """With out_of_bounds='linear', extrapolation is unclipped."""
        cal = IsotonicMeanCalibrator(out_of_bounds="linear")
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 2.0, 4.0])
        cal.fit(x, y)
        out = cal.transform(torch.tensor([-1.0, 3.0]))
        # Linear extrapolation: slope = 2.0
        # -1.0 → 0.0 + 2.0 * (-1.0) = -2.0
        # 3.0 → 2.0 + 2.0 * (3.0 - 1.0) = 6.0
        torch.testing.assert_close(out, torch.tensor([-2.0, 6.0]))


# ── PITCalibrator ────────────────────────────────────────────────────


class TestPITCalibrator:
    """PIT computation, fit, transform, and validation."""

    def test_pit_from_gaussian_values_in_unit_interval(self):
        pred_mean = torch.randn(200)
        pred_std = torch.full_like(pred_mean, 0.5)
        target = pred_mean + pred_std * torch.randn(200)
        pit = PITCalibrator.pit_from_gaussian(pred_mean, pred_std, target)
        assert torch.all(pit > 0.0)
        assert torch.all(pit < 1.0)

    def test_pit_from_gaussian_zero_std_clamped(self):
        """Zero std is clamped internally, PIT stays in (0, 1)."""
        pred_mean = torch.tensor([0.0, 1.0, 2.0])
        pred_std = torch.zeros(3)
        target = torch.tensor([0.0, 1.0, 2.0])
        pit = PITCalibrator.pit_from_gaussian(pred_mean, pred_std, target)
        assert torch.all(pit > 0.0)
        assert torch.all(pit < 1.0)
        # At exact match with clamped std, PIT ≈ 0.5
        torch.testing.assert_close(pit, torch.full_like(pit, 0.5), atol=1e-4, rtol=0.0)

    def test_pit_from_gaussian_exact_zero_residual(self):
        """Zero residual → PIT = Φ(0) = 0.5."""
        pred_mean = torch.randn(100)
        target = pred_mean.clone()
        pit = PITCalibrator.pit_from_gaussian(pred_mean, torch.ones_like(pred_mean), target)
        torch.testing.assert_close(pit, torch.full_like(pit, 0.5))

    def test_fit_transform_basic(self):
        pit = torch.linspace(0.01, 0.99, 200)
        cal = PITCalibrator().fit(pit)
        out = cal.transform(pit)
        assert out.shape == pit.shape
        assert torch.all(out > 0.0)
        assert torch.all(out < 1.0)

    def test_transform_before_fit_raises(self):
        cal = PITCalibrator()
        with pytest.raises(ValueError, match="fitted before transform"):
            cal.transform(torch.rand(10))

    def test_transform_preserves_shape(self):
        cal = PITCalibrator().fit(torch.rand(100))
        for shape in [(10,), (10, 1), (3, 4)]:
            out = cal.transform(torch.rand(shape))
            assert out.shape == shape

    def test_pit_edge_values_clamped(self):
        """PIT values at exact 0 or 1 are clamped to [eps, 1-eps]."""
        cal = PITCalibrator(eps=1e-6)
        cal.fit(torch.linspace(0.1, 0.9, 100))
        out = cal.transform(torch.tensor([0.0, 1.0]))
        assert torch.all(out >= 1e-6)
        assert torch.all(out <= 1.0 - 1e-6)

    def test_pit_calibrator_monotonicity(self):
        """PIT calibration should be monotonically non-decreasing."""
        pit_raw = torch.rand(500)
        cal = PITCalibrator().fit(pit_raw)
        x = torch.linspace(0.01, 0.99, 50)
        y = cal.transform(x)
        diffs = y[1:] - y[:-1]
        assert torch.all(diffs >= -1e-6)
