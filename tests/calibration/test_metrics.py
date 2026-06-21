"""
Unit tests for torchregress.calibration.metrics — calibration evaluation metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.calibration.metrics import (
    ExpectedCalibrationError,
    MarginalCalibrationError,
    _compute_histograms,
    bias,
    calibration_metrics_report,
    calibration_score,
    expected_calibration_error,
    marginal_calibration_error,
)

# ═══════════════════════════════════════════════════════════════════════════════
# _compute_histograms
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeHistograms:
    def test_basic(self) -> None:
        """Basic binning of samples."""
        samples = torch.tensor([[0.1, 0.3, 0.5], [0.2, 0.6, 0.9]], dtype=torch.float32)
        bin_edges = torch.tensor([0.0, 0.4, 0.8, 1.0], dtype=torch.float32)
        hists = _compute_histograms(samples, bin_edges)
        assert hists.shape == (2, 3)
        assert hists.dtype == torch.float32

    def test_counts_sum_to_samples(self) -> None:
        """Each row's histogram should sum to the number of samples."""
        samples = torch.randn(5, 100)
        bin_edges = torch.linspace(-3, 3, 11)
        hists = _compute_histograms(samples, bin_edges)
        assert hists.shape == (5, 10)
        assert torch.all(hists.sum(dim=1) == 100)

    def test_single_sample(self) -> None:
        """Single sample per row."""
        samples = torch.tensor([[0.5]])
        bin_edges = torch.tensor([0.0, 0.5, 1.0])
        hists = _compute_histograms(samples, bin_edges)
        assert hists.shape == (1, 2)

    def test_all_in_one_bin(self) -> None:
        """All samples fall in the same bin."""
        samples = torch.ones(3, 10) * 0.1
        bin_edges = torch.tensor([0.0, 0.5, 1.0])
        hists = _compute_histograms(samples, bin_edges)
        assert torch.all(hists[:, 0] == 10)
        assert torch.all(hists[:, 1] == 0)


# ═══════════════════════════════════════════════════════════════════════════════
# ExpectedCalibrationError (torchmetrics Metric)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExpectedCalibrationError:
    def test_default_construction(self) -> None:
        """Default n_bins=10 creates valid metric."""
        metric = ExpectedCalibrationError()
        assert metric.n_bins == 10

    def test_perfectly_calibrated_returns_near_zero(self) -> None:
        """When observed proportions match expected, ECE ≈ 0."""
        metric = ExpectedCalibrationError()
        n = 1000
        y_true = torch.rand(n)
        # Create perfectly calibrated quantile predictions
        y_pred = {
            0.1: torch.quantile(y_true, 0.1).repeat(n),
            0.5: torch.quantile(y_true, 0.5).repeat(n),
            0.9: torch.quantile(y_true, 0.9).repeat(n),
        }
        metric.update(y_pred, y_true)
        result = metric.compute()
        mace = float(result["mean_absolute_calibration_error"].item())
        assert mace < 0.05

    def test_overconfident_returns_positive_ece(self) -> None:
        """Predictions that are too tight produce nonzero ECE."""
        metric = ExpectedCalibrationError()
        n = 500
        y_true = torch.randn(n)
        # Overconfident: narrow predictions
        y_pred = {
            0.1: torch.full((n,), -0.5),
            0.5: torch.zeros(n),
            0.9: torch.full((n,), 0.5),
        }
        metric.update(y_pred, y_true)
        result = metric.compute()
        mace = float(result["mean_absolute_calibration_error"].item())
        assert mace > 0.0

    def test_returns_three_keys(self) -> None:
        """compute() returns mace, rmsce, max_ce."""
        metric = ExpectedCalibrationError()
        metric.update({0.25: torch.zeros(10), 0.75: torch.ones(10)}, torch.rand(10))
        result = metric.compute()
        assert "mean_absolute_calibration_error" in result
        assert "root_mean_squared_calibration_error" in result
        assert "maximum_calibration_error" in result

    def test_multiple_updates_accumulate(self) -> None:
        """Multiple update() calls accumulate state."""
        metric = ExpectedCalibrationError()
        metric.update({0.5: torch.zeros(30)}, torch.rand(30))
        metric.update({0.5: torch.zeros(20)}, torch.rand(20))
        result = metric.compute()
        assert float(result["maximum_calibration_error"].item()) >= 0.0

    def test_single_quantile(self) -> None:
        """Works with a single quantile level."""
        metric = ExpectedCalibrationError()
        metric.update({0.5: torch.zeros(20)}, torch.randn(20))
        result = metric.compute()
        assert float(result["mean_absolute_calibration_error"].item()) >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# MarginalCalibrationError (torchmetrics Metric)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarginalCalibrationError:
    def test_default_construction(self) -> None:
        """Default n_bins=20 creates valid metric."""
        metric = MarginalCalibrationError()
        assert metric.n_bins == 20

    def test_perfectly_calibrated_returns_near_zero(self) -> None:
        """When predictive CDF matches observed CDF, MCE ≈ 0."""
        metric = MarginalCalibrationError()
        n = 1000
        y_true = torch.randn(n)
        # Sample from the true distribution (perfect calibration)
        samples = torch.randn(100, n)
        metric.update(samples, y_true)
        result = metric.compute()
        mce = float(result["marginal_calibration_error"].item())
        assert mce < 0.05

    def test_biased_samples_return_positive_mce(self) -> None:
        """Shifted predictions produce nonzero MCE."""
        metric = MarginalCalibrationError()
        n = 500
        y_true = torch.randn(n)
        samples = torch.randn(50, n) + 2.0  # shifted by 2
        metric.update(samples, y_true)
        result = metric.compute()
        mce = float(result["marginal_calibration_error"].item())
        assert mce > 0.0

    def test_returns_three_keys(self) -> None:
        """compute() returns mce, rms_mce, max_mce."""
        metric = MarginalCalibrationError()
        metric.update(torch.randn(30, 20), torch.randn(20))
        result = metric.compute()
        assert "marginal_calibration_error" in result
        assert "root_mean_squared_mce" in result
        assert "maximum_marginal_calibration_error" in result

    def test_constant_values_handles_safely(self) -> None:
        """All-equal values (min==max) handled gracefully."""
        metric = MarginalCalibrationError()
        y_true = torch.ones(20)
        samples = torch.ones(10, 20)
        metric.update(samples, y_true)
        result = metric.compute()
        assert torch.isfinite(result["marginal_calibration_error"])


# ═══════════════════════════════════════════════════════════════════════════════
# expected_calibration_error (functional)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExpectedCalibrationErrorFunctional:
    def test_basic(self) -> None:
        """Returns the three calibration error metrics."""
        y_true = torch.rand(100)
        y_pred = {
            0.1: torch.full((100,), 0.1),
            0.5: torch.full((100,), 0.5),
            0.9: torch.full((100,), 0.9),
        }
        result = expected_calibration_error(y_pred, y_true)
        assert "mean_absolute_calibration_error" in result
        assert "root_mean_squared_calibration_error" in result
        assert "maximum_calibration_error" in result

    def test_with_diagnostics(self) -> None:
        """return_diagnostics=True adds bin_errors, expected/actual proportions."""
        y_true = torch.rand(50)
        y_pred = {0.3: torch.full((50,), 0.2), 0.7: torch.full((50,), 0.8)}
        result = expected_calibration_error(y_pred, y_true, return_diagnostics=True)
        assert "bin_errors" in result
        assert "expected_proportions" in result
        assert "actual_proportions" in result

    def test_as_numpy(self) -> None:
        """as_numpy=True returns numpy-compatible result."""
        y_true = np.random.rand(50)
        y_pred = {0.5: np.full(50, 0.5)}
        result = expected_calibration_error(y_pred, y_true, as_numpy=True)
        assert isinstance(result["mean_absolute_calibration_error"], (float, np.floating))

    def test_tensor_inputs(self) -> None:
        """Accepts torch.Tensor inputs directly."""
        y_true = torch.rand(30)
        y_pred = {0.25: torch.zeros(30), 0.75: torch.ones(30)}
        result = expected_calibration_error(y_pred, y_true)
        assert np.isfinite(float(result["mean_absolute_calibration_error"]))

    def test_perfect_calibration(self) -> None:
        """When quantile predictions match the data, ECE should be near zero."""
        n = 500
        y_true = torch.randn(n)
        q_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        y_pred = {q: torch.quantile(y_true, q).repeat(n) for q in q_levels}
        result = expected_calibration_error(y_pred, y_true)
        mace = float(result["mean_absolute_calibration_error"])
        assert mace < 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# marginal_calibration_error (functional)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarginalCalibrationErrorFunctional:
    def test_basic(self) -> None:
        """Returns the three MCE metrics."""
        y_true = torch.randn(100)
        samples = torch.randn(50, 100)
        result = marginal_calibration_error(samples, y_true)
        assert "marginal_calibration_error" in result
        assert "root_mean_squared_mce" in result
        assert "maximum_marginal_calibration_error" in result

    def test_with_diagnostics(self) -> None:
        """return_diagnostics=True adds bin_centers, observed/predicted CDFs."""
        y_true = torch.randn(60)
        samples = torch.randn(30, 60)
        result = marginal_calibration_error(samples, y_true, return_diagnostics=True)
        assert "bin_centers" in result
        assert "observed_cdf" in result
        assert "predicted_cdf" in result
        assert "abs_errors" in result

    def test_as_numpy(self) -> None:
        """as_numpy=True returns numpy-compatible result."""
        y_true = np.random.randn(50)
        samples = np.random.randn(20, 50)
        result = marginal_calibration_error(samples, y_true, as_numpy=True)
        assert isinstance(result["marginal_calibration_error"], (float, np.floating))

    def test_1d_y_true_raises_on_wrong_samples_dim(self) -> None:
        """MarginalCalibrationError needs [n_samples, batch] shaped samples."""
        y_true = torch.randn(20)
        samples = torch.randn(20)  # 1D, not 2D
        with pytest.raises(ValueError, match="must have shape"):
            marginal_calibration_error(samples, y_true)

    def test_multi_dimensional_y_true_reshapes(self) -> None:
        """Multi-dimensional y_true is flattened internally."""
        y_true = torch.randn(30, 2)
        samples = torch.randn(20, 30, 2)
        result = marginal_calibration_error(samples, y_true)
        assert np.isfinite(float(result["marginal_calibration_error"]))

    def test_custom_n_bins(self) -> None:
        """n_bins parameter changes resolution."""
        y_true = torch.randn(50)
        samples = torch.randn(10, 50)
        result = marginal_calibration_error(samples, y_true, n_bins=5)
        assert np.isfinite(float(result["marginal_calibration_error"]))


# ═══════════════════════════════════════════════════════════════════════════════
# calibration_score
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalibrationScore:
    def test_basic(self) -> None:
        """Converts mean/std to quantile predictions and computes ECE."""
        y_true = torch.randn(80)
        pred_mean = torch.zeros(80)
        pred_std = torch.ones(80)
        result = calibration_score(y_true, pred_mean, pred_std)
        assert "mean_absolute_calibration_error" in result

    def test_perfect_gaussian_calibration(self) -> None:
        """When mean/std match the data, score should be near zero."""
        n = 1000
        y_true = torch.randn(n)
        result = calibration_score(y_true, torch.zeros(n), torch.ones(n))
        mace = float(result["mean_absolute_calibration_error"])
        assert mace < 0.05

    def test_miscalibrated_returns_positive(self) -> None:
        """Overconfident std produces positive error."""
        n = 500
        y_true = torch.randn(n)
        result = calibration_score(y_true, torch.zeros(n), torch.full((n,), 0.1))
        mace = float(result["mean_absolute_calibration_error"])
        assert mace > 0.0

    def test_custom_n_levels(self) -> None:
        """n_levels parameter controls quantization granularity."""
        y_true = torch.randn(50)
        result = calibration_score(y_true, torch.zeros(50), torch.ones(50), n_levels=5)
        assert np.isfinite(float(result["mean_absolute_calibration_error"]))

    def test_as_numpy(self) -> None:
        """as_numpy=True returns numpy-compatible result."""
        y_true = np.random.randn(50)
        result = calibration_score(y_true, np.zeros(50), np.ones(50), as_numpy=True)
        assert isinstance(result["mean_absolute_calibration_error"], (float, np.floating))


# ═══════════════════════════════════════════════════════════════════════════════
# bias
# ═══════════════════════════════════════════════════════════════════════════════


class TestBias:
    def test_zero_bias(self) -> None:
        """Perfect predictions give zero bias."""
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        result = bias(y_pred, y_true)
        assert float(result.item() if torch.is_tensor(result) else result) == pytest.approx(0.0)

    def test_positive_bias(self) -> None:
        """Overprediction gives positive bias."""
        y_true = torch.zeros(10)
        y_pred = torch.ones(10)
        result = bias(y_pred, y_true)
        assert float(result.item() if torch.is_tensor(result) else result) == pytest.approx(1.0)

    def test_negative_bias(self) -> None:
        """Underprediction gives negative bias."""
        y_true = torch.ones(10)
        y_pred = torch.zeros(10)
        result = bias(y_pred, y_true)
        assert float(result.item() if torch.is_tensor(result) else result) == pytest.approx(-1.0)

    def test_as_numpy(self) -> None:
        """as_numpy=True returns float."""
        result = bias(np.array([1.0, 2.0]), np.array([1.0, 2.0]), as_numpy=True)
        assert isinstance(result, (float, np.floating))


# ═══════════════════════════════════════════════════════════════════════════════
# calibration_metrics_report
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalibrationMetricsReport:
    def test_from_quantiles(self) -> None:
        """Generate report from quantile predictions."""
        y_true = torch.rand(100)
        y_pred_quantiles = {
            0.1: torch.full((100,), 0.1),
            0.5: torch.full((100,), 0.5),
            0.9: torch.full((100,), 0.9),
        }
        result = calibration_metrics_report(None, y_true, y_pred_quantiles=y_pred_quantiles)
        assert "mean_absolute_calibration_error" in result

    def test_from_distribution(self) -> None:
        """Generate report from a torch Distribution."""
        y_true = torch.randn(100)
        dist = torch.distributions.Normal(torch.zeros(100), torch.ones(100))
        result = calibration_metrics_report(dist, y_true, n_samples=50)
        assert "marginal_calibration_error" in result

    def test_from_samples(self) -> None:
        """Generate report from raw samples tensor."""
        y_true = torch.randn(80)
        samples = torch.randn(40, 80)
        result = calibration_metrics_report(samples, y_true)
        assert "marginal_calibration_error" in result

    def test_from_dict(self) -> None:
        """Generate report from dict with loc/scale keys."""
        y_true = torch.randn(60)
        dist_dict = {"loc": torch.zeros(60), "scale": torch.ones(60)}
        result = calibration_metrics_report(dist_dict, y_true, n_samples=30)
        assert "marginal_calibration_error" in result

    def test_from_dict_with_mean_std(self) -> None:
        """Dict with mean/std keys also works."""
        y_true = torch.randn(60)
        dist_dict = {"mean": torch.zeros(60), "std": torch.ones(60)}
        result = calibration_metrics_report(dist_dict, y_true, n_samples=30)
        assert "marginal_calibration_error" in result

    def test_from_dict_missing_keys_raises(self) -> None:
        """Dict without loc/mean or scale/std raises ValueError."""
        y_true = torch.randn(10)
        with pytest.raises(ValueError, match="loc/mean"):
            calibration_metrics_report({"wrong": torch.zeros(10)}, y_true)

    def test_combined_quantiles_and_samples(self) -> None:
        """Report with both quantiles and samples includes both metrics."""
        y_true = torch.rand(50)
        y_pred_q = {0.3: torch.full((50,), 0.3), 0.7: torch.full((50,), 0.7)}
        samples = torch.randn(20, 50)
        result = calibration_metrics_report(samples, y_true, y_pred_quantiles=y_pred_q)
        assert "mean_absolute_calibration_error" in result
        assert "marginal_calibration_error" in result

    def test_none_dist_and_no_quantiles_returns_empty(self) -> None:
        """When nothing is provided, returns empty dict."""
        result = calibration_metrics_report(None, torch.rand(10))
        assert result == {}
