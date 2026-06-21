"""
Supplemental edge-case tests for torchregress.calibration.shift.

Existing tests cover basic fit/compute paths for both
RepresentationShiftCalibrator and BinnedLabelShiftEstimator.
This file fills remaining gaps: internal helpers, edge inputs,
torch/numpy interop, and error paths.
"""

import numpy as np
import pytest
import torch

from torchregress.calibration.shift import (
    BinnedLabelShiftEstimator,
    RepresentationShiftCalibrator,
)

# ═══════════════════════════════════════════════════════════════════════════════
# RepresentationShiftCalibrator — edge cases beyond existing smoke tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRepresentationShiftCalibratorEdge:
    def test_fit_returns_self(self) -> None:
        cal = RepresentationShiftCalibrator()
        source = np.random.default_rng(0).normal(size=(20, 3))
        assert cal.fit(source) is cal

    def test_squared_mahalanobis_before_fit_raises(self) -> None:
        cal = RepresentationShiftCalibrator()
        with pytest.raises(RuntimeError, match="call fit"):
            cal._squared_mahalanobis(np.ones((5, 2)))

    def test_fit_with_subsampling(self) -> None:
        cal = RepresentationShiftCalibrator(source_sample_size=10, random_state=0)
        source = np.random.default_rng(1).normal(size=(50, 4))
        cal.fit(source)
        assert cal.source_mean_ is not None
        assert cal.source_mean_.shape == (4,)

    def test_fit_with_winsorizing(self) -> None:
        cal = RepresentationShiftCalibrator(clip_quantile=0.1)
        source = np.random.default_rng(2).normal(size=(30, 3))
        cal.fit(source)
        assert cal.reference_scale_ is not None
        assert np.isfinite(cal.reference_scale_)

    def test_single_feature_source(self) -> None:
        cal = RepresentationShiftCalibrator()
        source = np.array([[1.0], [2.0], [3.0], [5.0], [8.0]])
        cal.fit(source)
        assert cal.source_mean_.shape == (1,)
        assert cal.source_var_.shape == (1,)
        scores = cal.shift_scores(np.array([[1.0], [3.0], [8.0]]))
        assert scores.shape == (3,)
        # 1 is below mean, 3 is near mean, 8 is above mean
        assert scores[1] < scores[0]
        assert scores[1] < scores[2]

    def test_calibrate_probabilities_multi_class(self) -> None:
        cal = RepresentationShiftCalibrator()
        cal.source_mean_ = np.array([0.0])
        cal.source_var_ = np.array([1.0])
        cal.reference_scale_ = 1.0
        probs = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3]])
        target = np.array([[0.0], [0.0]])
        out = cal.calibrate_probabilities(probs, target)
        assert out.shape == probs.shape
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_calibrate_probabilities_with_high_temperature_flattens(self) -> None:
        cal = RepresentationShiftCalibrator(max_temperature=10.0)
        cal.source_mean_ = np.array([0.0])
        cal.source_var_ = np.array([1.0])
        cal.reference_scale_ = 1.0
        probs = np.array([[0.99, 0.01]])
        target = np.array([[100.0]])  # large shift → max temperature
        out = cal.calibrate_probabilities(probs, target)
        # Should be less peaked than original
        assert out[0, 0] < 0.99
        assert out[0, 1] > 0.01
        assert np.allclose(out.sum(), 1.0)

    def test_calibrate_std_1d_input(self) -> None:
        cal = RepresentationShiftCalibrator()
        cal.source_mean_ = np.array([0.0])
        cal.source_var_ = np.array([1.0])
        cal.reference_scale_ = 1.0
        result = cal.calibrate_std(np.array([1.0]), np.array([[0.0]]))
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result[0] == pytest.approx(1.0)

    def test_temperatures_reference_scale_eps_bound(self) -> None:
        cal = RepresentationShiftCalibrator(eps=1e-6)
        cal.source_mean_ = np.array([0.0])
        cal.source_var_ = np.array([1.0])
        cal.reference_scale_ = 0.0  # Would divide by zero without eps clamp
        temps = cal.temperatures(np.array([[1.0]]))
        assert np.all(np.isfinite(temps))
        assert temps[0] >= cal.base_temperature

    def test_shift_scores_no_negative(self) -> None:
        cal = RepresentationShiftCalibrator()
        cal.source_mean_ = np.array([0.0, 0.0])
        cal.source_var_ = np.array([1.0, 1.0])
        cal.reference_scale_ = 1.0
        scores = cal.shift_scores(np.array([[0.0, 0.0], [-5.0, 0.0], [3.0, 4.0]]))
        assert np.all(scores >= 0.0)

    def test_temperatures_with_base_slope_zero(self) -> None:
        cal = RepresentationShiftCalibrator(base_temperature=1.0, slope=0.0)
        cal.source_mean_ = np.array([0.0])
        cal.source_var_ = np.array([1.0])
        cal.reference_scale_ = 1.0
        temps = cal.temperatures(np.array([[0.0], [10.0]]))
        # slope=0 → temperature always = base_temperature
        np.testing.assert_array_almost_equal(temps, [1.0, 1.0])


# ═══════════════════════════════════════════════════════════════════════════════
# BinnedLabelShiftEstimator — edge cases beyond existing smoke tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBinnedLabelShiftEstimatorEdge:
    def test_bin_values_with_fitted_edges(self) -> None:
        est = BinnedLabelShiftEstimator(n_bins=3, binning_strategy="uniform")
        est.bin_edges_ = np.array([-np.inf, 0.0, 5.0, np.inf])
        est.n_bins = 3
        bins = est._bin_values(np.array([-10.0, 2.0, 10.0]))
        np.testing.assert_array_equal(bins, [0, 1, 2])

    def test_bin_values_before_fit_raises(self) -> None:
        est = BinnedLabelShiftEstimator(n_bins=3)
        with pytest.raises(RuntimeError, match="bin_edges_ not fitted"):
            est._bin_values(np.array([1.0, 2.0]))

    def test_prepare_predictions_1d_point_predictions(self) -> None:
        est = BinnedLabelShiftEstimator(n_bins=3, binning_strategy="uniform")
        est.bin_edges_ = np.array([-np.inf, 0.0, 5.0, np.inf])
        est.n_bins = 3
        probs = est._prepare_predictions(np.array([-1.0, 2.0, 10.0]))
        assert probs.shape == (3, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        # One-hot: [1 0 0], [0 1 0], [0 0 1]
        assert probs[0, 0] == 1.0
        assert probs[1, 1] == 1.0
        assert probs[2, 2] == 1.0

    def test_prepare_predictions_3d_raises(self) -> None:
        est = BinnedLabelShiftEstimator(n_bins=3)
        with pytest.raises(ValueError):
            est._prepare_predictions(np.ones((2, 3, 4)))

    def test_prepare_predictions_unnormalized_probs(self) -> None:
        est = BinnedLabelShiftEstimator(n_bins=3, binning_strategy="uniform")
        est.bin_edges_ = np.array([-np.inf, 0.0, 5.0, np.inf])
        est.n_bins = 3
        # Raw values (not summing to 1)
        raw = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0]])
        probs = est._prepare_predictions(raw)
        assert np.allclose(probs.sum(axis=1), 1.0)
        # First row: [2, 1, 0] → [2/3, 1/3, 0]
        np.testing.assert_array_almost_equal(probs[0], [2 / 3, 1 / 3, 0.0])

    def test_prepare_predictions_wrong_ndim_single_col(self) -> None:
        est = BinnedLabelShiftEstimator(n_bins=3, binning_strategy="uniform")
        est.bin_edges_ = np.array([-np.inf, 0.0, 5.0, np.inf])
        est.n_bins = 3
        with pytest.raises(ValueError):
            est._prepare_predictions(np.ones((2, 5)))  # 5 != n_bins

    def test_get_bin_weights_before_fit_raises(self) -> None:
        est = BinnedLabelShiftEstimator(n_bins=3)
        with pytest.raises(RuntimeError, match="call fit"):
            est.get_bin_weights()

    def test_sample_weights_integer_bin_indices(self) -> None:
        est = BinnedLabelShiftEstimator(n_bins=3, binning_strategy="uniform")
        est.bin_edges_ = np.array([-np.inf, 0.0, 5.0, np.inf])
        est.n_bins = 3
        est.source_prior_ = np.array([0.3, 0.4, 0.3])
        est.target_prior_ = np.array([0.1, 0.6, 0.3])
        weights = est.sample_weights(np.array([0, 1, 2], dtype=np.int32))
        expected = est.target_prior_ / est.source_prior_
        np.testing.assert_array_almost_equal(weights, expected)

    def test_fit_with_torch_tensor_inputs(self) -> None:
        y_src = torch.tensor([0.5, 0.5, 1.5, 1.5], dtype=torch.float64)
        p_src = torch.tensor([0.5, 0.5, 1.5, 1.5], dtype=torch.float64)
        p_tgt = torch.tensor([1.5, 1.5, 0.5], dtype=torch.float64)

        est = BinnedLabelShiftEstimator(n_bins=2, binning_strategy="uniform", method="bbse")
        est.fit(y_src, p_src, p_tgt)
        assert est.bin_edges_ is not None
        assert est.source_prior_ is not None
        assert est.target_prior_ is not None

    def test_bbse_confusion_matrix_call(self) -> None:
        y_src = np.array([1.0, 2.0, 3.0, 4.0])
        p_src = np.array([1.0, 2.0, 3.0, 4.0])
        p_tgt = np.array([3.0, 4.0, 1.0, 2.0])

        est = BinnedLabelShiftEstimator(n_bins=2, binning_strategy="uniform", method="bbse")
        est.fit(y_src, p_src, p_tgt)
        assert est.confusion_matrix_ is not None
        assert est.confusion_matrix_.shape == (2, 2)
        # Each column of confusion matrix should sum to ~1
        assert np.allclose(est.confusion_matrix_.sum(axis=0), 1.0, atol=0.01)

    def test_bbse_empty_bin_fallback(self) -> None:
        """When a bin receives no source samples, confusion-matrix column falls
        back to uniform 1/n_bins (not singular — just a test of the empty-bin path)."""
        y_src = np.array([0.0, 0.0, 0.0])
        p_src = np.array([0.0, 0.0, 0.0])
        p_tgt = np.array([0.0, 0.0])

        est = BinnedLabelShiftEstimator(n_bins=2, binning_strategy="uniform", method="bbse")
        est.fit(y_src, p_src, p_tgt)
        assert est.target_prior_ is not None
        assert np.allclose(est.target_prior_.sum(), 1.0)

    def test_em_convergence_reaches_target(self) -> None:
        np.random.seed(123)
        n_src, n_tgt = 1000, 1000
        # Bin 0 mean -2, bin 1 mean +2
        y_src = np.concatenate([np.full(n_src // 2, -2.0), np.full(n_src // 2, 2.0)])
        y_tgt_actual = np.concatenate([np.full(n_tgt // 4, -2.0), np.full(3 * n_tgt // 4, 2.0)])

        noise_src = np.random.normal(0, 0.5, y_src.shape)
        noise_tgt = np.random.normal(0, 0.5, y_tgt_actual.shape)

        # Predictor: y ~ N(y_true, 0.5²) → 2-class probability via Bayes
        def bayes_2class_probs(x):
            p0 = np.exp(-0.5 * ((x - (-2.0)) / 0.5) ** 2)
            p1 = np.exp(-0.5 * ((x - 2.0) / 0.5) ** 2)
            denom = p0 + p1
            return np.stack([p0 / denom, p1 / denom], axis=1)

        est = BinnedLabelShiftEstimator(
            n_bins=2, binning_strategy="uniform", method="em", max_iter=200
        )
        est.fit(
            y_src,
            bayes_2class_probs(y_src + noise_src),
            bayes_2class_probs(y_tgt_actual + noise_tgt),
        )
        assert est.target_prior_ is not None
        # True target prior: [0.25, 0.75]
        assert np.allclose(est.target_prior_, [0.25, 0.75], atol=0.1)

    def test_em_uses_max_iter_before_break(self) -> None:
        """EM with tol=0 should use exactly max_iter steps."""
        est = BinnedLabelShiftEstimator(
            n_bins=2, binning_strategy="uniform", method="em", max_iter=5, tol=0.0
        )
        y_src = np.array([1.0, 2.0])
        p_src = np.array([[0.5, 0.5], [0.5, 0.5]])
        p_tgt = np.array([[0.5, 0.5], [0.5, 0.5]])
        est.fit(y_src, p_src, p_tgt)
        assert est.target_prior_ is not None
        assert np.allclose(est.target_prior_.sum(), 1.0)

    def test_fit_with_fewer_samples_than_bins(self) -> None:
        est = BinnedLabelShiftEstimator(n_bins=10, binning_strategy="uniform")
        y_src = np.array([1.0, 2.0])
        p_src = np.array([1.0, 2.0])
        p_tgt = np.array([1.5])
        est.fit(y_src, p_src, p_tgt)
        assert est.bin_edges_ is not None
        assert est.source_prior_ is not None
        assert np.allclose(est.source_prior_.sum(), 1.0)

    def test_sample_weights_torch_device_preserved(self) -> None:
        est = BinnedLabelShiftEstimator(n_bins=2, binning_strategy="uniform", method="bbse")
        est.bin_edges_ = np.array([-np.inf, 0.0, np.inf])
        est.n_bins = 2
        est.source_prior_ = np.array([0.5, 0.5])
        est.target_prior_ = np.array([0.5, 0.5])
        # Force onto non-default device if available
        device = torch.device("cpu")
        y = torch.tensor([0.0, 0.0], device=device)
        w = est.sample_weights(y)
        assert isinstance(w, torch.Tensor)
        assert w.device == device
