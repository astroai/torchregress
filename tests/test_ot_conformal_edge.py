"""
Supplemental edge-case tests for torchregress.test_time.ot_conformal.

Existing tests cover basic smoke-test paths for OTShiftReweighter,
WeightedSplitConformalAdapter, and OptimTransportCoverageGap.
This file fills gaps: internal helpers, parameter validation,
diagnostics contents, and boundary behavior.
"""

import pytest
import torch

from torchregress.test_time.ot_conformal import (
    OptimalTransportCoverageGap,
    OTShiftReweighter,
    WeightedSplitConformalAdapter,
    _as_1d_scores,
    _effective_sample_size_inv_square,
    _normalize_simplex,
    _uniform_ecdf_on_grid,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestAs1dScores:
    def test_tensor_passthrough(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        result = _as_1d_scores(t, name="test")
        torch.testing.assert_close(result, t)

    def test_non_tensor_conversion(self) -> None:
        result = _as_1d_scores([1.0, 2.0, 3.0], name="test")
        assert torch.is_tensor(result)
        assert result.shape == (3,)
        assert result.dtype == torch.float32

    def test_2d_flattened(self) -> None:
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = _as_1d_scores(t, name="test")
        assert result.shape == (4,)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must be non-empty"):
            _as_1d_scores(torch.tensor([]), name="scores")


class TestNormalizeSimplex:
    def test_already_normalized(self) -> None:
        w = torch.tensor([0.2, 0.3, 0.5])
        result = _normalize_simplex(w)
        torch.testing.assert_close(result, w)

    def test_unnormalized(self) -> None:
        w = torch.tensor([1.0, 2.0, 3.0])
        result = _normalize_simplex(w)
        torch.testing.assert_close(result.sum(), torch.tensor(1.0))
        torch.testing.assert_close(result, w / 6.0)

    def test_negative_values_clamped(self) -> None:
        w = torch.tensor([-1.0, 2.0, -3.0, 6.0])
        result = _normalize_simplex(w)
        torch.testing.assert_close(result[0], torch.tensor(0.0))
        torch.testing.assert_close(result[2], torch.tensor(0.0))
        torch.testing.assert_close(result.sum(), torch.tensor(1.0))

    def test_all_zeros(self) -> None:
        w = torch.zeros(5)
        result = _normalize_simplex(w)
        # All clamped to 0, sum clamped to eps, division gives 0/eps = 0
        torch.testing.assert_close(result, torch.zeros(5))


class TestUniformECDFOnGrid:
    def test_matches_manual(self) -> None:
        scores = torch.tensor([0.0, 1.0, 2.0])
        grid = torch.tensor([-0.5, 0.5, 1.5, 2.5])
        result = _uniform_ecdf_on_grid(scores, grid)
        # uniform weights [1/3, 1/3, 1/3]
        # at -0.5: 0, at 0.5: 1/3, at 1.5: 2/3, at 2.5: 1
        expected = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
        torch.testing.assert_close(result, expected)

    def test_single_score(self) -> None:
        scores = torch.tensor([0.5])
        grid = torch.tensor([0.0, 0.5, 1.0])
        result = _uniform_ecdf_on_grid(scores, grid)
        assert result[0] == 0.0
        assert result[1] == 1.0
        assert result[2] == 1.0


class TestEffectiveSampleSize:
    def test_uniform_weights(self) -> None:
        n = 50
        w = torch.ones(n) / n
        ess = _effective_sample_size_inv_square(w)
        torch.testing.assert_close(ess, torch.tensor(float(n)))

    def test_single_nonzero(self) -> None:
        w = torch.tensor([0.0, 1.0, 0.0])
        ess = _effective_sample_size_inv_square(w)
        torch.testing.assert_close(ess, torch.tensor(1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# OptimalTransportCoverageGap edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestOptimalTransportCoverageGapEdge:
    def test_invalid_n_grid(self) -> None:
        with pytest.raises(ValueError, match="n_grid must be at least 8"):
            OptimalTransportCoverageGap(n_grid=4)

    def test_hi_lo_equal_fallback(self) -> None:
        """When all scores are identical, hi==lo triggers fallback to lo+1."""
        scores = torch.full((10,), 3.0)
        result = OptimalTransportCoverageGap().estimate(
            calibration_scores=scores, target_score_summary=scores
        )
        assert result["l2_cdf_gap"] == pytest.approx(0.0, abs=1e-8)
        assert result["n_calibration"] == 10
        assert result["n_target"] == 10

    def test_keys_present(self) -> None:
        result = OptimalTransportCoverageGap().estimate(
            calibration_scores=torch.randn(20),
            target_score_summary=torch.randn(15),
        )
        assert set(result.keys()) == {"l2_cdf_gap", "ks_max_abs", "n_calibration", "n_target"}
        assert result["l2_cdf_gap"] >= 0.0
        assert 0.0 <= result["ks_max_abs"] <= 1.0

    def test_identical_distributions_small_gap(self) -> None:
        torch.manual_seed(0)
        s = torch.randn(100)
        result = OptimalTransportCoverageGap().estimate(
            calibration_scores=s, target_score_summary=s
        )
        assert result["l2_cdf_gap"] < 1e-4
        assert result["ks_max_abs"] < 1e-4

    def test_non_tensor_input(self) -> None:
        result = OptimalTransportCoverageGap().estimate(
            calibration_scores=[1.0, 2.0, 3.0],
            target_score_summary=[2.0, 3.0, 4.0],
        )
        assert "l2_cdf_gap" in result
        assert result["n_calibration"] == 3
        assert result["n_target"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# OTShiftReweighter edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestOTShiftReweighterEdge:
    def test_invalid_score_mode(self) -> None:
        with pytest.raises(ValueError, match="score_mode"):
            OTShiftReweighter(score_mode="regression")  # type: ignore[arg-type]

    def test_invalid_objective(self) -> None:
        with pytest.raises(ValueError, match="objective"):
            OTShiftReweighter(objective="l2")  # type: ignore[arg-type]

    def test_invalid_weight_parameterization(self) -> None:
        with pytest.raises(ValueError, match="weight_parameterization"):
            OTShiftReweighter(weight_parameterization="simplex")  # type: ignore[arg-type]

    def test_negative_entropy_penalty(self) -> None:
        with pytest.raises(ValueError, match="entropy_penalty"):
            OTShiftReweighter(entropy_penalty=-0.1)

    def test_fit_returns_self(self) -> None:
        rw = OTShiftReweighter(n_steps=20, learning_rate=0.1)
        cal = torch.randn(30)
        tgt = torch.randn(25)
        assert rw.fit(cal, tgt) is rw

    def test_diagnostics_keys(self) -> None:
        torch.manual_seed(0)
        rw = OTShiftReweighter(n_steps=30, learning_rate=0.1)
        rw.fit(torch.randn(30), torch.randn(20))
        assert "ess_inv_square" in rw.diagnostics_
        assert "cdf_l2_on_grid" in rw.diagnostics_
        assert rw.diagnostics_["ess_inv_square"] > 0

    def test_objective_value_stored(self) -> None:
        rw = OTShiftReweighter(n_steps=30, learning_rate=0.1)
        rw.fit(torch.randn(30), torch.randn(20))
        assert rw.objective_value_ is not None
        assert isinstance(rw.objective_value_, float)
        # Loss can be slightly negative due to entropy term; check finite
        assert abs(rw.objective_value_) < float("inf")

    def test_empty_calibration_raises(self) -> None:
        rw = OTShiftReweighter()
        with pytest.raises(ValueError, match="must be non-empty"):
            rw.fit(torch.tensor([]), torch.randn(5))

    def test_empty_target_raises(self) -> None:
        rw = OTShiftReweighter()
        with pytest.raises(ValueError, match="must be non-empty"):
            rw.fit(torch.randn(5), torch.tensor([]))

    def test_single_element_each(self) -> None:
        rw = OTShiftReweighter(n_steps=30, learning_rate=0.1)
        rw.fit(torch.tensor([0.5]), torch.tensor([0.6]))
        assert rw.weights_ is not None
        assert rw.weights_.numel() == 1
        torch.testing.assert_close(rw.weights_.sum(), torch.tensor(1.0))

    def test_high_entropy_penalty_flattens_weights(self) -> None:
        """Large entropy_penalty pushes weights toward uniform."""
        torch.manual_seed(0)
        cal = torch.randn(30)
        tgt = cal + 0.5  # slight shift

        rw_high = OTShiftReweighter(n_steps=80, learning_rate=0.1, entropy_penalty=10.0)
        rw_high.fit(cal, tgt)

        rw_low = OTShiftReweighter(n_steps=80, learning_rate=0.1, entropy_penalty=1e-6)
        rw_low.fit(cal, tgt)

        # High entropy penalty → weights closer to uniform (lower std)
        std_high = float(rw_high.weights_.std())
        std_low = float(rw_low.weights_.std())
        assert std_high < std_low


# ═══════════════════════════════════════════════════════════════════════════════
# WeightedSplitConformalAdapter edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeightedSplitConformalAdapterEdge:
    def test_alpha_near_bounds(self) -> None:
        # alpha close to 0
        ad = WeightedSplitConformalAdapter(alpha=0.01)
        assert ad.alpha == 0.01

        # alpha close to 1
        ad2 = WeightedSplitConformalAdapter(alpha=0.99)
        assert ad2.alpha == 0.99

    def test_predict_before_calibrate_raises(self) -> None:
        ad = WeightedSplitConformalAdapter(alpha=0.1)
        with pytest.raises(RuntimeError, match="calibrate"):
            ad.predict_from_test_scores(torch.randn(3, 2))

    def test_coverage_diagnostics_before_calibrate_raises(self) -> None:
        ad = WeightedSplitConformalAdapter(alpha=0.1)
        with pytest.raises(RuntimeError, match="calibrate"):
            ad.coverage_diagnostics(torch.randn(5), torch.ones(5))

    def test_calibrate_weights_shape_mismatch_raises(self) -> None:
        ad = WeightedSplitConformalAdapter(alpha=0.1)
        with pytest.raises(ValueError, match="weights must match"):
            ad.calibrate(torch.randn(5), torch.ones(3))

    def test_coverage_diagnostics_shape_mismatch_raises(self) -> None:
        ad = WeightedSplitConformalAdapter(alpha=0.1)
        ad.calibrate(torch.randn(5), torch.ones(5))
        with pytest.raises(ValueError, match="weights must match"):
            ad.coverage_diagnostics(torch.randn(5), torch.ones(3))

    def test_predict_1d_candidate_raises(self) -> None:
        ad = WeightedSplitConformalAdapter(alpha=0.1)
        ad.calibrate(torch.randn(10), torch.ones(10))
        with pytest.raises(ValueError, match="2-D"):
            ad.predict_from_test_scores(torch.randn(5))

    def test_calibrate_returns_self(self) -> None:
        ad = WeightedSplitConformalAdapter(alpha=0.1)
        result = ad.calibrate(torch.randn(10), torch.ones(10))
        assert result is ad

    def test_predict_all_above_threshold(self) -> None:
        scores = torch.randn(20)
        ad = WeightedSplitConformalAdapter(alpha=0.1).calibrate(scores, torch.ones(20))
        # All candidates well above threshold → no inclusion
        cand = torch.full((5, 3), float(ad.threshold_) + 10.0)
        mask = ad.predict_from_test_scores(cand)
        assert not mask.any()

    def test_predict_all_below_threshold(self) -> None:
        scores = torch.randn(20)
        ad = WeightedSplitConformalAdapter(alpha=0.1).calibrate(scores, torch.ones(20))
        # All candidates well below threshold → all included
        cand = torch.full((5, 3), float(ad.threshold_) - 10.0)
        mask = ad.predict_from_test_scores(cand)
        assert mask.all()

    def test_predict_at_threshold(self) -> None:
        """Scores exactly at threshold should be included (<=)."""
        scores = torch.randn(20)
        ad = WeightedSplitConformalAdapter(alpha=0.1).calibrate(scores, torch.ones(20))
        cand = torch.full((3, 2), ad.threshold_)
        mask = ad.predict_from_test_scores(cand)
        assert mask.all()

    def test_coverage_diagnostics_keys(self) -> None:
        scores = torch.randn(30)
        w = torch.ones(30)
        ad = WeightedSplitConformalAdapter(alpha=0.1).calibrate(scores, w)
        diag = ad.coverage_diagnostics(scores, w)
        expected_keys = {
            "alpha",
            "threshold",
            "n_calibration",
            "nominal_coverage",
            "weighted_empirical_coverage",
            "coverage_gap",
            "calibration_ess_inv_square",
        }
        assert set(diag.keys()) == expected_keys
        assert diag["coverage_gap"] == pytest.approx(
            diag["weighted_empirical_coverage"] - diag["nominal_coverage"]
        )
