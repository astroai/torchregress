"""
Supplemental edge-case tests for torchregress.test_time.label_shift.

Existing tests cover the main smoke-test paths. This file fills gaps:
dataclass integrity, internal helpers, gaussian bin edge cases,
convergence behavior, and error paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchregress.test_time.label_shift import (
    GaussianLabelShiftConfig,
    LabelShiftEMConfig,
    LabelShiftEstimate,
    PosteriorLabelShiftAdapter,
    _normalize_rows,
    _subsample_probabilities,
    apply_label_shift_correction,
    correct_gaussian_predictions_for_label_shift,
    estimate_target_prior_em,
    gaussian_bin_edges_from_targets,
    gaussian_bin_probabilities,
    gaussian_moments_from_binned_probabilities,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Dataclass integrity
# ═══════════════════════════════════════════════════════════════════════════════


class TestLabelShiftEMConfig:
    def test_defaults(self) -> None:
        """Defaults."""
        cfg = LabelShiftEMConfig()
        assert cfg.max_iter == 100
        assert cfg.tol == 1e-6
        assert cfg.eps == 1e-8

    def test_custom_values(self) -> None:
        """Custom values."""
        cfg = LabelShiftEMConfig(max_iter=10, tol=0.01, eps=1e-12)
        assert cfg.max_iter == 10
        assert cfg.tol == 0.01
        assert cfg.eps == 1e-12

    def test_frozen(self) -> None:
        """Frozen."""
        cfg = LabelShiftEMConfig()
        with pytest.raises(Exception, match="cannot assign"):
            cfg.max_iter = 200  # type: ignore[misc]

    def test_equality(self) -> None:
        """Equality."""
        assert LabelShiftEMConfig() == LabelShiftEMConfig()
        assert LabelShiftEMConfig(max_iter=10) != LabelShiftEMConfig(max_iter=20)


class TestLabelShiftEstimate:
    def test_fields(self) -> None:
        """Fields."""
        src = np.array([0.3, 0.7])
        tgt = np.array([0.5, 0.5])
        est = LabelShiftEstimate(src, tgt, 5, True)
        np.testing.assert_array_equal(est.source_prior, src)
        np.testing.assert_array_equal(est.target_prior, tgt)
        assert est.iterations == 5
        assert est.converged is True

    def test_frozen(self) -> None:
        """Frozen."""
        est = LabelShiftEstimate(np.array([0.5, 0.5]), np.array([0.5, 0.5]), 1, True)
        with pytest.raises(Exception, match="cannot assign"):
            est.iterations = 10  # type: ignore[misc]

    def test_not_converged(self) -> None:
        """Not converged."""
        est = LabelShiftEstimate(np.array([0.5, 0.5]), np.array([0.5, 0.5]), 100, False)
        assert est.converged is False
        assert est.iterations == 100


class TestGaussianLabelShiftConfig:
    def test_defaults(self) -> None:
        """Defaults."""
        cfg = GaussianLabelShiftConfig()
        assert cfg.n_bins == 32
        assert cfg.estimation_rows is None
        assert cfg.top_fraction == 0.5
        assert cfg.reference_size == 2048
        assert cfg.seed == 0
        assert cfg.eps == 1e-8

    def test_custom(self) -> None:
        """Custom."""
        cfg = GaussianLabelShiftConfig(n_bins=8, estimation_rows=100, top_fraction=0.3)
        assert cfg.n_bins == 8
        assert cfg.estimation_rows == 100
        assert cfg.top_fraction == 0.3

    def test_frozen(self) -> None:
        """Frozen."""
        cfg = GaussianLabelShiftConfig()
        with pytest.raises(Exception, match="cannot assign"):
            cfg.n_bins = 16  # type: ignore[misc]

    def test_top_fraction_none(self) -> None:
        """Top fraction none."""
        cfg = GaussianLabelShiftConfig(top_fraction=None)
        assert cfg.top_fraction is None


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeRows:
    def test_already_normalized(self) -> None:
        """Already normalized."""
        probs = np.array([[0.2, 0.8], [0.5, 0.5]])
        result = _normalize_rows(probs, eps=1e-8)
        np.testing.assert_array_almost_equal(result, probs)

    def test_unnormalized(self) -> None:
        """Unnormalized."""
        raw = np.array([[2.0, 3.0], [1.0, 1.0]])
        result = _normalize_rows(raw, eps=1e-8)
        np.testing.assert_array_almost_equal(result[0], [2 / 5, 3 / 5])
        np.testing.assert_array_almost_equal(result[1], [0.5, 0.5])

    def test_eps_clamping(self) -> None:
        """Zero values get clamped up by eps before normalization."""
        raw = np.array([[0.0, 1.0], [1e-12, 1.0]])
        result = _normalize_rows(raw, eps=1e-6)
        assert np.all(result >= 0.0)
        # Clamped then normalized: 0.0 -> eps; row becomes [eps/(1+eps), 1/(1+eps)]
        # First element is just under eps; check it's nonzero and rows sum to 1
        assert result[0, 0] > 0
        np.testing.assert_array_almost_equal(result.sum(axis=1), 1.0)


class TestSubsampleProbabilities:
    def test_no_subsample_when_none(self) -> None:
        """No subsample when none."""
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        p, w = _subsample_probabilities(probs, None, None, random_state=0)
        np.testing.assert_array_equal(p, probs)
        assert w is None

    def test_no_subsample_when_zero_or_negative(self) -> None:
        """No subsample when zero or negative."""
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        p, w = _subsample_probabilities(probs, None, 0, random_state=0)
        np.testing.assert_array_equal(p, probs)
        p2, _ = _subsample_probabilities(probs, None, -5, random_state=0)
        np.testing.assert_array_equal(p2, probs)

    def test_no_subsample_when_sample_size_exceeds_n(self) -> None:
        """No subsample when sample size exceeds n."""
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        p, _ = _subsample_probabilities(probs, None, 100, random_state=0)
        np.testing.assert_array_equal(p, probs)

    def test_subsamples_correctly(self) -> None:
        """Subsamples correctly."""
        probs = np.random.default_rng(0).uniform(size=(10, 3))
        probs = probs / probs.sum(axis=1, keepdims=True)
        p, w = _subsample_probabilities(probs, None, 5, random_state=42)
        assert p.shape == (5, 3)
        assert w is None
        # All rows should be from original
        for row in p:
            assert any(np.allclose(row, orig) for orig in probs)

    def test_subsample_with_weights(self) -> None:
        """Subsample with weights."""
        probs = np.random.default_rng(1).uniform(size=(10, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        weights = np.arange(10, dtype=float)
        p, w = _subsample_probabilities(probs, weights, 5, random_state=42)
        assert p.shape == (5, 2)
        assert w is not None
        assert w.shape == (5,)
        # Weights should be the same indices as probs were kept
        for pw, pw_row in enumerate(p):
            # Compare each row against all original rows
            row_match = np.all(np.isclose(probs, pw_row, atol=1e-10), axis=1)
            orig_idx = int(np.flatnonzero(row_match)[0])
            assert w[pw] == pytest.approx(weights[orig_idx])

    def test_reproducibility(self) -> None:
        """Reproducibility."""
        probs = np.random.default_rng(2).uniform(size=(20, 3))
        probs = probs / probs.sum(axis=1, keepdims=True)
        p1, _ = _subsample_probabilities(probs, None, 8, random_state=123)
        p2, _ = _subsample_probabilities(probs, None, 8, random_state=123)
        np.testing.assert_array_equal(p1, p2)

    def test_preserves_row_order(self) -> None:
        """Subsampled rows should be in their original relative order."""
        probs = np.arange(20, dtype=float).reshape(10, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)  # row i sums to 1
        p, _ = _subsample_probabilities(probs, None, 6, random_state=42)
        # First column identifies original row: row i → i*2 / norm
        indices = []
        for row in p:
            for i in range(10):
                if np.allclose(row, probs[i]):
                    indices.append(i)
                    break
        assert indices == sorted(indices)


# ═══════════════════════════════════════════════════════════════════════════════
# apply_label_shift_correction edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestApplyLabelShiftCorrectionEdge:
    def test_custom_eps(self) -> None:
        """Custom eps."""
        probs = np.array([[0.0, 1.0]])
        src = np.array([0.5, 0.5])
        tgt = np.array([0.5, 0.5])
        result = apply_label_shift_correction(probs, source_prior=src, target_prior=tgt, eps=1e-12)
        np.testing.assert_array_almost_equal(result.sum(), 1.0)

    def test_single_sample(self) -> None:
        """Single sample."""
        probs = np.array([[0.7, 0.3]])
        src = np.array([0.4, 0.6])
        tgt = np.array([0.6, 0.4])
        result = apply_label_shift_correction(probs, source_prior=src, target_prior=tgt)
        assert result.shape == (1, 2)
        np.testing.assert_array_almost_equal(result.sum(), 1.0)

    def test_many_classes(self) -> None:
        """Many classes."""
        n = 5
        probs = np.ones((3, n)) / n
        src = np.ones(n) / n
        tgt = np.ones(n) / n
        result = apply_label_shift_correction(probs, source_prior=src, target_prior=tgt)
        assert result.shape == (3, n)
        np.testing.assert_array_almost_equal(result, probs)


# ═══════════════════════════════════════════════════════════════════════════════
# estimate_target_prior_em edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEstimateTargetPriorEMEdge:
    def test_converges_quickly_with_balanced_data(self) -> None:
        """Converges quickly with balanced data."""
        probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        cfg = LabelShiftEMConfig(max_iter=50, tol=1e-6)
        est = estimate_target_prior_em(probs, source_prior=np.array([0.5, 0.5]), config=cfg)
        assert est.converged is True
        assert est.iterations <= 5  # should converge quickly

    def test_em_runs_multiple_iterations_with_tight_tol(self) -> None:
        """With an explicit source_prior far from the data, EM takes multiple
        steps before converging (even with very tight tol)."""
        # 50 samples all suggesting class 0 (target prior near [1, 0])
        probs = np.tile(np.array([0.95, 0.05]), (50, 1))
        # Source prior is opposite → EM must shift significantly
        source_prior = np.array([0.1, 0.9])
        cfg = LabelShiftEMConfig(max_iter=10, tol=1e-12)
        est = estimate_target_prior_em(probs, source_prior=source_prior, config=cfg)
        assert est.iterations > 1
        np.testing.assert_array_almost_equal(est.target_prior.sum(), 1.0)

    def test_single_sample(self) -> None:
        """Single sample."""
        probs = np.array([[0.7, 0.3]])
        est = estimate_target_prior_em(
            probs, source_prior=np.array([0.7, 0.3]), config=LabelShiftEMConfig(max_iter=10)
        )
        assert est.target_prior.shape == (2,)
        np.testing.assert_array_almost_equal(est.target_prior.sum(), 1.0)

    def test_many_classes(self) -> None:
        """Many classes."""
        n = 5
        probs = np.ones((10, n)) / n
        est = estimate_target_prior_em(
            probs, source_prior=np.ones(n) / n, config=LabelShiftEMConfig(max_iter=10)
        )
        assert est.target_prior.shape == (n,)
        # Uniform input → uniform target prior
        np.testing.assert_array_almost_equal(est.target_prior, np.ones(n) / n)

    def test_weighted_subsample_estimate(self) -> None:
        """Weighted subsample estimate."""
        probs = np.random.default_rng(0).uniform(size=(20, 3))
        probs = probs / probs.sum(axis=1, keepdims=True)
        weights = np.random.default_rng(1).uniform(0.5, 2.0, size=20)
        est = estimate_target_prior_em(
            probs,
            source_prior=np.array([1 / 3, 1 / 3, 1 / 3]),
            sample_weights=weights,
            sample_size=10,
            random_state=42,
        )
        assert est.target_prior.shape == (3,)
        np.testing.assert_array_almost_equal(est.target_prior.sum(), 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# PosteriorLabelShiftAdapter edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestPosteriorLabelShiftAdapterEdge:
    def test_with_sample_size(self) -> None:
        """With sample size."""
        probs = np.random.default_rng(0).uniform(size=(20, 3))
        probs = probs / probs.sum(axis=1, keepdims=True)
        adapter = PosteriorLabelShiftAdapter(
            source_prior=np.array([1 / 3, 1 / 3, 1 / 3]), sample_size=10, random_state=42
        )
        est = adapter.estimate(probs)
        corrected = adapter.transform(probs)
        assert corrected.shape == probs.shape
        assert est.target_prior.shape == (3,)

    def test_with_custom_config(self) -> None:
        """With custom config."""
        cfg = LabelShiftEMConfig(max_iter=5, tol=1e-10)
        adapter = PosteriorLabelShiftAdapter(source_prior=np.array([0.5, 0.5]), config=cfg)
        est = adapter.estimate(np.array([[0.5, 0.5], [0.5, 0.5]]))
        assert est.iterations <= 5

    def test_estimate_stores_last_estimate(self) -> None:
        """Estimate stores last estimate."""
        adapter = PosteriorLabelShiftAdapter(source_prior=np.array([0.5, 0.5]))
        probs = np.array([[0.8, 0.2], [0.2, 0.8]])
        assert adapter.last_estimate is None
        est = adapter.estimate(probs)
        assert adapter.last_estimate is est

    def test_transform_auto_estimate_when_no_last_estimate(self) -> None:
        """Transform auto estimate when no last estimate."""
        adapter = PosteriorLabelShiftAdapter(source_prior=np.array([0.5, 0.5]))
        probs = np.array([[0.8, 0.2]])
        # last_estimate is None, target_prior is None → should call estimate()
        corrected = adapter.transform(probs)
        assert corrected.shape == probs.shape
        assert adapter.last_estimate is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Gaussian bin operations — edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestGaussianBinEdgesEdge:
    def test_single_bin(self) -> None:
        """Single bin."""
        targets = np.array([0.0, 5.0, 10.0])
        edges = gaussian_bin_edges_from_targets(targets, n_bins=1)
        # n_bins=1: linspace(0, 1, 2) → [0, 1] quantiles → 2 edges, but unique may collapse
        # Falls to linspace(0, 10, 2) → [0, 10]
        assert edges.shape[0] >= 2

    def test_many_bins(self) -> None:
        """Many bins."""
        targets = np.linspace(0, 100, 1000)
        edges = gaussian_bin_edges_from_targets(targets, n_bins=50)
        assert edges.shape[0] == 51

    def test_all_same_value(self) -> None:
        """All same value."""
        targets = np.array([7.0, 7.0, 7.0])
        edges = gaussian_bin_edges_from_targets(targets, n_bins=3)
        assert edges[0] == 7.0
        assert edges[-1] == 8.0  # hi = lo + 1


class TestGaussianBinProbabilitiesEdge:
    def test_single_bin(self) -> None:
        """Single bin."""
        mean = np.array([0.0])
        std = np.array([1.0])
        edges = np.array([-np.inf, np.inf])
        probs = gaussian_bin_probabilities(mean, std, edges)
        assert probs.shape == (1, 1)
        np.testing.assert_array_almost_equal(probs[0, 0], 1.0)

    def test_eps_clamps_std(self) -> None:
        """std=0 should be clamped to eps."""
        mean = np.array([0.0])
        std = np.array([0.0])
        edges = np.array([-1.0, 0.0, 1.0])
        probs = gaussian_bin_probabilities(mean, std, edges, eps=1e-8)
        assert np.all(np.isfinite(probs))
        np.testing.assert_array_almost_equal(probs.sum(), 1.0)

    def test_wide_gaussian_flat_across_bins(self) -> None:
        """Very wide Gaussian yields roughly equal probabilities across bins."""
        mean = np.array([0.0])
        std = np.array([100.0])
        edges = np.array([-1.0, 0.0, 1.0])
        probs = gaussian_bin_probabilities(mean, std, edges)
        assert probs.shape == (1, 2)
        np.testing.assert_array_almost_equal(probs[0], [0.5, 0.5], decimal=2)


class TestGaussianMomentsEdge:
    def test_roundtrip_from_uniform(self) -> None:
        """Uniform bin probabilities → mean at center, std > 0."""
        probs = np.array([[0.5, 0.5], [0.25, 0.75]])
        edges = np.array([0.0, 1.0, 2.0])
        mean, std = gaussian_moments_from_binned_probabilities(probs, edges)
        assert mean.shape == (2,)
        assert std.shape == (2,)
        assert np.all(std > 0)

    def test_delta_spike_yields_near_zero_std(self) -> None:
        """One bin with probability 1 → std near eps bound."""
        probs = np.array([[1.0, 0.0]])
        edges = np.array([0.0, 1.0, 2.0])
        mean, std = gaussian_moments_from_binned_probabilities(probs, edges, eps=1e-8)
        assert mean[0] == pytest.approx(0.5)  # center of first bin
        assert std[0] < 0.01  # near-zero variance (clamped to eps)

    def test_many_bins(self) -> None:
        """Many bins."""
        n_bins = 10
        probs = np.ones((3, n_bins)) / n_bins
        edges = np.linspace(0, 1, n_bins + 1)
        mean, std = gaussian_moments_from_binned_probabilities(probs, edges)
        assert mean.shape == (3,)
        assert std.shape == (3,)


# ═══════════════════════════════════════════════════════════════════════════════
# correct_gaussian_predictions_for_label_shift edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestCorrectGaussianPredictionsEdge:
    def test_estimation_rows_subsamples(self) -> None:
        """Estimation rows subsamples."""
        mean = np.array([0.0, 1.0, 2.0])
        std = np.array([1.0, 1.0, 1.0])
        source_targets = np.linspace(0, 5, 50)
        cfg = GaussianLabelShiftConfig(n_bins=5, estimation_rows=20)
        c_mean, c_std, meta = correct_gaussian_predictions_for_label_shift(
            mean=mean, std=std, source_targets=source_targets, config=cfg
        )
        assert c_mean.shape == (3,)
        assert c_std.shape == (3,)
        assert meta["estimate_converged"] in (True, False)

    def test_without_features(self) -> None:
        """Without features."""
        mean = np.array([0.0, 1.0])
        std = np.array([0.5, 0.5])
        source_targets = np.linspace(0, 3, 20)
        cfg = GaussianLabelShiftConfig(n_bins=4, top_fraction=None)
        c_mean, c_std, meta = correct_gaussian_predictions_for_label_shift(
            mean=mean, std=std, source_targets=source_targets, features=None, config=cfg
        )
        assert c_mean.shape == (2,)
        assert meta["selected_rows"] > 0
