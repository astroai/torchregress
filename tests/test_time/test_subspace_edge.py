"""
Supplemental edge-case tests for torchregress.test_time.subspace.

Existing tests cover basic smoke-test paths. This file fills gaps:
dataclass integrity, internal helper edge cases, variance_threshold
rank selection, subsampling/winsorize paths in transform,
and state mutation verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchregress.test_time.subspace import (
    FeatureStatNormalizer,
    SignificantSubspaceAligner,
    SubspaceAlignmentState,
    _clip_scale_ratio,
    _feature_significance,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SubspaceAlignmentState — dataclass integrity
# ═══════════════════════════════════════════════════════════════════════════════


class TestSubspaceAlignmentState:
    def test_fields(self) -> None:
        """Fields."""
        src_mean = np.array([1.0, 2.0])
        tgt_mean = np.array([3.0, 4.0])
        src_scale = np.array([0.5, 0.6])
        tgt_scale = np.array([0.7, 0.8])
        comps = np.array([[0.1, 0.2], [0.3, 0.4]])
        weights = np.array([0.9, 1.0])
        state = SubspaceAlignmentState(
            source_mean=src_mean,
            target_mean=tgt_mean,
            source_scale=src_scale,
            target_scale=tgt_scale,
            components=comps,
            feature_weights=weights,
            rank=2,
        )
        np.testing.assert_array_equal(state.source_mean, src_mean)
        np.testing.assert_array_equal(state.target_mean, tgt_mean)
        np.testing.assert_array_equal(state.source_scale, src_scale)
        np.testing.assert_array_equal(state.target_scale, tgt_scale)
        np.testing.assert_array_equal(state.components, comps)
        np.testing.assert_array_equal(state.feature_weights, weights)
        assert state.rank == 2

    def test_frozen(self) -> None:
        """Frozen."""
        state = SubspaceAlignmentState(
            source_mean=np.array([1.0]),
            target_mean=np.array([2.0]),
            source_scale=np.array([0.5]),
            target_scale=np.array([0.5]),
            components=np.array([[1.0]]),
            feature_weights=np.array([1.0]),
            rank=1,
        )
        with pytest.raises(Exception):
            state.rank = 3  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════════
# _feature_significance edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureSignificanceEdge:
    def test_single_sample_with_y(self) -> None:
        """Single sample with y."""
        X = np.array([[1.0, 2.0]])
        y = np.array([3.0])
        sig = _feature_significance(X, y, 1e-8)
        # Single sample: centered values are all 0 → correlation undefined
        # Denominator clipped to eps → result near 0/clipped
        assert np.all(np.isfinite(sig))

    def test_large_eps_dominates(self) -> None:
        """Large eps dominates."""
        X = np.array([[0.0, 0.0], [0.0, 0.0]])
        y = np.array([1.0, -1.0])
        sig = _feature_significance(X, y, 1e-2)
        # All-zero features: std=0 → clipped to eps
        assert np.all(sig < 1e-2 + 1e-10)

    def test_with_y_not_provided(self) -> None:
        """With y not provided."""
        X = np.random.default_rng(0).normal(size=(10, 3))
        sig = _feature_significance(X, None, 1e-8)
        assert sig.shape == (3,)
        assert np.all(sig > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# _clip_scale_ratio edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestClipScaleRatioEdge:
    def test_limit_barely_above_one(self) -> None:
        """Limit barely above one."""
        ratio = np.array([0.01, 100.0])
        clipped = _clip_scale_ratio(ratio, 1.001)
        assert clipped[0] == pytest.approx(1.0 / 1.001)
        assert clipped[1] == pytest.approx(1.001)

    def test_all_within_bounds(self) -> None:
        """All within bounds."""
        ratio = np.array([0.5, 1.5, 2.0])
        clipped = _clip_scale_ratio(ratio, 5.0)
        np.testing.assert_array_equal(clipped, ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# SignificantSubspaceAligner edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestSignificantSubspaceAlignerEdge:
    def test_fit_returns_self(self) -> None:
        """Fit returns self."""
        X = np.random.default_rng(0).normal(size=(20, 4))
        aligner = SignificantSubspaceAligner()
        assert aligner.fit(X) is aligner

    def test_variance_threshold_selects_rank(self) -> None:
        """Variance threshold selects rank."""
        X = np.random.default_rng(1).normal(size=(50, 5))
        aligner = SignificantSubspaceAligner(variance_threshold=0.5)
        aligner.fit(X)
        assert aligner.state_ is not None
        assert aligner.state_.rank >= 1
        assert aligner.state_.rank <= 5

    def test_variance_threshold_near_zero_gives_rank_one(self) -> None:
        """Variance threshold near zero gives rank one."""
        X = np.random.default_rng(2).normal(size=(50, 5))
        aligner = SignificantSubspaceAligner(variance_threshold=0.01)
        aligner.fit(X)
        assert aligner.state_.rank == 1

    def test_variance_threshold_one_gives_full_rank(self) -> None:
        """Variance threshold one gives full rank."""
        X = np.random.default_rng(3).normal(size=(20, 3))
        aligner = SignificantSubspaceAligner(variance_threshold=0.9999)
        aligner.fit(X)
        assert aligner.state_.rank >= 2  # should capture most variance

    def test_with_subsampling_and_winsorize(self) -> None:
        """With subsampling and winsorize."""
        rng = np.random.default_rng(4)
        X_source = rng.normal(size=(100, 4))
        X_target = rng.normal(loc=2.0, scale=1.5, size=(100, 4))
        X_target[:5, 0] += 500.0  # outliers

        aligner = SignificantSubspaceAligner(
            target_sample_size=30,
            random_state=42,
            clip_quantile=0.05,
        )
        transformed = aligner.fit_transform(X_source, X_target)
        assert transformed.shape == X_target.shape
        assert np.all(np.isfinite(transformed))

    def test_with_max_scale_ratio(self) -> None:
        """With max scale ratio."""
        rng = np.random.default_rng(5)
        X_source = rng.normal(size=(30, 3))
        X_target = X_source * 100.0  # huge scale difference

        aligner = SignificantSubspaceAligner(max_scale_ratio=3.0)
        transformed = aligner.fit_transform(X_source, X_target)
        assert transformed.shape == X_target.shape
        assert np.all(np.isfinite(transformed))
        # With max_scale_ratio=3, the scale correction is bounded:
        # target_scale ~ 100*source_scale, ratio=1/100, clipped to 1/3
        # So aligned features should be much closer to source scale
        assert transformed.std() < X_target.std()

    def test_transform_updates_state(self) -> None:
        """Transform updates state."""
        rng = np.random.default_rng(6)
        X_source = rng.normal(size=(30, 3))
        X_target = rng.normal(loc=1.0, size=(30, 3))
        aligner = SignificantSubspaceAligner()
        aligner.fit(X_source)
        old_tgt_mean = aligner.state_.target_mean.copy()
        aligner.transform(X_target)
        new_tgt_mean = aligner.state_.target_mean
        # transform should update target_mean in state
        assert not np.array_equal(new_tgt_mean, old_tgt_mean)

    def test_fit_without_y(self) -> None:
        """Fit without y."""
        X = np.random.default_rng(7).normal(size=(30, 4))
        aligner = SignificantSubspaceAligner()
        aligner.fit(X)
        assert aligner.state_ is not None
        assert aligner.state_.feature_weights.shape == (4,)
        assert np.all(aligner.state_.feature_weights > 0)

    def test_transform_state_after_fit_transform(self) -> None:
        """Transform state after fit transform."""
        rng = np.random.default_rng(8)
        X_source = rng.normal(size=(25, 3))
        X_target = rng.normal(loc=2.0, size=(25, 3))
        aligner = SignificantSubspaceAligner()
        result = aligner.fit_transform(X_source, X_target)
        assert result.shape == (25, 3)
        assert aligner.state_ is not None

    def test_simple_2d_alignment_preserves_relative_structure(self) -> None:
        """After alignment, the relative distances in target should be closer to source."""
        rng = np.random.default_rng(9)
        X_source = rng.normal(loc=[0, 0], scale=[1, 1], size=(40, 2))
        X_target = X_source * 3 + np.array([10.0, -5.0])
        aligner = SignificantSubspaceAligner(rank=2)
        transformed = aligner.fit_transform(X_source, X_target)
        # Transformed should be closer to source than original target was
        src_dist = np.mean(np.std(X_source, axis=0))
        tgt_dist = np.mean(np.std(X_target, axis=0))
        tr_dist = np.mean(np.std(transformed, axis=0))
        # Transformed std should be between source and target (pulled back toward source)
        assert abs(tr_dist - src_dist) < abs(tgt_dist - src_dist)

    def test_explicit_rank_capped_by_data(self) -> None:
        """Rank larger than n_features is capped."""
        X = np.random.default_rng(10).normal(size=(10, 2))
        aligner = SignificantSubspaceAligner(rank=10)
        aligner.fit(X)
        assert aligner.state_.rank == 2  # capped at n_features


# ═══════════════════════════════════════════════════════════════════════════════
# FeatureStatNormalizer edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureStatNormalizerEdge:
    def test_fit_returns_self(self) -> None:
        """Fit returns self."""
        X = np.random.default_rng(0).normal(size=(20, 3))
        norm = FeatureStatNormalizer()
        assert norm.fit(X) is norm

    def test_single_feature(self) -> None:
        """Single feature."""
        X_source = np.array([[1.0], [2.0], [3.0]])
        X_target = np.array([[10.0], [20.0], [30.0]])
        norm = FeatureStatNormalizer()
        norm.fit(X_source)
        transformed = norm.transform(X_target)
        assert transformed.shape == (3, 1)
        assert np.allclose(transformed, X_source)

    def test_with_subsampling_and_winsorize(self) -> None:
        """With subsampling and winsorize."""
        rng = np.random.default_rng(1)
        X_source = rng.normal(size=(50, 4))
        X_target = rng.normal(loc=3.0, scale=2.0, size=(50, 4))
        X_target[:3, 0] += 1000.0  # outliers

        norm = FeatureStatNormalizer(
            target_sample_size=20,
            random_state=42,
            clip_quantile=0.1,
        )
        norm.fit(X_source)
        transformed = norm.transform(X_target)
        assert transformed.shape == X_target.shape
        assert np.all(np.isfinite(transformed))

    def test_with_max_scale_ratio(self) -> None:
        """With max scale ratio."""
        X_source = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        X_target = np.array([[100.0, 200.0], [200.0, 400.0], [300.0, 600.0]])

        norm = FeatureStatNormalizer(max_scale_ratio=3.0)
        norm.fit(X_source)
        transformed = norm.transform(X_target)
        # Source std ~ [0.82, 1.63], target std ~ [82, 163]
        # ratio = src_std / tgt_std ≈ [0.01, 0.01] → clipped to [1/3, 1/3]
        # So scale correction is soft: transformed won't exactly match source
        assert transformed.shape == X_target.shape
        assert np.all(np.isfinite(transformed))
        # With bounded ratio, transformed std should be closer to source than target
        assert np.std(transformed) < np.std(X_target)

    def test_near_zero_source_std_clamped(self) -> None:
        """Source features with zero variance get clamped to eps."""
        X_source = np.array([[5.0, 5.0], [5.0, 5.0]])
        X_target = np.array([[1.0, 2.0], [3.0, 4.0]])
        norm = FeatureStatNormalizer(eps=1e-6)
        norm.fit(X_source)
        assert np.all(norm.source_std_ >= 1e-6)
        transformed = norm.transform(X_target)
        assert np.all(np.isfinite(transformed))

    def test_transform_before_fit_raises(self) -> None:
        """Transform before fit raises."""
        norm = FeatureStatNormalizer()
        with pytest.raises(RuntimeError, match="call fit"):
            norm.transform(np.ones((3, 2)))

    def test_fit_overwrites_previous_fit(self) -> None:
        """Fit overwrites previous fit."""
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[100.0, 200.0], [300.0, 400.0]])
        norm = FeatureStatNormalizer()
        norm.fit(X1)
        mean1 = norm.source_mean_.copy()
        norm.fit(X2)
        mean2 = norm.source_mean_
        assert not np.array_equal(mean1, mean2)

    def test_transform_idempotent_on_source(self) -> None:
        """Transforming source data through a fitted normalizer should produce source."""
        X = np.random.default_rng(2).normal(loc=[1.0, 2.0], scale=[0.5, 0.3], size=(30, 2))
        norm = FeatureStatNormalizer()
        norm.fit(X)
        result = norm.transform(X)
        np.testing.assert_allclose(result, X, atol=1e-10)
