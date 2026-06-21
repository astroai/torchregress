"""
Supplemental edge-case tests for torchregress.test_time.selection.

Existing tests cover basic smoke-test paths. This file fills gaps:
dataclass integrity, internal helper edge cases, combined criteria,
single-sample paths, and convergence/consistency checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchregress.test_time.selection import (
    LocalConsistencyConfig,
    _sample_reference_indices,
    confidence_scores,
    entropy_scores,
    local_consistency_weights,
    pseudo_label_targets,
    select_high_confidence,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Dataclass integrity
# ═══════════════════════════════════════════════════════════════════════════════


class TestLocalConsistencyConfig:
    def test_defaults(self) -> None:
        """Defaults."""
        cfg = LocalConsistencyConfig()
        assert cfg.k == 5
        assert cfg.temperature == 1.0
        assert cfg.reference_size is None
        assert cfg.max_exact_rows == 4096
        assert cfg.query_chunk_size == 2048
        assert cfg.random_state == 0
        assert cfg.eps == 1e-8

    def test_custom(self) -> None:
        """Custom."""
        cfg = LocalConsistencyConfig(k=10, temperature=2.0, reference_size=512)
        assert cfg.k == 10
        assert cfg.temperature == 2.0
        assert cfg.reference_size == 512

    def test_frozen(self) -> None:
        """Frozen."""
        cfg = LocalConsistencyConfig()
        with pytest.raises(Exception):
            cfg.k = 10  # type: ignore[misc]

    def test_equality(self) -> None:
        """Equality."""
        assert LocalConsistencyConfig() == LocalConsistencyConfig()
        assert LocalConsistencyConfig(k=3) != LocalConsistencyConfig(k=5)


# ═══════════════════════════════════════════════════════════════════════════════
# _sample_reference_indices edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestSampleReferenceIndicesEdge:
    def test_zero_reference_size(self) -> None:
        """Zero reference size."""
        idx = _sample_reference_indices(10, 0, random_state=42)
        np.testing.assert_array_equal(idx, np.arange(10))

    def test_negative_reference_size(self) -> None:
        """Negative reference size."""
        idx = _sample_reference_indices(10, -5, random_state=42)
        np.testing.assert_array_equal(idx, np.arange(10))

    def test_reference_size_none(self) -> None:
        """Reference size none."""
        idx = _sample_reference_indices(10, None, random_state=42)
        np.testing.assert_array_equal(idx, np.arange(10))

    def test_reproducibility(self) -> None:
        """Reproducibility."""
        a = _sample_reference_indices(100, 20, random_state=42)
        b = _sample_reference_indices(100, 20, random_state=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        """Different seeds differ."""
        a = _sample_reference_indices(100, 20, random_state=42)
        b = _sample_reference_indices(100, 20, random_state=123)
        assert not np.array_equal(a, b)

    def test_sorted_output(self) -> None:
        """Sorted output."""
        idx = _sample_reference_indices(100, 20, random_state=42)
        assert np.all(np.diff(idx) > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# confidence_scores edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfidenceScoresEdge:
    def test_single_sample(self) -> None:
        """Single sample."""
        result = confidence_scores(np.array([[0.7, 0.3]]))
        assert result.shape == (1,)
        assert float(result[0]) == 0.7

    def test_all_zeros_assumes_uniform(self) -> None:
        """All zeros assumes uniform."""
        probs = np.array([[0.0, 0.0, 0.0]])
        result = confidence_scores(probs)
        assert float(result[0]) == 0.0

    def test_3d_input(self) -> None:
        """3d input."""
        # 3D input: axis=1 gives max over n_classes (axis=1)
        probs = np.array([[[0.1, 0.9], [0.8, 0.2]]])
        # This is shape (1, 2, 2) — .max(axis=1) gives (1, 2)
        result = confidence_scores(probs)
        assert result.shape == (1, 2)

    def test_many_classes(self) -> None:
        """Many classes."""
        n = 50
        probs = np.ones((3, n)) / n
        result = confidence_scores(probs)
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, 1.0 / n)


# ═══════════════════════════════════════════════════════════════════════════════
# entropy_scores edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEntropyScoresEdge:
    def test_custom_eps(self) -> None:
        """Custom eps."""
        probs = np.array([[0.0, 1.0]])
        result = entropy_scores(probs, eps=1e-12)
        assert np.all(np.isfinite(result))

    def test_single_sample(self) -> None:
        """Single sample."""
        result = entropy_scores(np.array([[0.7, 0.3]]))
        assert result.shape == (1,)
        assert float(result[0]) > 0

    def test_uniform_probs(self) -> None:
        """Uniform probabilities yield maximum entropy = log(n_classes)."""
        n = 5
        probs = np.ones((3, n)) / n
        result = entropy_scores(probs)
        expected = np.log(n)
        np.testing.assert_array_almost_equal(result, expected)

    def test_deterministic_probs_zero_entropy(self) -> None:
        """Deterministic probs zero entropy."""
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = entropy_scores(probs)
        np.testing.assert_array_almost_equal(result, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# pseudo_label_targets edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestPseudoLabelTargetsEdge:
    def test_single_sample(self) -> None:
        """Single sample."""
        labels, weights = pseudo_label_targets(np.array([[0.3, 0.7]]))
        assert labels.shape == (1,)
        assert weights.shape == (1,)
        assert int(labels[0]) == 1
        assert float(weights[0]) == 0.7

    def test_ties_breaks_to_first_max(self) -> None:
        """argmax returns first occurrence on ties."""
        labels, weights = pseudo_label_targets(np.array([[0.5, 0.5]]))
        assert int(labels[0]) == 0
        assert float(weights[0]) == 0.5

    def returns_float_weights(self) -> None:
        labels, weights = pseudo_label_targets(np.array([[0.8, 0.2]]))
        assert weights.dtype == float or np.issubdtype(weights.dtype, np.floating)


# ═══════════════════════════════════════════════════════════════════════════════
# select_high_confidence edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestSelectHighConfidenceEdge:
    def test_combined_criteria(self) -> None:
        """Combined criteria."""
        probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9], [0.55, 0.45]])
        mask = select_high_confidence(probs, min_confidence=0.7, max_entropy=0.5)
        # Row 0: conf=0.9, entropy low → True
        # Row 1: conf=0.6 < 0.7 → False
        # Row 2: conf=0.9, entropy low → True
        # Row 3: conf=0.55 < 0.7 → False
        np.testing.assert_array_equal(mask, [True, False, True, False])

    def test_top_fraction_one_keeps_all(self) -> None:
        """Top fraction one keeps all."""
        probs = np.random.default_rng(0).uniform(size=(10, 3))
        probs = probs / probs.sum(axis=1, keepdims=True)
        mask = select_high_confidence(probs, top_fraction=1.0)
        assert mask.sum() == 10

    def test_top_fraction_with_min_confidence(self) -> None:
        """top_fraction and min_confidence combined: AND logic."""
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.95, 0.05]])
        mask = select_high_confidence(probs, min_confidence=0.85, top_fraction=0.5)
        # min_confidence: rows [0, 3] qualify
        # top_fraction=0.5 of 4 = 2, top 2 confidences = rows 3 (0.95) and 0 (0.9)
        # Intersection = [0, 3]
        assert mask[0] and mask[3]
        assert not mask[1] and not mask[2]

    def test_min_count_forces_selection(self) -> None:
        """When no samples pass, min_count forces top-N by confidence."""
        probs = np.array([[0.5, 0.5], [0.4, 0.6], [0.3, 0.7]])
        mask = select_high_confidence(probs, min_confidence=0.99, min_count=2)
        assert mask.sum() == 2
        # Confidence scores: [0.5, 0.6, 0.7]; top 2 are rows 1 and 2
        assert mask[1]
        assert mask[2]
        assert not mask[0]

    def test_min_count_with_top_fraction(self) -> None:
        """Min count with top fraction."""
        probs = np.array([[0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7]])
        # top_fraction=0.25 of 4 = 1, but min_count=3 → should select at least 3
        mask = select_high_confidence(probs, top_fraction=0.25, min_count=3)
        assert mask.sum() == 3

    def test_invalid_top_fraction_zero(self) -> None:
        """Invalid top fraction zero."""
        with pytest.raises(ValueError, match="top_fraction must be"):
            select_high_confidence(np.ones((3, 2)), top_fraction=0.0)

    def test_invalid_top_fraction_negative(self) -> None:
        """Invalid top fraction negative."""
        with pytest.raises(ValueError, match="top_fraction must be"):
            select_high_confidence(np.ones((3, 2)), top_fraction=-0.5)

    def test_no_criteria_keeps_all(self) -> None:
        """No criteria keeps all."""
        probs = np.ones((10, 3))
        mask = select_high_confidence(probs)
        assert mask.sum() == 10
        assert mask.all()


# ═══════════════════════════════════════════════════════════════════════════════
# local_consistency_weights edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestLocalConsistencyWeightsEdge:
    def test_with_reference_size(self) -> None:
        """With reference size."""
        features = np.random.default_rng(0).normal(size=(50, 4))
        probs = np.random.default_rng(1).uniform(size=(50, 3))
        probs = probs / probs.sum(axis=1, keepdims=True)
        cfg = LocalConsistencyConfig(k=5, reference_size=20, random_state=42)
        weights = local_consistency_weights(features, probs, cfg)
        assert weights.shape == (50,)
        assert np.all(weights >= 0)
        assert np.all(np.isfinite(weights))

    def test_weights_mean_one(self) -> None:
        """Weights mean one."""
        features = np.random.default_rng(2).normal(size=(20, 3))
        probs = np.random.default_rng(3).uniform(size=(20, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        weights = local_consistency_weights(features, probs)
        assert weights.shape == (20,)
        np.testing.assert_allclose(weights.mean(), 1.0, atol=1e-10)

    def test_single_sample_returns_ones(self) -> None:
        """Single sample returns ones."""
        features = np.array([[0.0, 0.0]])
        probs = np.array([[0.5, 0.5]])
        weights = local_consistency_weights(features, probs)
        np.testing.assert_array_equal(weights, [1.0])

    def test_chunked_matches_unchunked(self) -> None:
        """Chunked matches unchunked."""
        rng = np.random.default_rng(4)
        features = rng.normal(size=(32, 4))
        probs = rng.uniform(size=(32, 3))
        probs = probs / probs.sum(axis=1, keepdims=True)
        cfg_chunked = LocalConsistencyConfig(k=5, query_chunk_size=8, random_state=0)
        cfg_full = LocalConsistencyConfig(k=5, query_chunk_size=None, random_state=0)
        w_chunked = local_consistency_weights(features, probs, cfg_chunked)
        w_full = local_consistency_weights(features, probs, cfg_full)
        np.testing.assert_allclose(w_chunked, w_full, atol=1e-12)

    def test_temperature_higher_flattens_weights(self) -> None:
        """Higher temperature → more uniform weights (all closer to 1)."""
        features = np.random.default_rng(5).normal(size=(30, 3))
        probs = np.random.default_rng(6).uniform(size=(30, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        w_cold = local_consistency_weights(features, probs, LocalConsistencyConfig(temperature=0.1))
        w_hot = local_consistency_weights(features, probs, LocalConsistencyConfig(temperature=10.0))
        # Hotter should have less variance (closer to uniform)
        assert w_hot.std() < w_cold.std()

    def test_identical_features_give_high_weights(self) -> None:
        """When all features are identical, agreement should be high everywhere."""
        features = np.tile(np.array([0.0, 0.0]), (10, 1))
        probs = np.random.default_rng(7).uniform(size=(10, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        weights = local_consistency_weights(features, probs, LocalConsistencyConfig(k=3))
        assert np.all(weights > 0)
        np.testing.assert_allclose(weights.mean(), 1.0, atol=1e-10)

    def test_raises_on_mismatched_shapes(self) -> None:
        """Raises on mismatched shapes."""
        with pytest.raises(ValueError, match="matching batch"):
            local_consistency_weights(np.ones((3, 2)), np.ones((4, 2)))

    def test_raises_on_non_2d_features(self) -> None:
        """Raises on non 2d features."""
        with pytest.raises(ValueError, match="matching batch"):
            local_consistency_weights(np.ones(5), np.ones((5, 2)))

    def test_max_exact_rows_guard(self) -> None:
        """Small dataset uses exact self-reference (no subsampling)."""
        features = np.random.default_rng(8).normal(size=(10, 2))
        probs = np.random.default_rng(9).uniform(size=(10, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        cfg = LocalConsistencyConfig(k=3, max_exact_rows=100, reference_size=None)
        weights = local_consistency_weights(features, probs, cfg)
        assert weights.shape == (10,)
        np.testing.assert_allclose(weights.mean(), 1.0, atol=1e-10)
