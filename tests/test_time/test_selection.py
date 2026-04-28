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


def test_sample_reference_indices():
    # If reference_size is None, returns all indices
    idx = _sample_reference_indices(10, None, random_state=42)
    np.testing.assert_array_equal(idx, np.arange(10))

    # If reference_size <= 0, returns all indices
    idx = _sample_reference_indices(10, 0, random_state=42)
    np.testing.assert_array_equal(idx, np.arange(10))
    idx = _sample_reference_indices(10, -5, random_state=42)
    np.testing.assert_array_equal(idx, np.arange(10))

    # If reference_size >= n_rows, returns all indices
    idx = _sample_reference_indices(10, 10, random_state=42)
    np.testing.assert_array_equal(idx, np.arange(10))
    idx = _sample_reference_indices(10, 15, random_state=42)
    np.testing.assert_array_equal(idx, np.arange(10))

    # Happy path
    idx = _sample_reference_indices(10, 5, random_state=42)
    assert len(idx) == 5
    assert np.all(np.diff(idx) > 0)  # Sorted
    assert np.all(idx >= 0) and np.all(idx < 10)


def test_entropy_scores():
    probs = np.array([[1.0, 0.0], [0.5, 0.5], [0.1, 0.9]])
    scores = entropy_scores(probs)
    # entropy of [1.0, 0.0] should be ~0
    assert np.isclose(scores[0], 0.0, atol=1e-5)
    # entropy of [0.5, 0.5] should be -log(0.5)
    assert np.isclose(scores[1], -np.log(0.5), atol=1e-5)

    # Test normalization
    probs_unnorm = np.array([[2.0, 2.0]])
    scores_unnorm = entropy_scores(probs_unnorm)
    assert np.isclose(scores_unnorm[0], -np.log(0.5), atol=1e-5)


def test_confidence_scores():
    probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    scores = confidence_scores(probs)
    np.testing.assert_array_almost_equal(scores, [0.8, 0.7, 0.5])


def test_pseudo_label_targets():
    probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    labels, weights = pseudo_label_targets(probs)
    np.testing.assert_array_equal(labels, [0, 1, 0])
    np.testing.assert_array_almost_equal(weights, [0.8, 0.7, 0.5])


def test_select_high_confidence():
    probs = np.array(
        [
            [0.9, 0.1],  # conf: 0.9, entropy: low
            [0.6, 0.4],  # conf: 0.6, entropy: mid
            [0.5, 0.5],  # conf: 0.5, entropy: high
            [0.2, 0.8],  # conf: 0.8, entropy: low
        ]
    )

    # Min confidence
    mask = select_high_confidence(probs, min_confidence=0.75)
    np.testing.assert_array_equal(mask, [True, False, False, True])

    # Max entropy
    mask = select_high_confidence(probs, max_entropy=0.6)
    np.testing.assert_array_equal(mask, [True, False, False, True])

    # Top fraction
    mask = select_high_confidence(probs, top_fraction=0.5)
    np.testing.assert_array_equal(mask, [True, False, False, True])

    # Top fraction ValueError
    with pytest.raises(ValueError, match="top_fraction must be in"):
        select_high_confidence(probs, top_fraction=1.5)

    with pytest.raises(ValueError, match="top_fraction must be in"):
        select_high_confidence(probs, top_fraction=-0.1)

    # Min count
    mask = select_high_confidence(probs, min_confidence=0.95, min_count=2)
    # None have >=0.95 conf natively except none, wait, max is 0.9.
    # So 0 passed conf threshold. min_count=2 ensures top 2 are selected
    np.testing.assert_array_equal(mask, [True, False, False, True])


def test_local_consistency_weights():
    features = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

    # With defaults
    weights = local_consistency_weights(features, probs)
    assert weights.shape == (4,)
    # Normalized weights should mean to 1.0
    assert np.isclose(weights.mean(), 1.0)

    # Single feature row
    weights_single = local_consistency_weights(features[:1], probs[:1])
    np.testing.assert_array_equal(weights_single, [1.0])

    # Feature / probs mismatch
    with pytest.raises(ValueError, match="matching batch dimensions"):
        local_consistency_weights(features, probs[:3])

    # Try custom config
    config = LocalConsistencyConfig(k=1, temperature=0.5, reference_size=2, query_chunk_size=2)
    weights_custom = local_consistency_weights(features, probs, config=config)
    assert weights_custom.shape == (4,)
