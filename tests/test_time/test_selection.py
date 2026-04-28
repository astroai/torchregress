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


def test_entropy_scores():
    probs = np.array([[1.0, 0.0], [0.5, 0.5], [0.1, 0.9], [0.0, 0.0]])
    scores = entropy_scores(probs)
    assert len(scores) == 4
    assert np.isclose(scores[0], 0, atol=1e-5)
    assert np.isclose(scores[1], -np.log(0.5), atol=1e-5)
    assert np.isclose(scores[2], -0.1 * np.log(0.1) - 0.9 * np.log(0.9), atol=1e-5)
    assert np.isclose(scores[3], -np.log(0.5), atol=1e-5)


def test_confidence_scores():
    probs = np.array([[1.0, 0.0], [0.5, 0.5], [0.1, 0.9]])
    scores = confidence_scores(probs)
    assert np.allclose(scores, [1.0, 0.5, 0.9])


def test_pseudo_label_targets():
    probs = np.array([[0.8, 0.2], [0.4, 0.6], [0.5, 0.5]])
    labels, weights = pseudo_label_targets(probs)
    assert np.array_equal(labels, [0, 1, 0])
    assert np.allclose(weights, [0.8, 0.6, 0.5])


def test_select_high_confidence():
    probs = np.array(
        [
            [0.9, 0.1],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.1, 0.9],
        ]
    )

    mask = select_high_confidence(probs, min_confidence=0.8)
    assert np.array_equal(mask, [True, False, False, True])

    max_ent = -0.7 * np.log(0.7) - 0.3 * np.log(0.3)
    mask = select_high_confidence(probs, max_entropy=max_ent)
    assert np.array_equal(mask, [True, False, False, True])

    mask = select_high_confidence(probs, top_fraction=0.5)
    assert np.array_equal(mask, [True, False, False, True])

    mask = select_high_confidence(probs, min_confidence=0.95, min_count=1)
    assert mask.sum() == 1
    assert mask[0] or mask[3]

    with pytest.raises(ValueError, match="top_fraction must be in"):
        select_high_confidence(probs, top_fraction=1.5)


def test_local_consistency_weights():
    features = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0]])
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])

    config = LocalConsistencyConfig(k=2)
    weights = local_consistency_weights(features, probs, config)
    assert weights.shape == (3,)
    assert np.isclose(weights.mean(), 1.0)

    # Test batch dim mismatch
    with pytest.raises(ValueError, match="matching batch dimensions"):
        local_consistency_weights(np.array([[1.0]]), probs)

    # Test shape[0] == 1
    weights_single = local_consistency_weights(features[:1], probs[:1])
    assert np.array_equal(weights_single, [1.0])

    # Test chunk size and large exact
    large_features = np.random.randn(10, 2)
    large_probs = np.random.uniform(size=(10, 2))
    large_probs = large_probs / large_probs.sum(axis=1, keepdims=True)
    config_chunk = LocalConsistencyConfig(query_chunk_size=2)
    weights_chunked = local_consistency_weights(large_features, large_probs, config_chunk)
    assert weights_chunked.shape == (10,)


def test_sample_reference_indices():
    idx = _sample_reference_indices(10, None, random_state=42)
    assert np.array_equal(idx, np.arange(10))

    idx = _sample_reference_indices(10, 5, random_state=42)
    assert len(idx) == 5
    assert np.all(np.diff(idx) > 0)
    assert len(np.unique(idx)) == 5

    idx_invalid = _sample_reference_indices(10, 15, random_state=42)
    assert np.array_equal(idx_invalid, np.arange(10))
