"""Tests for the causal TTA benchmark harness."""

from __future__ import annotations

import numpy as np
import pytest

from torchregress.prediction import PredictiveBatch
from torchregress.test_time.benchmark import CausalTTAHarness


class MockAdapter:
    def __init__(self) -> None:
        self.unlabeled_updates: list[np.ndarray] = []
        self.labeled_updates: list[tuple[np.ndarray, np.ndarray]] = []
        self.predict_count = 0

    def predict_distribution(self, X: np.ndarray) -> PredictiveBatch:
        self.predict_count += 1
        # Predict y = 2 * X with std = 1
        mean = 2.0 * X.reshape(-1)
        std = np.ones_like(mean)
        return PredictiveBatch(
            point=mean,
            mean=mean,
            std=std,
            extra={"family": "gaussian"},
        )

    def adapt_unlabeled_target(self, X: np.ndarray) -> None:
        self.unlabeled_updates.append(X)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.labeled_updates.append((X, y))


def test_harness_causal_evaluation_immediate_feedback() -> None:
    adapter = MockAdapter()
    harness = CausalTTAHarness(delay=0, alpha=0.1)

    # 5 steps, batch_size=2
    stream_X = np.arange(10, dtype=np.float32).reshape(-1, 1)
    stream_y = 2.0 * stream_X + 0.1  # slightly off

    metrics = harness.evaluate(adapter, stream_X, stream_y, batch_size=2)

    # Predictions check
    assert adapter.predict_count == 5
    # Since delay=0, labeled updates should have happened for all 5 batches (immediately after each predict)
    assert len(adapter.labeled_updates) == 5
    assert len(adapter.unlabeled_updates) == 5

    # Check update content
    # First batch: X=[0, 1], y=[0.1, 2.1]
    np.testing.assert_allclose(adapter.unlabeled_updates[0], [[0], [1]])
    np.testing.assert_allclose(adapter.labeled_updates[0][0], [[0], [1]])
    np.testing.assert_allclose(adapter.labeled_updates[0][1], [[0.1], [2.1]])

    # Metrics check
    # predict is 2.0 * X, y is 2.0 * X + 0.1. Error is constant -0.1.
    assert np.isclose(metrics["RMSE"], 0.1, atol=1e-5)
    assert np.isclose(metrics["MAE"], 0.1, atol=1e-5)
    # std = 1.0, alpha = 0.1 => z = 1.64485
    # y = pred + 0.1, which is within [pred - 1.645, pred + 1.645]
    assert metrics["coverage"] == 1.0
    assert np.isclose(metrics["mean_width"], 2.0 * 1.6448536269514722, atol=1e-5)
    assert "NLL" in metrics
    assert "winkler_score" in metrics


def test_harness_causal_evaluation_delayed_feedback() -> None:
    adapter = MockAdapter()
    # 2 steps delay
    harness = CausalTTAHarness(delay=2, alpha=0.05)

    stream_X = np.arange(10, dtype=np.float32).reshape(-1, 1)
    stream_y = 2.0 * stream_X

    harness.evaluate(adapter, stream_X, stream_y, batch_size=2)

    assert adapter.predict_count == 5
    # With 2 steps delay:
    # t=0: predict batch 0, adapt_unlabeled batch 0, enqueue batch 0. len(pending)=1 <= 2 -> no labeled update.
    # t=1: predict batch 1, adapt_unlabeled batch 1, enqueue batch 1. len(pending)=2 <= 2 -> no labeled update.
    # t=2: predict batch 2, adapt_unlabeled batch 2, enqueue batch 2. len(pending)=3 > 2 -> update batch 0.
    # t=3: predict batch 3, adapt_unlabeled batch 3, enqueue batch 3. len(pending)=4 > 2 -> update batch 1.
    # t=4: predict batch 4, adapt_unlabeled batch 4, enqueue batch 4. len(pending)=5 > 2 -> update batch 2.
    assert len(adapter.labeled_updates) == 3
    assert len(adapter.unlabeled_updates) == 5

    # Check updated batches: batch 0, batch 1, batch 2
    np.testing.assert_allclose(adapter.labeled_updates[0][0], [[0], [1]])
    np.testing.assert_allclose(adapter.labeled_updates[1][0], [[2], [3]])
    np.testing.assert_allclose(adapter.labeled_updates[2][0], [[4], [5]])


def test_harness_causal_evaluation_unlabeled_tta() -> None:
    adapter = MockAdapter()
    # delay = None -> unlabeled only
    harness = CausalTTAHarness(delay=-1, alpha=0.1)

    stream_X = np.arange(10, dtype=np.float32).reshape(-1, 1)
    stream_y = 2.0 * stream_X

    harness.evaluate(adapter, stream_X, stream_y, batch_size=2)

    assert adapter.predict_count == 5
    assert len(adapter.unlabeled_updates) == 5
    # Labeled update should never be called
    assert len(adapter.labeled_updates) == 0


def test_harness_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        CausalTTAHarness(alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        CausalTTAHarness(alpha=1.0)
