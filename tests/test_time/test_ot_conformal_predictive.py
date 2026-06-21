"""
Unit tests for torchregress.test_time.ot_conformal_predictive —
weighted_split_classification_predictive_batch.
"""

from __future__ import annotations

import pytest
import torch

from torchregress.test_time.ot_conformal import WeightedSplitConformalAdapter
from torchregress.test_time.ot_conformal_predictive import (
    weighted_split_classification_predictive_batch,
)


def _make_adapter(
    alpha: float = 0.1, calibration_scores: torch.Tensor | None = None
) -> WeightedSplitConformalAdapter:
    adapter = WeightedSplitConformalAdapter(alpha=alpha)
    if calibration_scores is not None:
        n = calibration_scores.numel()
        adapter.calibrate(calibration_scores, torch.ones(n))
    return adapter


# ═══════════════════════════════════════════════════════════════════════════════
# Basic correctness
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeightedSplitPredictiveBatch:
    def test_returns_predictive_batch(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
        scores = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert batch.point is not None
        assert batch.mean is not None
        assert batch.std is not None
        assert batch.extra is not None

    def test_label_inclusion_mask_in_extra(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
        scores = torch.tensor([[0.2, 0.8], [0.4, 0.1], [0.9, 0.3]])
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert batch.extra is not None
        assert "label_inclusion_mask" in batch.extra
        mask = batch.extra["label_inclusion_mask"]
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool

    def test_alpha_in_extra(self) -> None:
        adapter = _make_adapter(alpha=0.2, calibration_scores=torch.tensor([0.1, 0.3, 0.5]))
        scores = torch.tensor([[0.1, 0.2]])
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert batch.extra is not None
        assert batch.extra["alpha"] == 0.2

    def test_threshold_in_extra(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
        scores = torch.tensor([[0.2, 0.4]])
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert batch.extra is not None
        assert "threshold" in batch.extra
        assert batch.extra["threshold"] is not None
        assert isinstance(batch.extra["threshold"], float)

    def test_set_sizes_are_integer_counts(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
        scores = torch.tensor(
            [
                [0.1, 0.2, 0.9],  # 2 below threshold (<= 0.4 with q=0.8? let's just check shape)
                [0.8, 0.9, 0.05],  # 1 below
            ]
        )
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        set_sizes = batch.point
        assert set_sizes is not None
        assert set_sizes.shape == (2, 1)
        assert torch.all(set_sizes >= 0)
        assert torch.all(set_sizes <= scores.shape[1])

    def test_mean_equals_point(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.3, 0.5]))
        scores = torch.tensor([[0.2, 0.4, 0.6], [0.1, 0.3, 0.8]])
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert torch.equal(batch.mean, batch.point)

    def test_std_is_zero(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.3, 0.5]))
        scores = torch.tensor([[0.2, 0.4]])
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert torch.all(batch.std == 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Optional diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeightedSplitPredictiveBatchDiagnostics:
    def test_gap_diagnostics_included(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
        scores = torch.tensor([[0.2, 0.4]])
        gap = {"coverage_gap": 0.05, "wasserstein": 0.03}
        batch = weighted_split_classification_predictive_batch(adapter, scores, gap_diagnostics=gap)
        assert batch.extra is not None
        assert "shift_gap_diagnostics" in batch.extra
        assert batch.extra["shift_gap_diagnostics"] is gap

    def test_gap_diagnostics_none_not_included(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.3, 0.5]))
        scores = torch.tensor([[0.2, 0.4]])
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert batch.extra is not None
        assert "shift_gap_diagnostics" not in batch.extra

    def test_calibration_ess_included(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.3, 0.5]))
        scores = torch.tensor([[0.2, 0.4]])
        batch = weighted_split_classification_predictive_batch(
            adapter, scores, calibration_ess_inv_square=0.8
        )
        assert batch.extra is not None
        assert "calibration_ess_inv_square" in batch.extra
        assert batch.extra["calibration_ess_inv_square"] == 0.8

    def test_calibration_ess_none_not_included(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.3, 0.5]))
        scores = torch.tensor([[0.2, 0.4]])
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert batch.extra is not None
        assert "calibration_ess_inv_square" not in batch.extra


# ═══════════════════════════════════════════════════════════════════════════════
# Input validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeightedSplitPredictiveBatchValidation:
    def test_1d_scores_raises(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.3, 0.5]))
        scores = torch.tensor([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="2-D"):
            weighted_split_classification_predictive_batch(adapter, scores)

    def test_3d_scores_raises(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.3, 0.5]))
        scores = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="2-D"):
            weighted_split_classification_predictive_batch(adapter, scores)

    def test_before_calibrate_raises(self) -> None:
        adapter = WeightedSplitConformalAdapter(alpha=0.1)
        scores = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
        with pytest.raises(RuntimeError, match="calibrate"):
            weighted_split_classification_predictive_batch(adapter, scores)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeightedSplitPredictiveBatchEdge:
    def test_single_sample_single_class(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.tensor([0.1, 0.3, 0.5]))
        scores = torch.tensor([[0.2]])
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert batch.extra is not None
        assert batch.extra["label_inclusion_mask"].shape == (1, 1)

    def test_many_classes(self) -> None:
        adapter = _make_adapter(calibration_scores=torch.linspace(0, 1, 20))
        scores = torch.rand(5, 50)
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert batch.extra is not None
        assert batch.extra["label_inclusion_mask"].shape == (5, 50)

    def test_all_scores_below_threshold(self) -> None:
        """All candidates pass → all labels included."""
        adapter = _make_adapter(calibration_scores=torch.tensor([10.0, 20.0, 30.0]))
        scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert batch.extra is not None
        mask = batch.extra["label_inclusion_mask"]
        assert torch.all(mask)
        assert torch.all(batch.point == scores.shape[1])

    def test_all_scores_above_threshold(self) -> None:
        """All candidates fail → no labels included."""
        adapter = _make_adapter(calibration_scores=torch.tensor([1.0, 2.0, 3.0]))
        scores = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert batch.extra is not None
        mask = batch.extra["label_inclusion_mask"]
        assert not torch.any(mask)
        assert torch.all(batch.point == 0)

    def test_mixed_inclusion(self) -> None:
        """Some labels pass, some fail."""
        adapter = _make_adapter(calibration_scores=torch.tensor([0.2, 0.4, 0.6]))
        scores = torch.tensor([[0.1, 0.5, 0.3], [0.7, 0.2, 0.8]])
        batch = weighted_split_classification_predictive_batch(adapter, scores)
        assert batch.extra is not None
        # Some labels pass, some fail — set sizes vary row by row
        sizes = batch.point.squeeze(-1)
        assert sizes[0] != sizes[1]
