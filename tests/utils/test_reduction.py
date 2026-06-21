"""
Unit tests for torchregress.utils.reduction — shared reduction helpers.
"""

from __future__ import annotations

import pytest
import torch

from torchregress.utils.reduction import (
    REDUCERS,
    _reduce_max,
    _reduce_mean,
    _reduce_min,
    _reduce_none,
    _reduce_sum,
    reduce_per_sample,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Individual reducers
# ═══════════════════════════════════════════════════════════════════════════════


class TestReduceMean:
    def test_no_weights(self) -> None:
        """_reduce_mean without weights computes plain mean."""
        values = torch.tensor([1.0, 2.0, 3.0])
        result = _reduce_mean(values, None)
        assert float(result.item()) == pytest.approx(2.0)

    def test_with_weights(self) -> None:
        """_reduce_mean with weights computes weighted mean."""
        values = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([2.0, 1.0, 0.0])
        result = _reduce_mean(values, weights)
        # (1*2 + 2*1 + 3*0) / (2+1+0) = 4/3 ≈ 1.333
        assert float(result.item()) == pytest.approx(4.0 / 3.0)


class TestReduceSum:
    def test_no_weights(self) -> None:
        """_reduce_sum without weights computes plain sum."""
        values = torch.tensor([1.0, 2.0, 3.0])
        result = _reduce_sum(values, None)
        assert float(result.item()) == pytest.approx(6.0)

    def test_with_weights(self) -> None:
        """_reduce_sum with weights computes weighted sum."""
        values = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([2.0, 1.0, 0.5])
        result = _reduce_sum(values, weights)
        assert float(result.item()) == pytest.approx(1 * 2 + 2 * 1 + 3 * 0.5)


class TestReduceNone:
    def test_no_weights(self) -> None:
        """_reduce_none without weights returns values unchanged."""
        values = torch.tensor([1.0, 2.0, 3.0])
        result = _reduce_none(values, None)
        assert torch.equal(result, values)

    def test_with_weights(self) -> None:
        """_reduce_none with weights returns values * weights."""
        values = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([2.0, 1.0, 0.5])
        result = _reduce_none(values, weights)
        expected = values * weights
        assert torch.equal(result, expected)


class TestReduceMax:
    def test_no_weights(self) -> None:
        """_reduce_max without weights returns plain max."""
        values = torch.tensor([1.0, 5.0, 3.0])
        result = _reduce_max(values, None)
        assert float(result.item()) == pytest.approx(5.0)

    def test_with_weights(self) -> None:
        """_reduce_max with weights returns max of weighted values."""
        values = torch.tensor([10.0, 2.0, 3.0])
        weights = torch.tensor([0.1, 5.0, 1.0])
        result = _reduce_max(values, weights)
        assert float(result.item()) == pytest.approx(10.0)  # 2*5=10 > 10*0.1=1


class TestReduceMin:
    def test_no_weights(self) -> None:
        """_reduce_min without weights returns plain min."""
        values = torch.tensor([1.0, 5.0, 3.0])
        result = _reduce_min(values, None)
        assert float(result.item()) == pytest.approx(1.0)

    def test_with_weights(self) -> None:
        """_reduce_min with weights returns min of weighted values."""
        values = torch.tensor([10.0, 2.0, 3.0])
        weights = torch.tensor([0.1, 5.0, 1.0])
        result = _reduce_min(values, weights)
        assert float(result.item()) == pytest.approx(1.0)  # 10*0.1=1 < 2*5=10


# ═══════════════════════════════════════════════════════════════════════════════
# REDUCERS dict
# ═══════════════════════════════════════════════════════════════════════════════


class TestReducersDict:
    def test_contains_all_keys(self) -> None:
        """REDUCERS has mean, sum, none, max, min."""
        assert set(REDUCERS.keys()) == {"mean", "sum", "none", "max", "min"}

    def test_all_callable(self) -> None:
        """All REDUCERS values are callable."""
        for name, func in REDUCERS.items():
            assert callable(func), f"{name} is not callable"


# ═══════════════════════════════════════════════════════════════════════════════
# reduce_per_sample
# ═══════════════════════════════════════════════════════════════════════════════


class TestReducePerSample:
    def test_mean_reduction_no_mask(self) -> None:
        """Mean reduction without mask."""
        nll = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = reduce_per_sample(nll, None, None, "mean")
        assert float(result.item()) == pytest.approx(2.5)

    def test_sum_reduction_no_mask(self) -> None:
        """Sum reduction without mask."""
        nll = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = reduce_per_sample(nll, None, None, "sum")
        assert float(result.item()) == pytest.approx(10.0)

    def test_none_reduction_no_mask(self) -> None:
        """None reduction returns per-sample values."""
        nll = torch.tensor([1.0, 2.0, 3.0])
        result = reduce_per_sample(nll, None, None, "none")
        assert torch.equal(result, nll)

    def test_weighted_mean(self) -> None:
        """Mean reduction with weights."""
        nll = torch.tensor([1.0, 2.0, 3.0, 4.0])
        weights = torch.tensor([1.0, 1.0, 1.0, 0.0])
        result = reduce_per_sample(nll, None, weights, "mean")
        assert float(result.item()) == pytest.approx(2.0)

    def test_mask_excludes_samples(self) -> None:
        """Mask excludes specific samples."""
        nll = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, True, False, False])
        result = reduce_per_sample(nll, mask, None, "mean")
        assert float(result.item()) == pytest.approx(1.5)

    def test_mask_with_weights(self) -> None:
        """Mask + weights work together."""
        nll = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, True, False, False])
        weights = torch.tensor([2.0, 1.0, 1.0, 1.0])
        result = reduce_per_sample(nll, mask, weights, "mean")
        # selected: nll[0]=1, nll[1]=2; weights: 2, 1 → (1*2+2*1)/(2+1) = 4/3
        assert float(result.item()) == pytest.approx(4.0 / 3.0)

    def test_none_reduction_with_mask(self) -> None:
        """None reduction with mask returns zeros for masked-out."""
        nll = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, False, True])
        result = reduce_per_sample(nll, mask, None, "none")
        assert float(result[0].item()) == pytest.approx(1.0)
        assert float(result[1].item()) == pytest.approx(0.0)
        assert float(result[2].item()) == pytest.approx(3.0)

    def test_none_reduction_with_mask_and_weights(self) -> None:
        """None reduction with mask + weights returns weighted for selected, zero for masked."""
        nll = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, False, True])
        weights = torch.tensor([0.5, 1.0, 2.0])
        result = reduce_per_sample(nll, mask, weights, "none")
        assert float(result[0].item()) == pytest.approx(0.5)
        assert float(result[1].item()) == pytest.approx(0.0)
        assert float(result[2].item()) == pytest.approx(6.0)

    def test_sum_reduction_with_mask(self) -> None:
        """Sum reduction with mask."""
        nll = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        result = reduce_per_sample(nll, mask, None, "sum")
        assert float(result.item()) == pytest.approx(4.0)

    def test_sum_reduction_with_mask_and_weights(self) -> None:
        """Sum reduction with mask + weights."""
        nll = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        weights = torch.tensor([2.0, 1.0, 3.0, 1.0])
        result = reduce_per_sample(nll, mask, weights, "sum")
        assert float(result.item()) == pytest.approx(1 * 2 + 3 * 3)

    def test_mask_dtype_int(self) -> None:
        """Mask with int dtype (nonzero = True)."""
        nll = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([1, 0, 1])  # ints
        result = reduce_per_sample(nll, mask, None, "mean")
        assert float(result.item()) == pytest.approx(2.0)

    def test_mask_multi_dim(self) -> None:
        """Multi-dimensional mask is reduced via .all(dim=-1)."""
        nll = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([[True, True], [False, True], [True, True]])
        result = reduce_per_sample(nll, mask, None, "mean")
        # all(dim=-1): [True, False, True] → selects rows 0 and 2
        assert float(result.item()) == pytest.approx(2.0)

    def test_max_reduction(self) -> None:
        """Max reduction."""
        nll = torch.tensor([1.0, 5.0, 3.0])
        result = reduce_per_sample(nll, None, None, "max")
        assert float(result.item()) == pytest.approx(5.0)

    def test_min_reduction(self) -> None:
        """Min reduction."""
        nll = torch.tensor([1.0, 5.0, 3.0])
        result = reduce_per_sample(nll, None, None, "min")
        assert float(result.item()) == pytest.approx(1.0)

    def test_weights_mismatched_batch_raises(self) -> None:
        """Weights with wrong batch size raises ValueError."""
        nll = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, True, True])
        weights = torch.tensor([1.0, 2.0])  # wrong size
        with pytest.raises(ValueError, match="weights must match"):
            reduce_per_sample(nll, mask, weights, "mean")
