"""
Unit tests for torchregress.utils.reduction — shared reduction helpers.
"""

from __future__ import annotations

import pytest
import torch

from torchregress.utils.reduction import (
    REDUCERS,
    _reduce_mean,
    _reduce_none,
    _reduce_sum,
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


# ═══════════════════════════════════════════════════════════════════════════════
# REDUCERS dict
# ═══════════════════════════════════════════════════════════════════════════════


class TestReducersDict:
    def test_contains_all_keys(self) -> None:
        """REDUCERS has mean, sum, none."""
        assert set(REDUCERS.keys()) == {"mean", "sum", "none"}

    def test_all_callable(self) -> None:
        """All REDUCERS values are callable."""
        for name, func in REDUCERS.items():
            assert callable(func), f"{name} is not callable"
