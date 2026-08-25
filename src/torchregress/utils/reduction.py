"""Shared reduction helpers for losses and per-sample NLL vectors."""

from __future__ import annotations

from typing import Callable, Optional

import torch


def _safe_denominator(denom: torch.Tensor) -> torch.Tensor:
    """Replace a non-positive denominator with one so empty reductions give 0, not NaN."""
    return torch.where(denom > 0, denom, torch.ones_like(denom))


def _reduce_mean(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is not None:
        # Zero-weight-safe normalization (A9): an all-zero weight vector yields
        # 0 instead of inf/NaN, replacing the previous ``clamp(min=1.0)`` which
        # silently under-normalized small positive weight sums.
        w_sum = torch.sum(weights)
        return torch.sum(values * weights) / _safe_denominator(w_sum)
    return torch.mean(values)


def _reduce_sum(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is not None:
        return torch.sum(values * weights)
    return torch.sum(values)


def _reduce_none(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is not None:
        return values * weights
    return values


REDUCERS: dict[str, Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]] = {
    "mean": _reduce_mean,
    "sum": _reduce_sum,
    "none": _reduce_none,
}


__all__ = ["_safe_denominator"]
