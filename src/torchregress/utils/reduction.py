"""Shared reduction helpers for losses and per-sample NLL vectors."""

from __future__ import annotations

from typing import Callable, Optional

import torch


def _reduce_mean(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is not None:
        return torch.sum(values * weights) / torch.sum(weights).clamp(min=1.0)
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


def reduce_per_sample(
    nll: torch.Tensor,
    sample_mask: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
    reduction: str,
) -> torch.Tensor:
    """Apply mask/weight/reduction to a per-sample NLL vector."""
    reducer = REDUCERS[reduction]

    if sample_mask is not None:
        if sample_mask.dtype != torch.bool:
            sample_mask = sample_mask > 0
        if sample_mask.dim() > 1:
            sample_mask = sample_mask.all(dim=-1)

    if sample_mask is not None and weights is not None:
        masked_weights = weights.to(device=nll.device, dtype=nll.dtype)
        if masked_weights.shape[0] != nll.shape[0]:
            raise ValueError("weights must match batch size")
        masked_nll = nll[sample_mask]
        masked_weights = masked_weights[sample_mask]
        if reduction == "none":
            result = torch.zeros_like(nll)
            result[sample_mask] = masked_nll * masked_weights
            return result
        if reduction == "sum":
            return torch.sum(masked_nll * masked_weights)
        return torch.sum(masked_nll * masked_weights) / torch.sum(masked_weights).clamp(min=1.0)

    if sample_mask is not None:
        masked_nll = nll[sample_mask]
        if reduction == "none":
            result = torch.zeros_like(nll)
            result[sample_mask] = masked_nll
            return result
        if reduction == "sum":
            return torch.sum(masked_nll)
        return torch.sum(masked_nll) / sample_mask.sum().clamp(min=1)

    return reducer(nll, weights)


__all__ = ["reduce_per_sample"]
