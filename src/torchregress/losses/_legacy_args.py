"""Backward-compatible positional argument resolution for loss forwards.

Remove when callers no longer pass ``(y_pred, target, mask, weights, extra)`` with
mis-ordered trailing covariance / log-variance tensors (target: next major release).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import Tensor


def is_legacy_mask_argument(extra: Any, mask: Any) -> bool:
    """Heuristic for legacy ``forward(..., mask, weights, extra)`` ordering.

    When a loss ignores a trailing covariance/log-var slot, callers historically
    passed a mask tensor there. A genuine covariance has trailing square dims
    ``[..., D, D]``; masks are broadcast-compatible with targets.
    """
    if extra is None or not isinstance(extra, torch.Tensor):
        return False
    if extra.dtype == torch.bool:
        return True
    if extra.dtype.is_floating_point is False:
        return False
    if extra.dim() >= 2 and extra.shape[-1] == extra.shape[-2]:
        return False
    return True


def resolve_legacy_cov_mask_weights(
    covariance_matrices: Optional[Tensor],
    mask: Optional[Tensor],
    weights: Optional[Tensor],
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Reinterpret a misplaced mask tensor in the covariance slot."""
    if is_legacy_mask_argument(covariance_matrices, mask):
        legacy_mask = covariance_matrices
        legacy_weights = mask if isinstance(mask, torch.Tensor) else None
        return legacy_mask, legacy_weights, None
    return mask, weights, covariance_matrices


def resolve_legacy_log_variance_kwarg(
    log_variance: Any,
    mask: Optional[Tensor],
    weights: Optional[Tensor],
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[bool]]:
    """Reinterpret legacy ``forward(..., mask, weights, log_variance)`` ordering."""
    if isinstance(log_variance, torch.Tensor):
        return (
            log_variance,
            mask if isinstance(mask, torch.Tensor) else None,
            weights if isinstance(weights, bool) else None,
        )
    return mask, weights, log_variance if isinstance(log_variance, (bool, type(None))) else None
