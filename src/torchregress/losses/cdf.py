"""Binned cumulative-distribution losses.

This module keeps the generic binned-CDF objective used by ordinal and
multi-bin regression models.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from .base import BaseLoss
from .loss_registry import register_regression_loss


def cumulative_cdf_loss(
    logits: Tensor,
    target: Tensor,
    bin_centers: Tensor,
) -> Tensor:
    """Per-sample bin-CDF squared error on a fixed bin grid.

    Args:
        logits: ``[B, K]`` bin logits.
        target: ``[B]`` or ``[B, 1]`` target values.
        bin_centers: ``[K]`` increasing bin centers.

    Returns:
        ``[B]`` loss values.
    """
    centers = torch.as_tensor(bin_centers, dtype=logits.dtype, device=logits.device)
    if centers.ndim != 1 or centers.numel() < 2 or not bool(torch.isfinite(centers).all()):
        raise ValueError("bin_centers must be finite and 1-D with at least two values")
    if not bool(torch.all(centers[1:] > centers[:-1])):
        raise ValueError("bin_centers must be strictly increasing")
    if logits.ndim != 2 or logits.shape[-1] != centers.numel() or target.numel() != logits.shape[0]:
        raise ValueError("logits must be [B, K] and target must contain B scalar values")
    y = target.reshape(-1).to(dtype=logits.dtype, device=logits.device)
    probability = torch.softmax(logits, dim=-1)
    cdf = probability.cumsum(dim=-1)
    step = (centers.unsqueeze(0) >= y.unsqueeze(1)).to(dtype=logits.dtype)
    mean_width = (centers[-1] - centers[0]) / max(centers.numel() - 1, 1)
    return mean_width * ((cdf - step) ** 2).sum(dim=-1)


@register_regression_loss("cdf")
class CumulativeCDFLoss(BaseLoss):
    """Binned cumulative-distribution loss for ordinal/tabular targets."""

    bin_centers: Tensor

    def __init__(self, bin_centers: Tensor | list[float], reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        centers = torch.as_tensor(bin_centers, dtype=torch.float32)
        if centers.dim() != 1 or centers.numel() < 2 or not bool(torch.isfinite(centers).all()):
            raise ValueError("bin_centers must be a 1-D sequence with at least two values")
        if not bool(torch.all(centers[1:] > centers[:-1])):
            raise ValueError("bin_centers must be strictly increasing")
        self.register_buffer("bin_centers", centers)

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        if y_pred.dim() != 2 or y_pred.shape[-1] != self.bin_centers.numel():
            raise ValueError(f"y_pred must be [B, {self.bin_centers.numel()}]")
        loss = cumulative_cdf_loss(y_pred, target, self.bin_centers)
        return self._reduce(loss, mask, weights)
