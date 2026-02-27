"""Ordinal regression losses for ordered discrete targets."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..utils.ordinal import labels_to_levels
from ..utils.validation import validate_weights
from .base import BaseLoss
from .loss_registry import register_regression_loss


def _validate_classification_inputs(
    y_pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if y_pred.dim() < 2:
        raise ValueError("y_pred must have at least 2 dimensions")

    expected_target_shape = (y_pred.shape[0], *y_pred.shape[2:])
    if target.shape != expected_target_shape:
        raise ValueError(f"target shape {tuple(target.shape)} must match {expected_target_shape}")

    if mask is not None and mask.shape != target.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match target shape")

    target_i = target.long()
    if torch.any(target_i < 0) or torch.any(target_i >= y_pred.shape[1]):
        raise ValueError("target contains class indices outside valid range")
    return y_pred, target_i


@register_regression_loss("ordinal_cross_entropy")
class OrdinalCrossEntropyLoss(BaseLoss):
    """Cross-entropy loss for ordinal targets represented as class indices."""

    def __init__(
        self,
        reduction: str = "mean",
        class_weight: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(reduction=reduction)
        if class_weight is not None and class_weight.dim() != 1:
            raise ValueError("class_weight must be a 1D tensor")
        self.class_weight = class_weight.detach().float() if class_weight is not None else None
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        y_pred, target_i = _validate_classification_inputs(y_pred, target, mask)

        if weights is not None:
            weights = validate_weights(weights, y_pred.shape[0])

        class_weight = self.class_weight
        if class_weight is not None:
            class_weight = class_weight.to(device=y_pred.device, dtype=y_pred.dtype)
            if class_weight.numel() != y_pred.shape[1]:
                raise ValueError(
                    f"class_weight length {class_weight.numel()} must match "
                    f"num_classes {y_pred.shape[1]}"
                )

        loss = F.cross_entropy(
            y_pred,
            target_i,
            reduction="none",
            weight=class_weight,
            label_smoothing=self.label_smoothing,
        )
        return self._reduce(loss, mask=mask, weights=weights)


@register_regression_loss("cumulative_link")
class CumulativeLinkLoss(BaseLoss):
    """Ordinal loss over cumulative logits ``P(y > k)`` for ``k=0..K-2``."""

    def __init__(
        self,
        reduction: str = "mean",
        level_weights: Tensor | None = None,
        pos_weight: Tensor | None = None,
    ) -> None:
        super().__init__(reduction=reduction)
        self.level_weights = level_weights.detach().float() if level_weights is not None else None
        self.pos_weight = pos_weight.detach().float() if pos_weight is not None else None

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        if y_pred.dim() < 2:
            raise ValueError("y_pred must have at least 2 dimensions")

        logits = y_pred.movedim(1, -1)
        expected_target_shape = logits.shape[:-1]
        if target.shape != expected_target_shape:
            raise ValueError(
                f"target shape {tuple(target.shape)} must match {expected_target_shape}"
            )
        if mask is not None and mask.shape != target.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} must match target shape")

        num_classes = logits.shape[-1] + 1
        target_i = target.long()
        levels = labels_to_levels(target_i, num_classes=num_classes).to(dtype=y_pred.dtype)

        if weights is not None:
            weights = validate_weights(weights, logits.shape[0])

        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=y_pred.device, dtype=y_pred.dtype)
            if pos_weight.numel() not in {1, logits.shape[-1]}:
                raise ValueError(
                    f"pos_weight length {pos_weight.numel()} must be 1 or {logits.shape[-1]}"
                )

        per_level = F.binary_cross_entropy_with_logits(
            logits,
            levels,
            reduction="none",
            pos_weight=pos_weight,
        )

        level_weights = self.level_weights
        if level_weights is not None:
            level_weights = level_weights.to(device=y_pred.device, dtype=y_pred.dtype)
            if level_weights.numel() != logits.shape[-1]:
                raise ValueError(
                    f"level_weights length {level_weights.numel()} must match {logits.shape[-1]}"
                )
            per_level = per_level * level_weights.view((1,) * (per_level.dim() - 1) + (-1,))

        loss = per_level.mean(dim=-1)
        return self._reduce(loss, mask=mask, weights=weights)


@register_regression_loss("coral")
class CORALLoss(CumulativeLinkLoss):
    """CORAL ordinal loss (cumulative-link with CORAL encoding semantics)."""


__all__ = [
    "OrdinalCrossEntropyLoss",
    "CumulativeLinkLoss",
    "CORALLoss",
]
