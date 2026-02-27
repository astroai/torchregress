"""Losses for uncertain or weak ground-truth supervision."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from ..utils.propensity import ipw_weights
from .base import RegressionLoss
from .loss_registry import register_regression_loss


def _split_gaussian_params(y_pred: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    if isinstance(y_pred, tuple):
        if len(y_pred) != 2:
            raise ValueError("Gaussian tuple prediction must be (mean, log_variance)")
        mean, log_var = y_pred
        return mean, log_var
    # Allow mean-only predictions for convenience by assuming unit-variance
    # logits (log-variance = 0). This keeps API compatibility with generic
    # regression call sites that pass a single-output tensor.
    if y_pred.shape[-1] == 1 or y_pred.shape[-1] % 2 != 0:
        return y_pred, torch.zeros_like(y_pred)
    n_targets = y_pred.shape[-1] // 2
    mean = y_pred[..., :n_targets]
    log_var = y_pred[..., n_targets:]
    return mean, log_var


def _combine_external_weights(
    *,
    weights: Optional[Tensor],
    propensity_weights: Optional[Tensor],
    propensity_scores: Optional[Tensor],
    propensity_clip: float,
) -> Optional[Tensor]:
    resolved_propensity = propensity_weights
    if resolved_propensity is None and propensity_scores is not None:
        min_p = max(1.0 / max(propensity_clip, 1.0), 1e-4)
        max_p = min(1.0 - min_p, 1.0 - 1e-4)
        resolved_propensity = ipw_weights(
            propensity_scores,
            clip_min=min_p,
            clip_max=max_p,
        )
    if resolved_propensity is None:
        return weights
    if weights is None:
        return resolved_propensity
    return weights * resolved_propensity


def _gaussian_nll(
    mean: Tensor,
    variance: Tensor,
    target: Tensor,
    *,
    min_variance: float = 1e-8,
) -> Tensor:
    safe_var = variance.clamp_min(min_variance)
    return 0.5 * (
        torch.log(safe_var) + (target - mean) ** 2 / safe_var + math.log(2.0 * math.pi)
    )


@register_regression_loss("noisy_target_gaussian_nll")
class NoisyTargetGaussianNLL(RegressionLoss):
    """Gaussian NLL with additive target-noise variance support.

    This loss is useful when labels are uncertain (e.g., noisy annotators, uncertain
    spectroscopy matches, weak labels). Target uncertainty is added to predictive
    variance before scoring.
    """

    def __init__(
        self,
        reduction: str = "mean",
        min_variance: float = 1e-8,
        propensity_clip: float = 20.0,
    ) -> None:
        super().__init__(reduction=reduction)
        self.min_variance = min_variance
        self.propensity_clip = propensity_clip

    def forward(
        self,
        y_pred: Tensor | Tuple[Tensor, Tensor],
        target: Tensor,
        target_variance: Optional[Tensor] = None,
        propensity_weights: Optional[Tensor] = None,
        propensity_scores: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        mean, log_var = _split_gaussian_params(y_pred)
        if mean.shape != target.shape:
            raise ValueError(
                f"target shape {target.shape} must match predicted mean shape {mean.shape}"
            )

        pred_var = torch.exp(log_var).clamp_min(self.min_variance)
        if target_variance is None:
            target_var = torch.zeros_like(pred_var)
        else:
            target_var = target_variance.to(pred_var.device, pred_var.dtype)
            if target_var.shape != pred_var.shape:
                raise ValueError(
                    "target_variance shape must match predicted variance shape "
                    f"{pred_var.shape}, got {target_var.shape}"
                )
            target_var = target_var.clamp_min(0.0)

        total_var = pred_var + target_var
        loss = _gaussian_nll(mean, total_var, target, min_variance=self.min_variance)
        effective_weights = _combine_external_weights(
            weights=weights,
            propensity_weights=propensity_weights,
            propensity_scores=propensity_scores,
            propensity_clip=self.propensity_clip,
        )
        return self._reduce_with_mask(loss, mask, effective_weights)


@register_regression_loss("consistency_reg")
class ConsistencyRegLoss(RegressionLoss):
    """Point-regression loss with teacher/student consistency regularization."""

    def __init__(
        self,
        consistency_weight: float = 1.0,
        base_loss: str = "mse",
        detach_teacher: bool = True,
        reduction: str = "mean",
        propensity_clip: float = 20.0,
    ) -> None:
        super().__init__(reduction=reduction)
        self.consistency_weight = consistency_weight
        self.base_loss = base_loss.lower()
        self.detach_teacher = detach_teacher
        self.propensity_clip = propensity_clip
        if self.base_loss not in {"mse", "mae"}:
            raise ValueError(f"base_loss must be one of ['mse', 'mae'], got {base_loss!r}")

    def _base_error(self, y_pred: Tensor, target: Tensor) -> Tensor:
        if self.base_loss == "mae":
            return torch.abs(y_pred - target)
        return (y_pred - target) ** 2

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        teacher_pred: Optional[Tensor] = None,
        propensity_weights: Optional[Tensor] = None,
        propensity_scores: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        self._validate_inputs(y_pred, target, mask)
        teacher_tensor = target if teacher_pred is None else teacher_pred
        if teacher_tensor.shape != y_pred.shape:
            raise ValueError("teacher_pred shape must match y_pred shape")
        teacher = teacher_tensor.detach() if self.detach_teacher else teacher_tensor
        supervised = self._base_error(y_pred, target)
        consistency = (y_pred - teacher) ** 2
        loss = supervised + self.consistency_weight * consistency
        effective_weights = _combine_external_weights(
            weights=weights,
            propensity_weights=propensity_weights,
            propensity_scores=propensity_scores,
            propensity_clip=self.propensity_clip,
        )
        return self._reduce_with_mask(loss, mask, effective_weights)


@register_regression_loss("pseudo_label_nll")
class PseudoLabelNLL(RegressionLoss):
    """Gaussian NLL with observed + pseudo-label blending.

    This loss supports partial labels and pseudo-label confidence weights.
    """

    def __init__(
        self,
        pseudo_weight: float = 0.5,
        reduction: str = "mean",
        min_variance: float = 1e-8,
        detach_pseudo_labels: bool = True,
        propensity_clip: float = 20.0,
    ) -> None:
        super().__init__(reduction=reduction)
        self.pseudo_weight = pseudo_weight
        self.min_variance = min_variance
        self.detach_pseudo_labels = detach_pseudo_labels
        self.propensity_clip = propensity_clip

    def forward(
        self,
        y_pred: Tensor | Tuple[Tensor, Tensor],
        target: Optional[Tensor] = None,
        *,
        pseudo_target: Optional[Tensor] = None,
        pseudo_confidence: Optional[Tensor] = None,
        label_mask: Optional[Tensor] = None,
        propensity_weights: Optional[Tensor] = None,
        propensity_scores: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        mean, log_var = _split_gaussian_params(y_pred)
        pred_var = torch.exp(log_var).clamp_min(self.min_variance)
        zero = torch.zeros_like(mean)
        blended_loss = torch.zeros_like(mean)
        blend_weight = torch.zeros_like(mean)

        if target is not None:
            if target.shape != mean.shape:
                raise ValueError("target shape must match predicted mean shape")
            supervised_nll = _gaussian_nll(mean, pred_var, target, min_variance=self.min_variance)
            supervised_weight = torch.ones_like(supervised_nll)
            if label_mask is not None:
                if label_mask.shape != supervised_nll.shape:
                    raise ValueError("label_mask shape must match target shape")
                supervised_weight = label_mask.to(supervised_nll.dtype)
            blended_loss = blended_loss + supervised_nll * supervised_weight
            blend_weight = blend_weight + supervised_weight

        if pseudo_target is not None:
            pseudo = pseudo_target.detach() if self.detach_pseudo_labels else pseudo_target
            if pseudo.shape != mean.shape:
                raise ValueError("pseudo_target shape must match predicted mean shape")
            pseudo_nll = _gaussian_nll(mean, pred_var, pseudo, min_variance=self.min_variance)
            pseudo_w = torch.full_like(pseudo_nll, self.pseudo_weight)
            if pseudo_confidence is not None:
                if pseudo_confidence.shape != pseudo_nll.shape:
                    raise ValueError("pseudo_confidence shape must match pseudo_target shape")
                pseudo_w = pseudo_w * pseudo_confidence.clamp(min=0.0)
            if label_mask is not None:
                pseudo_w = pseudo_w * (~label_mask).to(pseudo_w.dtype)
            blended_loss = blended_loss + pseudo_nll * pseudo_w
            blend_weight = blend_weight + pseudo_w

        if target is None and pseudo_target is None:
            raise ValueError("At least one of target or pseudo_target must be provided")

        loss = torch.where(blend_weight > 0, blended_loss / blend_weight, zero)
        effective_weights = _combine_external_weights(
            weights=weights,
            propensity_weights=propensity_weights,
            propensity_scores=propensity_scores,
            propensity_clip=self.propensity_clip,
        )
        return self._reduce_with_mask(loss, mask, effective_weights)


__all__ = [
    "NoisyTargetGaussianNLL",
    "ConsistencyRegLoss",
    "PseudoLabelNLL",
]
