"""
Transform losses for regression tasks.
"""

from typing import Any, Optional

from torch import Tensor

from ..utils.transform import (
    boxcox_inverse,
    boxcox_transform,
    log_inverse,
    log_transform,
    sqrt_inverse,
    sqrt_transform,
)
from .base import RegressionLoss


class LogTransformLoss(RegressionLoss):
    """
    Loss on log-transformed targets (variance-stabilizing for multiplicative noise).
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__(reduction)
        self.eps = eps

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        self._validate_inputs(y_pred, target, mask)
        y_pred_t = log_transform(y_pred, self.eps)
        y_true_t = log_transform(target, self.eps)
        loss = (y_pred_t - y_true_t) ** 2
        return self._reduce_with_mask(loss, mask, weights)

    def inverse(self, y_transformed: Tensor) -> Tensor:
        """
        Inverse of log transform: map transformed values back to original scale.
        """
        return log_inverse(y_transformed, self.eps)


class BoxCoxTransformLoss(RegressionLoss):
    """
    Loss on Box-Cox transformed targets.

    If lambda ~= 0, uses log transform.
    """

    def __init__(self, lam: float = 0.0, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__(reduction)
        self.lam = lam
        self.eps = eps

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        self._validate_inputs(y_pred, target, mask)
        y_pred_t = boxcox_transform(y_pred, self.lam, self.eps)
        y_true_t = boxcox_transform(target, self.lam, self.eps)
        loss = (y_pred_t - y_true_t) ** 2
        return self._reduce_with_mask(loss, mask, weights)

    def inverse(self, y_transformed: Tensor) -> Tensor:
        """
        Inverse of Box-Cox transform: map transformed values back to original scale.
        """
        return boxcox_inverse(y_transformed, self.lam, self.eps)


class SqrtTransformLoss(RegressionLoss):
    """
    Loss on square-root transformed targets (variance-stabilizing for Poisson-like data).
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__(reduction)
        self.eps = eps

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        self._validate_inputs(y_pred, target, mask)
        y_pred_t = sqrt_transform(y_pred, self.eps)
        y_true_t = sqrt_transform(target, self.eps)
        loss = (y_pred_t - y_true_t) ** 2
        return self._reduce_with_mask(loss, mask, weights)

    def inverse(self, y_transformed: Tensor) -> Tensor:
        """
        Inverse of sqrt transform: map transformed values back to original scale.
        """
        return sqrt_inverse(y_transformed, self.eps)
