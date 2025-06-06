"""
Target transformation losses: log, Box-Cox, variance-stabilizing.
"""

import torch
from torch import Tensor
from .base import RegressionLoss
from ..utils.transform import log_transform, log_inverse, boxcox_transform, boxcox_inverse, sqrt_transform, sqrt_inverse


class LogTransformLoss(RegressionLoss):
    """
    Loss on log-transformed targets.
    Applies squared error in log space.
    """
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__(reduction)
        self.eps = eps

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        mask=None,
        weights=None,
    ) -> Tensor:
        self._validate_inputs(y_pred, y_true, mask)
        y_pred_t = log_transform(y_pred, self.eps)
        y_true_t = log_transform(y_true, self.eps)
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
        y_true: Tensor,
        mask=None,
        weights=None,
    ) -> Tensor:
        self._validate_inputs(y_pred, y_true, mask)
        y_pred_t = boxcox_transform(y_pred, self.lam, self.eps)
        y_true_t = boxcox_transform(y_true, self.lam, self.eps)
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
        y_true: Tensor,
        mask=None,
        weights=None,
    ) -> Tensor:
        self._validate_inputs(y_pred, y_true, mask)
        y_pred_t = sqrt_transform(y_pred, self.eps)
        y_true_t = sqrt_transform(y_true, self.eps)
        loss = (y_pred_t - y_true_t) ** 2
        return self._reduce_with_mask(loss, mask, weights)

    def inverse(self, y_transformed: Tensor) -> Tensor:
        """
        Inverse of sqrt transform: map transformed values back to original scale.
        """
        return sqrt_inverse(y_transformed, self.eps)
