"""
Target transformation utilities and losses for regression tasks.
"""

from typing import Any, Optional

import torch
from torch import Tensor


# Transform utilities (keeping existing functions)
def log_transform(x: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Apply log transform with epsilon stability.
    """
    return torch.log(x + eps)


def log_inverse(y: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Inverse of log transform.
    """
    return torch.clamp(torch.exp(y) - eps, min=0.0)


def boxcox_transform(x: Tensor, lam: float = 0.0, eps: float = 1e-6) -> Tensor:
    """
    Apply Box-Cox transform.
    """
    x_shift = x + eps
    if abs(lam) < 1e-8:
        return torch.log(x_shift)
    return (torch.pow(x_shift, lam) - 1.0) / lam


def boxcox_inverse(y: Tensor, lam: float = 0.0, eps: float = 1e-6) -> Tensor:
    """
    Inverse of Box-Cox transform.
    """
    if abs(lam) < 1e-8:
        return torch.clamp(torch.exp(y) - eps, min=0.0)
    return torch.clamp((y * lam + 1.0).pow(1.0 / lam) - eps, min=0.0)


def sqrt_transform(x: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Apply square-root transform.
    """
    return torch.sqrt(x + eps)


def sqrt_inverse(y: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Inverse of square-root transform.
    """
    return torch.clamp(y.pow(2) - eps, min=0.0)


# Loss classes (moved from losses/target_transform.py)
from ..losses.base import RegressionLoss


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
