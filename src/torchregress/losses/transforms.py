"""Transformed-target losses for skewed or heteroscedastic targets."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from ..utils.transform import (
    BoxCoxTransform,
    LogTransform,
    SqrtTransform,
    TargetTransform,
    YeoJohnsonTransform,
    make_target_transform,
)
from .base import RegressionLoss
from .loss_registry import register_regression_loss

BasePointLoss = Literal["mse", "mae", "huber"]


def _pointwise_error(
    y_pred: Tensor,
    target: Tensor,
    *,
    base_loss: BasePointLoss,
    delta: float,
) -> Tensor:
    residual = y_pred - target
    if base_loss == "mse":
        return residual.square()
    if base_loss == "mae":
        return residual.abs()
    if base_loss == "huber":
        abs_residual = residual.abs()
        quadratic = torch.minimum(abs_residual, torch.full_like(abs_residual, delta))
        linear = abs_residual - quadratic
        return 0.5 * quadratic.square() + delta * linear
    raise ValueError(f"Unsupported base_loss {base_loss!r}")


@register_regression_loss("transformed_target")
class TransformedTargetLoss(RegressionLoss):
    """Apply a target-space transform before computing a pointwise loss."""

    def __init__(
        self,
        transform: str | TargetTransform = "identity",
        *,
        base_loss: BasePointLoss = "mse",
        delta: float = 1.0,
        reduction: str = "mean",
        **transform_kwargs: float,
    ) -> None:
        super().__init__(reduction=reduction)
        if isinstance(transform, str):
            self.transform_name = transform
            self.transform = make_target_transform(transform, **transform_kwargs)
        else:
            self.transform_name = type(transform).__name__
            self.transform = transform
        self.base_loss = base_loss
        self.delta = delta

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        self._validate_inputs(y_pred, target, mask)
        y_pred_t = self.transform.forward(y_pred)
        target_t = self.transform.forward(target)
        loss = _pointwise_error(y_pred_t, target_t, base_loss=self.base_loss, delta=self.delta)
        return self._reduce(loss, mask, weights)

    def inverse(self, y_transformed: Tensor) -> Tensor:
        """Map transformed predictions back to the original target scale."""
        return self.transform.inverse(y_transformed)


@register_regression_loss("log_transform")
class LogTransformLoss(TransformedTargetLoss):
    """Regression loss in log space for positive targets."""

    def __init__(
        self,
        eps: float = 1e-6,
        *,
        base_loss: BasePointLoss = "mse",
        delta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            LogTransform(eps=eps),
            base_loss=base_loss,
            delta=delta,
            reduction=reduction,
        )
        self.eps = eps


@register_regression_loss("boxcox_transform")
class BoxCoxTransformLoss(TransformedTargetLoss):
    """Regression loss in Box-Cox transformed space for positive targets."""

    def __init__(
        self,
        lam: float = 0.0,
        eps: float = 1e-6,
        *,
        base_loss: BasePointLoss = "mse",
        delta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            BoxCoxTransform(lam=lam, eps=eps),
            base_loss=base_loss,
            delta=delta,
            reduction=reduction,
        )
        self.lam = lam
        self.eps = eps


@register_regression_loss("sqrt_transform")
class SqrtTransformLoss(TransformedTargetLoss):
    """Regression loss in square-root space for positive targets."""

    def __init__(
        self,
        eps: float = 1e-6,
        *,
        base_loss: BasePointLoss = "mse",
        delta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            SqrtTransform(eps=eps),
            base_loss=base_loss,
            delta=delta,
            reduction=reduction,
        )
        self.eps = eps


@register_regression_loss("yeojohnson_transform")
class YeoJohnsonTransformLoss(TransformedTargetLoss):
    """Regression loss in Yeo-Johnson space for signed targets."""

    def __init__(
        self,
        lam: float = 1.0,
        *,
        base_loss: BasePointLoss = "mse",
        delta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            YeoJohnsonTransform(lam=lam),
            base_loss=base_loss,
            delta=delta,
            reduction=reduction,
        )
        self.lam = lam


__all__ = [
    "TransformedTargetLoss",
    "LogTransformLoss",
    "BoxCoxTransformLoss",
    "SqrtTransformLoss",
    "YeoJohnsonTransformLoss",
]
