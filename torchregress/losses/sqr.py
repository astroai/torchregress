"""
Simultaneous Quantile Regression (SQR) Loss.
"""

from typing import Any, Optional

from torch import Tensor

from .base import RegressionLoss
from .loss_registry import register_regression_loss
from ..utils.quantile import quantile_loss


@register_regression_loss("sqr")
class SQRLoss(RegressionLoss):
    """
    Simultaneous Quantile Regression Loss (SQR).
    Encourages distribution-free uncertainty estimation via twin quantiles.
    """

    def __init__(
        self,
        lower_quantile: float = 0.1,
        upper_quantile: float = 0.9,
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        if target.dim() == 1:
            target = target.unsqueeze(1)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(1)

        n_features = target.shape[-1]
        if y_pred.shape[-1] != 2 * n_features:
            raise ValueError(
                f"SQRLoss expects y_pred shape [..., {2 * n_features}], got {list(y_pred.shape)}"
            )

        lower_pred = y_pred[..., :n_features]
        upper_pred = y_pred[..., n_features:]

        if lower_pred.shape != target.shape:
            raise ValueError(
                f"Target shape {list(target.shape)} must match prediction shape {list(lower_pred.shape)}"
            )
        if mask is not None and mask.shape != target.shape:
            raise ValueError(f"Mask shape {list(mask.shape)} must match target shape {list(target.shape)}")

        lower_loss_val = quantile_loss(lower_pred, target, self.lower_quantile)
        upper_loss_val = quantile_loss(upper_pred, target, self.upper_quantile)
        loss = lower_loss_val + upper_loss_val
        return self._reduce_with_mask(loss, mask, weights)
