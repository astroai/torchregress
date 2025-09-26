"""
Simultaneous Quantile Regression (SQR) Loss.
"""

from typing import Any, Optional

from torch import Tensor

from .base import RegressionLoss
from .quantile import QuantileLoss


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
        self.lower_loss = QuantileLoss(quantile=lower_quantile, reduction="none")
        self.upper_loss = QuantileLoss(quantile=upper_quantile, reduction="none")

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        self._validate_inputs(y_pred, target, mask)
        # Expect y_pred shaped [batch, 2*features]
        if y_pred.dim() == 2 and y_pred.shape[1] == 2 * target.shape[1]:
            n = target.shape[1]
            lower_pred = y_pred[:, :n]
            upper_pred = y_pred[:, n:]
        else:
            raise ValueError("SQRLoss expects y_pred shape [batch, 2*features]")
        lower_loss_val = self.lower_loss(lower_pred, target, mask=mask)
        upper_loss_val = self.upper_loss(upper_pred, target, mask=mask)
        loss = lower_loss_val + upper_loss_val
        return self._reduce(loss, mask, weights)
