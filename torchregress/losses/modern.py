"""
Modern regression losses: Simultaneous Quantile Regression (SQR), Barron, Conformal, DeepAR.
"""

import torch
from torch import Tensor
from typing import Optional
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
        y_true: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        self._validate_inputs(y_pred, y_true, mask)
        # Expect y_pred shaped [batch, 2*features]
        if y_pred.dim() == 2 and y_pred.shape[1] == 2 * y_true.shape[1]:
            n = y_true.shape[1]
            lower_pred = y_pred[:, :n]
            upper_pred = y_pred[:, n:]
        else:
            raise ValueError(
                "SQRLoss expects y_pred shape [batch, 2*features]"
            )
        l = self.lower_loss(lower_pred, y_true, mask=mask)
        u = self.upper_loss(upper_pred, y_true, mask=mask)
        loss = l + u
        return self._reduce(loss, mask, weights)


class BarronLoss(RegressionLoss):
    """
    Barron loss: generalization of L1/L2 tunable via alpha.
    See Barron (2019).
    """
    def __init__(
        self,
        alpha: float = 1.0,
        scale: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        self.alpha = alpha
        self.scale = scale

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        self._validate_inputs(y_pred, y_true, mask)
        error = (y_pred - y_true) / self.scale
        a = self.alpha
        if a == 2.0:
            loss = 0.5 * error ** 2
        elif a == 0.0:
            loss = torch.log(0.5 * error ** 2 + 1.0)
        else:
            loss = (torch.abs(a - 2.0) / a) * (
                (error ** 2 / torch.abs(a - 2.0) + 1.0) ** (a / 2.0) - 1.0
            )
        loss = loss * (self.scale ** 2)
        return self._reduce(loss, mask, weights)


class ConformalLoss(RegressionLoss):
    """
    Placeholder for conformal prediction loss.
    """
    def __init__(self):
        super().__init__("mean")
        raise NotImplementedError("ConformalLoss is not implemented yet.")


class DeepARLoss(RegressionLoss):
    """
    Placeholder for DeepAR-style autoregressive loss.
    """
    def __init__(self):
        super().__init__("mean")
        raise NotImplementedError("DeepARLoss is not implemented yet.")
