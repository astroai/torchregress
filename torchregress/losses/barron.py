"""
Barron loss: generalization of L1/L2 tunable via alpha.
See Barron (2019).
"""

from typing import Optional

import torch
from torch import Tensor

from .base import RegressionLoss


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
            loss = 0.5 * error**2
        elif a == 0.0:
            loss = torch.log(0.5 * error**2 + 1.0)
        else:
            loss = (abs(a - 2.0) / a) * (
                (error**2 / abs(a - 2.0) + 1.0) ** (a / 2.0) - 1.0
            )
        loss = loss * (self.scale**2)
        return self._reduce(loss, mask, weights)
