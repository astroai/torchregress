"""
Conformal prediction loss via quantile & calibration.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple
from .base import RegressionLoss
from .quantile import QuantileLoss


class ConformalLoss(RegressionLoss):
    """Conformalized Quantile Regression (CQR).
    Train lower (α/2) and upper (1−α/2) quantile losses,
    then calibrate interval width on hold-out data."""
    def __init__(
        self,
        alpha: float = 0.1,
        reduction: str = "mean",
        base_loss_reduction: str = "none",
    ) -> None:
        super().__init__(reduction=reduction)
        self.alpha = alpha
        self.lower_q = alpha / 2
        self.upper_q = 1 - alpha / 2
        self.lower_loss = QuantileLoss(self.lower_q, reduction=base_loss_reduction)
        self.upper_loss = QuantileLoss(self.upper_q, reduction=base_loss_reduction)
        self.tau: Optional[Tensor] = None

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Training loss: sum of lower and upper quantile losses."""
        # Validate and split predictions
        n_feat = y_true.shape[-1] if y_true.dim() > 1 else 1
        if y_pred.dim() >= 1 and y_pred.shape[-1] == 2 * n_feat:
            lower_pred = y_pred[..., :n_feat]
            upper_pred = y_pred[..., n_feat:]
        else:
            raise ValueError(
                f"ConformalLoss expects y_pred shape [...,2*features], got {y_pred.shape}"
            )
        # Compute quantile losses
        l = self.lower_loss(lower_pred, y_true, mask=mask, weights=weights)
        u = self.upper_loss(upper_pred, y_true, mask=mask, weights=weights)
        loss = l + u
        return self._reduce_with_mask(loss, mask, weights)

    def calibrate(
        self,
        lower_pred: Tensor,
        upper_pred: Tensor,
        y_true: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Calibrate hold-out: compute τ so intervals cover ≥1−α."""
        if mask is not None:
            lower_pred = lower_pred[mask]
            upper_pred = upper_pred[mask]
            y_true = y_true[mask]
        # Residuals outside predicted interval
        resid = torch.maximum(lower_pred - y_true, y_true - upper_pred)
        resid = torch.clamp(resid, min=0.0)
        resid_flat = resid.view(-1)
        self.tau = torch.quantile(resid_flat, 1 - self.alpha)
        return self.tau

    def predict_interval(
        self,
        lower_pred: Tensor,
        upper_pred: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return calibrated intervals: [lower−τ, upper+τ]."""
        if self.tau is None:
            raise ValueError("Call calibrate() before predict_interval().")
        return lower_pred - self.tau, upper_pred + self.tau
