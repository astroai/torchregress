"""
Conformal prediction loss via quantile & calibration.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..utils.validation import validate_range
from .base import RegressionLoss
from .quantile import QuantileLoss

# Note: torchcp is imported in conformal_advanced.py where it's actually used
# This file doesn't need torchcp since it implements basic CQR from scratch


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
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Training loss: sum of lower and upper quantile losses."""
        # Validate and split predictions
        n_feat = target.shape[-1] if target.dim() > 1 else 1
        if y_pred.dim() >= 1 and y_pred.shape[-1] == 2 * n_feat:
            lower_pred = y_pred[..., :n_feat]
            upper_pred = y_pred[..., n_feat:]
        else:
            raise ValueError(
                f"ConformalLoss expects y_pred shape [...,2*features], got {y_pred.shape}"
            )
        # Compute quantile losses
        l = self.lower_loss(lower_pred, target, mask=mask, weights=weights)
        u = self.upper_loss(upper_pred, target, mask=mask, weights=weights)
        loss = l + u
        return self._reduce_with_mask(loss, mask, weights)

    def calibrate(
        self,
        lower_pred: Tensor,
        upper_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Calibrate hold-out: compute τ so intervals cover ≥1−α."""
        if mask is not None:
            lower_pred = lower_pred[mask]
            upper_pred = upper_pred[mask]
            target = target[mask]
        # Residuals outside predicted interval
        resid = torch.maximum(lower_pred - target, target - upper_pred)
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


# Note: The following classes would ideally integrate with torchcp when available
# For now, they provide basic conformal prediction functionality


class AdaptiveConformalLoss(ConformalLoss):
    """
    Adaptive Conformal Prediction for regression.

    Adjusts the confidence level based on the difficulty of recent predictions.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.01,
        reduction: str = "mean",
        base_loss_reduction: str = "none",
    ) -> None:
        super().__init__(alpha, reduction, base_loss_reduction)
        self.gamma = gamma  # Learning rate for adaptation
        self.alpha_history = [alpha]  # Track alpha values over time
        self.residual_history: List[float] = []  # Track residuals for adaptation

    def calibrate(
        self,
        lower_pred: Tensor,
        upper_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Calibrate with adaptive adjustment based on recent performance."""
        if mask is not None:
            lower_pred = lower_pred[mask]
            upper_pred = upper_pred[mask]
            target = target[mask]

        # Calculate residuals
        resid = torch.maximum(lower_pred - target, target - upper_pred)
        resid = torch.clamp(resid, min=0.0)
        resid_flat = resid.view(-1)

        # Store residuals for adaptation
        self.residual_history.extend(resid_flat.detach().cpu().numpy())

        # Keep only recent history (last 1000 points)
        if len(self.residual_history) > 1000:
            self.residual_history = self.residual_history[-1000:]

        # Adapt alpha based on recent coverage
        if len(self.residual_history) > 10:
            recent_coverage = np.mean(np.array(self.residual_history) <= 0)
            target_coverage = 1 - self.alpha
            error = target_coverage - recent_coverage

            # Adjust alpha (smaller alpha -> wider intervals -> better coverage)
            new_alpha = max(0.01, min(0.5, self.alpha + self.gamma * error))
            self.alpha = float(new_alpha)  # type: ignore
            self.alpha_history.append(float(new_alpha))  # type: ignore

            # Update quantiles
            self.lower_q = self.alpha / 2
            self.upper_q = 1 - self.alpha / 2
            self.lower_loss = QuantileLoss(float(self.lower_q), reduction=self.lower_loss.reduction)  # type: ignore
            self.upper_loss = QuantileLoss(float(self.upper_q), reduction=self.upper_loss.reduction)  # type: ignore

        # Compute tau as before
        self.tau = torch.quantile(resid_flat, 1 - self.alpha)
        return self.tau


class ConformalizedQuantileLoss(RegressionLoss):
    """
    Conformalized Quantile Regression (CQR) with multiple quantiles.

    Extends basic CQR to work with multiple quantile levels simultaneously.
    """

    def __init__(
        self,
        quantiles: Union[Tuple[float, float], list] = (0.05, 0.95),
        alpha: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        if isinstance(quantiles, tuple):
            quantiles = list(quantiles)
        self.quantiles = sorted(quantiles)
        self.alpha = alpha
        self.lower_q = quantiles[0]
        self.upper_q = quantiles[-1]
        self.lower_loss = QuantileLoss(self.lower_q, reduction="none")
        self.upper_loss = QuantileLoss(self.upper_q, reduction="none")
        self.tau: Optional[Tensor] = None

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Training loss: sum of lower and upper quantile losses."""
        # Validate and split predictions
        n_feat = target.shape[-1] if target.dim() > 1 else 1
        expected_features = len(self.quantiles) * n_feat
        if y_pred.dim() >= 1 and y_pred.shape[-1] == expected_features:
            # Split predictions for each quantile
            pred_list = []
            for i in range(len(self.quantiles)):
                start_idx = i * n_feat
                end_idx = (i + 1) * n_feat
                pred_list.append(y_pred[..., start_idx:end_idx])

            lower_pred = pred_list[0]
            upper_pred = pred_list[-1]
        else:
            raise ValueError(
                f"ConformalizedQuantileLoss expects y_pred shape [...,{expected_features}], got {y_pred.shape}"
            )

        # Compute quantile losses
        l = self.lower_loss(lower_pred, target, mask=mask, weights=weights)
        u = self.upper_loss(upper_pred, target, mask=mask, weights=weights)
        loss = l + u
        return self._reduce_with_mask(loss, mask, weights)

    def calibrate(
        self,
        quantile_preds: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Calibrate using multiple quantile predictions."""
        if mask is not None:
            quantile_preds = quantile_preds[mask]
            target = target[mask]

        # Extract lower and upper predictions
        n_feat = target.shape[-1] if target.dim() > 1 else 1
        lower_pred = quantile_preds[..., :n_feat]
        upper_pred = quantile_preds[..., -n_feat:]

        # Residuals outside predicted interval
        resid = torch.maximum(lower_pred - target, target - upper_pred)
        resid = torch.clamp(resid, min=0.0)
        resid_flat = resid.view(-1)
        self.tau = torch.quantile(resid_flat, 1 - self.alpha)
        return self.tau

    def predict_interval(
        self,
        quantile_preds: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return calibrated intervals from multiple quantile predictions."""
        if self.tau is None:
            raise ValueError("Call calibrate() before predict_interval().")

        n_feat = quantile_preds.shape[-1] // len(self.quantiles)
        lower_pred = quantile_preds[..., :n_feat]
        upper_pred = quantile_preds[..., -n_feat:]
        return lower_pred - self.tau, upper_pred + self.tau


class MultiDimensionalConformalLoss(RegressionLoss):
    """
    Multi-dimensional conformal prediction for multi-output regression.

    Calibrates prediction intervals separately for each output dimension.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.alpha = alpha
        self.taus: Optional[Tensor] = None
        self._is_calibrated = False

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Training loss for multi-dimensional conformal prediction."""
        # Validate inputs
        self._validate_inputs(y_pred, target, mask)

        # For training, we can use a simple MSE loss or other base loss
        # In practice, this would be combined with quantile losses
        loss = (y_pred - target) ** 2
        return self._reduce_with_mask(loss, mask, weights)

    def calibrate(
        self,
        lower_preds: Tensor,
        upper_preds: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Calibrate separate taus for each output dimension."""
        if mask is not None:
            lower_preds = lower_preds[mask]
            upper_preds = upper_preds[mask]
            target = target[mask]

        n_features = target.shape[-1]
        taus = []

        # Calculate separate tau for each feature
        for i in range(n_features):
            lower_pred = lower_preds[..., i]
            upper_pred = upper_preds[..., i]
            y_i = target[..., i]

            # Residuals outside predicted interval for this feature
            resid = torch.maximum(lower_pred - y_i, y_i - upper_pred)
            resid = torch.clamp(resid, min=0.0)
            resid_flat = resid.view(-1)
            tau_i = torch.quantile(resid_flat, 1 - self.alpha)
            taus.append(tau_i)

        self.taus = torch.stack(taus)
        self._is_calibrated = True
        return self.taus

    def predict_intervals(
        self,
        lower_preds: Tensor,
        upper_preds: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return calibrated intervals using dimension-specific taus."""
        if not self._is_calibrated or self.taus is None:
            raise ValueError("Call calibrate() before predict_intervals().")

        # Expand taus to match the dimensions of predictions
        taus = self.taus
        while taus.dim() < lower_preds.dim():
            taus = taus.unsqueeze(0)

        taus_expanded = taus.expand_as(lower_preds)
        lower_intervals = lower_preds - taus_expanded
        upper_intervals = upper_preds + taus_expanded
        return lower_intervals, upper_intervals
