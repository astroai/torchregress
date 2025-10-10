"""
Conformal prediction loss via quantile & calibration.
"""

from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torchcp.regression.predictor import ACIPredictor, SplitPredictor
from torchcp.regression.score import ABS as AbsScore
from torchcp.regression.score import CQR as CQRScore

from .base import RegressionLoss


class ConformalLoss(RegressionLoss):
    """
    Wrapper for torchcp conformal methods.

    Provides access to state-of-the-art conformal prediction methods including:
    - Split Conformal Prediction
    - Conformalized Quantile Regression (CQR)
    - Adaptive Conformal Inference (ACI)

    This wrapper provides a unified interface to torchcp predictors while maintaining
    compatibility with the torchregress loss function API.
    """

    def __init__(
        self,
        method: str = "cqr",
        alpha: float = 0.1,
        reduction: str = "mean",
        model: Optional[Module] = None,
        gamma: float = 0.01,  # Needed for ACI
    ) -> None:
        super().__init__(reduction=reduction)
        self.method = method.lower()
        self.alpha = alpha
        self._predictor: Optional[Any] = None
        self._is_calibrated = False
        self.q_hat: Optional[Tensor] = None
        self.model = model
        self.gamma = gamma

        self._init_predictor()

    def _init_predictor(self) -> None:
        """Initialize the appropriate torchcp predictor."""
        if self.method == "split":
            self._predictor = SplitPredictor(score_function=AbsScore())
        elif self.method == "cqr":
            self._predictor = SplitPredictor(score_function=CQRScore())
        elif self.method == "aci":
            if self.model is None:
                raise ValueError("ACIPredictor requires a 'model' to be provided.")
            self._predictor = ACIPredictor(score_function=AbsScore(), model=self.model, gamma=self.gamma)
        else:
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Supported methods: 'split', 'cqr', 'aci'"
            )

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Training loss for conformal prediction.
        """
        if self.method != "cqr":
            self._validate_inputs(y_pred, target, mask)

        if self.method == "cqr":
            n_feat = target.shape[-1] if target.dim() > 1 else 1
            if y_pred.shape[-1] != 2 * n_feat:
                raise ValueError(
                    f"CQR expects y_pred shape [..., 2*features], got {y_pred.shape}"
                )
            lower_pred = y_pred[..., :n_feat]
            upper_pred = y_pred[..., n_feat:]
            lower_q = self.alpha / 2
            upper_q = 1 - self.alpha / 2
            lower_err = target - lower_pred
            lower_loss = torch.maximum(lower_q * lower_err, (lower_q - 1) * lower_err)
            upper_err = target - upper_pred
            upper_loss = torch.maximum(upper_q * upper_err, (upper_q - 1) * upper_err)
            loss = lower_loss + upper_loss
        else:
            loss = (y_pred - target) ** 2

        return self._reduce_with_mask(loss, mask, weights)

    def calibrate(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> None:
        """
        Calibrate the conformal predictor on hold-out data.
        """
        if self._predictor is None:
            raise RuntimeError("Predictor not initialized")

        if mask is not None:
            mask_1d = mask.all(dim=-1) if mask.dim() > 1 else mask
            y_pred = y_pred[mask_1d]
            target = target[mask_1d]

        scores = self._predictor.calculate_score(y_pred, target)
        q_hat_scalar = torch.quantile(scores, 1 - self.alpha)
        self.q_hat = q_hat_scalar.unsqueeze(0)  # Ensure q_hat is a 1D tensor
        self._is_calibrated = True

    def predict_interval(
        self,
        y_pred: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Make predictions with conformal prediction intervals.
        """
        if not self._is_calibrated or self.q_hat is None:
            raise RuntimeError(
                "Predictor must be calibrated before making predictions. Call calibrate() first."
            )
        if self._predictor is None:
            raise RuntimeError("Predictor not initialized")

        if self.method == "cqr":
            n_feat = y_pred.shape[-1] // 2
            lower = y_pred[..., :n_feat] - self.q_hat
            upper = y_pred[..., n_feat:] + self.q_hat
            return lower, upper

        intervals = self._predictor.score_function.generate_intervals(y_pred, self.q_hat)

        if intervals.dim() > 1 and intervals.shape[1] == 1:
            intervals = intervals.squeeze(1)

        lower, upper = intervals[..., 0], intervals[..., 1]

        if lower.dim() == 1:
            lower = lower.unsqueeze(-1)
            upper = upper.unsqueeze(-1)

        return lower, upper


class AdaptiveConformalLoss(ConformalLoss):
    """
    Adaptive Conformal Prediction for regression.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        reduction: str = "mean",
        model: Optional[Module] = None,
        gamma: float = 0.01,
    ) -> None:
        if model is None:
            raise ValueError("AdaptiveConformalLoss requires a 'model' to be provided.")
        super().__init__(method="aci", alpha=alpha, reduction=reduction, model=model, gamma=gamma)


class ConformalizedQuantileLoss(ConformalLoss):
    """
    Conformalized Quantile Regression (CQR).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__(method="cqr", alpha=alpha, reduction=reduction)


class MultiDimensionalConformalLoss(ConformalLoss):
    """
    Multi-dimensional conformal prediction for multi-output regression.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        # For multi-dimensional, we use 'split' as the base, but override calibration
        super().__init__(method="split", alpha=alpha, reduction=reduction)

    def calibrate(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> None:
        """Calibrate separate taus for each output dimension."""
        if self._predictor is None:
            raise RuntimeError("Predictor not initialized")

        if mask is not None:
            mask_1d = mask.all(dim=-1) if mask.dim() > 1 else mask
            y_pred = y_pred[mask_1d]
            target = target[mask_1d]

        n_features = target.shape[-1]
        q_hats = []

        for i in range(n_features):
            y_pred_i = y_pred[..., i].unsqueeze(-1)
            target_i = target[..., i].unsqueeze(-1)
            scores = self._predictor.calculate_score(y_pred_i, target_i)
            q_hat_i = torch.quantile(scores, 1 - self.alpha)
            q_hats.append(q_hat_i)

        self.q_hat = torch.stack(q_hats)
        self._is_calibrated = True

    def predict_intervals(
        self,
        y_pred: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return calibrated intervals using dimension-specific taus."""
        if not self._is_calibrated or self.q_hat is None:
            raise ValueError("Call calibrate() before predict_intervals().")

        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        return lower, upper
