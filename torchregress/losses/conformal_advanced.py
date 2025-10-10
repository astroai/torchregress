"""
Advanced conformal prediction methods using torchcp.

This module provides wrappers around the torchcp library for state-of-the-art
conformal prediction methods in regression.
"""

from typing import Any, Optional, Union

import torch
from torch import Tensor

# Direct import as per CLAUDE.md policy
from torchcp.regression import ACIPredictor, CQR, SplitPredictor

from .base import RegressionLoss


class TorchCPConformalLoss(RegressionLoss):
    """
    Wrapper for torchcp conformal methods.

    Provides access to state-of-the-art conformal prediction methods including:
    - Split Conformal Prediction
    - Conformalized Quantile Regression (CQR)
    - Adaptive Conformal Inference (ACI)

    This wrapper provides a unified interface to torchcp predictors while maintaining
    compatibility with the torchregress loss function API.

    Args:
        method: Conformal method to use. Options:
            - 'split': Split Conformal Prediction
            - 'cqr': Conformalized Quantile Regression
            - 'aci': Adaptive Conformal Inference
        alpha: Miscoverage level (1-alpha is target coverage). Default: 0.1
        reduction: Loss reduction method ('mean', 'sum', 'none'). Default: 'mean'

    Notes:
        - Requires torchcp >= 1.2.0
        - The predictor must be calibrated before making predictions
        - Use calibrate() method with calibration data before predict()

    Examples:
        >>> # Create a conformal loss for CQR
        >>> loss_fn = TorchCPConformalLoss(method='cqr', alpha=0.1)
        >>>
        >>> # Train your model normally
        >>> for epoch in range(n_epochs):
        >>>     for x, y in train_loader:
        >>>         pred = model(x)
        >>>         loss = loss_fn(pred, y)
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>
        >>> # Calibrate on hold-out set
        >>> with torch.no_grad():
        >>>     cal_preds = model(x_cal)
        >>>     loss_fn.calibrate(cal_preds, y_cal)
        >>>
        >>> # Make predictions with uncertainty
        >>> with torch.no_grad():
        >>>     test_preds = model(x_test)
        >>>     lower, upper = loss_fn.predict(test_preds)
    """

    def __init__(
        self,
        method: str = "cqr",
        alpha: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.method = method.lower()
        self.alpha = alpha
        self._predictor: Optional[Any] = None
        self._is_calibrated = False

        # Initialize appropriate torchcp predictor
        self._init_predictor()

    def _init_predictor(self) -> None:
        """Initialize the appropriate torchcp predictor."""
        if self.method == "split":
            self._predictor = SplitPredictor()
        elif self.method == "cqr":
            self._predictor = CQR()
        elif self.method == "aci":
            self._predictor = ACIPredictor()
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

        For CQR, this expects y_pred to contain [lower_quantile, upper_quantile].
        For split/ACI, this uses MSE loss during training.

        Args:
            y_pred: Model predictions. For CQR: [..., 2*features]. For others: [..., features]
            target: Ground truth values [..., features]
            mask: Optional boolean mask [..., features]
            weights: Optional sample weights [...]
            **kwargs: Additional arguments

        Returns:
            Training loss (scalar or per-sample depending on reduction)
        """
        self._validate_inputs(y_pred, target, mask)

        if self.method == "cqr":
            # For CQR, compute quantile losses
            n_feat = target.shape[-1] if target.dim() > 1 else 1
            if y_pred.shape[-1] != 2 * n_feat:
                raise ValueError(
                    f"CQR expects y_pred shape [..., 2*features], got {y_pred.shape}"
                )

            lower_pred = y_pred[..., :n_feat]
            upper_pred = y_pred[..., n_feat:]

            # Quantile loss for lower and upper bounds
            lower_q = self.alpha / 2
            upper_q = 1 - self.alpha / 2

            # Pinball loss for quantiles
            lower_err = target - lower_pred
            lower_loss = torch.maximum(lower_q * lower_err, (lower_q - 1) * lower_err)

            upper_err = target - upper_pred
            upper_loss = torch.maximum(upper_q * upper_err, (upper_q - 1) * upper_err)

            loss = lower_loss + upper_loss
        else:
            # For split/ACI, use MSE during training
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

        Args:
            y_pred: Predictions on calibration set
            target: True values on calibration set
            mask: Optional mask for missing values

        Raises:
            ValueError: If inputs are invalid
        """
        if self._predictor is None:
            raise RuntimeError("Predictor not initialized")

        # Apply mask if provided
        if mask is not None:
            if mask.dim() > 1:
                # For multi-dimensional mask, only use samples where all features are valid
                mask_1d = mask.all(dim=-1)
            else:
                mask_1d = mask
            y_pred = y_pred[mask_1d]
            target = target[mask_1d]

        # Convert to numpy for torchcp
        y_pred_np = y_pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # Calibrate the predictor
        self._predictor.calibrate(y_pred_np, target_np, alpha=self.alpha)
        self._is_calibrated = True

    def predict(
        self,
        y_pred: Tensor,
        return_pi: bool = True,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        Make predictions with conformal prediction intervals.

        Args:
            y_pred: Model predictions
            return_pi: If True, return (lower, upper). If False, return point predictions.

        Returns:
            If return_pi=True: (lower_bounds, upper_bounds)
            If return_pi=False: point_predictions

        Raises:
            RuntimeError: If predictor is not calibrated
        """
        if not self._is_calibrated:
            raise RuntimeError(
                "Predictor must be calibrated before making predictions. "
                "Call calibrate() first."
            )

        if self._predictor is None:
            raise RuntimeError("Predictor not initialized")

        # Convert to numpy for torchcp
        y_pred_np = y_pred.detach().cpu().numpy()

        # Get prediction sets
        prediction_sets = self._predictor.predict(y_pred_np, alpha=self.alpha)

        # Convert back to torch tensors
        device = y_pred.device
        dtype = y_pred.dtype

        if return_pi:
            # prediction_sets is typically [lower, upper]
            if isinstance(prediction_sets, tuple) and len(prediction_sets) == 2:
                lower = torch.tensor(prediction_sets[0], device=device, dtype=dtype)
                upper = torch.tensor(prediction_sets[1], device=device, dtype=dtype)
                return lower, upper
            else:
                # If prediction_sets is a different format, adapt accordingly
                # This is a fallback for different torchcp versions
                lower = torch.tensor(prediction_sets[:, 0], device=device, dtype=dtype)
                upper = torch.tensor(prediction_sets[:, 1], device=device, dtype=dtype)
                return lower, upper
        else:
            # Return point predictions (midpoint of interval)
            if isinstance(prediction_sets, tuple) and len(prediction_sets) == 2:
                lower = torch.tensor(prediction_sets[0], device=device, dtype=dtype)
                upper = torch.tensor(prediction_sets[1], device=device, dtype=dtype)
                return (lower + upper) / 2
            else:
                pred = torch.tensor(prediction_sets, device=device, dtype=dtype)
                return pred.mean(dim=-1) if pred.dim() > 1 else pred


def create_conformal_predictor(
    method: str = "cqr",
    alpha: float = 0.1,
    reduction: str = "mean",
) -> TorchCPConformalLoss:
    """
    Factory function to create a torchcp-based conformal predictor.

    Args:
        method: Conformal method ('split', 'cqr', 'aci')
        alpha: Miscoverage level (target coverage is 1-alpha)
        reduction: Loss reduction method

    Returns:
        TorchCPConformalLoss instance

    Examples:
        >>> # Create CQR predictor with 90% coverage
        >>> predictor = create_conformal_predictor(method='cqr', alpha=0.1)
        >>>
        >>> # Create Split Conformal predictor with 95% coverage
        >>> predictor = create_conformal_predictor(method='split', alpha=0.05)
    """
    return TorchCPConformalLoss(method=method, alpha=alpha, reduction=reduction)
