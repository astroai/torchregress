"""
Advanced ensemble methods for regression with uncertainty quantification.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..losses.base import RegressionLoss


class BayesianModelAveraging(RegressionLoss):
    """
    Bayesian Model Averaging for ensemble regression.

    Combines predictions from multiple models using Bayesian weighting.
    """

    def __init__(
        self,
        n_models: int,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.n_models = n_models
        # Initialize model weights (logits) uniformly
        self.model_weights = torch.nn.Parameter(torch.zeros(n_models))
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(
        self,
        y_pred: Union[Tensor, List[Tensor]],
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calculate BMA loss using weighted average of model predictions.

        Args:
            y_pred: Either a tensor of shape [batch, n_models, features] or
                   a list of n_models tensors each of shape [batch, features]
            target: Ground truth targets [batch, features]
            mask: Optional mask
            weights: Optional weights
        """
        # Get model weights (probabilities)
        model_probs = self.softmax(self.model_weights)

        if isinstance(y_pred, list):
            # List of tensors - stack them
            y_pred = torch.stack(y_pred, dim=1)

        # Weighted average of predictions
        weighted_pred = torch.sum(y_pred * model_probs.view(1, -1, 1), dim=1)

        # Calculate MSE loss
        loss = (weighted_pred - target) ** 2
        return self._reduce_with_mask(loss, mask, weights)

    def get_model_weights(self) -> Tensor:
        """Get the current model weights (probabilities)."""
        return self.softmax(self.model_weights)

    def predict_with_uncertainty(
        self, y_pred: Union[Tensor, List[Tensor]]
    ) -> Tuple[Tensor, Tensor]:
        """
        Get predictions with uncertainty estimates.

        Returns:
            Tuple of (mean_prediction, variance_estimate)
        """
        model_probs = self.softmax(self.model_weights)

        if isinstance(y_pred, list):
            y_pred = torch.stack(y_pred, dim=1)

        # Weighted mean
        mean_pred = torch.sum(y_pred * model_probs.view(1, -1, 1), dim=1)

        # Variance estimate (law of total variance)
        # Var(Y) = E[Var(Y|Model)] + Var(E[Y|Model])
        individual_vars = torch.var(y_pred, dim=1)  # Variance across models
        mean_vars = torch.sum(individual_vars * model_probs.view(1, -1, 1), dim=1)

        # Variance of means
        mean_diffs = y_pred - mean_pred.unsqueeze(1)
        var_of_means = torch.sum((mean_diffs**2) * model_probs.view(1, -1, 1), dim=1)

        total_variance = mean_vars + var_of_means
        return mean_pred, total_variance


class StackingEnsemble(RegressionLoss):
    """
    Stacking ensemble with meta-learner for regression.

    Uses a meta-learner to combine base model predictions.
    """

    def __init__(
        self,
        n_models: int,
        n_features: int,
        meta_learner: Optional[nn.Module] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.n_models = n_models
        self.n_features = n_features

        # Default meta-learner: linear combination with bias
        if meta_learner is None:
            # Input: n_models * n_features (concatenated predictions)
            # Output: n_features (final prediction)
            self.meta_learner = nn.Linear(n_models * n_features, n_features)
        else:
            self.meta_learner = meta_learner

    def forward(
        self,
        y_pred: Union[Tensor, List[Tensor]],
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calculate stacking ensemble loss.

        Args:
            y_pred: Either a tensor of shape [batch, n_models, features] or
                   a list of n_models tensors each of shape [batch, features]
            target: Ground truth targets [batch, features]
            mask: Optional mask
            weights: Optional weights
        """
        if isinstance(y_pred, list):
            # List of tensors - concatenate them
            y_pred = torch.cat(y_pred, dim=1)
        else:
            # Tensor of shape [batch, n_models, features] - flatten
            y_pred = y_pred.view(y_pred.shape[0], -1)

        # Apply meta-learner
        combined_pred = self.meta_learner(y_pred)

        # Calculate loss
        loss = (combined_pred - target) ** 2
        return self._reduce_with_mask(loss, mask, weights)

    def predict(self, y_pred: Union[Tensor, List[Tensor]]) -> Tensor:
        """Get final predictions from the ensemble."""
        if isinstance(y_pred, list):
            y_pred = torch.cat(y_pred, dim=1)
        else:
            y_pred = y_pred.view(y_pred.shape[0], -1)

        return self.meta_learner(y_pred)


class DynamicEnsembleWeighting(RegressionLoss):
    """
    Dynamic ensemble weighting based on recent performance.

    Adjusts model weights dynamically based on recent prediction accuracy.
    """

    def __init__(
        self,
        n_models: int,
        window_size: int = 100,
        learning_rate: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.n_models = n_models
        self.window_size = window_size
        self.learning_rate = learning_rate

        # Initialize model weights uniformly
        self.model_weights = torch.nn.Parameter(torch.ones(n_models) / n_models)
        self.softmax = torch.nn.Softmax(dim=0)

        # Performance tracking
        self.prediction_history: List[Tensor] = []
        self.target_history: List[Tensor] = []

    def forward(
        self,
        y_pred: Union[Tensor, List[Tensor]],
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calculate dynamically weighted ensemble loss.

        Args:
            y_pred: Either a tensor of shape [batch, n_models, features] or
                   a list of n_models tensors each of shape [batch, features]
            target: Ground truth targets [batch, features]
            mask: Optional mask
            weights: Optional weights
        """
        # Store predictions and targets for dynamic weighting
        if isinstance(y_pred, list):
            y_pred_tensor = torch.stack(y_pred, dim=1)
        else:
            y_pred_tensor = y_pred

        self.prediction_history.append(y_pred_tensor.detach())
        self.target_history.append(target.detach())

        # Keep only recent history
        if len(self.prediction_history) > self.window_size:
            self.prediction_history = self.prediction_history[-self.window_size :]
            self.target_history = self.target_history[-self.window_size :]

        # Update weights based on recent performance
        self._update_weights()

        # Get current weights
        current_weights = self.softmax(self.model_weights)

        # Weighted average of predictions
        weighted_pred = torch.sum(y_pred_tensor * current_weights.view(1, -1, 1), dim=1)

        # Calculate loss
        loss = (weighted_pred - target) ** 2
        return self._reduce_with_mask(loss, mask, weights)

    def _update_weights(self) -> None:
        """Update model weights based on recent performance."""
        if len(self.prediction_history) < 2:
            return

        # Calculate recent errors for each model
        recent_errors = []
        for i in range(self.n_models):
            model_errors = []
            for pred_batch, target_batch in zip(self.prediction_history, self.target_history):
                model_pred = pred_batch[:, i, :]  # [batch, features]
                error = torch.mean((model_pred - target_batch) ** 2, dim=1)  # [batch]
                model_errors.append(error)

            # Average error across batches
            avg_error = torch.mean(torch.cat(model_errors))
            recent_errors.append(avg_error)

        # Convert to tensor
        errors = torch.stack(recent_errors)  # [n_models]

        # Update weights using gradient descent (lower error = higher weight)
        with torch.no_grad():
            # Negative gradient because we want to minimize errors
            self.model_weights.grad = -errors * self.learning_rate
            self.model_weights.data += self.model_weights.grad

    def get_model_weights(self) -> Tensor:
        """Get the current model weights."""
        return self.softmax(self.model_weights)

    def predict_with_weights(self, y_pred: Union[Tensor, List[Tensor]]) -> Tuple[Tensor, Tensor]:
        """
        Get predictions with current model weights.

        Returns:
            Tuple of (weighted_prediction, model_weights)
        """
        if isinstance(y_pred, list):
            y_pred = torch.stack(y_pred, dim=1)

        current_weights = self.softmax(self.model_weights)
        weighted_pred = torch.sum(y_pred * current_weights.view(1, -1, 1), dim=1)
        return weighted_pred, current_weights


class EnsembleCalibration(RegressionLoss):
    """
    Ensemble calibration combining multiple calibration methods.

    Applies different calibration strategies to ensemble predictions.
    """

    def __init__(
        self,
        n_models: int,
        calibration_method: str = "temperature",
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.n_models = n_models
        self.calibration_method = calibration_method

        # Create separate calibrators for each model
        if calibration_method == "temperature":
            from .advanced_calibration import TemperatureScalingCalibration

            self.calibrators = torch.nn.ModuleList(
                [TemperatureScalingCalibration() for _ in range(n_models)]
            )
        elif calibration_method == "isotonic":
            from .advanced_calibration import IsotonicRegressionCalibration

            self.calibrators = torch.nn.ModuleList(
                [IsotonicRegressionCalibration() for _ in range(n_models)]
            )
        else:
            raise ValueError(f"Unknown calibration method: {calibration_method}")

        self._is_fitted = False

    def fit(
        self,
        predictions: Union[Tensor, List[Tensor]],
        targets: Tensor,
    ) -> None:
        """
        Fit calibration for each ensemble member.

        Args:
            predictions: Model predictions from ensemble members
            targets: Ground truth targets
        """
        if isinstance(predictions, list):
            pred_list = predictions
        else:
            # Split tensor into list
            pred_list = [predictions[:, i, :] for i in range(self.n_models)]

        # Fit each calibrator
        for i in range(self.n_models):
            self.calibrators[i].fit(pred_list[i], targets)

        self._is_fitted = True

    def forward(
        self,
        y_pred: Union[Tensor, List[Tensor]],
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply calibrated ensemble prediction and calculate loss."""
        if not self._is_fitted:
            raise ValueError("Call fit() before using ensemble calibration.")

        if isinstance(y_pred, list):
            pred_list = y_pred
        else:
            pred_list = [y_pred[:, i, :] for i in range(self.n_models)]

        # Apply calibration to each model's predictions
        calibrated_preds = []
        for i in range(self.n_models):
            if isinstance(pred_list[i], tuple) and len(pred_list[i]) == 2:
                # Distributional predictions
                mean, var = pred_list[i]
                calibrated_mean = mean  # Mean typically unchanged
                if self.calibration_method == "temperature":
                    calibrated_var = self.calibrators[i].scale_uncertainty(var)
                else:
                    calibrated_var = var
                calibrated_preds.append((calibrated_mean, calibrated_var))
            else:
                # Point predictions
                calibrated_pred = self.calibrators[i].calibrate_predictions(pred_list[i])
                calibrated_preds.append(calibrated_pred)

        # Simple average for now (could be enhanced with weighted averaging)
        if isinstance(calibrated_preds[0], tuple) and len(calibrated_preds[0]) == 2:
            # Distributional predictions
            means = [p[0] for p in calibrated_preds]
            vars = [p[1] for p in calibrated_preds]
            avg_mean = torch.mean(torch.stack(means), dim=0)
            avg_var = torch.mean(torch.stack(vars), dim=0)
            # Combined loss
            nll = 0.5 * (torch.log(avg_var) + (target - avg_mean) ** 2 / avg_var)
            loss = nll
        else:
            # Point predictions
            avg_pred = torch.mean(torch.stack(calibrated_preds), dim=0)
            loss = (avg_pred - target) ** 2

        return self._reduce_with_mask(loss, mask, weights)

    def predict_calibrated(
        self, y_pred: Union[Tensor, List[Tensor]]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Get calibrated ensemble predictions."""
        if not self._is_fitted:
            raise ValueError("Call fit() before getting calibrated predictions.")

        if isinstance(y_pred, list):
            pred_list = y_pred
        else:
            pred_list = [y_pred[:, i, :] for i in range(self.n_models)]

        # Apply calibration to each model's predictions
        calibrated_preds = []
        for i in range(self.n_models):
            if isinstance(pred_list[i], tuple) and len(pred_list[i]) == 2:
                # Distributional predictions
                mean, var = pred_list[i]
                calibrated_mean = mean
                if self.calibration_method == "temperature":
                    calibrated_var = self.calibrators[i].scale_uncertainty(var)
                else:
                    calibrated_var = var
                calibrated_preds.append((calibrated_mean, calibrated_var))
            else:
                # Point predictions
                calibrated_pred = self.calibrators[i].calibrate_predictions(pred_list[i])
                calibrated_preds.append(calibrated_pred)

        # Return averaged predictions
        if isinstance(calibrated_preds[0], tuple) and len(calibrated_preds[0]) == 2:
            # Distributional predictions
            means = [p[0] for p in calibrated_preds]
            vars = [p[1] for p in calibrated_preds]
            avg_mean = torch.mean(torch.stack(means), dim=0)
            avg_var = torch.mean(torch.stack(vars), dim=0)
            return avg_mean, avg_var
        else:
            # Point predictions
            avg_pred = torch.mean(torch.stack(calibrated_preds), dim=0)
            return avg_pred
