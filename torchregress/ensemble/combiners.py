"""
Advanced ensemble methods for regression with uncertainty quantification.
"""

from typing import List, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor


class BayesianModelAveraging(nn.Module):
    """
    Bayesian Model Averaging for ensemble regression.

    Combines predictions from multiple models using Bayesian weighting.
    """

    def __init__(
        self,
        models: List[nn.Module],
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        # Initialize model weights (logits) uniformly
        self.model_weights = nn.Parameter(torch.zeros(self.n_models))

    def forward(self, x: Tensor) -> Tensor:
        """
        Calculate BMA loss using weighted average of model predictions.
        """
        # Get model weights (probabilities)
        model_probs = torch.softmax(self.model_weights, dim=0)

        # Get predictions from all models
        preds = torch.stack([model(x) for model in self.models], dim=1)

        # Weighted average of predictions
        weighted_pred = torch.sum(preds * model_probs.view(1, -1, 1), dim=1)

        return weighted_pred

    def get_model_weights(self) -> Tensor:
        """Get the current model weights (probabilities)."""
        return torch.softmax(self.model_weights, dim=0)

    def predict_with_uncertainty(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get predictions with uncertainty estimates.
        """
        model_probs = torch.softmax(self.model_weights, dim=0)

        preds = torch.stack([model(x) for model in self.models], dim=1)

        # Weighted mean
        mean_pred = torch.sum(preds * model_probs.view(1, -1, 1), dim=1)

        # Variance estimate (law of total variance)
        individual_vars = torch.var(preds, dim=1)  # Variance across models
        mean_vars = torch.sum(individual_vars * model_probs.view(1, -1, 1), dim=1)

        # Variance of means
        mean_diffs = preds - mean_pred.unsqueeze(1)
        var_of_means = torch.sum((mean_diffs**2) * model_probs.view(1, -1, 1), dim=1)

        total_variance = mean_vars + var_of_means
        return mean_pred, total_variance


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble with meta-learner for regression.

    Uses a meta-learner to combine base model predictions.
    """

    def __init__(
        self,
        models: List[nn.Module],
        meta_learner: nn.Module,
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.meta_learner = meta_learner

    def forward(self, x: Tensor) -> Tensor:
        """
        Calculate stacking ensemble loss.
        """
        preds = torch.cat([model(x) for model in self.models], dim=1)
        return cast(Tensor, self.meta_learner(preds))


class DynamicEnsembleWeighting(nn.Module):
    """
    Dynamic ensemble weighting based on recent performance.

    Adjusts model weights dynamically based on recent prediction accuracy.
    """

    def __init__(
        self,
        models: List[nn.Module],
        window_size: int = 100,
        learning_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.window_size = window_size
        self.learning_rate = learning_rate

        # Initialize model weights uniformly
        self.model_weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)

        # Performance tracking
        self.prediction_history: List[Tensor] = []
        self.target_history: List[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        """
        Calculate dynamically weighted ensemble loss.
        """
        # Store predictions and targets for dynamic weighting
        preds = torch.stack([model(x) for model in self.models], dim=1)

        # Get current weights
        current_weights = torch.softmax(self.model_weights, dim=0)

        # Weighted average of predictions
        weighted_pred = torch.sum(preds * current_weights.view(1, -1, 1), dim=1)

        return weighted_pred

    def update_weights(self, y_pred: Tensor, target: Tensor) -> None:
        """Update model weights based on recent performance."""
        self.prediction_history.append(y_pred.detach())
        self.target_history.append(target.detach())

        # Keep only recent history
        if len(self.prediction_history) > self.window_size:
            self.prediction_history = self.prediction_history[-self.window_size :]
            self.target_history = self.target_history[-self.window_size :]

        if len(self.prediction_history) < 2:
            return

        # Vectorized error calculation
        # [total_samples, n_models, features]
        all_preds = torch.cat(self.prediction_history, dim=0)
        # [total_samples, features]
        all_targets = torch.cat(self.target_history, dim=0)

        # Calculate MSE for each model across all samples in history
        # [total_samples, n_models, features] - [total_samples, 1, features]
        diff = all_preds - all_targets.unsqueeze(1)
        # Mean over features: [total_samples, n_models]
        mse = torch.mean(diff**2, dim=2)
        # Mean over samples: [n_models]
        errors = torch.mean(mse, dim=0)

        # Update weights using gradient descent (lower error = higher weight)
        with torch.no_grad():
            # Negative gradient because we want to minimize errors
            self.model_weights.grad = -errors * self.learning_rate
            self.model_weights.data += self.model_weights.grad

    def get_model_weights(self) -> Tensor:
        """Get the current model weights."""
        return torch.softmax(self.model_weights, dim=0)
