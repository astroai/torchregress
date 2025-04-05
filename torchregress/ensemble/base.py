"""
Base classes for ensemble models.

This module provides foundation classes and abstractions for all ensemble techniques
in the torchregress library.
"""

import torch
import torch.nn as nn
from typing import Dict, Union
from copy import deepcopy


class BaseEnsembleModel(nn.Module):
    """
    Base class for ensemble models.

    This class provides common functionality for different ensemble techniques.

    Args:
        base_model: Base model class or instance to ensemble
        ensemble_size: Number of ensemble members
        device: Device to use
    """

    def __init__(
        self,
        base_model: Union[nn.Module, type],
        ensemble_size: int = 5,
        device: str = "cpu",
        **base_model_kwargs,
    ) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        self.device = device

        # Create ensemble members
        self.models = nn.ModuleList()
        for i in range(ensemble_size):
            if isinstance(base_model, type):
                # If base_model is a class, instantiate it with kwargs
                model = base_model(**base_model_kwargs)
            else:
                # Otherwise, make a deep copy of the provided instance
                model = deepcopy(base_model)
            self.models.append(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes predictions from all ensemble members.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            List of predictions from each ensemble member
        """
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return outputs

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with uncertainty estimates.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Dictionary with mean and variance of predictions
        """
        with torch.no_grad():
            # Get predictions from all ensemble members
            predictions = self.forward(x)

            # Stack predictions [ensemble_size, batch_size, output_dim]
            stacked_preds = torch.stack(predictions)

            # Calculate mean across ensemble dimension
            mean = torch.mean(stacked_preds, dim=0)

            # Calculate variance across ensemble dimension
            variance = torch.var(stacked_preds, dim=0, unbiased=True)

            return {"mean": mean, "variance": variance}

    def predict_with_uncertainties(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with epistemic and aleatoric uncertainty estimates.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        with torch.no_grad():
            # For standard ensemble, this is the same as predict
            return self.predict(x)
