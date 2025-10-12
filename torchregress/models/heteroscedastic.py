"""
Gaussian regression network utilities for heteroscedastic models.

This module provides models that predict both mean and variance.
"""

from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torchregress.losses.gaussian import GaussianNLLLoss


class DualHeadRegressionModel(nn.Module):
    """
    MLP with shared hidden layers and separate heads for mean and log-variance.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int] = [64, 64],
        activation: Callable[..., nn.Module] = nn.ReLU,
        log_var_init: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        self.shared = nn.Sequential(*layers)
        # Heads
        self.mean_head = nn.Linear(in_dim, output_dim)
        self.logvar_head = nn.Linear(in_dim, output_dim)
        # Initialize log-variance bias
        nn.init.constant_(self.logvar_head.bias, log_var_init)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns (mean, log_var) tensors of shape [batch, output_dim].
        """
        h = self.shared(x)
        mean = self.mean_head(h)
        log_var = self.logvar_head(h)
        return mean, log_var


def create_dual_head_regression(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int] = [64, 64],
    activation: Callable[..., nn.Module] = nn.ReLU,
    log_var_init: float = 0.0,
) -> Tuple[nn.Module, nn.Module]:
    """
    Convenience: returns DualHeadRegressionModel and GaussianNLLLoss.

    The model outputs (mean, log_var) tuple for heteroscedastic regression.
    """
    model = DualHeadRegressionModel(input_dim, output_dim, hidden_sizes, activation, log_var_init)

    # Model predicts variance, so no fixed_variance
    loss_fn = GaussianNLLLoss()
    return model, loss_fn
