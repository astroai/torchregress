"""
Heteroscedastic regression network utilities.
"""

from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torchregress.losses.gaussian import HeteroscedasticGaussianLoss


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
    Convenience: returns DualHeadRegressionModel and HeteroscedasticGaussianLoss.
    """
    model = DualHeadRegressionModel(input_dim, output_dim, hidden_sizes, activation, log_var_init)

    # Use predicted log_var, so learnable_variance=False
    loss_fn = HeteroscedasticGaussianLoss(learnable_variance=False)
    return model, loss_fn
