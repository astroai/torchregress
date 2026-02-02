"""
MC-Dropout for uncertainty estimation in neural networks.

MC-Dropout (Monte Carlo Dropout) is a simple technique for obtaining uncertainty
estimates by enabling dropout at inference time and running multiple forward passes.

Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers in a model for MC-Dropout inference."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def disable_dropout(model: nn.Module) -> None:
    """Disable dropout layers in a model."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


class MCDropoutWrapper(nn.Module):
    """
    Wrapper for applying MC-Dropout to any model with dropout layers.

    This wrapper enables dropout at inference time and performs multiple
    forward passes to estimate predictive uncertainty.

    Args:
        model: Base model with dropout layers
        n_samples: Number of MC samples for inference (default: 30)
        dropout_rate: If provided, replaces existing dropout rates

    Example:
        >>> base_model = nn.Sequential(
        ...     nn.Linear(10, 64),
        ...     nn.ReLU(),
        ...     nn.Dropout(0.2),
        ...     nn.Linear(64, 1)
        ... )
        >>> mc_model = MCDropoutWrapper(base_model, n_samples=50)
        >>> mean, std = mc_model.predict_with_uncertainty(x)
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 30,
        dropout_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.n_samples = n_samples

        # Optionally update dropout rates
        if dropout_rate is not None:
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout_rate

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward pass (dropout disabled)."""
        self.model.eval()
        return self.model(x)

    def mc_forward(self, x: Tensor, n_samples: Optional[int] = None) -> Tensor:
        """
        MC-Dropout forward pass with multiple samples.

        Args:
            x: Input tensor [batch_size, ...]
            n_samples: Number of samples (uses default if None)

        Returns:
            Stacked predictions [n_samples, batch_size, output_dim]
        """
        n = n_samples or self.n_samples
        enable_dropout(self.model)

        samples = []
        for _ in range(n):
            with torch.no_grad():
                pred = self.model(x)
            samples.append(pred)

        return torch.stack(samples, dim=0)

    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Make prediction with uncertainty estimate.

        Args:
            x: Input tensor [batch_size, ...]
            n_samples: Number of MC samples (uses default if None)

        Returns:
            Tuple of (mean, std) predictions
        """
        samples = self.mc_forward(x, n_samples)

        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        return mean, std

    def predict_interval(
        self,
        x: Tensor,
        confidence: float = 0.95,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute prediction intervals using MC-Dropout samples.

        Args:
            x: Input tensor [batch_size, ...]
            confidence: Confidence level (default 0.95)
            n_samples: Number of MC samples (uses default if None)

        Returns:
            Tuple of (lower, upper) bounds
        """
        samples = self.mc_forward(x, n_samples)

        alpha = 1 - confidence
        lower = torch.quantile(samples, alpha / 2, dim=0)
        upper = torch.quantile(samples, 1 - alpha / 2, dim=0)

        return lower, upper


class MCDropoutModel(nn.Module):
    """
    Neural network with built-in MC-Dropout support.

    This model includes dropout layers and provides methods for
    uncertainty estimation via MC-Dropout.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        dropout_rate: Dropout probability (default: 0.2)
        n_samples: Number of MC samples for inference (default: 30)

    Example:
        >>> model = MCDropoutModel(input_dim=10, hidden_dims=[64, 32], output_dim=1)
        >>> mean, std = model.predict_with_uncertainty(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout_rate: float = 0.2,
        n_samples: int = 30,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward pass."""
        return self.network(x)

    def mc_forward(self, x: Tensor, n_samples: Optional[int] = None) -> Tensor:
        """
        MC-Dropout forward pass with multiple samples.

        Args:
            x: Input tensor [batch_size, ...]
            n_samples: Number of samples (uses default if None)

        Returns:
            Stacked predictions [n_samples, batch_size, output_dim]
        """
        n = n_samples or self.n_samples
        enable_dropout(self)

        samples = []
        for _ in range(n):
            with torch.no_grad():
                pred = self.network(x)
            samples.append(pred)

        return torch.stack(samples, dim=0)

    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Make prediction with uncertainty estimate.

        Args:
            x: Input tensor [batch_size, ...]
            n_samples: Number of MC samples (uses default if None)

        Returns:
            Tuple of (mean, std) predictions
        """
        samples = self.mc_forward(x, n_samples)
        return samples.mean(dim=0), samples.std(dim=0)

    def predict_interval(
        self,
        x: Tensor,
        confidence: float = 0.95,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute prediction intervals using MC-Dropout samples.

        Args:
            x: Input tensor [batch_size, ...]
            confidence: Confidence level (default 0.95)
            n_samples: Number of MC samples (uses default if None)

        Returns:
            Tuple of (lower, upper) bounds
        """
        samples = self.mc_forward(x, n_samples)
        alpha = 1 - confidence
        lower = torch.quantile(samples, alpha / 2, dim=0)
        upper = torch.quantile(samples, 1 - alpha / 2, dim=0)
        return lower, upper
