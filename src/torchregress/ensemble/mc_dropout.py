"""
MC-Dropout for uncertainty estimation in neural networks.

MC-Dropout (Monte Carlo Dropout) is a simple technique for obtaining uncertainty
estimates by enabling dropout at inference time and running multiple forward passes.

Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
"""

from contextlib import contextmanager
from typing import Iterator, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers in a model for MC-Dropout inference."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@contextmanager
def _module_mode(model: nn.Module, dropout_train: bool) -> Iterator[None]:
    """Temporarily set BatchNorm/Dropout modules to a requested mode, then restore.

    ``dropout_train=True`` puts Dropout layers in train mode (stochastic passes)
    while keeping BatchNorm-family modules in eval mode so running statistics
    are not mutated during "inference" (TR-ENS-03). ``dropout_train=False``
    puts everything in eval mode. The snapshot is always restored on exit.
    """
    snapshot = {m: m.training for m in model.modules()}
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            module.eval()
        elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            if dropout_train:
                module.train()
            else:
                module.eval()
    try:
        yield
    finally:
        for module, was_training in snapshot.items():
            module.training = was_training


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

    References
    ----------
    .. [1] Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation:
       Representing Model Uncertainty in Deep Learning. In *ICML 2016*.
       https://arxiv.org/abs/1506.02142
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
        """Standard forward pass (dropout disabled, model mode restored after)."""
        with _module_mode(self.model, dropout_train=False):
            return cast(Tensor, self.model(x))

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

        # TR-ENS-03: dropout active but BatchNorm kept in eval so batch stats
        # and running statistics are untouched; original module modes restored.
        with torch.no_grad(), _module_mode(self.model, dropout_train=True):
            repeat_dims = [n] + [1] * (x.dim() - 1)
            x_expanded = x.repeat(*repeat_dims)
            preds = self.model(x_expanded)
            return preds.view(n, x.shape[0], *preds.shape[1:])

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
