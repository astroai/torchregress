"""
Shared test model factories for ensemble tests.

These classes are used by both ``test_ensemble.py`` and
``test_ensemble_consistency.py`` to avoid duplicated model definitions.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleMLP(nn.Module):
    """Simple MLP for point-regression ensemble tests.

    Args:
        input_size: Number of input features.
        hidden_size: Hidden layer width.
        output_size: Number of output features.
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 8, output_size: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeteroscedasticMLP(nn.Module):
    """MLP that returns (mean, log_variance) tuples for heteroscedastic
    ensemble tests.

    Args:
        input_size: Number of input features.
        hidden_size: Hidden layer width.
        output_size: Number of output features.
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 8, output_size: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_size, output_size)
        self.logvar_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mean_head(h), self.logvar_head(h)


class ConstantLogitModel(nn.Module):
    """Model that returns fixed logits regardless of input."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits.clone().detach().view(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(x.shape[0], -1)


class ConstantMDNModel(nn.Module):
    """Model that returns fixed MDN parameters regardless of input."""

    def __init__(self, packed: torch.Tensor):
        super().__init__()
        self.register_buffer("packed", packed.clone().detach().view(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.packed.expand(x.shape[0], -1)
