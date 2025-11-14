"""
Poisson distribution layer for regression tasks.
"""
import torch
import torch.nn as nn
from torch.distributions import Poisson as TorchPoisson


class Poisson(nn.Module):
    """
    An nn.Module that takes features from a backbone and produces a Poisson
    distribution object.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> TorchPoisson:
        """
        Computes the distribution object from the input features.
        """
        rate = torch.exp(self.layer(x))  # Ensure rate is positive
        return TorchPoisson(rate)
