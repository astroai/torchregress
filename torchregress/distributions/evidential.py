"""
Evidential distribution layer for regression tasks.
"""
import torch
import torch.nn as nn


class Evidential(nn.Module):
    """
    An nn.Module that produces the parameters for an Evidential distribution.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.layer = nn.Linear(in_features, out_features * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the evidential parameters from the input features.
        """
        return self.layer(x)
