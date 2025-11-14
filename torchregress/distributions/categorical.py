"""
Categorical distribution layer for classification tasks.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical as TorchCategorical


class Categorical(nn.Module):
    """
    An nn.Module that takes features from a backbone and produces a Categorical
    distribution object.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> TorchCategorical:
        """
        Computes the distribution object from the input features.
        """
        logits = self.layer(x)
        return TorchCategorical(logits=logits)
