"""
This module provides wrapper classes for creating probabilistic models.
"""
import torch.nn as nn


class ProbabilisticModel(nn.Module):
    """
    A wrapper class that combines a backbone model with a distribution head.

    This allows for a clean separation between the feature extraction part of a
    model (the backbone) and the part that defines the output distribution
    (the distribution head).

    Args:
        backbone (nn.Module): The feature extraction model.
        distribution_head (nn.Module): A module from `torchregress.distributions`
            that takes the backbone's output and produces a `torch.distributions`
            object.
    """

    def __init__(self, backbone: nn.Module, distribution_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.distribution_head = distribution_head

    def forward(self, x):
        """
        The forward pass of the probabilistic model.
        """
        features = self.backbone(x)
        return self.distribution_head(features)
