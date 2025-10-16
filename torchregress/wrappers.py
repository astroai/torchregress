"""
High-level wrappers and convenience functions for torchregress.

This module provides simplified interfaces for common regression tasks,
including model creation, loss function setup, and ensemble methods.
"""

from typing import Any

# Import base losses
from .losses.base import (
    BaseLoss,
    WeightedLossWrapper,
)


def wrap_pytorch_loss(loss_class: type, **kwargs: Any) -> BaseLoss:
    """
    Wrap any PyTorch loss function with torchregress's masking and weighting capabilities.

    Args:
        loss_class: PyTorch loss class (e.g., nn.MSELoss, nn.L1Loss)
        **kwargs: Arguments to pass to the loss constructor

    Returns:
        A wrapped loss function with masking and weighting support
    """
    return WeightedLossWrapper(loss_class, **kwargs)