"""
Mixture Density Network (MDN) loss functions.
"""
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import MixtureSameFamily

from .base import DistributionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("mdn")
class MixtureDensityLoss(DistributionLoss):
    """
    Negative Log-Likelihood loss for Mixture Density Networks.
    """

    def forward(
        self,
        y_pred: MixtureSameFamily,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the MDN NLL loss.

        Args:
            y_pred: A `torch.distributions.MixtureSameFamily` object.
            target: The ground truth target values.
            mask: An optional boolean mask.
            weights: Optional sample weights.
        """
        if not isinstance(y_pred, MixtureSameFamily):
            raise TypeError(f"y_pred must be a MixtureSameFamily distribution, but got {type(y_pred)}")

        nll = -y_pred.log_prob(target)
        return self._reduce_with_mask(nll, mask, weights)
