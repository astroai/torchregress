"""
Poisson loss function for regression tasks.
"""
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Poisson as TorchPoisson

from .base import DistributionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("poisson_nll")
class PoissonNLLLoss(DistributionLoss):
    """
    Poisson Negative Log-Likelihood loss for count data.
    """

    def forward(
        self,
        y_pred: TorchPoisson,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the Poisson NLL loss.

        Args:
            y_pred: A `torch.distributions.Poisson` object.
            target: The ground truth target values (counts).
            mask: An optional boolean mask.
            weights: Optional sample weights.
        """
        if not isinstance(y_pred, TorchPoisson):
            raise TypeError(f"y_pred must be a Poisson distribution, but got {type(y_pred)}")

        nll = -y_pred.log_prob(target)
        return self._reduce_with_mask(nll, mask, weights)
