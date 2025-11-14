"""
Gaussian loss functions for regression tasks.
"""
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Distribution

from .base import DistributionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("gaussian_nll")
class GaussianNLLLoss(DistributionLoss):
    """
    Calculates the Negative Log-Likelihood of a target given a predicted
    Normal or MultivariateNormal distribution.
    """

    def forward(
        self,
        y_pred: Distribution,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the Gaussian NLL loss.

        Args:
            y_pred: A `torch.distributions.Normal` or `MultivariateNormal` object.
            target: The ground truth target values.
            mask: An optional boolean mask.
            weights: Optional sample weights.
        """
        if not isinstance(y_pred, (torch.distributions.Normal, torch.distributions.MultivariateNormal)):
            raise TypeError(f"y_pred must be a Normal or MultivariateNormal distribution, but got {type(y_pred)}")

        nll = -y_pred.log_prob(target)
        return self._reduce_with_mask(nll, mask, weights)
