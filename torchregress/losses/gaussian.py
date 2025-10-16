"""
Gaussian loss functions for regression tasks.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from .base import DistributionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("gaussian_nll")
class GaussianNLLLoss(DistributionLoss):
    """
    Gaussian Negative Log-Likelihood loss.

    This loss supports both diagonal and full covariance matrices, as well as
    fixed and learned variances.

    Args:
        covariance_type: Type of covariance, one of ["diagonal", "full"].
        min_variance: Minimum variance for numerical stability.
    """

    def __init__(
        self,
        covariance_type: str = "diagonal",
        min_variance: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        if covariance_type not in ["diagonal", "full"]:
            raise ValueError(f"covariance_type must be 'diagonal' or 'full', got {covariance_type}")
        self.covariance_type = covariance_type
        self.min_variance = min_variance

    def forward(
        self,
        y_pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the Gaussian NLL loss.

        For diagonal covariance, y_pred can be:
        - A tensor of shape (N, 2*D), where the first D elements are the mean and the next D are the log variance.
        - A tuple of two tensors of shape (N, D), for the mean and log variance.

        For full covariance, y_pred must be a tuple of:
        - A tensor of shape (N, D) for the mean.
        - A tensor of shape (N, D, D) for the covariance matrix.
        """
        if self.covariance_type == "diagonal":
            mean, var = self._extract_diag_params(y_pred)
            nll = self._diag_nll(target, mean, var)
        else:  # full
            if not isinstance(y_pred, (tuple, list)) or len(y_pred) != 2:
                raise ValueError("For full covariance, y_pred must be a tuple of (mean, cov)")
            mean, cov = y_pred
            nll = self._full_nll(target, mean, cov)

        return self._reduce_with_mask(nll, mask, weights)

    def _extract_diag_params(
        self, y_pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(y_pred, (tuple, list)):
            mean, log_var = y_pred
        else:
            mean, log_var = torch.chunk(y_pred, 2, dim=-1)
        var = torch.exp(log_var).clamp(min=self.min_variance)
        return mean, var

    def _diag_nll(self, target: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log(2 * math.pi * var) + (target - mean) ** 2 / var)

    def _full_nll(self, target: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        try:
            mvn = MultivariateNormal(mean, cov)
            return -mvn.log_prob(target)
        except RuntimeError as e:
            # Fallback for numerical issues
            if "cholesky" in str(e).lower():
                cov = cov + torch.eye(cov.shape[-1], device=cov.device) * 1e-6
                mvn = MultivariateNormal(mean, cov)
                return -mvn.log_prob(target)
            else:
                raise e