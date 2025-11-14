"""
Normal distribution layer for regression tasks.
"""
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal as TorchNormal, MultivariateNormal


class Normal(nn.Module):
    """
    An nn.Module that takes features from a backbone and produces a Normal
    or MultivariateNormal distribution object.

    Args:
        in_features (int): The number of input features from the backbone.
        out_features (int): The dimensionality of the target variable.
        covariance_type (str): Type of covariance, one of ["diagonal", "full"].
            Defaults to "diagonal".
        min_variance (float): A small value to clamp the variance for stability.
            Defaults to 1e-6.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        covariance_type: str = "diagonal",
        min_variance: float = 1e-6,
    ) -> None:
        super().__init__()
        if covariance_type not in ["diagonal", "full"]:
            raise ValueError(f"covariance_type must be 'diagonal' or 'full', got {covariance_type}")

        self.out_features = out_features
        self.covariance_type = covariance_type
        self.min_variance = min_variance

        self.mean_layer = nn.Linear(in_features, out_features)

        if self.covariance_type == "diagonal":
            self.log_var_layer = nn.Linear(in_features, out_features)
        else:  # full covariance
            self.cov_layer = nn.Linear(in_features, out_features * out_features)

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        """
        Computes the distribution object from the input features.

        Args:
            x (torch.Tensor): The feature tensor from the backbone model.

        Returns:
            A torch.distributions.Normal or torch.distributions.MultivariateNormal object.
        """
        mean = self.mean_layer(x)

        if self.covariance_type == "diagonal":
            log_var = self.log_var_layer(x)
            var = torch.exp(log_var).clamp(min=self.min_variance)
            return TorchNormal(loc=mean, scale=var.sqrt())
        else:  # full covariance
            cov_flat = self.cov_layer(x)
            cov = cov_flat.view(-1, self.out_features, self.out_features)
            cov = cov @ cov.transpose(-1, -2)
            cov = cov + torch.eye(self.out_features, device=cov.device) * self.min_variance
            return MultivariateNormal(loc=mean, covariance_matrix=cov)
