"""
Tweedie distribution loss functions for regression.

The Tweedie distribution is a family of probability distributions that includes
many common distributions like the normal, Poisson, and gamma distributions.
It's defined through the variance function V(μ) = μ^p where p is the power parameter.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import RegressionLoss

class TweedieLoss(RegressionLoss):
    """
    Tweedie loss function for regression.
    
    The Tweedie distribution is a family of distributions defined by the variance function:
    V(μ) = μ^p
    
    Common special cases:
    - p=0: Normal distribution
    - p=1: Poisson distribution
    - p=2: Gamma distribution
    - p=3: Inverse Gaussian distribution
    - 1<p<2: Compound Poisson-Gamma (useful for mixed discrete-continuous data)
    
    Args:
        p: Power parameter defining the variance function V(μ) = μ^p
        eps: Small constant for numerical stability
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        link: Link function, 'log' or 'identity'. Default is 'log' for p>=1, 'identity' for p=0.
    """
    def __init__(
        self, 
        p: float = 1.5, 
        eps: float = 1e-8,
        reduction: str = 'mean',
        link: Optional[str] = None
    ):
        super().__init__(reduction=reduction)
        self.p = p
        self.eps = eps
        
        # Set default link function based on p
        if link is None:
            self.link = 'identity' if p == 0 else 'log'
        else:
            if link not in ['identity', 'log']:
                raise ValueError(f"link must be 'identity' or 'log', got {link}")
            self.link = link
        
        # Validate p value
        if p < 0:
            raise ValueError(f"Power parameter p must be non-negative, got {p}")
        if 0 < p < 1:
            raise ValueError(f"Power parameter p between 0 and 1 is not supported, got {p}")
        
    def _get_mean(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Get mean parameter from prediction based on link function.
        
        Args:
            y_pred: Model predictions
            
        Returns:
            Mean parameter μ
        """
        if self.link == 'log':
            mu = torch.exp(y_pred)
        else:  # identity
            mu = y_pred
        return torch.clamp(mu, min=self.eps)
    
    def _normal_loss(self, y_true: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Normal distribution loss (p=0).
        
        Args:
            y_true: Ground truth values
            mu: Mean parameter
            
        Returns:
            Loss tensor
        """
        return (y_true - mu)**2 / 2
    
    def _poisson_loss(self, y_true: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Poisson distribution loss (p=1).
        
        Args:
            y_true: Ground truth values
            mu: Mean parameter
            
        Returns:
            Loss tensor
        """
        zero_mask = y_true == 0
        non_zero_mask = ~zero_mask
        
        loss = torch.zeros_like(y_true)
        if torch.any(non_zero_mask):
            loss[non_zero_mask] = y_true[non_zero_mask] * torch.log(
                y_true[non_zero_mask] / mu[non_zero_mask] + self.eps
            ) - (y_true[non_zero_mask] - mu[non_zero_mask])
        loss[zero_mask] = mu[zero_mask]
        return loss
    
    def _gamma_loss(self, y_true: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Gamma distribution loss (p=2).
        
        Args:
            y_true: Ground truth values
            mu: Mean parameter
            
        Returns:
            Loss tensor
        """
        return torch.log(mu / (y_true + self.eps) + self.eps) + y_true / (mu + self.eps) - 1
    
    def _inverse_gaussian_loss(self, y_true: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Inverse Gaussian loss (p=3).
        
        Args:
            y_true: Ground truth values
            mu: Mean parameter
            
        Returns:
            Loss tensor
        """
        return (y_true - mu)**2 / (y_true * mu**2 + self.eps)
    
    def _compound_poisson_loss(self, y_true: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Compound Poisson-Gamma loss (1<p<2).
        
        Args:
            y_true: Ground truth values
            mu: Mean parameter
            
        Returns:
            Loss tensor
        """
        zero_mask = y_true == 0
        non_zero_mask = ~zero_mask
        
        loss = torch.zeros_like(y_true)
        
        # Constants for readability
        p1 = 1 - self.p
        p2 = 2 - self.p
        
        if torch.any(non_zero_mask):
            term1 = y_true[non_zero_mask]**(p2) / (p1 * p2)
            term2 = y_true[non_zero_mask] * mu[non_zero_mask]**(p1) / p1
            term3 = mu[non_zero_mask]**(p2) / p2
            loss[non_zero_mask] = 2 * (term1 - term2 + term3)
        
        loss[zero_mask] = 2 * mu[zero_mask]**(p2) / p2
        return loss
    
    def forward(self, 
               y_true: torch.Tensor, 
               y_pred: torch.Tensor, 
               mask: Optional[torch.Tensor] = None,
               weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate Tweedie loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values (log(μ) if link=='log', μ if link=='identity') [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Tweedie loss value
        """
        # Validate inputs
        self._validate_inputs(y_true, y_pred, mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Get mean parameter μ
        mu = self._get_mean(y_pred)
        
        # Calculate loss based on Tweedie deviance
        if self.p == 0:  # Normal distribution
            loss = self._normal_loss(y_true, mu)
        elif self.p == 1:  # Poisson distribution
            loss = self._poisson_loss(y_true, mu)
        elif self.p == 2:  # Gamma distribution
            loss = self._gamma_loss(y_true, mu)
        elif self.p == 3:  # Inverse Gaussian
            loss = self._inverse_gaussian_loss(y_true, mu)
        elif 1 < self.p < 2:  # Compound Poisson-Gamma
            loss = self._compound_poisson_loss(y_true, mu)
        else:
            raise ValueError(f"Tweedie power parameter p={self.p} not supported. "
                           f"Must be 0, 1, 2, 3, or between 1 and 2.")
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class GammaLoss(TweedieLoss):
    """
    Gamma loss for regression.
    
    The Gamma distribution is useful for modeling positive continuous variables
    with constant coefficient of variation.
    
    Args:
        eps: Small constant for numerical stability
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        link: Link function, 'log' or 'identity'
    """
    def __init__(
        self, 
        eps: float = 1e-8,
        reduction: str = 'mean',
        link: str = 'log'
    ):
        super().__init__(p=2, eps=eps, reduction=reduction, link=link)


class InverseGaussianLoss(TweedieLoss):
    """
    Inverse Gaussian loss for regression.
    
    The Inverse Gaussian distribution is useful for modeling positive continuous 
    right-skewed variables with variance proportional to the cube of the mean.
    
    Args:
        eps: Small constant for numerical stability
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        link: Link function, 'log' or 'identity'
    """
    def __init__(
        self, 
        eps: float = 1e-8,
        reduction: str = 'mean',
        link: str = 'log'
    ):
        super().__init__(p=3, eps=eps, reduction=reduction, link=link)


class CompoundPoissonLoss(TweedieLoss):
    """
    Compound Poisson-Gamma loss for regression with excess zeros.
    
    This loss is useful for modeling data with:
    1. A point mass at zero
    2. A continuous, right-skewed distribution for positive values
    
    Examples: insurance claims, precipitation, etc.
    
    Args:
        p: Power parameter between 1 and 2 (typically 1.5-1.7 works well)
        eps: Small constant for numerical stability
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        link: Link function, 'log' or 'identity'
    """
    def __init__(
        self, 
        p: float = 1.5,
        eps: float = 1e-8,
        reduction: str = 'mean',
        link: str = 'log'
    ):
        if not (1 < p < 2):
            raise ValueError(f"For CompoundPoissonLoss, p must be between 1 and 2, got {p}")
            
        super().__init__(p=p, eps=eps, reduction=reduction, link=link)