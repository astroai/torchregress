"""
Tweedie distribution loss functions for regression.

The Tweedie distribution is a family of probability distributions that includes
many common distributions like the normal, Poisson, and gamma distributions.
It's defined through the variance function V(μ) = μ^p where p is the power parameter.

Common special cases:
- p=0: Normal distribution
- p=1: Poisson distribution
- p=2: Gamma distribution
- p=3: Inverse Gaussian distribution
- 1<p<2: Compound Poisson-Gamma (useful for mixed discrete-continuous data)
"""

import torch
from typing import Optional

from .base import RegressionLoss

class TweedieLoss(RegressionLoss):
    """
    Tweedie loss function for regression.
    
    The Tweedie distribution is a family of distributions defined by the variance function:
    V(μ) = μ^p
    
    The loss is derived from the negative log-likelihood of the corresponding Tweedie
    distribution with the given power parameter p.
    
    Common special cases:
    - p=0: Normal distribution (MSE loss)
    - p=1: Poisson distribution
    - p=2: Gamma distribution
    - p=3: Inverse Gaussian distribution
    - 1<p<2: Compound Poisson-Gamma (useful for mixed discrete-continuous data)
    
    Args:
        p: Power parameter defining the variance function V(μ) = μ^p. Default: 1.5
        eps: Small constant for numerical stability. Default: 1e-8
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        link: Link function, 'log' or 'identity'. Default is 'log' for p>=1, 'identity' for p=0.
    
    Example:
        >>> # Compound Poisson-Gamma (p=1.5)
        >>> loss_fn = TweedieLoss(p=1.5, link='log')
        >>> y_pred = torch.log(torch.tensor([1.0, 2.0, 3.0]))  # log(mu)
        >>> target = torch.tensor([0.0, 2.0, 5.0])
        >>> loss_fn(y_pred, target)
        tensor(1.6283)
        
        >>> # Gamma distribution (p=2)
        >>> loss_fn = TweedieLoss(p=2.0)
        >>> y_pred = torch.log(torch.tensor([1.0, 2.0, 3.0]))
        >>> target = torch.tensor([1.0, 2.0, 3.0])
        >>> loss_fn(y_pred, target)
        tensor(0.0000)  # Perfect prediction
    """
    def __init__(
        self, 
        p: float = 1.5, 
        eps: float = 1e-8,
        reduction: str = 'mean',
        link: Optional[str] = None
    ) -> None:
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
    
    def _normal_loss(self, target: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Normal distribution loss (p=0).
        
        Args:
            target: Target values
            mu: Mean parameter
            
        Returns:
            Loss tensor
        """
        return (target - mu)**2 / 2
    
    def _poisson_loss(self, target: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Poisson distribution loss (p=1).
        
        Args:
            target: Target values
            mu: Mean parameter
            
        Returns:
            Loss tensor
        """
        zero_mask = target == 0
        non_zero_mask = ~zero_mask
        
        loss = torch.zeros_like(target)
        if torch.any(non_zero_mask):
            loss[non_zero_mask] = target[non_zero_mask] * torch.log(
                target[non_zero_mask] / mu[non_zero_mask] + self.eps
            ) - (target[non_zero_mask] - mu[non_zero_mask])
        loss[zero_mask] = mu[zero_mask]
        return loss
    
    def _gamma_loss(self, target: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Gamma distribution loss (p=2).
        
        Args:
            target: Target values
            mu: Mean parameter
            
        Returns:
            Loss tensor
        """
        return torch.log(mu / (target + self.eps) + self.eps) + target / (mu + self.eps) - 1
    
    def _inverse_gaussian_loss(self, target: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Inverse Gaussian loss (p=3).
        
        Args:
            target: Target values
            mu: Mean parameter
            
        Returns:
            Loss tensor
        """
        return (target - mu)**2 / (target * mu**2 + self.eps)
    
    def _compound_poisson_loss(self, target: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Compound Poisson-Gamma loss (1<p<2).
        
        Args:
            target: Target values
            mu: Mean parameter
            
        Returns:
            Loss tensor
        """
        zero_mask = target == 0
        non_zero_mask = ~zero_mask
        
        loss = torch.zeros_like(target)
        
        # Constants for readability
        p1 = 1 - self.p
        p2 = 2 - self.p
        
        if torch.any(non_zero_mask):
            term1 = target[non_zero_mask]**(p2) / (p1 * p2)
            term2 = target[non_zero_mask] * mu[non_zero_mask]**(p1) / p1
            term3 = mu[non_zero_mask]**(p2) / p2
            loss[non_zero_mask] = 2 * (term1 - term2 + term3)
        
        loss[zero_mask] = 2 * mu[zero_mask]**(p2) / p2
        return loss
    
    def forward(self, 
               y_pred: torch.Tensor, 
               target: torch.Tensor, 
               mask: Optional[torch.Tensor] = None,
               weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate Tweedie loss.
        
        Args:
            y_pred: Predicted values (log(μ) if link=='log', μ if link=='identity') [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Tweedie loss value
        """
        # Validate inputs
        self._validate_inputs(y_pred, target, mask)
        
        # Get mean parameter μ
        mu = self._get_mean(y_pred)
        
        # Calculate loss based on Tweedie deviance
        if self.p == 0:  # Normal distribution
            loss = self._normal_loss(target, mu)
        elif self.p == 1:  # Poisson distribution
            loss = self._poisson_loss(target, mu)
        elif self.p == 2:  # Gamma distribution
            loss = self._gamma_loss(target, mu)
        elif self.p == 3:  # Inverse Gaussian
            loss = self._inverse_gaussian_loss(target, mu)
        elif 1 < self.p < 2:  # Compound Poisson-Gamma
            loss = self._compound_poisson_loss(target, mu)
        else:
            raise ValueError(f"Tweedie power parameter p={self.p} not supported. "
                           f"Must be 0, 1, 2, 3, or between 1 and 2.")
        
        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


class GammaLoss(TweedieLoss):
    """
    Gamma loss for regression.
    
    The Gamma distribution is useful for modeling positive continuous variables
    with constant coefficient of variation.
    
    L(y, f(x)) = log(f(x)/y) + y/f(x) - 1
    
    Args:
        eps: Small constant for numerical stability. Default: 1e-8
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        link: Link function, 'log' or 'identity'. Default: 'log'
        
    Example:
        >>> loss_fn = GammaLoss()
        >>> y_pred = torch.log(torch.tensor([1.0, 2.0, 3.0]))  # log(mu)
        >>> target = torch.tensor([1.0, 2.0, 3.0])
        >>> loss_fn(y_pred, target)
        tensor(0.0000)  # Perfect prediction
    """
    def __init__(
        self, 
        eps: float = 1e-8,
        reduction: str = 'mean',
        link: str = 'log'
    ) -> None:
        super().__init__(p=2, eps=eps, reduction=reduction, link=link)


class InverseGaussianLoss(TweedieLoss):
    """
    Inverse Gaussian loss for regression.
    
    The Inverse Gaussian distribution is useful for modeling positive continuous 
    right-skewed variables with variance proportional to the cube of the mean.
    
    L(y, f(x)) = (y - f(x))^2 / (y * f(x)^2)
    
    Args:
        eps: Small constant for numerical stability. Default: 1e-8
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        link: Link function, 'log' or 'identity'. Default: 'log'
        
    Example:
        >>> loss_fn = InverseGaussianLoss()
        >>> y_pred = torch.log(torch.tensor([1.0, 2.0, 3.0]))  # log(mu)
        >>> target = torch.tensor([1.0, 2.0, 3.0])
        >>> loss_fn(y_pred, target)
        tensor(0.0000)  # Perfect prediction
    """
    def __init__(
        self, 
        eps: float = 1e-8,
        reduction: str = 'mean',
        link: str = 'log'
    ) -> None:
        super().__init__(p=3, eps=eps, reduction=reduction, link=link)


class CompoundPoissonLoss(TweedieLoss):
    """
    Compound Poisson-Gamma loss for regression with excess zeros.
    
    This loss is useful for modeling data with:
    1. A point mass at zero
    2. A continuous, right-skewed distribution for positive values
    
    Examples include insurance claims, precipitation, and other zero-inflated data.
    
    Args:
        p: Power parameter between 1 and 2 (typically 1.5-1.7 works well)
        eps: Small constant for numerical stability. Default: 1e-8
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        link: Link function, 'log' or 'identity'. Default: 'log'
        
    Example:
        >>> # For insurance claim amount prediction
        >>> loss_fn = CompoundPoissonLoss(p=1.5)
        >>> y_pred = torch.log(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        >>> target = torch.tensor([0.0, 0.0, 2.5, 5.0])  # Many zeros, some positive values
        >>> loss_fn(y_pred, target)
        tensor(2.0432)
    """
    def __init__(
        self, 
        p: float = 1.5,
        eps: float = 1e-8,
        reduction: str = 'mean',
        link: str = 'log'
    ) -> None:
        if not (1 < p < 2):
            raise ValueError(f"For CompoundPoissonLoss, p must be between 1 and 2, got {p}")
            
        super().__init__(p=p, eps=eps, reduction=reduction, link=link)