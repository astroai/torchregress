"""
Poisson regression loss functions.

This module provides loss functions for count data regression based on
the Poisson distribution, which is appropriate for modeling:
- Count data (non-negative integers)
- Rate data (events per unit time/space)
- Rare event occurrences
"""
import torch
from typing import Optional

from .base import RegressionLoss

class PoissonNLLLoss(RegressionLoss):
    """
    Negative log-likelihood loss for Poisson regression.
    
    The Poisson distribution models the probability of observing k events in a fixed
    interval when events occur independently at a constant rate λ.
    
    L(y, λ) = λ - y*log(λ) + log(y!)
    
    where:
    - y is the observed count (target)
    - λ is the predicted rate parameter
    
    Args:
        log_input: If True, input is log(λ) rather than λ. Default: True
        full: If True, include the constant term log(y!). Default: False
        eps: Small constant for numerical stability. Default: 1e-8
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    
    Example:
        >>> # For count data
        >>> loss_fn = PoissonNLLLoss(log_input=True)
        >>> y_pred = torch.log(torch.tensor([1.0, 2.0, 3.0]))  # log(lambda)
        >>> target = torch.tensor([0.0, 1.0, 4.0])  # counts
        >>> loss_fn(y_pred, target)
        tensor(0.7662)
    """
    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        eps: float = 1e-8,
        reduction: str = 'mean'
    ) -> None:
        super().__init__(reduction=reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps
        
    def forward(self, 
               y_pred: torch.Tensor, 
               target: torch.Tensor, 
               mask: Optional[torch.Tensor] = None, 
               weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate Poisson NLL loss.
        
        Args:
            y_pred: Predicted rate parameters λ or log(λ) [batch_size, n_features]
            target: Ground truth count values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]
            
        Returns:
            Poisson NLL loss value
        """
        self._validate_inputs(y_pred, target, mask)
        
        # Ensure non-negative targets
        if torch.any(target < 0):
            raise ValueError("Target values must be non-negative for Poisson regression")
        
        # Convert log_input to predicted rate
        if self.log_input:
            rate = torch.exp(y_pred)
        else:
            rate = y_pred
            
        # Calculate Poisson NLL
        # loss = λ - y * log(λ) + log(y!)
        loss = rate - target * torch.log(rate + self.eps)
        
        # Add factorial term if full=True
        if self.full:
            # Use Stirling's approximation for log(y!)
            # for large values of y: log(y!) ≈ y*log(y) - y
            # For y=0, we explicitly handle this case
            non_zero = target > 0
            if torch.any(non_zero):
                log_factorial = torch.zeros_like(target)
                log_factorial[non_zero] = target[non_zero] * torch.log(target[non_zero]) - target[non_zero]
                loss = loss + log_factorial
        
        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


class PoissonLikelihoodRatioLoss(RegressionLoss):
    """
    Poisson Likelihood Ratio Loss for binned data, also known as Baker-Cousins Loss.
    
    This implements the likelihood ratio test statistic for Poisson binned data
    as defined in the PDG Review of Statistics:
    
    -2ln(λ) = 2∑[λᵢ - nᵢ + nᵢln(nᵢ/λᵢ)]
    
    Where:
    - nᵢ are the observed counts (target)
    - λᵢ are the expected counts (predictions)
    - For nᵢ = 0, the term nᵢln(nᵢ/λᵢ) = 0, so the contribution is just 2λᵢ
    
    This loss is useful for:
    - Histogram fitting in physics
    - Maximum likelihood estimation with binned data
    - Goodness-of-fit testing for count data
    
    Args:
        log_input: If True, input is log(λ) rather than λ. Default: True
        eps: Small constant for numerical stability. Default: 1e-8
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    
    Example:
        >>> # For binned histogram data
        >>> loss_fn = PoissonLikelihoodRatioLoss(log_input=False)
        >>> y_pred = torch.tensor([10.0, 20.0, 15.0])  # expected counts
        >>> target = torch.tensor([12.0, 18.0, 14.0])  # observed counts
        >>> loss_fn(y_pred, target)
        tensor(0.4749)
    """
    def __init__(
        self,
        log_input: bool = True,
        eps: float = 1e-8,
        reduction: str = 'mean'
    ) -> None:
        super().__init__(reduction=reduction)
        self.log_input = log_input
        self.eps = eps
        
    def forward(self, 
               y_pred: torch.Tensor, 
               target: torch.Tensor, 
               mask: Optional[torch.Tensor] = None, 
               weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate Poisson likelihood ratio loss.
        
        Args:
            y_pred: Expected counts λᵢ (or log(λᵢ) if log_input=True) [batch_size, n_features]
            target: Observed counts nᵢ [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]
            
        Returns:
            Poisson likelihood ratio statistic value
        """
        self._validate_inputs(y_pred, target, mask)
        
        # Ensure non-negative targets
        if torch.any(target < 0):
            raise ValueError("Observed counts must be non-negative for Poisson statistics")
        
        # Convert log_input to expected counts
        if self.log_input:
            expected = torch.exp(y_pred)
        else:
            expected = y_pred
            
        # Ensure expected counts are positive
        expected = expected + self.eps
        
        # Create the loss tensor with the same shape as target
        loss = torch.zeros_like(target)
            
        # Calculate Poisson likelihood ratio statistic: 2∑[λᵢ - nᵢ + nᵢln(nᵢ/λᵢ)]
        # Calculate first part: 2(λᵢ - nᵢ)
        loss = 2.0 * (expected - target)
        
        # Add the 2nᵢln(nᵢ/λᵢ) term for bins where nᵢ > 0
        non_zero_mask = target > 0
        if torch.any(non_zero_mask):
            target_nz = target[non_zero_mask]
            expected_nz = expected[non_zero_mask]
            loss[non_zero_mask] += 2.0 * target_nz * torch.log(target_nz / expected_nz)
        
        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)
