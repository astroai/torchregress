"""
Poisson regression loss functions.

This module provides loss functions for count data regression based on
the Poisson distribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict

from .base import RegressionLoss
from ..utils.validation import validate_positive


class PoissonNLLLoss(RegressionLoss):
    """
    Negative log-likelihood loss for Poisson regression.
    
    The Poisson distribution models the probability of observing k events in a fixed
    interval when events occur independently at a constant rate λ.
    
    This loss implements log(y!) - y*log(λ) + λ, where y is the observed count and
    λ is the predicted rate.
    
    Args:
        log_input: Whether model output is log(λ) rather than λ
        full: Whether to include the constant term log(y!)
        eps: Small constant for numerical stability
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """
    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        eps: float = 1e-8,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Poisson NLL loss.
        
        Args:
            y_true: Ground truth count values [batch_size, n_features]
            y_pred: Predicted rate parameters λ or log(λ) [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]
            
        Returns:
            Poisson NLL loss value
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Ensure non-negative targets
        if torch.any(y_true < 0):
            raise ValueError("Target values must be non-negative for Poisson regression")
        
        # Convert log_input to predicted rate
        if self.log_input:
            rate = torch.exp(y_pred)
        else:
            rate = y_pred
            
        # Calculate Poisson NLL
        # loss = λ - y * log(λ) + log(y!)
        loss = rate - y_true * torch.log(rate + self.eps)
        
        # Add factorial term if full=True
        if self.full:
            # Use Stirling's approximation for log(y!)
            # for large values of y: log(y!) ≈ y*log(y) - y
            # For y=0, we explicitly handle this case
            non_zero = y_true > 0
            if torch.any(non_zero):
                log_factorial = torch.zeros_like(y_true)
                log_factorial[non_zero] = y_true[non_zero] * torch.log(y_true[non_zero]) - y_true[non_zero]
                loss = loss + log_factorial
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
        
        return self._reduce(loss, mask)

class PoissonLikelihoodRatioLoss(RegressionLoss):
    """
    Poisson Likelihood Ratio Loss for binned data, also known as Baker-Cousins Loss.
    
    This implements the likelihood ratio test statistic for Poisson binned data
    as defined in the PDG Review of Statistics:
    
    -2ln(λ) = 2∑[λᵢ - nᵢ + nᵢln(nᵢ/λᵢ)]
    
    Where:
    - nᵢ are the observed counts
    - λᵢ are the expected counts
    - For nᵢ = 0, the term nᵢln(nᵢ/λᵢ) = 0, so the contribution is just 2λᵢ
    
    Args:
        log_input: Whether model output is log(λ) rather than λ
        eps: Small constant for numerical stability
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """
    def __init__(
        self,
        log_input: bool = True,
        eps: float = 1e-8,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        self.log_input = log_input
        self.eps = eps
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Poisson likelihood ratio loss.
        
        Args:
            y_true: Observed counts nᵢ [batch_size, n_features]
            y_pred: Expected counts λᵢ (or log(λᵢ) if log_input=True) [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]
            
        Returns:
            Poisson likelihood ratio statistic value
        """
        # Make a copy of the mask for later use
        original_mask = mask
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Ensure non-negative targets
        if torch.any(y_true < 0):
            raise ValueError("Observed counts must be non-negative for Poisson statistics")
        
        # Convert log_input to expected counts
        if self.log_input:
            expected = torch.exp(y_pred)
        else:
            expected = y_pred
            
        # Ensure expected counts are positive
        expected = expected + self.eps
        
        # Create the loss tensor with the same shape as y_true
        loss = torch.zeros_like(y_true)
            
        # Calculate Poisson likelihood ratio statistic: 2∑[λᵢ - nᵢ + nᵢln(nᵢ/λᵢ)]
        # Apply computation only to unmasked entries
        compute_mask = torch.ones_like(y_true, dtype=torch.bool)
        if mask is not None:
            compute_mask = mask
        
        # Calculate first part: 2(λᵢ - nᵢ)
        loss[compute_mask] = 2.0 * (expected[compute_mask] - y_true[compute_mask])
        
        # Add the 2nᵢln(nᵢ/λᵢ) term for bins where nᵢ > 0 and not masked
        non_zero_mask = y_true > 0
        if mask is not None:
            non_zero_mask = non_zero_mask & mask
        
        if torch.any(non_zero_mask):
            y_nz = y_true[non_zero_mask]
            expected_nz = expected[non_zero_mask]
            loss[non_zero_mask] += 2.0 * y_nz * torch.log(y_nz / expected_nz)
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, original_mask)
            loss = loss * weights
        
        return self._reduce(loss, original_mask)

