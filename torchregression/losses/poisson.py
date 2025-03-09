"""
Poisson regression loss functions.

This module provides loss functions for count data regression based on
the Poisson distribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union

from .base import RegressionLoss, MaskedLoss
from ..utils.validation import validate_positive
from ..utils.tensor_ops import apply_mask, masked_reduction

class PoissonNLL(MaskedLoss):
    """
    Negative log-likelihood for Poisson distributed targets.
    
    For count data, the Poisson distribution models:
    P(Y=y|λ) = λ^y * e^(-λ) / y!
    
    The NLL loss is: λ - y*log(λ) + log(y!)
    
    Args:
        log_input: If True, input is assumed to be log(lambda) instead of lambda
        eps: Small constant for numerical stability
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, log_input: bool = False, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.log_input = log_input
        self.eps = eps
    
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate Poisson negative log-likelihood loss.
        
        Args:
            y_true: Target values (should be non-negative counts)
            y_pred: Predicted values (lambda or log-lambda)
            mask: Optional mask for valid values
            weights: Optional sample weights
            
        Returns:
            Loss tensor
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Apply log if needed
        if self.log_input:
            log_lambda = y_pred
            # Avoid overflow in exp(log_lambda)
            log_lambda = torch.clamp(log_lambda, max=20)
            rate = torch.exp(log_lambda)
        else:
            rate = y_pred
            log_lambda = torch.log(rate + self.eps)
        
        # Calculate NLL: λ - y*log(λ), ignoring the factorial term as it's constant
        loss = rate - y_true * log_lambda
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
        
        # Apply final reduction
        return self._reduce(loss, mask)


# Alias with PyTorch-like naming
PoissonNLLLoss = PoissonNLL


class TruncatedPoissonNLL(MaskedLoss):
    """
    Negative log-likelihood for truncated Poisson distribution.
    
    This is useful when the data is truncated (e.g., zero counts are not observed).
    
    Args:
        min_value: Minimum observed value (truncation point)
        log_input: If True, input is assumed to be log(lambda) instead of lambda
        eps: Small constant for numerical stability
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, min_value: int = 1, log_input: bool = False, 
                 eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.min_value = min_value
        self.log_input = log_input
        self.eps = eps
    
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate truncated Poisson negative log-likelihood.
        
        Args:
            y_true: Target values (should be >= min_value)
            y_pred: Predicted values (lambda or log-lambda)
            mask: Optional mask for valid values
            weights: Optional sample weights
            
        Returns:
            Loss tensor
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Apply log if needed
        if self.log_input:
            log_lambda = y_pred
            # Avoid overflow in exp(log_lambda)
            log_lambda = torch.clamp(log_lambda, max=20)
            rate = torch.exp(log_lambda)
        else:
            rate = y_pred
            log_lambda = torch.log(rate + self.eps)
        
        # Calculate standard Poisson NLL: λ - y*log(λ)
        standard_nll = rate - y_true * log_lambda
        
        # Calculate truncation adjustment: -log(1 - sum_{i=0}^{min_value-1} p(i))
        # For min_value=1 (truncated at zero), this becomes: -log(1 - e^(-λ))
        if self.min_value == 1:
            # For zero-truncated Poisson (most common case)
            adjustment = -torch.log(1.0 - torch.exp(-rate) + self.eps)
        else:
            # For general truncation point, calculate the CDF up to min_value-1
            # This would be more complex, involving the incomplete gamma function
            # Here's a simple approximation for small truncation points
            adjustment = torch.zeros_like(rate)
            factorial = 1
            for i in range(self.min_value):
                adjustment += (rate ** i) * torch.exp(-rate) / factorial
                factorial *= (i + 1)
            adjustment = -torch.log(1.0 - adjustment + self.eps)
        
        # Combine the standard NLL with the adjustment
        loss = standard_nll + adjustment
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
        
        # Apply final reduction
        return self._reduce(loss, mask)


class ZeroInflatedPoissonNLL(MaskedLoss):
    """
    Negative log-likelihood for Zero-Inflated Poisson (ZIP) distribution.
    
    The ZIP model has two parameters:
    - λ (rate parameter of the Poisson distribution)
    - π (probability of structural zeros)
    
    Args:
        log_input: If True, input is assumed to be log(lambda) instead of lambda
        eps: Small constant for numerical stability
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, log_input: bool = False, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.log_input = log_input
        self.eps = eps
    
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate ZIP negative log-likelihood.
        
        Args:
            y_true: Target values (counts)
            y_pred: Predicted values, should be a tuple of (rate, zero_prob) or a tensor
                   with shape [..., 2] where the last dimension contains [rate, zero_logit]
            mask: Optional mask for valid values
            weights: Optional sample weights
            
        Returns:
            Loss tensor
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        
        # Extract rate and zero probability
        if isinstance(y_pred, tuple) and len(y_pred) == 2:
            rate_pred, zero_logit = y_pred
        elif y_pred.shape[-1] == 2:
            rate_pred = y_pred[..., 0]
            zero_logit = y_pred[..., 1]
        else:
            raise ValueError("ZIP loss expects y_pred to be a tuple (rate, zero_logit) "
                             "or a tensor with shape [..., 2]")
        
        rate_pred = self._apply_mask(rate_pred, mask)
        zero_logit = self._apply_mask(zero_logit, mask)
        
        # Convert to rates and probabilities
        if self.log_input:
            log_rate = rate_pred
            rate = torch.exp(torch.clamp(log_rate, max=20))
        else:
            rate = rate_pred
            log_rate = torch.log(rate + self.eps)
        
        # Calculate zero probability using sigmoid
        zero_prob = torch.sigmoid(zero_logit)
        
        # Calculate log probability
        # For y=0: log(π + (1-π) * e^(-λ))
        # For y>0: log(1-π) + y*log(λ) - λ - log(y!)
        
        # Create a mask for zeros
        is_zero = (y_true == 0)
        
        # Initialize log probabilities
        log_prob = torch.zeros_like(y_true)
        
        # For zero counts
        if is_zero.any():
            log_prob_zero = torch.log(zero_prob + (1 - zero_prob) * torch.exp(-rate) + self.eps)
            log_prob = torch.where(is_zero, log_prob_zero, log_prob)
        
        # For non-zero counts
        non_zero = ~is_zero
        if non_zero.any():
            log_prob_nonzero = (torch.log(1 - zero_prob + self.eps) + 
                                y_true * log_rate - rate)
            log_prob = torch.where(non_zero, log_prob_nonzero, log_prob)
        
        # Calculate negative log likelihood
        loss = -log_prob
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
        
        # Apply final reduction
        return self._reduce(loss, mask)


# Factory functions for easy instantiation
def poisson_nll_loss(log_input: bool = False, eps: float = 1e-8, reduction: str = 'mean') -> PoissonNLL:
    """Create a Poisson NLL loss."""
    return PoissonNLL(log_input=log_input, eps=eps, reduction=reduction)

def truncated_poisson_loss(min_value: int = 1, log_input: bool = False, 
                          eps: float = 1e-8, reduction: str = 'mean') -> TruncatedPoissonNLL:
    """Create a truncated Poisson NLL loss."""
    return TruncatedPoissonNLL(min_value=min_value, log_input=log_input, 
                             eps=eps, reduction=reduction)

def zero_inflated_poisson_loss(log_input: bool = False, eps: float = 1e-8, 
                              reduction: str = 'mean') -> ZeroInflatedPoissonNLL:
    """Create a Zero-Inflated Poisson NLL loss."""
    return ZeroInflatedPoissonNLL(log_input=log_input, eps=eps, reduction=reduction)

