"""
Poisson-Gaussian mixture loss functions for signal processing applications.

This module provides loss functions that model signals as a mixture of
Poisson process (for count/shot noise) and Gaussian readout noise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union

from .base import MaskedLoss
from .poisson import PoissonNLL

class PoissonGaussianMixtureLoss(MaskedLoss):
    """
    Negative log-likelihood loss for a mixture of Gaussian (readout noise)
    and Poisson (count) noise, common in imaging and signal processing.
    
    This models the signal as a combination of:
    y ~ Poisson(λ) + N(0, σ²)
    
    Args:
        eps: Small constant for numerical stability
        learn_variance: Whether to learn the Gaussian variance parameter
        initial_variance: Initial value for Gaussian variance
        min_variance: Minimum variance value for numerical stability
        log_input: Whether y_pred is provided as log(lambda)
        mixture_weights: How to weight the mixture components:
            - If None: Equal weighting (0.5, 0.5)
            - If float: Fixed weighting (mixture_weights, 1-mixture_weights)
            - If 'learn': Learn the mixture weights
        extra_variance_model: Whether to include a separate learned variance term
        reduction: Method for loss reduction
    """
    def __init__(self, eps: float = 1e-8, learn_variance: bool = False,
                 initial_variance: float = 1.0, min_variance: float = 1e-6,
                 log_input: bool = False, mixture_weights: Optional[Union[float, str]] = None,
                 extra_variance_model: bool = False, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.eps = eps
        self.learn_variance = learn_variance
        self.initial_variance = initial_variance
        self.min_variance = min_variance
        self.log_input = log_input
        self.mixture_weights = mixture_weights
        self.extra_variance_model = extra_variance_model
        
        # Initialize learnable parameters if needed
        if learn_variance:
            self.log_variance = nn.Parameter(torch.ones(1) * torch.log(torch.tensor(initial_variance)))
            
        if mixture_weights == 'learn':
            self.weight_logit = nn.Parameter(torch.zeros(1))
            
        # Initialize Poisson loss
        self.poisson_loss = PoissonNLL(log_input=log_input, reduction='none')


    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
              mask: Optional[torch.Tensor] = None, 
              extra_var: Optional[torch.Tensor] = None):
        """
        Calculate the Poisson-Gaussian mixture loss.
        
        Args:
            y_true: Target values
            y_pred: Predicted values (lambda for Poisson component)
            mask: Optional mask for invalid values
            extra_var: Optional per-sample variance term
            
        Returns:
            Loss value
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Get lambda (rate) parameter
        if self.log_input:
            lam = torch.exp(y_pred)
        else:
            lam = y_pred
            
        # Calculate Poisson component
        poisson_nll = self.poisson_loss(y_true, y_pred)
        
        # Calculate Gaussian component (using squared error)
        residuals = y_true - lam
        
        # Get variance parameter
        if self.learn_variance:
            variance = torch.exp(self.log_variance).clamp(min=self.min_variance)
        else:
            variance = torch.tensor(self.initial_variance, device=y_true.device)
            
        # Add extra variance if provided
        if extra_var is not None and self.extra_variance_model:
            variance = variance + extra_var
            
        # Calculate Gaussian NLL: 0.5 * (log(2π) + log(σ²) + (y-μ)²/σ²)
        gaussian_nll = 0.5 * (torch.log(2 * torch.pi * variance + self.eps) + 
                             residuals**2 / (variance + self.eps))
        
        # Calculate mixture weights
        if self.mixture_weights is None:
            # Equal weights (0.5, 0.5)
            poisson_weight = 0.5
            gaussian_weight = 0.5
        elif self.mixture_weights == 'learn':
            # Learned weights using sigmoid to ensure they sum to 1
            poisson_weight = torch.sigmoid(self.weight_logit)
            gaussian_weight = 1 - poisson_weight
        else:
            # Fixed weights
            poisson_weight = self.mixture_weights
            gaussian_weight = 1 - self.mixture_weights
            
        # Calculate weighted mixture
        mixture_nll = poisson_weight * poisson_nll + gaussian_weight * gaussian_nll
        
        # Apply reduction
        return self._reduce(mixture_nll, mask)


def poisson_gaussian_mixture_loss(learn_variance: bool = False, initial_variance: float = 1.0,
                                  log_input: bool = False, mixture_weights: Optional[Union[float, str]] = None,
                                  extra_variance_model: bool = False, **kwargs) -> PoissonGaussianMixtureLoss:
    """
    Create a Poisson-Gaussian mixture loss function.
    
    Args:
        learn_variance: Whether to learn the Gaussian variance
        initial_variance: Initial value for Gaussian variance
        log_input: Whether inputs are in log space
        mixture_weights: How to handle mixture weights
        extra_variance_model: Whether to include extra variance terms
        **kwargs: Additional parameters for the loss
        
    Returns:
        PoissonGaussianMixtureLoss instance
    """
    return PoissonGaussianMixtureLoss(
        learn_variance=learn_variance,
        initial_variance=initial_variance,
        log_input=log_input,
        mixture_weights=mixture_weights,
        extra_variance_model=extra_variance_model,
        **kwargs
    )


class EnhancedPoissonGaussianMixtureLoss(MaskedLoss):
    """
    Advanced Poisson-Gaussian mixture loss with additional features for scientific applications.
    
    This extends the basic Poisson-Gaussian mixture with:
    1. Support for gain/scaling factor in the model
    2. Handling of offset/bias term
    3. Optional calibration parameters
    4. Multiple variance components
    
    Model: y ~ Poisson(g * λ + b) + N(0, σ₁² + σ₂² * λ)
    
    Args:
        gain: Fixed gain/scaling factor or 'learn' to make it learnable
        offset: Fixed offset/bias or 'learn' to make it learnable
        read_noise: Constant variance component (σ₁²)
        shot_noise: Signal-dependent variance component (σ₂²)
        log_input: Whether inputs are in log space
        calibration: Whether to include calibration parameters
        reduction: Method for loss reduction
    """
    def __init__(self, gain: Union[float, str] = 1.0, offset: Union[float, str] = 0.0,
                read_noise: Union[float, str] = 1.0, shot_noise: Union[float, str] = 0.0,
                log_input: bool = False, calibration: bool = False, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.log_input = log_input
        self.calibration = calibration
        self.eps = 1e-8
        
        # Configure gain parameter
        self.learn_gain = (gain == 'learn')
        if self.learn_gain:
            self.log_gain = nn.Parameter(torch.tensor(0.0))  # Initialize with gain=1.0
        else:
            self.registerBuffer('fixed_gain', torch.tensor(float(gain)))
            
        # Configure offset parameter
        self.learn_offset = (offset == 'learn')
        if self.learn_offset:
            self.offset = nn.Parameter(torch.tensor(0.0))  # Initialize with offset=0.0
        else:
            self.registerBuffer('fixed_offset', torch.tensor(float(offset)))
            
        # Configure read noise (constant variance component)
        self.learn_read_noise = (read_noise == 'learn')
        if self.learn_read_noise:
            # Initialize with log(1.0) = 0.0
            self.log_read_noise = nn.Parameter(torch.tensor(0.0))
        else:
            self.registerBuffer('fixed_read_noise', torch.tensor(float(read_noise)))
            
        # Configure shot noise (signal-dependent variance component)
        self.learn_shot_noise = (shot_noise == 'learn')
        if self.learn_shot_noise:
            # Initialize with log(small value) for stability
            self.log_shot_noise = nn.Parameter(torch.tensor(-4.0))  # exp(-4) ≈ 0.018
        else:
            self.registerBuffer('fixed_shot_noise', torch.tensor(float(shot_noise)))
            
        # Calibration parameters (optional)
        if self.calibration:
            # Initializing intercept (add) and slope (mult) parameters
            self.calib_add = nn.Parameter(torch.tensor(0.0))
            self.calib_mult = nn.Parameter(torch.tensor(1.0))
            
        # Initialize Poisson NLL
        self.poisson_nll = PoissonNLL(eps=self.eps, log_input=False)  # We'll handle log_input ourselves
        
    def forward(self, y_true, y_pred, mask=None):
        """
        Calculate enhanced Poisson-Gaussian mixture loss.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted rate parameters or log(rate)
            mask: Optional mask for masked loss
            
        Returns:
            Negative log-likelihood loss
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Convert from log space if needed
        if self.log_input:
            rate = torch.exp(y_pred)
        else:
            rate = y_pred
            
        # Apply calibration if enabled
        if self.calibration:
            rate = self.calib_mult * rate + self.calib_add
            
        # Get gain parameter
        if self.learn_gain:
            gain = torch.exp(self.log_gain)
        else:
            gain = self.fixed_gain.to(y_true.device)
            
        # Get offset parameter
        if self.learn_offset:
            offset = self.offset
        else:
            offset = self.fixed_offset.to(y_true.device)
            
        # Apply gain and offset to rate
        scaled_rate = gain * rate + offset
        scaled_rate = torch.clamp(scaled_rate, min=self.eps)  # Ensure positive rate
        
        # Calculate Poisson component
        poisson_loss = self.poisson_nll(y_true, scaled_rate)
        
        # Calculate variance components for Gaussian
        if self.learn_read_noise:
            read_var = torch.exp(self.log_read_noise)
        else:
            read_var = self.fixed_read_noise.to(y_true.device)
            
        if self.learn_shot_noise:
            shot_coef = torch.exp(self.log_shot_noise)
        else:
            shot_coef = self.fixed_shot_noise.to(y_true.device)
            
        # Total variance: read noise + shot noise * signal
        total_var = read_var + shot_coef * scaled_rate
        total_var = torch.clamp(total_var, min=self.eps)
        
        # Calculate Gaussian component
        squared_error = (y_true - scaled_rate) ** 2
        gaussian_loss = 0.5 * (
            squared_error / total_var + 
            torch.log(total_var) + 
            torch.log(torch.tensor(2 * np.pi, device=y_true.device))
        )
        
        # Combine losses - use equal weight for simplicity
        # Could be extended to learn weights if needed
        loss = 0.5 * (poisson_loss + gaussian_loss)
        
        return self._reduce(loss, mask)
        

def enhanced_poisson_gaussian_loss(**kwargs) -> EnhancedPoissonGaussianMixtureLoss:
    """
    Create an enhanced Poisson-Gaussian mixture loss.
    
    Args:
        **kwargs: Parameters passed to EnhancedPoissonGaussianMixtureLoss
        
    Returns:
        EnhancedPoissonGaussianMixtureLoss instance
    """
    return EnhancedPoissonGaussianMixtureLoss(**kwargs)
