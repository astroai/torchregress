"""
Poisson-Gaussian mixture loss functions for signal processing applications.

This module provides loss functions that model signals as a mixture of
Poisson process (for count/shot noise) and Gaussian readout noise.
"""
import torch
import torch.nn as nn
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
        self.log_input = log_input
        self.min_variance = min_variance
        self.learn_mixture_weights = (mixture_weights == 'learn')
        self.extra_variance_model = extra_variance_model
        
        # Handle mixture weights configuration
        if mixture_weights is None:
            # Equal weighting (0.5, 0.5)
            self.register_buffer('fixed_mixture_weight', torch.tensor(0.5, dtype=torch.float32))
            self.learn_mixture_weights = False
        elif isinstance(mixture_weights, (int, float)) and not self.learn_mixture_weights:
            # Fixed weighting provided by user
            self.register_buffer('fixed_mixture_weight', torch.tensor(float(mixture_weights), dtype=torch.float32))
        elif self.learn_mixture_weights:
            # Learnable weight initialized to 0.5 (logit = 0)
            self.mixture_logit = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError("mixture_weights must be None, a float, or 'learn'")

        # Gaussian variance - fixed or learnable
        if self.learn_variance:
            self.log_variance = nn.Parameter(torch.tensor(np.log(initial_variance)))
        else:
            self.register_buffer('fixed_variance', torch.tensor(initial_variance, dtype=torch.float32))
            
        # Extra variance parameter for heteroscedastic noise model
        if self.extra_variance_model:
            # Initialize with small value (log scale for numerical stability)
            self.log_extra_var_multiplier = nn.Parameter(torch.tensor(-2.0))  # exp(-2) ≈ 0.135

        # Initialize Poisson NLL loss
        self.poisson_nll = PoissonNLL(eps=self.eps, log_input=self.log_input)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
              mask: Optional[torch.Tensor] = None, 
              extra_var: Optional[torch.Tensor] = None):
        """
        Calculate the Poisson-Gaussian mixture NLL loss.
        
        Args:
            y_true: Ground truth values [batch_size, n_features]
            y_pred: Predicted Poisson rates [batch_size, n_features]
                   If log_input=True, these are log(lambda) values
            mask: Optional mask for masked calculations
            extra_var: Optional per-sample variance predictions when using extra_variance_model
            
        Returns:
            Negative log-likelihood loss
        """
        # Apply mask
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Poisson component calculation
        poisson_loss = self.poisson_nll(y_true, y_pred, mask=None)  # Mask already applied
        
        # Convert predictions to linear space if needed
        if self.log_input:
            y_pred_linear = torch.exp(y_pred)
        else:
            y_pred_linear = y_pred
        
        # Get Gaussian variance - either fixed or learned
        if self.learn_variance:
            variance = torch.exp(self.log_variance) + self.eps
            variance = torch.clamp(variance, min=self.min_variance)
        else:
            variance = self.fixed_variance.to(y_true.device)
        
        # Gaussian component calculation 
        squared_error = (y_true - y_pred_linear) ** 2
        gaussian_loss = 0.5 * (
            squared_error / variance + 
            torch.log(variance) + 
            torch.log(torch.tensor(2 * np.pi, device=y_true.device))
        )
        
        # Get mixture weights
        if self.learn_mixture_weights:
            # Convert logit to probabilities
            poisson_weight = torch.sigmoid(self.mixture_logit)
            gaussian_weight = 1.0 - poisson_weight
        else:
            poisson_weight = self.fixed_mixture_weight.to(y_true.device)
            gaussian_weight = 1.0 - poisson_weight
            
        # Calculate base mixture loss
        loss = poisson_weight * poisson_loss + gaussian_weight * gaussian_loss
        
        # Apply extra variance model if enabled and provided
        if self.extra_variance_model and extra_var is not None:
            # Apply mask to extra_var if needed
            if mask is not None:
                extra_var = self._apply_mask(extra_var, mask)
                
            # Scale the extra variance with learned multiplier
            scaled_extra_var = torch.exp(self.log_extra_var_multiplier) * extra_var
            scaled_extra_var = torch.clamp(scaled_extra_var, min=self.eps)
            
            # Apply the extra variance - dividing the loss by (1 + scaled_var)
            # This effectively reduces the weight of samples with higher predicted variance
            loss = loss / (1.0 + scaled_extra_var)
            
            # Add log variance term (from Gaussian log likelihood formula)
            loss = loss + 0.5 * torch.log(1.0 + scaled_extra_var)
        
        return self._reduce(loss, mask)


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
            self.register_buffer('fixed_gain', torch.tensor(float(gain)))
            
        # Configure offset parameter
        self.learn_offset = (offset == 'learn')
        if self.learn_offset:
            self.offset = nn.Parameter(torch.tensor(0.0))  # Initialize with offset=0.0
        else:
            self.register_buffer('fixed_offset', torch.tensor(float(offset)))
            
        # Configure read noise (constant variance component)
        self.learn_read_noise = (read_noise == 'learn')
        if self.learn_read_noise:
            # Initialize with log(1.0) = 0.0
            self.log_read_noise = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('fixed_read_noise', torch.tensor(float(read_noise)))
            
        # Configure shot noise (signal-dependent variance component)
        self.learn_shot_noise = (shot_noise == 'learn')
        if self.learn_shot_noise:
            # Initialize with log(small value) for stability
            self.log_shot_noise = nn.Parameter(torch.tensor(-4.0))  # exp(-4) ≈ 0.018
        else:
            self.register_buffer('fixed_shot_noise', torch.tensor(float(shot_noise)))
            
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
