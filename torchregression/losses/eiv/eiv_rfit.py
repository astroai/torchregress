import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses.base import MaskedLoss
from .robust import HuberLoss, L1Loss
from typing import Union, Optional, List, Callable, Dict, Literal
from .eiv_utils import prepare_param

class RobustEIVLoss(MaskedLoss):
    """
    Robust Errors-in-Variables Loss that uses multiple forward passes with systematically 
    varied inputs to account for uncertainties in the input features.
    
    This approach is inspired by RFit and GFitter methods (EPJ C60, 543-583, 2009, arXiv:0811.0009)
    but uses direct sampling rather than profiling. It's particularly useful for models where
    analytical gradient propagation is difficult or unstable.
    
    Args:
        model (Callable): The model function f(x) that predicts y.
        base_loss (str): Base loss function ('huber', 'l1', or 'mse'). Default: 'huber'.
        delta (float): Delta parameter for Huber loss (if used). Default: 1.0.
        variation_fn (Callable): Function to generate input variations.
            Signature: (x_batch, sigma_x, n_samples, **variation_params) -> List[torch.Tensor]
        sigma_x (float or torch.Tensor): Standard deviation for input variations.
        n_samples (int): Number of samples to generate. Default: 10.
        variation_params (dict): Additional keyword arguments for variation_fn. Default: {}.
        aggregation (str): How to aggregate losses: 'mean', 'median', 'max', or 'quantile'. Default: 'median'.
        quantile (float): Quantile level for 'quantile' aggregation (0-1). Default: 0.95.
        reduction (str): 'none', 'mean', or 'sum'. Default: 'mean'.
    """
    def __init__(
        self, 
        model: Callable,
        base_loss: str = 'huber',
        delta: float = 1.0,
        variation_fn: Callable = None,
        sigma_x: Union[float, torch.Tensor] = 1.0,
        n_samples: int = 10,
        variation_params: Dict = None,
        aggregation: Literal['mean', 'median', 'max', 'quantile'] = 'median',
        quantile: float = 0.95,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.model = model
        
        # Set up base loss function
        if base_loss.lower() == 'huber':
            self.base_loss_fn = HuberLoss(delta=delta, reduction='none')
        elif base_loss.lower() == 'l1':
            self.base_loss_fn = L1Loss(reduction='none')
        elif base_loss.lower() == 'mse':
            self.base_loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError("base_loss must be 'huber', 'l1', or 'mse'")

        # Validate variation function
        if variation_fn is None:
            raise ValueError("variation_fn must be provided")
        self.variation_fn = variation_fn
        
        # Store parameters
        self.sigma_x = sigma_x
        self.n_samples = n_samples
        self.variation_params = variation_params if variation_params is not None else {}
        
        # Validate aggregation method
        self.aggregation = aggregation.lower()
        if self.aggregation not in ('mean', 'median', 'max', 'quantile'):
            raise ValueError("aggregation must be 'mean', 'median', 'max', or 'quantile'")
        
        # Validate reduction method
        self.reduction = reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction: {reduction}")

        # Validate quantile parameter
        self.quantile = quantile
        if self.aggregation == 'quantile' and (quantile is None or not 0 <= quantile <= 1):
            raise ValueError('quantile value must be between 0 and 1')

    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate the robust EIV loss.

        Args:
            x_obs: Observed features with noise. Shape: (batch_size, n_features_x)
            y_true: Observed targets with noise. Shape: (batch_size, n_features_y)
            mask: Optional mask. Shape: (batch_size, n_features_y)

        Returns:
            loss: The robust EIV loss
        """
        # Apply mask to ground truth
        y_true = self._apply_mask(y_true, mask)
        batch_size = y_true.shape[0]
        device = x_obs.device
        
        # Prepare sigma_x parameter
        sigma_x = prepare_param(self.sigma_x, x_obs.shape[1], device)
        
        # Generate input variations using the provided function
        varied_inputs = self.variation_fn(
            x_obs, 
            sigma_x=sigma_x, 
            n_samples=self.n_samples,
            **self.variation_params
        )
        
        # Multiple forward passes with varied inputs
        all_losses = []
        for x_varied in varied_inputs:
            # Get predictions for this variation
            with torch.set_grad_enabled(True):
                y_pred = self.model(x_varied)
            
            # Apply mask to predictions
            y_pred = self._apply_mask(y_pred, mask)
            
            # Calculate loss for this variation
            batch_loss = self.base_loss_fn(y_true, y_pred)
            
            # Reduce across features if needed
            if batch_loss.dim() > 1:
                batch_loss = torch.mean(batch_loss, dim=tuple(range(1, batch_loss.dim())))
                
            all_losses.append(batch_loss)
        
        # Stack all losses [n_variations, batch_size]
        all_losses = torch.stack(all_losses)
        
        # Aggregate across variations (dimension 0)
        if self.aggregation == 'mean':
            per_sample_loss = torch.mean(all_losses, dim=0)
        elif self.aggregation == 'median':
            per_sample_loss = torch.median(all_losses, dim=0).values
        elif self.aggregation == 'max':
            per_sample_loss = torch.max(all_losses, dim=0).values
        elif self.aggregation == 'quantile':
            per_sample_loss = torch.quantile(all_losses, self.quantile, dim=0)
        
        # Apply final reduction
        if self.reduction == 'mean':
            return torch.mean(per_sample_loss)
        elif self.reduction == 'sum':
            return torch.sum(per_sample_loss)
        else:  # 'none'
            return per_sample_loss
            
    def set_aggregation(self, aggregation, quantile=None):
        """
        Update the aggregation method.
        
        Args:
            aggregation: New aggregation method ('mean', 'median', 'max', or 'quantile')
            quantile: New quantile value (required if aggregation='quantile')
        """
        if aggregation not in ('mean', 'median', 'max', 'quantile'):
            raise ValueError("aggregation must be 'mean', 'median', 'max', or 'quantile'")
            
        self.aggregation = aggregation
        
        if aggregation == 'quantile':
            if quantile is None or not 0 <= quantile <= 1:
                raise ValueError('quantile value must be between 0 and 1')
            self.quantile = quantile


# Common variation functions

def gaussian_variation(x_batch, sigma_x, n_samples=10, scale_factor=1.0, **kwargs):
    """
    Generate Gaussian variations of input features.
    
    Args:
        x_batch: Input tensor [batch_size, n_features]
        sigma_x: Standard deviation of noise
        n_samples: Number of variations to generate
        scale_factor: Additional scaling factor for the noise
        
    Returns:
        List of varied input tensors
    """
    batch_size, n_features = x_batch.shape
    device = x_batch.device
    
    variations = []
    for _ in range(n_samples):
        # Generate Gaussian noise
        if isinstance(sigma_x, (int, float)):
            noise = torch.randn(batch_size, n_features, device=device) * sigma_x * scale_factor
        else:  # tensor case
            noise = torch.randn(batch_size, n_features, device=device) * sigma_x.view(-1) * scale_factor
            
        # Add noise to create variation
        x_varied = x_batch + noise
        variations.append(x_varied)
        
    return variations


def uniform_variation(x_batch, sigma_x, n_samples=10, range_factor=1.732, **kwargs):
    """
    Generate uniform variations of input features.
    
    Args:
        x_batch: Input tensor [batch_size, n_features]
        sigma_x: Standard deviation parameter (used to determine range)
        n_samples: Number of variations to generate
        range_factor: Factor to convert sigma to uniform range (default: sqrt(3))
        
    Returns:
        List of varied input tensors
    """
    batch_size, n_features = x_batch.shape
    device = x_batch.device
    
    variations = []
    for _ in range(n_samples):
        # Generate uniform noise with range approximately matched to sigma
        half_range = sigma_x * range_factor
        
        if isinstance(half_range, (int, float)):
            noise = torch.rand(batch_size, n_features, device=device) * 2 * half_range - half_range
        else:  # tensor case
            noise = torch.rand(batch_size, n_features, device=device) * 2 * half_range.view(-1) - half_range.view(-1)
            
        # Add noise to create variation
        x_varied = x_batch + noise
        variations.append(x_varied)
        
    return variations


def bootstrap_variation(x_batch, sigma_x=None, n_samples=10, **kwargs):
    """
    Generate bootstrap variations by resampling with replacement.
    
    Args:
        x_batch: Input tensor [batch_size, n_features]
        sigma_x: Not used, included for API consistency
        n_samples: Number of variations to generate
        
    Returns:
        List of varied input tensors
    """
    batch_size = x_batch.shape[0]
    device = x_batch.device
    
    variations = []
    for _ in range(n_samples):
        # Sample indices with replacement
        indices = torch.randint(0, batch_size, (batch_size,), device=device)
        
        # Create resampled batch
        x_varied = x_batch[indices]
        variations.append(x_varied)
        
    return variations

"""
Robust fitting methods for Error-in-Variables (EIV) regression.

This module provides implementations of robust EIV models that handle
both measurement error and outliers simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, List, Tuple, Callable

from ..base import MaskedLoss
from ...utils.tensor_ops import prepare_param


def gaussian_variation(x: torch.Tensor, sigma_x: Union[float, torch.Tensor], 
                     n_samples: int) -> torch.Tensor:
    """
    Generate Gaussian variations of input samples.
    
    Args:
        x: Input tensor [batch_size, n_features]
        sigma_x: Standard deviation(s) of the noise
        n_samples: Number of variations to generate
        
    Returns:
        Tensor of variations [n_samples, batch_size, n_features]
    """
    batch_size, n_features = x.shape
    device = x.device
    
    # Handle different types of sigma_x
    if isinstance(sigma_x, (int, float)):
        # Scalar case
        noise = torch.randn(n_samples, batch_size, n_features, device=device) * sigma_x
    elif isinstance(sigma_x, torch.Tensor):
        if sigma_x.ndim == 0:
            # Scalar tensor
            noise = torch.randn(n_samples, batch_size, n_features, device=device) * sigma_x
        elif sigma_x.ndim == 1:
            # Vector - apply different noise to each feature
            if sigma_x.shape[0] != n_features:
                raise ValueError(f"sigma_x shape {sigma_x.shape} doesn't match features {n_features}")
            noise = torch.randn(n_samples, batch_size, n_features, device=device) * sigma_x.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"sigma_x must be scalar or vector, got tensor with {sigma_x.ndim} dimensions")
    else:
        raise TypeError(f"sigma_x must be float or tensor, got {type(sigma_x)}")
    
    # Create variations by adding noise to original samples
    variations = x.unsqueeze(0) + noise
    
    return variations


def uniform_variation(x: torch.Tensor, range_x: Union[float, torch.Tensor], 
                    n_samples: int) -> torch.Tensor:
    """
    Generate uniform variations of input samples.
    
    Args:
        x: Input tensor [batch_size, n_features]
        range_x: Range of the uniform noise (+/-)
        n_samples: Number of variations to generate
        
    Returns:
        Tensor of variations [n_samples, batch_size, n_features]
    """
    batch_size, n_features = x.shape
    device = x.device
    
    # Handle different types of range_x
    if isinstance(range_x, (int, float)):
        # Scalar case
        noise = torch.rand(n_samples, batch_size, n_features, device=device) * (2 * range_x) - range_x
    elif isinstance(range_x, torch.Tensor):
        if range_x.ndim == 0:
            # Scalar tensor
            noise = torch.rand(n_samples, batch_size, n_features, device=device) * (2 * range_x) - range_x
        elif range_x.ndim == 1:
            # Vector - apply different noise to each feature
            if range_x.shape[0] != n_features:
                raise ValueError(f"range_x shape {range_x.shape} doesn't match features {n_features}")
            # Expand for broadcasting
            expanded_range = range_x.unsqueeze(0).unsqueeze(0).expand(n_samples, batch_size, -1)
            noise = torch.rand(n_samples, batch_size, n_features, device=device) * (2 * expanded_range) - expanded_range
        else:
            raise ValueError(f"range_x must be scalar or vector, got tensor with {range_x.ndim} dimensions")
    else:
        raise TypeError(f"range_x must be float or tensor, got {type(range_x)}")
    
    # Create variations by adding noise to original samples
    variations = x.unsqueeze(0) + noise
    
    return variations


def bootstrap_variation(x: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Generate bootstrap variations of input samples.
    
    Args:
        x: Input tensor [batch_size, n_features]
        n_samples: Number of variations to generate
        
    Returns:
        Tensor of variations [n_samples, batch_size, n_features]
    """
    batch_size, _ = x.shape
    device = x.device
    
    # Sample with replacement
    variations = []
    for _ in range(n_samples):
        # Generate random indices with replacement
        indices = torch.randint(0, batch_size, (batch_size,), device=device)
        
        # Sample from original data
        bootstrap_sample = x[indices]
        variations.append(bootstrap_sample)
    
    # Stack along new dimension
    return torch.stack(variations)


class RobustEIVLoss(MaskedLoss):
    """
    Robust Error-in-Variables loss that uses data variations to reduce sensitivity to both 
    measurement errors and outliers.
    
    This loss generates multiple input variations and applies robust loss functions
    to handle outliers and measurement errors simultaneously.
    
    Args:
        model: Model function f(x) that predicts y
        base_loss: Base loss function type ('huber', 'l1', 'mse')
        delta: Delta parameter for Huber loss
        variation_fn: Function to generate input variations
        sigma_x: Parameter for variation function (e.g., standard deviation for Gaussian noise)
        n_samples: Number of variations to generate
        variation_params: Additional parameters for variation function
        aggregation: How to aggregate losses: 'mean', 'median', 'max', 'quantile'
        quantile: Quantile to use if aggregation='quantile'
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(
        self,
        model: Callable,
        base_loss: str = 'huber',
        delta: float = 1.0,
        variation_fn: Callable = gaussian_variation,
        sigma_x: Union[float, torch.Tensor] = 1.0,
        n_samples: int = 10,
        variation_params: Dict = None,
        aggregation: str = 'median',
        quantile: float = 0.95,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        self.model = model
        self.base_loss = base_loss
        self.delta = delta
        self.variation_fn = variation_fn
        self.sigma_x = sigma_x
        self.n_samples = n_samples
        self.variation_params = variation_params or {}
        self.aggregation = aggregation
        self.quantile = quantile
        
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate robust EIV loss.
        
        Args:
            x_obs: Observed features with noise [batch_size, n_features_x]
            y_true: Observed targets [batch_size, n_features_y]
            mask: Optional boolean mask [batch_size, n_features_y]
            
        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        y_true = self._apply_mask(y_true, mask)
        batch_size, n_features_y = y_true.shape
        device = x_obs.device
        
        # Create variations of input based on measurement error model
        if self.variation_fn == gaussian_variation:
            # Pass sigma_x as argument for Gaussian variations
            x_variations = self.variation_fn(x_obs, self.sigma_x, self.n_samples, **self.variation_params)
        elif self.variation_fn == uniform_variation:
            # Similar for uniform variations
            x_variations = self.variation_fn(x_obs, self.sigma_x, self.n_samples, **self.variation_params)
        else:
            # Pass generic variation parameters
            x_variations = self.variation_fn(x_obs, self.n_samples, **self.variation_params)
        
        # Flatten for batch processing
        x_flat = x_variations.reshape(-1, x_variations.shape[-1])
        
        # Get predictions for all variations
        with torch.no_grad():  # No need for gradients in the variation evaluation
            y_preds_flat = self.model(x_flat)
            
            # Reshape back to [n_samples, batch_size, n_features_y]
            y_preds = y_preds_flat.reshape(self.n_samples, batch_size, n_features_y)
        
        # Calculate loss for each variation
        variation_losses = []
        
        for i in range(self.n_samples):
            y_pred = y_preds[i]
            
            # Calculate residuals
            residuals = y_true - y_pred
            
            # Apply different base loss functions
            if self.base_loss == 'huber':
                # Huber loss: quadratic for small residuals, linear for large ones
                abs_residuals = torch.abs(residuals)
                loss_i = torch.where(
                    abs_residuals <= self.delta,
                    0.5 * residuals**2,
                    self.delta * (abs_residuals - 0.5 * self.delta)
                )
            elif self.base_loss == 'l1':
                # L1/MAE loss
                loss_i = torch.abs(residuals)
            else:  # 'mse'
                # MSE loss
                loss_i = 0.5 * residuals**2
                
            # Sum across feature dimension
            variation_losses.append(torch.sum(loss_i, dim=1))
            
        # Stack losses [n_samples, batch_size]
        all_losses = torch.stack(variation_losses)
        
        # Aggregate losses across variations
        if self.aggregation == 'mean':
            # Simple average across variations
            agg_loss = torch.mean(all_losses, dim=0)
        elif self.aggregation == 'median':
            # More robust to extreme variations
            agg_loss = torch.median(all_losses, dim=0).values
        elif self.aggregation == 'max':
            # Worst-case variation
            agg_loss = torch.max(all_losses, dim=0).values
        elif self.aggregation == 'quantile':
            # Specific quantile across variations (e.g., 95th percentile)
            agg_loss = torch.quantile(all_losses, self.quantile, dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(agg_loss)
        elif self.reduction == 'sum':
            return torch.sum(agg_loss)
        else:  # 'none'
            return agg_loss