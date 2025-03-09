import torch
import torch.nn as nn
import torch.nn.functional as F
from ..losses.base import MaskedLoss
from ..losses.robust import HuberLoss, L1Loss
from typing import Union, Optional, List, Callable, Dict, Literal
from ..utils.tensor_ops import prepare_param
from ..utils.validation import validate_reduction

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
        self.reduction = validate_reduction(reduction)

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

These methods provide resistance to outliers and model misspecification
when working with noisy input and output variables.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union, Dict, Any, List, Tuple

from ..losses.base import MaskedLoss
from ..losses.robust import HuberLoss, L1Loss
from ..utils.tensor_ops import prepare_sigma

def gaussian_variation(
    x: torch.Tensor,
    sigma_x: Union[float, torch.Tensor],
    n_samples: int = 10,
) -> List[torch.Tensor]:
    """
    Generate Gaussian random variations of input tensor.
    
    Args:
        x: Input tensor [batch_size, n_features]
        sigma_x: Standard deviation of noise
        n_samples: Number of variations to generate
        
    Returns:
        List of tensors with Gaussian variations
    """
    device = x.device
    batch_size, n_features = x.shape
    
    variations = []
    for _ in range(n_samples):
        # Generate noise
        if isinstance(sigma_x, torch.Tensor) and sigma_x.ndim > 0:
            if sigma_x.ndim == 1:
                # Vector sigma - scale noise for each feature
                noise = torch.randn(batch_size, n_features, device=device) * sigma_x.to(device)
            else:
                # Matrix sigma - use multivariate normal
                noise = torch.empty_like(x)
                for i in range(batch_size):
                    noise[i] = torch.distributions.MultivariateNormal(
                        loc=torch.zeros(n_features, device=device),
                        covariance_matrix=sigma_x.to(device)
                    ).sample()
        else:
            # Scalar sigma - same noise level for all features
            noise = torch.randn(batch_size, n_features, device=device) * sigma_x
            
        variations.append(x + noise)
    
    return variations

def uniform_variation(
    x: torch.Tensor,
    width: Union[float, torch.Tensor],
    n_samples: int = 10,
) -> List[torch.Tensor]:
    """
    Generate uniform random variations of input tensor.
    
    Args:
        x: Input tensor [batch_size, n_features]
        width: Width of uniform distribution (total width = 2*width)
        n_samples: Number of variations to generate
        
    Returns:
        List of tensors with uniform variations
    """
    device = x.device
    batch_size, n_features = x.shape
    
    variations = []
    for _ in range(n_samples):
        # Generate noise
        if isinstance(width, torch.Tensor) and width.ndim > 0:
            if width.ndim == 1:
                # Vector width - scale noise for each feature
                noise = (torch.rand(batch_size, n_features, device=device) * 2 - 1) * width.to(device)
            else:
                # Matrix width not supported for uniform variation
                raise ValueError("Matrix width not supported for uniform_variation")
        else:
            # Scalar width - same noise level for all features
            noise = (torch.rand(batch_size, n_features, device=device) * 2 - 1) * width
            
        variations.append(x + noise)
    
    return variations

def bootstrap_variation(
    x: torch.Tensor,
    n_samples: int = 10,
) -> List[torch.Tensor]:
    """
    Generate bootstrap resampled variations of input tensor.
    
    Args:
        x: Input tensor [batch_size, n_features]
        n_samples: Number of variations to generate
        
    Returns:
        List of tensors with bootstrap variations
    """
    batch_size = x.shape[0]
    
    variations = []
    for _ in range(n_samples):
        # Generate random indices with replacement
        indices = torch.randint(0, batch_size, (batch_size,), device=x.device)
        
        # Create bootstrapped sample
        variations.append(x[indices])
    
    return variations

class RobustEIVLoss(MaskedLoss):
    """
    Robust Error-in-Variables loss that is resistant to outliers.
    
    This loss uses a variation-based approach: it generates multiple variants
    of the input by adding noise, then aggregates the results in a robust way.
    
    Args:
        model: Model function that predicts y from x
        base_loss: Base loss function ('mse', 'huber', 'l1', etc.)
        delta: Delta parameter for Huber loss
        variation_fn: Function to generate input variations
        sigma_x: Standard deviation for input variations
        n_samples: Number of variations to generate
        variation_params: Additional parameters for variation function
        aggregation: How to aggregate losses ('mean', 'median', 'trimmed', etc.)
        quantile: Quantile for quantile-based aggregation
        reduction: 'none' | 'mean' | 'sum'
        
    Reference:
        Inspired by "Robust Fitting in Computer Vision: Easy or Hard?"
        by Meer, Mintz, and Rosenfeld.
    """
    def __init__(
        self,
        model: Callable,
        base_loss: str = 'huber',
        delta: float = 1.0,
        variation_fn: Optional[Callable] = None,
        sigma_x: Union[float, torch.Tensor] = 1.0,
        n_samples: int = 10,
        variation_params: Dict[str, Any] = None,
        aggregation: str = 'median',
        quantile: float = 0.95,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        
        self.model = model
        self.n_samples = n_samples
        self.sigma_x = sigma_x
        self.aggregation = aggregation
        self.quantile = quantile
        
        # Set variation function
        if variation_fn is None:
            self.variation_fn = gaussian_variation
        else:
            self.variation_fn = variation_fn
            
        # Set variation parameters
        if variation_params is None:
            self.variation_params = {}
        else:
            self.variation_params = variation_params
            
        # Set base loss function
        if base_loss == 'mse':
            self.base_loss_fn = nn.MSELoss(reduction='none')
        elif base_loss == 'huber':
            self.base_loss_fn = HuberLoss(delta=delta, reduction='none')
        elif base_loss == 'l1':
            self.base_loss_fn = L1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown base_loss: {base_loss}")
    
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate robust EIV loss.
        
        Args:
            x_obs: Observed features with noise [batch_size, n_features_x]
            y_true: Observed targets [batch_size, n_features_y]
            mask: Optional boolean mask [batch_size, n_features_y]
            
        Returns:
            Robust EIV loss tensor
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        
        batch_size = x_obs.shape[0]
        
        # Generate variations of input
        variation_params = {'sigma_x': self.sigma_x, 'n_samples': self.n_samples, **self.variation_params}
        variations = self.variation_fn(x_obs, **variation_params)
        
        # Calculate losses for each variation
        all_losses = []
        
        with torch.no_grad():  # No need for gradients during variation evaluation
            for x_var in variations:
                # Forward pass through model
                y_pred = self.model(x_var)
                
                # Apply mask to predictions if provided
                if mask is not None:
                    y_pred = self._apply_mask(y_pred, mask)
                
                # Calculate loss (element-wise, without reduction)
                loss = self.base_loss_fn(y_true, y_pred)
                
                # Average across feature dimensions, keeping batch dimension
                if loss.dim() > 1:
                    loss = torch.mean(loss, dim=list(range(1, loss.dim())))
                
                all_losses.append(loss)
        
        # Stack losses [n_samples, batch_size]
        stacked_losses = torch.stack(all_losses)
        
        # Aggregate losses based on specified method
        if self.aggregation == 'mean':
            aggregated_loss = torch.mean(stacked_losses, dim=0)
        elif self.aggregation == 'median':
            aggregated_loss = torch.median(stacked_losses, dim=0)[0]
        elif self.aggregation == 'min':
            aggregated_loss = torch.min(stacked_losses, dim=0)[0]
        elif self.aggregation == 'max':
            aggregated_loss = torch.max(stacked_losses, dim=0)[0]
        elif self.aggregation == 'quantile':
            # Sort losses along sample dimension
            sorted_losses, _ = torch.sort(stacked_losses, dim=0)
            
            # Calculate index for the quantile
            quantile_idx = min(
                int(self.n_samples * self.quantile),
                self.n_samples - 1
            )
            
            # Select the specified quantile
            aggregated_loss = sorted_losses[quantile_idx]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        # Apply final reduction
        if self.reduction == 'mean':
            return torch.mean(aggregated_loss)
        elif self.reduction == 'sum':
            return torch.sum(aggregated_loss)
        else:  # 'none'
            return aggregated_loss