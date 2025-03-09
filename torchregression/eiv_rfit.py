import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import MaskedLoss
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