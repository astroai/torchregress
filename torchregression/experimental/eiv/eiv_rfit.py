import torch
import torch.nn as nn
from typing import Callable, Literal, Dict, Union, List, Optional
from ..base import MaskedLoss
from ..robust import HuberLoss, L1Loss
from .eiv_utils import prepare_sigma, generate_perturbed_samples

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
        batch_size (int): Batch size for processing variations. Default: None (process all at once).
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
        batch_size: Optional[int] = None,
        variation_params: Dict = None,
        aggregation: Literal['mean', 'median', 'max', 'quantile'] = 'median',
        quantile: float = 0.95,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
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
            # Default to Gaussian variation if none provided
            variation_fn = gaussian_variation
            
        self.variation_fn = variation_fn
        
        # Store parameters
        self.sigma_x = sigma_x
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.variation_params = variation_params if variation_params is not None else {}
        
        # Validate aggregation method
        self.aggregation = aggregation.lower()
        if self.aggregation not in ('mean', 'median', 'max', 'quantile'):
            raise ValueError("aggregation must be 'mean', 'median', 'max', or 'quantile'")
        
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
        sigma_x = prepare_sigma(self.sigma_x, x_obs.shape[1], device)
        
        # Generate input variations using the provided function
        varied_inputs = self.variation_fn(
            x_obs, 
            sigma_x=sigma_x, 
            n_samples=self.n_samples,
            **self.variation_params
        )
        
        # Process in batches if batch_size is specified
        if self.batch_size is not None and self.batch_size < self.n_samples:
            all_losses = self._batch_process_variations(varied_inputs, y_true, mask)
        else:
            # Process all variations at once
            all_losses = self._process_variations(varied_inputs, y_true, mask)
        
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

    def _process_variations(self, varied_inputs, y_true, mask=None):
        """Process all variations and calculate losses."""
        all_losses = []
        
        # Multiple forward passes with varied inputs
        for x_varied in varied_inputs:
            # Get predictions for this variation
            try:
                with torch.set_grad_enabled(True):
                    y_pred = self.model(x_varied)
                    
                # Apply mask to predictions if needed
                if mask is not None:
                    y_pred = self._apply_mask(y_pred, mask)
                
                # Calculate loss for this variation
                batch_loss = self.base_loss_fn(y_true, y_pred)
                
                # Reduce across features if needed
                if batch_loss.dim() > 1:
                    batch_loss = torch.mean(batch_loss, dim=tuple(range(1, batch_loss.dim())))
                    
                all_losses.append(batch_loss)
            except Exception as e:
                # Handle errors during forward pass
                print(f"Warning: Error during variation processing: {e}")
                # Skip this variation
                continue
                
        # Stack all losses [n_variations, batch_size]
        if not all_losses:
            raise RuntimeError("All variations failed to process")
        
        return torch.stack(all_losses)

    def _batch_process_variations(self, varied_inputs, y_true, mask=None):
        """Process variations in smaller batches to save memory."""
        all_losses = []
        n_variations = len(varied_inputs)
        
        # Process variations in batches
        for start_idx in range(0, n_variations, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_variations)
            batch_inputs = varied_inputs[start_idx:end_idx]
            
            # Process this batch of variations
            batch_losses = self._process_variations(batch_inputs, y_true, mask)
            all_losses.append(batch_losses)
        
        # Concatenate results from all batches
        return torch.cat(all_losses, dim=0)
            
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
            noise = torch.randn(batch_size, n_features, device=device) * sigma_x * scale_factor
            
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
            noise = torch.rand(batch_size, n_features, device=device) * 2 * half_range - half_range
            
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


def structured_variation(x_batch, sigma_x, n_samples=10, correlation=0.5, **kwargs):
    """
    Generate variations with correlated feature noise.
    
    Args:
        x_batch: Input tensor [batch_size, n_features]
        sigma_x: Standard deviation of noise
        n_samples: Number of variations to generate
        correlation: Correlation coefficient between adjacent features (0-1)
        
    Returns:
        List of varied input tensors
    """
    batch_size, n_features = x_batch.shape
    device = x_batch.device
    
    # Convert sigma_x to tensor if not already
    if isinstance(sigma_x, (int, float)):
        sigma_x = torch.tensor(sigma_x, device=device).expand(n_features)
    
    # Create correlation matrix
    corr_matrix = torch.eye(n_features, device=device)
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                # Correlation decreases with feature distance
                dist = abs(i - j)
                corr_matrix[i, j] = correlation ** dist
    
    # Create covariance matrix from correlation matrix and sigmas
    sigma_matrix = torch.diag(sigma_x)
    cov_matrix = sigma_matrix @ corr_matrix @ sigma_matrix
    
    # Add small value to diagonal for numerical stability
    cov_matrix = cov_matrix + torch.eye(n_features, device=device) * 1e-6
    
    # Generate correlated variations
    variations = []
    for _ in range(n_samples):
        # Generate multivariate normal samples
        mvn = torch.distributions.MultivariateNormal(
            loc=torch.zeros(n_features, device=device),
            covariance_matrix=cov_matrix
        )
        noise = mvn.sample((batch_size,))
        
        # Add correlated noise
        x_varied = x_batch + noise
        variations.append(x_varied)
    
    return variations


def adversarial_variation(x_batch, sigma_x, n_samples=10, model=None, loss_fn=None, 
                         step_size=0.01, norm_constraint=2.0, **kwargs):
    """
    Generate adversarial variations by finding inputs that maximize the loss.
    
    Args:
        x_batch: Input tensor [batch_size, n_features]
        sigma_x: Standard deviation parameter (as scale for perturbations)
        n_samples: Number of variations to generate
        model: Model to generate adversarial examples against
        loss_fn: Loss function to maximize
        step_size: Step size for gradient ascent
        norm_constraint: Maximum L2 norm of perturbations
        
    Returns:
        List of varied input tensors
    """
    if model is None or loss_fn is None:
        # Fall back to Gaussian if no model or loss function provided
        return gaussian_variation(x_batch, sigma_x, n_samples)
    
    batch_size, n_features = x_batch.shape
    device = x_batch.device
    
    # Scale factor for noise magnitude
    if isinstance(sigma_x, (int, float)):
        scale = sigma_x
    else:
        scale = torch.mean(sigma_x).item()
    
    variations = []
    
    # Get predictions for original inputs
    with torch.no_grad():
        y_pred = model(x_batch)
    
    # Generate adversarial variations
    for _ in range(n_samples):
        # Start with small random perturbation
        delta = torch.randn_like(x_batch) * scale * 0.1
        delta.requires_grad_(True)
        
        # Perform gradient ascent to maximize loss
        for _ in range(3):  # Only a few steps needed
            # Forward pass
            x_adv = x_batch + delta
            pred_adv = model(x_adv)
            
            # Maximize loss (gradient ascent)
            loss = -loss_fn(pred_adv, y_pred)  # Negative for maximization
            loss.backward()
            
            # Update delta with normalized gradient
            with torch.no_grad():
                grad_norm = torch.norm(delta.grad, dim=1, keepdim=True)
                delta.grad /= torch.clamp(grad_norm, min=1e-8)
                delta += step_size * delta.grad
                
                # Project back to norm constraint
                delta_norm = torch.norm(delta, dim=1, keepdim=True)
                delta *= torch.min(
                    torch.ones_like(delta_norm),
                    norm_constraint * scale / torch.clamp(delta_norm, min=1e-8)
                )
                
                delta.grad.zero_()
        
        # Create adversarial variation
        x_varied = (x_batch + delta).detach()
        variations.append(x_varied)
    
    return variations
