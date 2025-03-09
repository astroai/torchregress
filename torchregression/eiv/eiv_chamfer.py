"""
Chamfer distance-based Error-in-Variables loss implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union, Dict, List

from ..losses.base import MaskedLoss
from ..utils.tensor_ops import prepare_sigma

class ChamferEIVLoss(MaskedLoss):
    """
    Chamfer distance-based Error-in-Variables loss.
    
    This approach finds the closest point on the model manifold to each observation,
    which can provide unbiased estimates especially for highly nonlinear models.
    
    Args:
        model: The model function f(x) that predicts y
        method: Method for finding closest point ('monte_carlo', 'optimization')
        n_samples: Number of Monte Carlo samples (for 'monte_carlo' method)
        optim_steps: Number of optimization steps (for 'optimization' method)
        optim_lr: Learning rate for optimization (for 'optimization' method)
        sigma_x: Standard deviation of feature noise (for sampling)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(
        self,
        model: Callable,
        method: str = 'monte_carlo',
        n_samples: int = 100,
        optim_steps: int = 50,
        optim_lr: float = 0.01,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        self.model = model
        
        if method not in ['monte_carlo', 'optimization']:
            raise ValueError("method must be 'monte_carlo' or 'optimization'")
        
        self.method = method
        self.n_samples = n_samples
        self.optim_steps = optim_steps
        self.optim_lr = optim_lr
        self.sigma_x = sigma_x
    
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate the Chamfer EIV loss.
        
        Args:
            x_obs: Observed features with noise. Shape: [batch_size, n_features_x]
            y_true: Observed targets. Shape: [batch_size, n_features_y]
            mask: Optional mask. Shape: [batch_size, n_features_y]
            
        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        
        # Choose method for finding closest points
        if self.method == 'monte_carlo':
            loss = self._monte_carlo_chamfer(x_obs, y_true, mask)
        else:  # optimization
            loss = self._optimization_chamfer(x_obs, y_true, mask)
            
        return loss
    
    def _monte_carlo_chamfer(self, x_obs, y_true, mask=None):
        """
        Monte Carlo method for finding the closest point on the model manifold.
        
        Generate samples around each observed x_obs, find the closest predicted y to y_true.
        """
        batch_size, n_features_x = x_obs.shape
        n_features_y = y_true.shape[1]
        device = x_obs.device
        
        # Prepare sigma parameter
        sigma_x = prepare_sigma(self.sigma_x, n_features_x, device)
        if sigma_x is None or sigma_x == 0:
            sigma_x = 0.1  # Default value if not provided
            
        # Generate samples around each observed x_obs
        all_x_samples = []
        
        for _ in range(self.n_samples):
            # Generate noise
            if isinstance(sigma_x, torch.Tensor) and sigma_x.ndim > 0:
                # Vector or matrix case
                if sigma_x.ndim == 1:
                    # Vector case
                    noise = torch.randn(batch_size, n_features_x, device=device) * sigma_x.unsqueeze(0)
                else:
                    # Matrix case - generate multivariate normal
                    mvn = torch.distributions.MultivariateNormal(
                        loc=torch.zeros(n_features_x, device=device),
                        covariance_matrix=sigma_x
                    )
                    noise = mvn.sample((batch_size,))
            else:
                # Scalar case
                noise = torch.randn(batch_size, n_features_x, device=device) * sigma_x
                
            # Create sample
            x_sample = x_obs + noise
            all_x_samples.append(x_sample)
            
        # Concatenate all samples
        x_samples = torch.cat(all_x_samples, dim=0)  # [batch_size * n_samples, n_features_x]
        
        # Forward pass through the model for all samples
        with torch.no_grad():
            y_samples = self.model(x_samples)  # [batch_size * n_samples, n_features_y]
            
        # Reshape to [batch_size, n_samples, n_features_y]
        y_samples = y_samples.view(self.n_samples, batch_size, n_features_y)
        y_samples = y_samples.transpose(0, 1)  # [batch_size, n_samples, n_features_y]
        
        # Calculate distance from each sample to the true y
        y_true_expanded = y_true.unsqueeze(1).expand(-1, self.n_samples, -1)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand(-1, self.n_samples, -1)
            squared_distances = torch.where(
                mask_expanded,
                (y_samples - y_true_expanded) ** 2,
                torch.zeros_like(y_samples)
            )
            
            # Calculate valid elements per batch for normalization
            if self.reduction == 'mean':
                valid_elements = mask_expanded.sum(dim=(1, 2), keepdim=True)
                # Add epsilon to prevent division by zero
                valid_elements = torch.clamp(valid_elements, min=1.0)
            
        else:
            squared_distances = (y_samples - y_true_expanded) ** 2
            
        # Sum over feature dimension
        distances = torch.sum(squared_distances, dim=2)  # [batch_size, n_samples]
        
        # Find the minimum distance for each batch element
        min_distances, _ = torch.min(distances, dim=1)  # [batch_size]
        
        # Apply reduction
        if self.reduction == 'mean':
            if mask is not None:
                return torch.sum(min_distances) / valid_elements.sum()
            else:
                return torch.mean(min_distances)
        elif self.reduction == 'sum':
            return torch.sum(min_distances)
        else:  # 'none'
            return min_distances
    
    def _optimization_chamfer(self, x_obs, y_true, mask=None):
        """
        Optimization method for finding the closest point on the model manifold.
        
        Start from x_obs and optimize to minimize distance to y_true.
        """
        batch_size = x_obs.shape[0]
        device = x_obs.device
        
        # Create parameter to optimize, starting from the observed x
        x_opt = nn.Parameter(x_obs.clone())
        
        # Create optimizer
        optimizer = torch.optim.Adam([x_opt], lr=self.optim_lr)
        
        # Optimization loop
        for _ in range(self.optim_steps):
            optimizer.zero_grad()
            
            # Forward pass through the model
            y_pred = self.model(x_opt)
            
            # Apply mask if provided
            if mask is not None:
                y_pred = self._apply_mask(y_pred, mask)
                y_true_masked = self._apply_mask(y_true, mask)
                # Calculate MSE loss
                optim_loss = F.mse_loss(y_pred, y_true_masked, reduction='sum') / mask.sum()
            else:
                # Calculate MSE loss
                optim_loss = F.mse_loss(y_pred, y_true, reduction='mean')
                
            # Add regularization to stay close to the original x
            # This helps prevent the optimization from going too far from the observation
            reg_lambda = 0.01  # Low weight to avoid biasing the solution
            optim_loss = optim_loss + reg_lambda * F.mse_loss(x_opt, x_obs, reduction='mean')
            
            # Backward and optimize
            optim_loss.backward()
            optimizer.step()
        
        # Final forward pass with optimized x
        with torch.no_grad():
            y_opt = self.model(x_opt)
            
            # Apply mask if provided
            if mask is not None:
                y_opt = self._apply_mask(y_opt, mask)
                y_true_masked = self._apply_mask(y_true, mask)
                # Calculate squared distances
                squared_dist = (y_opt - y_true_masked) ** 2
                # Get valid count for each batch element
                valid_count = mask.sum(dim=1, keepdim=True)
                valid_count = torch.clamp(valid_count, min=1.0)
                # Sum over features and divide by valid count
                distances = torch.sum(squared_dist, dim=1) / valid_count.squeeze()
            else:
                # Calculate squared distances
                squared_dist = (y_opt - y_true) ** 2
                # Sum over features
                distances = torch.sum(squared_dist, dim=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(distances)
        elif self.reduction == 'sum':
            return torch.sum(distances)
        else:  # 'none'
            return distances


class HybridEIVChamferLoss(MaskedLoss):
    """
    Hybrid EIV-Chamfer loss that combines a standard EIV loss with a Chamfer loss.
    
    The combination allows balancing between fast but approximate EIV methods and
    more accurate but slower Chamfer-based methods.
    
    Args:
        eiv_loss: An instance of a standard EIV loss
        chamfer_loss: An instance of ChamferEIVLoss
        alpha: Weight for the EIV component (0-1)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(
        self,
        eiv_loss: MaskedLoss,
        chamfer_loss: ChamferEIVLoss,
        alpha: float = 0.5,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
            
        self.eiv_loss = eiv_loss
        self.chamfer_loss = chamfer_loss
        self.alpha = alpha
    
    def forward(self, x_obs, y_true, mask=None, **kwargs):
        """
        Calculate the hybrid loss.
        
        Args:
            x_obs: Observed features with noise. Shape: [batch_size, n_features_x]
            y_true: Observed targets. Shape: [batch_size, n_features_y]
            mask: Optional mask. Shape: [batch_size, n_features_y]
            **kwargs: Additional arguments for component losses
            
        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        # Set both component losses to use 'none' reduction
        original_eiv_reduction = self.eiv_loss.reduction
        original_chamfer_reduction = self.chamfer_loss.reduction
        
        self.eiv_loss.reduction = 'none'
        self.chamfer_loss.reduction = 'none'
        
        # Calculate component losses
        eiv_losses = self.eiv_loss(x_obs, y_true, mask=mask, **kwargs)
        chamfer_losses = self.chamfer_loss(x_obs, y_true, mask=mask)
        
        # Restore original reductions
        self.eiv_loss.reduction = original_eiv_reduction
        self.chamfer_loss.reduction = original_chamfer_reduction
        
        # Combine losses
        combined_losses = self.alpha * eiv_losses + (1 - self.alpha) * chamfer_losses
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(combined_losses)
        elif self.reduction == 'sum':
            return torch.sum(combined_losses)
        else:  # 'none'
            return combined_losses
