"""
Chamfer distance-based Error-in-Variables (EIV) regression.

This module provides implementations of EIV regression losses based on
the Chamfer distance, particularly useful for point cloud matching and
when inputs have significant measurement error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, List, Tuple, Callable

from ..base import MaskedLoss
from ...utils.tensor_ops import prepare_param, prepare_sigma


def chamfer_distance(x: torch.Tensor, y: torch.Tensor, bidirectional: bool = True) -> torch.Tensor:
    """
    Calculate Chamfer distance between point clouds.
    
    Args:
        x: First point cloud [batch_size, n_points_x, n_features]
        y: Second point cloud [batch_size, n_points_y, n_features]
        bidirectional: Whether to compute distance in both directions
        
    Returns:
        Chamfer distance [batch_size]
    """
    # Calculate pairwise distances
    # For each point in x, find the closest point in y
    batch_size, n_points_x, n_features = x.shape
    n_points_y = y.shape[1]
    
    # Reshape for broadcasting
    x_expanded = x.unsqueeze(2)  # [batch_size, n_points_x, 1, n_features]
    y_expanded = y.unsqueeze(1)  # [batch_size, 1, n_points_y, n_features]
    
    # Calculate squared distances
    squared_dist = torch.sum((x_expanded - y_expanded)**2, dim=3)  # [batch_size, n_points_x, n_points_y]
    
    # Find minimum distance for each point in x to any point in y
    x_to_y_min, _ = torch.min(squared_dist, dim=2)  # [batch_size, n_points_x]
    x_to_y = torch.mean(x_to_y_min, dim=1)  # [batch_size]
    
    if not bidirectional:
        return x_to_y
    
    # Find minimum distance for each point in y to any point in x
    y_to_x_min, _ = torch.min(squared_dist, dim=1)  # [batch_size, n_points_y]
    y_to_x = torch.mean(y_to_x_min, dim=1)  # [batch_size]
    
    # Combine for bidirectional Chamfer distance
    return 0.5 * (x_to_y + y_to_x)


class ChamferEIVLoss(MaskedLoss):
    """
    Error-in-Variables loss using Chamfer distance.
    
    This loss is used when seeking to match a model's manifold rather than just
    point-wise predictions, making it robust to input uncertainty.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of noise in the features (for sampling)
        method: Method for finding closest point ('monte_carlo', 'optimization')
        n_samples: Number of Monte Carlo samples (for 'monte_carlo' method)
        optim_steps: Number of optimization steps (for 'optimization' method)
        optim_lr: Learning rate for optimization (for 'optimization' method)
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(
        self,
        model: Callable,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        method: str = 'monte_carlo',
        n_samples: int = 100,
        optim_steps: int = 50,
        optim_lr: float = 0.01,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        self.model = model
        self.sigma_x = sigma_x
        self.method = method
        self.n_samples = n_samples
        self.optim_steps = optim_steps
        self.optim_lr = optim_lr
        
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate Chamfer EIV loss.
        
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
        
        # Handle different methods for finding closest point on the manifold
        if self.method == 'monte_carlo':
            # Generate samples around observed x to approximate the manifold
            if self.sigma_x is None:
                # Default noise level if not specified
                sigma_x = torch.ones(x_obs.shape[1], device=device) * 0.1
            else:
                sigma_x = prepare_sigma(self.sigma_x, x_obs.shape[1], device)
                
            # Generate samples
            samples = []
            for _ in range(self.n_samples):
                noise = torch.randn_like(x_obs) * sigma_x
                x_sample = x_obs + noise
                samples.append(x_sample)
            
            # Stack samples [n_samples, batch_size, n_features_x]
            x_samples = torch.stack(samples)
            
            # Reshape for batch processing
            x_flat = x_samples.reshape(-1, x_samples.shape[-1])
            
            # Get predictions for all samples
            with torch.no_grad():
                y_preds_flat = self.model(x_flat)
                
                # Reshape back to [n_samples, batch_size, n_features_y]
                y_preds = y_preds_flat.reshape(self.n_samples, batch_size, n_features_y)
                
                # Transpose to [batch_size, n_samples, n_features_y] for Chamfer distance
                y_preds = y_preds.transpose(0, 1)
                
            # Reshape y_true to [batch_size, 1, n_features_y] for Chamfer distance
            y_true_expanded = y_true.unsqueeze(1)
            
            # Calculate Chamfer distance from y_true to manifold samples
            chamfer_loss = chamfer_distance(y_true_expanded, y_preds, bidirectional=True)
            
        elif self.method == 'optimization':
            # Find closest point on the manifold through optimization
            chamfer_loss = self._optimization_method(x_obs, y_true)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(chamfer_loss)
        elif self.reduction == 'sum':
            return torch.sum(chamfer_loss)
        else:  # 'none'
            return chamfer_loss
    
    def _optimization_method(self, x_obs, y_true):
        """Find closest point on the model manifold through optimization."""
        batch_size, n_features_y = y_true.shape
        device = x_obs.device
        
        # Create optimizable inputs initialized to observed values
        x_opt = x_obs.detach().clone().requires_grad_(True)
        
        # Create optimizer
        optimizer = torch.optim.Adam([x_opt], lr=self.optim_lr)
        
        # Run optimization to find closest point on the manifold
        for _ in range(self.optim_steps):
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.model(x_opt)
            
            # L2 distance between prediction and target
            loss = torch.sum((y_pred - y_true)**2, dim=1).mean()
            
            # Add regularization to stay close to original x
            if self.sigma_x is not None:
                sigma_x = prepare_sigma(self.sigma_x, x_obs.shape[1], device)
                reg_term = torch.sum(((x_opt - x_obs) / sigma_x)**2, dim=1).mean()
                loss = loss + 0.5 * reg_term
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
        
        # Final forward pass
        with torch.no_grad():
            y_pred = self.model(x_opt)
            
            # Calculate squared distance to target
            chamfer_loss = torch.sum((y_pred - y_true)**2, dim=1)
        
        return chamfer_loss


class HybridEIVChamferLoss(MaskedLoss):
    """
    Hybrid loss combining standard EIV and Chamfer distance.
    
    This loss combines the analytical EIV loss with Chamfer distance for
    better handling of complex error structures.
    
    Args:
        eiv_loss: Standard EIV loss instance
        chamfer_loss: Chamfer EIV loss instance
        alpha: Weight for EIV component (0-1)
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(
        self,
        eiv_loss: MaskedLoss,
        chamfer_loss: ChamferEIVLoss,
        alpha: float = 0.5,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        self.eiv_loss = eiv_loss
        self.chamfer_loss = chamfer_loss
        self.alpha = alpha
        
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate hybrid EIV-Chamfer loss.
        
        Args:
            x_obs: Observed features with noise [batch_size, n_features_x]
            y_true: Observed targets [batch_size, n_features_y]
            mask: Optional boolean mask [batch_size, n_features_y]
            
        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        # Ensure both losses use 'none' reduction for proper weighting
        original_eiv_reduction = self.eiv_loss.reduction
        original_chamfer_reduction = self.chamfer_loss.reduction
        
        self.eiv_loss.reduction = 'none'
        self.chamfer_loss.reduction = 'none'
        
        # Calculate both losses
        eiv_loss_val = self.eiv_loss(x_obs, y_true, mask)
        chamfer_loss_val = self.chamfer_loss(x_obs, y_true, mask)
        
        # Restore original reduction settings
        self.eiv_loss.reduction = original_eiv_reduction
        self.chamfer_loss.reduction = original_chamfer_reduction
        
        # Combine losses with alpha weighting
        combined_loss = self.alpha * eiv_loss_val + (1.0 - self.alpha) * chamfer_loss_val
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(combined_loss)
        elif self.reduction == 'sum':
            return torch.sum(combined_loss)
        else:  # 'none'
            return combined_loss
