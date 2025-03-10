"""
Quantile Loss functions for Error-in-Variables regression.

This module provides implementations of Quantile regression losses that
account for measurement error in input features.
"""
import torch
from typing import Callable, Union

from .eiv_standard import BaseEIVLoss
from .eiv_utils import prepare_covariance, generate_perturbed_samples

class QuantileEIVLoss(BaseEIVLoss):
    """
    Quantile Error-in-Variables Loss.
    
    This loss combines quantile regression with input uncertainty handling
    by generating multiple perturbed versions of inputs, running the model 
    on each, averaging predictions, and applying the quantile (pinball) loss.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise (scalar, vector, or matrix)
        quantile: Quantile level to estimate (0 to 1, default: 0.5 for median)
        n_samples: Number of perturbed samples to generate
        perturb_method: Method for perturbing inputs ('gaussian', 'uniform')
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability
    """
    def __init__(
        self, 
        model: Callable, 
        sigma_x: Union[float, torch.Tensor],
        quantile: float = 0.5,
        n_samples: int = 20,
        perturb_method: str = 'gaussian',
        reduction: str = 'mean',
        eps: float = 1e-8
    ):
        super().__init__(model, sigma_x, None, reduction, eps)
        
        # Validate quantile value
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantile must be between 0 and 1, got {quantile}")
            
        self.quantile = quantile
        self.n_samples = n_samples
        self.perturb_method = perturb_method
        
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate Quantile EIV loss.
        
        Args:
            x_obs: Observed features with noise [batch_size, n_features_x]
            y_true: Observed targets [batch_size, n_features_y]
            mask: Optional boolean mask [batch_size, n_features_y]
            
        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        # Apply mask to targets if provided
        y_true = self._apply_mask(y_true, mask)
        
        batch_size, n_features_x = x_obs.shape
        device = x_obs.device
        
        # Prepare noise parameters
        sigma_x_tensor = prepare_covariance(self.sigma_x, n_features_x, device)
        
        # Generate perturbed samples
        perturbed_samples = generate_perturbed_samples(
            x_obs, sigma_x_tensor, self.n_samples, perturb_method=self.perturb_method
        )
            
        # Stack perturbed samples and reshape for batch processing
        x_perturbed = torch.stack(perturbed_samples)  # [n_samples, batch_size, n_features_x]
        x_flat = x_perturbed.reshape(-1, n_features_x)  # [n_samples * batch_size, n_features_x]
        
        # Forward pass for all samples
        with torch.no_grad():
            y_preds_flat = self.model(x_flat)
            
            # Get output feature dimension
            n_features_y = y_preds_flat.shape[1] if y_preds_flat.dim() > 1 else 1
            
            # Reshape predictions 
            if y_preds_flat.dim() == 1:
                # Handle scalar output case
                y_preds = y_preds_flat.reshape(self.n_samples, batch_size, 1)
            else:
                y_preds = y_preds_flat.reshape(self.n_samples, batch_size, n_features_y)
                
            # Apply mask if provided
            if mask is not None:
                mask_expanded = mask.unsqueeze(0).expand(self.n_samples, -1, -1)
                y_preds = torch.where(mask_expanded, y_preds, torch.zeros_like(y_preds))
        
        # Average predictions across samples
        mean_pred = torch.mean(y_preds, dim=0)  # [batch_size, n_features_y]
        
        # Calculate quantile (pinball) loss
        diff = y_true - mean_pred
        loss = torch.max(self.quantile * diff, (self.quantile - 1.0) * diff)
        
        # Sum across features if multi-dimensional output
        loss = torch.sum(loss, dim=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss


class MultiQuantileEIVLoss(BaseEIVLoss):
    """
    Multi-Quantile Error-in-Variables Loss.
    
    This loss estimates multiple quantiles simultaneously, which is useful
    for prediction intervals. It handles input uncertainty through data
    augmentation and ensemble averaging.
    
    Args:
        model: Model function that outputs multiple quantiles
        sigma_x: Standard deviation of feature noise
        quantiles: List of quantile levels to estimate (each between 0 and 1)
        n_samples: Number of perturbed samples
        perturb_method: Method for perturbing inputs ('gaussian', 'uniform')
        reduction: 'none' | 'mean' | 'sum'
        eps: Small value for numerical stability
    """
    def __init__(
        self, 
        model: Callable, 
        sigma_x: Union[float, torch.Tensor],
        quantiles: list = [0.1, 0.5, 0.9],
        n_samples: int = 20,
        perturb_method: str = 'gaussian',
        reduction: str = 'mean',
        eps: float = 1e-8
    ):
        super().__init__(model, sigma_x, None, reduction, eps)
        self.quantiles = torch.tensor(quantiles)
        self.n_quantiles = len(quantiles)
        self.n_samples = n_samples
        self.perturb_method = perturb_method
        
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate Multi-Quantile EIV loss.
        
        Args:
            x_obs: Observed features with noise [batch_size, n_features_x]
            y_true: Observed targets [batch_size, n_features_y]
            mask: Optional boolean mask [batch_size, n_features_y]
            
        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        # Apply mask to targets if provided
        y_true = self._apply_mask(y_true, mask)
        
        batch_size, n_features_x = x_obs.shape
        device = x_obs.device
        y_true = y_true.to(device)
        self.quantiles = self.quantiles.to(device)
        
        # Prepare noise parameters
        sigma_x_tensor = prepare_covariance(self.sigma_x, n_features_x, device)
        
        # Generate perturbed samples
        perturbed_samples = generate_perturbed_samples(
            x_obs, sigma_x_tensor, self.n_samples, perturb_method=self.perturb_method
        )
            
        # Stack perturbed samples and reshape for batch processing
        x_perturbed = torch.stack(perturbed_samples)  # [n_samples, batch_size, n_features_x]
        x_flat = x_perturbed.reshape(-1, n_features_x)  # [n_samples * batch_size, n_features_x]
        
        # Forward pass for all samples
        with torch.no_grad():
            y_preds_flat = self.model(x_flat)  # Should output [n_samples * batch_size, n_features_y, n_quantiles]
            
            # Reshape predictions
            n_features_y = y_true.shape[1]
            y_preds = y_preds_flat.reshape(self.n_samples, batch_size, n_features_y, self.n_quantiles)
            
            # Apply mask if provided
            if mask is not None:
                mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand(
                    self.n_samples, -1, -1, self.n_quantiles)
                y_preds = torch.where(mask_expanded, y_preds, torch.zeros_like(y_preds))
        
        # Average predictions across samples (ensemble averaging)
        mean_preds = torch.mean(y_preds, dim=0)  # [batch_size, n_features_y, n_quantiles]
        
        # Calculate quantile loss for each quantile
        # Expand y_true for broadcasting with quantile dimension
        y_expanded = y_true.unsqueeze(-1).expand(-1, -1, self.n_quantiles)  # [batch_size, n_features_y, n_quantiles]
        diff = y_expanded - mean_preds  # [batch_size, n_features_y, n_quantiles]
        
        # Calculate pinball loss for each quantile
        q_expanded = self.quantiles.view(1, 1, -1).expand(batch_size, n_features_y, -1)
        pinball_loss = torch.max(q_expanded * diff, (q_expanded - 1.0) * diff)
        
        # Sum across features and quantiles
        loss = torch.sum(pinball_loss, dim=(1, 2))
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss