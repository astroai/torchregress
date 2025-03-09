"""
Quantile regression loss functions.

Quantile regression provides a more complete view of the conditional distribution,
useful for estimating prediction intervals and handling heteroscedastic data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple, Dict

from .base import RegressionLoss
from ..utils.validation import validate_range

class QuantileLoss(RegressionLoss):
    """
    Quantile regression loss function.
    
    The quantile loss is asymmetric:
    - For points below the prediction: q * |y - f(x)|
    - For points above the prediction: (1-q) * |y - f(x)|
    
    where q is the quantile level (0 < q < 1).
    
    Args:
        quantile: Quantile level (0 < q < 1). Default: 0.5 (median)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, quantile: float = 0.5, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.quantile = validate_range(quantile, 0.0, 1.0, "quantile")
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate quantile loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Quantile loss
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Calculate asymmetric absolute error
        indicator = (residuals >= 0).float()
        loss = torch.abs(residuals) * (self.quantile * indicator + (1 - self.quantile) * (1 - indicator))
        
        # Apply sample weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class MultiQuantileLoss(RegressionLoss):
    """
    Loss for multiple quantile levels simultaneously.
    
    This loss is useful for models that predict multiple quantiles at once,
    such as when generating prediction intervals.
    
    Args:
        quantiles: List of quantile levels in ascending order
        joint_prediction: Whether predictions are passed as a joint tensor
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(
        self, 
        quantiles: Union[List[float], torch.Tensor],
        joint_prediction: bool = True,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        
        # Convert list to tensor if needed
        if isinstance(quantiles, list):
            quantiles = torch.tensor(quantiles, dtype=torch.float32)
        
        # Validate quantile levels
        self.register_buffer('quantiles', validate_range(quantiles, 0.0, 1.0, "quantiles"))
        self.num_quantiles = self.quantiles.size(0)
        self.joint_prediction = joint_prediction
    
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate combined quantile loss for multiple levels.
        
        Args:
            y_true: Ground truth values [batch_size, n_features]
            y_pred: When joint_prediction=True: [batch_size, num_quantiles, n_features]
                   or [batch_size, n_features * num_quantiles]
                   Otherwise: List of quantile predictions, each [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]
            
        Returns:
            Combined quantile loss value
        """
        batch_size = y_true.shape[0]
        n_features = y_true.shape[1] if y_true.dim() > 1 else 1
        device = y_true.device
        
        # Reshape y_true for broadcasting if it's a 1D tensor
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
            
        # Handle mask and weights
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(1)
        if weights is not None and weights.dim() == 1:
            weights = weights.unsqueeze(1)
            
        # Process predictions based on format
        if self.joint_prediction:
            # Handle joint predictions
            if isinstance(y_pred, torch.Tensor):
                # Process based on prediction shape
                if y_pred.dim() == 3 and y_pred.shape[1] == self.num_quantiles:
                    # [batch_size, num_quantiles, n_features] format
                    quantile_preds = y_pred
                elif y_pred.dim() == 2 and y_pred.shape[1] == n_features * self.num_quantiles:
                    # [batch_size, n_features * num_quantiles] format
                    # Reshape to [batch_size, num_quantiles, n_features]
                    quantile_preds = y_pred.reshape(batch_size, self.num_quantiles, n_features)
                else:
                    raise ValueError(
                        f"Expected y_pred shape to be either "
                        f"[batch_size, {self.num_quantiles}, {n_features}] or "
                        f"[batch_size, {n_features * self.num_quantiles}], "
                        f"got {y_pred.shape}"
                    )
            else:
                raise TypeError("With joint_prediction=True, y_pred must be a tensor")
        else:
            # Handle separate predictions (list of tensors)
            if isinstance(y_pred, (list, tuple)) and len(y_pred) == self.num_quantiles:
                # Stack predictions [batch_size, num_quantiles, n_features]
                quantile_preds = torch.stack(y_pred, dim=1)
            else:
                raise TypeError(
                    f"With joint_prediction=False, y_pred must be a list or tuple "
                    f"of {self.num_quantiles} tensors"
                )
        
        # Calculate loss for each quantile level
        losses = []
        for i, q in enumerate(self.quantiles):
            # Extract predictions for this quantile
            level_preds = quantile_preds[:, i]
            
            # Calculate residuals
            residuals = y_true - level_preds
            
            # Calculate asymmetric absolute error
            indicator = (residuals >= 0).float()
            level_loss = torch.abs(residuals) * (q * indicator + (1 - q) * (1 - indicator))
            
            # Apply mask if provided
            if mask is not None:
                level_loss = level_loss * mask
                
            # Apply sample weights if provided
            if weights is not None:
                level_loss = level_loss * weights
                
            # Reduce across features
            level_loss = torch.mean(level_loss, dim=1)
            losses.append(level_loss)
        
        # Stack losses for all quantile levels [batch_size, num_quantiles]
        stacked_losses = torch.stack(losses, dim=1)
        
        # Average across quantile levels for each sample
        combined_loss = torch.mean(stacked_losses, dim=1)
        
        # Apply final reduction
        if self.reduction == 'mean':
            return torch.mean(combined_loss)
        elif self.reduction == 'sum':
            return torch.sum(combined_loss)
        else:  # 'none'
            return combined_loss


class QuantileCrossover(RegressionLoss):
    """
    Loss that encourages proper crossover of quantile curves.
    
    In quantile regression, we expect lower quantiles to be below higher ones.
    This loss adds a penalty when this constraint is violated.
    
    Args:
        quantiles: List of quantile levels in ascending order
        base_loss: Base quantile loss coefficient
        crossover_penalty: Coefficient for crossover penalty term
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(
        self,
        quantiles: Union[List[float], torch.Tensor],
        base_loss: float = 1.0,
        crossover_penalty: float = 10.0,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        # Ensure quantiles are sorted in ascending order
        if isinstance(quantiles, list):
            quantiles = sorted(quantiles)
            quantiles_tensor = torch.tensor(quantiles, dtype=torch.float32)
        else:
            sorted_indices = torch.argsort(quantiles)
            quantiles_tensor = quantiles[sorted_indices]
            
        self.register_buffer('quantiles', quantiles_tensor)
        self.num_quantiles = len(quantiles)
        self.base_loss = base_loss
        self.crossover_penalty = crossover_penalty
        
        # Create individual quantile losses
        self.quantile_losses = nn.ModuleList([
            QuantileLoss(quantile=q, reduction='none')
            for q in quantiles
        ])
    
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate quantile loss with crossover penalty.
        
        Args:
            y_true: Ground truth values [batch_size, n_features]
            y_pred: Predicted quantiles [batch_size, num_quantiles, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features] or [batch_size]
            
        Returns:
            Loss combining standard quantile loss and crossover penalty
        """
        batch_size, n_features = y_true.shape[0], y_true.shape[-1]
        device = y_true.device
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        
        # Shape validation for y_pred
        if y_pred.shape[1] != self.num_quantiles:
            raise ValueError(
                f"Expected y_pred shape [batch_size, {self.num_quantiles}, n_features], "
                f"got shape {y_pred.shape}"
            )
            
        # Calculate standard quantile losses
        base_losses = []
        for i, loss_fn in enumerate(self.quantile_losses):
            level_preds = y_pred[:, i]
            level_loss = loss_fn(y_true, level_preds, mask, weights)
            base_losses.append(level_loss)
            
        stacked_base_losses = torch.stack(base_losses, dim=0)  # [num_quantiles, batch_size]
        
        # Calculate crossover penalties
        crossover_penalties = torch.zeros(batch_size, device=device)
        
        for i in range(self.num_quantiles - 1):
            # Lower quantile should be <= higher quantile
            lower_preds = y_pred[:, i]    # Lower quantile predictions
            higher_preds = y_pred[:, i+1] # Higher quantile predictions
            
            # Calculate violation: ReLU(lower - higher)
            violations = F.relu(lower_preds - higher_preds)
            
            # Apply mask if provided
            if mask is not None:
                violations = violations * mask
                
            # Sum violations across features
            sample_violations = torch.sum(violations, dim=-1)
            crossover_penalties += sample_violations
        
        # Final loss is weighted combination of base loss and crossover penalty
        total_base_loss = torch.mean(stacked_base_losses, dim=0)  # Mean across quantiles
        final_loss = self.base_loss * total_base_loss + self.crossover_penalty * crossover_penalties
        
        # Apply final reduction
        if self.reduction == 'mean':
            return torch.mean(final_loss)
        elif self.reduction == 'sum':
            return torch.sum(final_loss)
        else:  # 'none'
            return final_loss


class TiltedLoss(QuantileLoss):
    """
    Tilted absolute value loss (alias for QuantileLoss for legacy compatibility).
    
    Args:
        tau: Quantile level (0 < tau < 1). Default: 0.5 (median)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, tau: float = 0.5, reduction: str = 'mean'):
        super().__init__(quantile=tau, reduction=reduction)