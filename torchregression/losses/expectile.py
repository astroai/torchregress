"""
Expectile regression loss functions.

Expectile regression provides a richer description of the conditional distribution
than standard mean regression, similar to quantile regression.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple

from .base import RegressionLoss
from ..utils.validation import validate_range

class ExpectileLoss(RegressionLoss):
    """
    Expectile regression loss function.
    
    Expectiles are defined via asymmetric least squares that generalize
    the mean in a similar way as quantiles generalize the median.
    
    Args:
        expectile: Expectile level (0 < τ < 1). Default: 0.5 (mean)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, expectile: float = 0.5, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.expectile = validate_range(expectile, 0.0, 1.0, "expectile")
        
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, mask: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate expectile loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Expectile loss
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Calculate asymmetric squared error
        indicator = (residuals >= 0).float()
        loss = residuals**2 * (self.expectile * indicator + (1 - self.expectile) * (1 - indicator))
        
        # Apply sample weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class MultiExpectileLoss(RegressionLoss):
    """
    Loss for multiple expectile levels simultaneously.
    
    This loss is useful for models that predict multiple expectiles at once,
    providing a more complete description of the conditional distribution.
    
    Args:
        expectiles: List of expectile levels in ascending order
        joint_prediction: Whether predictions are passed as a joint tensor
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(
        self, 
        expectiles: Union[List[float], torch.Tensor],
        joint_prediction: bool = True,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        
        # Convert list to tensor if needed
        if isinstance(expectiles, list):
            expectiles = torch.tensor(expectiles, dtype=torch.float32)
        
        # Validate expectile levels
        self.register_buffer('expectiles', validate_range(expectiles, 0.0, 1.0, "expectiles"))
        self.num_expectiles = self.expectiles.size(0)
        self.joint_prediction = joint_prediction
    
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate combined expectile loss for multiple levels.
        
        Args:
            y_true: Ground truth values [batch_size, n_features]
            y_pred: When joint_prediction=True: [batch_size, num_expectiles, n_features]
                   or [batch_size, n_features * num_expectiles]
                   Otherwise: List of expectile predictions, each [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]
            
        Returns:
            Combined expectile loss value
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
                if y_pred.dim() == 3 and y_pred.shape[1] == self.num_expectiles:
                    # [batch_size, num_expectiles, n_features] format
                    expectile_preds = y_pred
                elif y_pred.dim() == 2 and y_pred.shape[1] == n_features * self.num_expectiles:
                    # [batch_size, n_features * num_expectiles] format
                    # Reshape to [batch_size, num_expectiles, n_features]
                    expectile_preds = y_pred.reshape(batch_size, self.num_expectiles, n_features)
                else:
                    raise ValueError(
                        f"Expected y_pred shape to be either "
                        f"[batch_size, {self.num_expectiles}, {n_features}] or "
                        f"[batch_size, {n_features * self.num_expectiles}], "
                        f"got {y_pred.shape}"
                    )
            else:
                raise TypeError("With joint_prediction=True, y_pred must be a tensor")
        else:
            # Handle separate predictions (list of tensors)
            if isinstance(y_pred, (list, tuple)) and len(y_pred) == self.num_expectiles:
                # Stack predictions [batch_size, num_expectiles, n_features]
                expectile_preds = torch.stack(y_pred, dim=1)
            else:
                raise TypeError(
                    f"With joint_prediction=False, y_pred must be a list or tuple "
                    f"of {self.num_expectiles} tensors"
                )
        
        # Calculate loss for each expectile level
        losses = []
        for i, expectile in enumerate(self.expectiles):
            # Extract predictions for this expectile
            level_preds = expectile_preds[:, i]
            
            # Calculate residuals
            residuals = y_true - level_preds
            
            # Calculate asymmetric squared error
            indicator = (residuals >= 0).float()
            level_loss = residuals**2 * (expectile * indicator + (1 - expectile) * (1 - indicator))
            
            # Apply mask if provided
            if mask is not None:
                level_loss = level_loss * mask
                
            # Apply sample weights if provided
            if weights is not None:
                level_loss = level_loss * weights
                
            # Reduce across features
            level_loss = torch.mean(level_loss, dim=1)
            losses.append(level_loss)
        
        # Stack losses for all expectile levels [batch_size, num_expectiles]
        stacked_losses = torch.stack(losses, dim=1)
        
        # Average across expectile levels for each sample
        combined_loss = torch.mean(stacked_losses, dim=1)
        
        # Apply final reduction
        if self.reduction == 'mean':
            return torch.mean(combined_loss)
        elif self.reduction == 'sum':
            return torch.sum(combined_loss)
        else:  # 'none'
            return combined_loss


class AsymmetricLeastSquaresLoss(ExpectileLoss):
    """
    Asymmetric least squares loss (alias for ExpectileLoss for legacy compatibility).
    
    Args:
        tau: Expectile level (0 < tau < 1). Default: 0.5 (mean)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, tau: float = 0.5, reduction: str = 'mean'):
        super().__init__(expectile=tau, reduction=reduction)


class ExpectileCrossover(RegressionLoss):
    """
    Loss that ensures proper ordering of expectile curves.
    
    In expectile regression, we expect lower expectiles to be below higher ones.
    This loss adds a penalty when this constraint is violated.
    
    Args:
        expectiles: List of expectile levels in ascending order
        base_loss: Base expectile loss coefficient
        crossover_penalty: Coefficient for crossover penalty term
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(
        self,
        expectiles: Union[List[float], torch.Tensor],
        base_loss: float = 1.0,
        crossover_penalty: float = 10.0,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        # Ensure expectiles are sorted in ascending order
        if isinstance(expectiles, list):
            expectiles = sorted(expectiles)
            expectiles_tensor = torch.tensor(expectiles, dtype=torch.float32)
        else:
            sorted_indices = torch.argsort(expectiles)
            expectiles_tensor = expectiles[sorted_indices]
            
        self.register_buffer('expectiles', expectiles_tensor)
        self.num_expectiles = len(expectiles)
        self.base_loss = base_loss
        self.crossover_penalty = crossover_penalty
        
        # Create individual expectile losses
        self.expectile_losses = nn.ModuleList([
            ExpectileLoss(expectile=e, reduction='none')
            for e in expectiles
        ])
    
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate expectile loss with crossover penalty.
        
        Args:
            y_true: Ground truth values [batch_size, n_features]
            y_pred: Predicted expectiles [batch_size, num_expectiles, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features] or [batch_size]
        
        Returns:
            Loss combining standard expectile loss and crossover penalty
        """
        batch_size, n_features = y_true.shape[0], y_true.shape[-1]
        device = y_true.device
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        
        # Shape validation for y_pred
        if y_pred.shape[1] != self.num_expectiles:
            raise ValueError(
                f"Expected y_pred shape [batch_size, {self.num_expectiles}, n_features], "
                f"got shape {y_pred.shape}"
            )
            
        # Calculate standard expectile losses
        base_losses = []
        for i, loss_fn in enumerate(self.expectile_losses):
            level_preds = y_pred[:, i]
            level_loss = loss_fn(y_true, level_preds, mask, weights)
            base_losses.append(level_loss)
            
        stacked_base_losses = torch.stack(base_losses, dim=0)  # [num_expectiles, batch_size]
        
        # Calculate crossover penalties
        crossover_penalties = torch.zeros(batch_size, device=device)
        
        for i in range(self.num_expectiles - 1):
            # Lower expectile should be <= higher expectile
            lower_preds = y_pred[:, i]    # Lower expectile predictions
            higher_preds = y_pred[:, i+1] # Higher expectile predictions
            
            # Calculate violation: ReLU(lower - higher)
            violations = F.relu(lower_preds - higher_preds)
            
            # Apply mask if provided
            if mask is not None:
                violations = violations * mask
                
            # Sum violations across features
            sample_violations = torch.sum(violations, dim=-1)
            crossover_penalties += sample_violations
        
        # Final loss is weighted combination of base loss and crossover penalty
        total_base_loss = torch.mean(stacked_base_losses, dim=0)  # Mean across expectiles
        final_loss = self.base_loss * total_base_loss + self.crossover_penalty * crossover_penalties
        
        # Apply final reduction
        if self.reduction == 'mean':
            return torch.mean(final_loss)
        elif self.reduction == 'sum':
            return torch.sum(final_loss)
        else:  # 'none'
            return final_loss