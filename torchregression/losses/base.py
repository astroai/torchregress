"""
Base loss functions and abstract classes for regression tasks.

This module provides the foundation classes for all loss functions
in the torchregression library.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Dict, Callable

from ..utils.tensor_ops import apply_mask, masked_reduction
from ..utils.validation import validate_reduction

class MaskedLoss(nn.Module):
    """
    Base class for losses that support optional masking.
    
    This class provides common functionality for handling masked losses,
    where certain elements in the inputs should be ignored.
    
    Args:
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = validate_reduction(reduction)
    
    def _apply_mask(self, tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply an optional mask to a tensor.
        
        When mask is provided, this returns tensor.clone() for further modification.
        Otherwise, it returns the original tensor unchanged.
        
        Args:
            tensor: Input tensor
            mask: Optional boolean mask with the same shape as tensor
            
        Returns:
            Tensor (either original or cloned)
        """
        if mask is None:
            return tensor
            
        # Ensure mask has the same shape as tensor
        if mask.shape != tensor.shape:
            raise ValueError(f"Mask shape {mask.shape} must match tensor shape {tensor.shape}")
            
        # Return a copy for further modification
        return tensor.clone()
    
    def _validate_inputs(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
                       mask: Optional[torch.Tensor] = None) -> None:
        """
        Validate input tensors.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            mask: Optional boolean mask
            
        Raises:
            ValueError: If inputs have incompatible shapes
        """
        # Check shape compatibility
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true shape {y_true.shape} must match y_pred shape {y_pred.shape}")
            
        # Check mask if provided
        if mask is not None and mask.shape != y_true.shape:
            raise ValueError(f"Mask shape {mask.shape} must match target shape {y_true.shape}")
    
    def _reduce(self, loss: torch.Tensor, mask: Optional[torch.Tensor],
              weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply reduction to the loss tensor.
        
        Args:
            loss: Loss tensor
            mask: Optional boolean mask
            weights: Optional sample weights
            
        Returns:
            Reduced loss according to the reduction method
        """
        if self.reduction == 'none':
            return loss
            
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
            
        if mask is not None:
            # Count valid elements for mean calculation
            valid_count = mask.sum()
            
            if valid_count == 0:
                # All elements are masked, return zero loss
                return torch.tensor(0.0, device=loss.device)
                
            if self.reduction == 'mean':
                return loss.sum() / valid_count
            else:  # 'sum'
                return loss.sum()
        else:
            # No mask
            if self.reduction == 'mean':
                return loss.mean()
            else:  # 'sum'
                return loss.sum()


class TorchLossWrapper(MaskedLoss):
    """
    Wrapper for PyTorch loss functions to add masking and weighting support.
    
    Args:
        loss_fn: PyTorch loss function class (nn.Module) or instance
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        **kwargs: Additional arguments to pass to the loss function
    """
    def __init__(self, 
                loss_fn: Union[Callable, nn.Module], 
                reduction: str = 'mean', 
                **kwargs):
        super().__init__(reduction='none')  # We'll handle reduction ourselves
        
        # Handle both class and instance cases
        if isinstance(loss_fn, type):
            # It's a class, instantiate it with reduction='none'
            kwargs['reduction'] = 'none'
            self.torch_loss = loss_fn(**kwargs)
        else:
            # It's already an instance
            self.torch_loss = loss_fn
            if hasattr(self.torch_loss, 'reduction'):
                self.torch_loss.reduction = 'none'
                
        self.reduction = validate_reduction(reduction)
            
    def forward(self, 
               y_true: torch.Tensor, 
               y_pred: torch.Tensor, 
               mask: Optional[torch.Tensor] = None, 
               weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate loss using the wrapped PyTorch loss function.
        """
        # Apply mask if provided
        y_true_masked = self._apply_mask(y_true, mask)
        y_pred_masked = self._apply_mask(y_pred, mask)
        
        # PyTorch losses expect (input, target) order
        loss = self.torch_loss(y_pred_masked, y_true_masked)
        
        # Handle weights and reduction
        return self._reduce(loss, mask, weights)


class RegressionLoss(MaskedLoss):
    """
    Base class for regression losses.
    
    This extends MaskedLoss with additional functionality specific 
    to regression tasks.
    
    Args:
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction)
    
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
             mask: Optional[torch.Tensor] = None,
             weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate loss.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            mask: Optional boolean mask
            weights: Optional sample weights
            
        Returns:
            Loss value
        """
        raise NotImplementedError("Subclasses must implement this method")


class DistributionLoss(MaskedLoss):
    """
    Base class for distributional losses.
    
    This class supports losses that estimate full probability 
    distributions rather than just point predictions.
    """
    def _extract_distribution_parameters(self, y_pred):
        """Extract distribution parameters from model outputs."""
        raise NotImplementedError
        
    def _calculate_nll(self, y_true, params, mask=None):
        """Calculate negative log likelihood."""
        raise NotImplementedError

