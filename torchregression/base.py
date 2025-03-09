import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLoss(nn.Module):
    """Base class for loss functions that support masking and weighting."""
    def __init__(self, reduction='mean'):
        """
        Initialize the loss function.
        
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                'none': no reduction will be applied.
                'mean': the sum of the output will be divided by the number of elements.
                'sum': the output will be summed.
                Default: 'mean'
        """
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'none', 'mean', or 'sum'")
        self.reduction = reduction

    def _apply_mask(self, x, mask):
        """Applies the mask to the input tensor, handling None gracefully."""
        if mask is not None:
            return torch.where(mask, x, torch.zeros_like(x))
        return x

    def _reduce(self, loss, mask=None, weights=None):
        """
        Reduces the loss tensor, handling masking, weighting and averaging.
        
        Args:
            loss: Per-element loss values
            mask: Optional boolean mask indicating valid elements
            weights: Optional weights for each element
            
        Returns:
            Scalar loss value, reduced according to the reduction method
        """
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
            
        if self.reduction == 'none':
            return loss
            
        # For masked mean/sum, we need to handle normalization
        if mask is not None:
            if self.reduction == 'sum':
                return torch.sum(loss)
            else:  # 'mean'
                # Get the sum of valid elements for normalization
                # Add small epsilon to prevent division by zero
                valid_elements = torch.sum(mask.float()) + 1e-10
                return torch.sum(loss) / valid_elements
        else:
            if self.reduction == 'sum':
                return torch.sum(loss)
            else:  # 'mean'
                return torch.mean(loss)

class WeightedMSELoss(MaskedLoss):
    """
    Weighted Mean Squared Error Loss.
    
    Applies weights to each element of the squared error.
    Weights can be provided per feature or per sample and feature.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)

    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate weighted MSE loss.
        
        Args:
            y_true: Ground truth values [batch_size, n_features]
            y_pred: Predicted values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Weights for each element, either [n_features] or [batch_size, n_features]
            
        Returns:
            Scalar weighted MSE loss
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate MSE loss
        mse_loss = F.mse_loss(y_true, y_pred, reduction='none')
        
        # Process weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            # Ensure weights are properly broadcast to match input dimensions
            if weights.ndim == 1:
                weights = weights.unsqueeze(0).expand_as(mse_loss)
                
        return self._reduce(mse_loss, mask, weights)

