"""
Extensions to PyTorch loss functions with added features.

This module provides enhanced versions of PyTorch loss functions with
support for masking, weighted loss, and other useful features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

from .base import MaskedLoss, RegressionLoss, TorchLossWrapper

class MSELoss(RegressionLoss):
    """
    Mean Squared Error loss with masking support.
    
    This extends PyTorch's MSELoss with support for masks and per-sample weights.
    
    Args:
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
    
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate MSE loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            MSE loss
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate squared error
        squared_error = (y_true - y_pred)**2
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            squared_error = squared_error * weights
        
        return self._reduce(squared_error, mask)


class L1Loss(RegressionLoss):
    """L1 Loss (Mean Absolute Error) with masking and weighting support."""
    def forward(self, y_true, y_pred, mask=None, weights=None):
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Calculate residuals and take absolute value
        residuals = self._calculate_residuals(y_true, y_pred, mask)
        abs_error = torch.abs(residuals)
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            abs_error = abs_error * weights
        
        return self._reduce(abs_error, mask)


class SmoothL1Loss(RegressionLoss):
    """Smooth L1 Loss with masking and weighting support."""
    def __init__(self, beta=1.0, reduction='mean'):
        super().__init__(reduction=reduction)
        self.beta = beta
    
    def forward(self, y_true, y_pred, mask=None, weights=None):
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask to get masked values
        y_true_masked = self._apply_mask(y_true, mask)
        y_pred_masked = self._apply_mask(y_pred, mask)
        
        # Calculate Huber loss directly
        diff = torch.abs(y_true_masked - y_pred_masked)
        loss = torch.where(
            diff < self.beta,
            0.5 * (diff ** 2) / self.beta,
            diff - 0.5 * self.beta
        )
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
        
        return self._reduce(loss, mask)


class BCELoss(MaskedLoss):
    """
    Binary Cross Entropy loss with masking support.
    
    This extends PyTorch's BCELoss with support for masks and per-sample weights.
    
    Args:
        pos_weight: Weight for the positive class
        weight: Global weighting
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small constant for numerical stability
    """
    def __init__(self, pos_weight=None, weight=None, reduction='mean', eps=1e-8):
        super().__init__(reduction=reduction)
        self.pos_weight = pos_weight
        self.weight = weight
        self.eps = eps
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate binary cross entropy loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...] (0 or 1)
            y_pred: Predicted probabilities [batch_size, ...] (0 to 1)
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional sample weights [batch_size, ...] or [batch_size]
            
        Returns:
            BCE loss value
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Clamp predictions for numerical stability
        y_pred = torch.clamp(y_pred, min=self.eps, max=1-self.eps)
        
        # Calculate BCELoss
        if self.pos_weight is not None:
            pos_weight = self.pos_weight
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, device=y_pred.device, dtype=y_pred.dtype)
            loss = -(pos_weight * y_true * torch.log(y_pred) + 
                   (1 - y_true) * torch.log(1 - y_pred))
        else:
            loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        
        # Apply global weight if provided
        if self.weight is not None:
            loss = loss * self.weight
            
        # Apply sample weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


# Create wrapped versions of common PyTorch losses
MaskedMSELoss = TorchLossWrapper(nn.MSELoss)
MaskedL1Loss = TorchLossWrapper(nn.L1Loss)
MaskedCrossEntropyLoss = TorchLossWrapper(nn.CrossEntropyLoss)
MaskedBCELoss = TorchLossWrapper(nn.BCELoss)
MaskedBCEWithLogitsLoss = TorchLossWrapper(nn.BCEWithLogitsLoss)
MaskedKLDivLoss = TorchLossWrapper(nn.KLDivLoss)
MaskedNLLLoss = TorchLossWrapper(nn.NLLLoss)
