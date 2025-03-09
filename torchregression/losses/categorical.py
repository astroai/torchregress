"""
Loss functions for categorical outputs and classification-as-regression tasks.

This module provides loss functions that are useful for converting regression
tasks to classification tasks and vice versa, as well as handling categorical data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Literal, Tuple

from .base import MaskedLoss
from ..utils.validation import validate_tensor_shapes

class MaskedSoftmaxCELoss(MaskedLoss):
    """
    Cross-entropy loss with softmax, supporting masked values and probabilistic targets.
    
    Args:
        reduction: 'none' | 'mean' | 'sum'
        ignore_index: Specifies target value that is ignored
        label_smoothing: Float in [0, 1] for label smoothing
    """
    def __init__(self, reduction: str = 'mean', ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__(reduction=reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate softmax cross-entropy loss.
        
        Args:
            y_true: Either class indices [batch_size] or probabilities [batch_size, n_classes]
            y_pred: Logits [batch_size, n_classes]
            mask: Optional mask [batch_size]
            weights: Optional weights [batch_size] or [n_classes]
            
        Returns:
            Loss value after applying reduction
        """
        # Apply mask if provided
        if mask is not None:
            y_pred = y_pred.clone()
            
            # If y_true is class indices, just use mask
            if y_true.dim() == 1 or (y_true.dim() == 2 and y_true.shape[1] == 1):
                # Set predictions to ignore_index where mask is False
                if mask.dim() < y_pred.dim():
                    mask = mask.unsqueeze(-1).expand_as(y_pred)
                y_pred[~mask] = self.ignore_index
            else:
                # If y_true has probabilities, apply mask to both
                if mask.dim() < y_true.dim():
                    mask = mask.unsqueeze(-1).expand_as(y_true)
                y_true = y_true.clone()
                y_true[~mask] = 0
                y_pred[~mask] = self.ignore_index
        
        # Check if y_true has one-hot or soft labels
        if y_true.dim() == y_pred.dim():
            # Using KL-div with log_softmax for soft targets
            log_probs = F.log_softmax(y_pred, dim=-1)
            loss = F.kl_div(
                log_probs,
                y_true,
                reduction='none',
                log_target=False
            )
            
            # Sum over class dimension
            loss = loss.sum(dim=-1)
        else:
            # Using cross_entropy for hard targets
            loss = F.cross_entropy(
                y_pred,
                y_true,
                weight=weights if weights is not None and weights.dim() > 0 and weights.size(0) == y_pred.shape[1] else None,
                ignore_index=self.ignore_index,
                reduction='none',
                label_smoothing=self.label_smoothing
            )
            
            # Apply per-sample weights if provided
            if weights is not None and weights.dim() == 1 and weights.size(0) == y_pred.shape[0]:
                loss = loss * weights
        
        # Apply final reduction
        return self._reduce(loss, mask)

# For backward compatibility
SoftmaxCELoss = MaskedSoftmaxCELoss


class HistogramLoss(MaskedLoss):
    """
    Histogram loss for regression-as-classification with binning.
    
    This loss converts regression to a soft histogram prediction,
    useful for handling complex or multi-modal output distributions.
    
    Args:
        bins: Number of bins or array of bin edges
        min_value: Minimum value for auto-generated bins
        max_value: Maximum value for auto-generated bins
        sigma: Standard deviation for soft binning
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(
        self, 
        bins: Union[int, torch.Tensor] = 10, 
        min_value: float = 0.0, 
        max_value: float = 1.0,
        sigma: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        
        # Set up bins
        if isinstance(bins, int):
            self.n_bins = bins
            self.register_buffer('bin_edges', torch.linspace(min_value, max_value, bins + 1))
        else:
            self.n_bins = len(bins) - 1
            self.register_buffer('bin_edges', bins)
            
        # Calculate bin centers
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.register_buffer('bin_centers', bin_centers)
        
        # Sigma controls the smoothness of bin assignments
        self.sigma = sigma
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate histogram loss.
        
        Args:
            y_true: Ground truth values [batch_size, 1]
            y_pred: Predicted histogram logits [batch_size, n_bins]
            mask: Optional mask [batch_size]
            weights: Optional weights [batch_size]
            
        Returns:
            Loss value after applying reduction
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Check input shapes
        if y_pred.shape[-1] != self.n_bins:
            raise ValueError(f"Expected y_pred last dimension to be {self.n_bins}, got {y_pred.shape[-1]}")
            
        batch_size = y_true.shape[0]
        
        # Convert targets to soft bin assignments
        # Calculate distance to bin centers [batch_size, n_bins]
        y_true_expanded = y_true.expand(-1, self.n_bins)
        distances = (y_true_expanded - self.bin_centers.unsqueeze(0)) ** 2
        
        # Convert distances to probabilities using Gaussian kernel
        target_probs = torch.exp(-distances / (2 * self.sigma**2))
        # Normalize to sum to 1 over bin dimension
        target_probs = target_probs / (torch.sum(target_probs, dim=1, keepdim=True) + 1e-10)
        
        # Apply softmax to predicted logits
        pred_probs = F.softmax(y_pred, dim=1)
        
        # Calculate KL divergence loss
        loss = F.kl_div(
            torch.log(pred_probs + 1e-10),
            target_probs,
            reduction='none',
            log_target=False
        )
        
        # Sum over bin dimension
        loss = torch.sum(loss, dim=1)
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)
    
    def decode_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Decode histogram predictions back to scalar values.
        
        Args:
            y_pred: Predicted histogram logits or probabilities [batch_size, n_bins]
            
        Returns:
            Scalar predictions [batch_size, 1]
        """
        # Apply softmax if not already probabilities
        if torch.min(y_pred) < 0 or torch.max(y_pred) > 1 + 1e-4:
            y_pred = F.softmax(y_pred, dim=1)
            
        # Weighted average of bin centers
        weighted_values = y_pred * self.bin_centers.unsqueeze(0)
        return torch.sum(weighted_values, dim=1, keepdim=True)


class RegressionAsClassificationLoss(MaskedLoss):
    """
    Convert regression to classification by binning values.
    
    This loss bins continuous output values and treats them as classes,
    allowing regression to be treated as a classification problem.
    
    Args:
        bins: Number of bins or array of bin edges
        min_value: Minimum value for auto-generated bins
        max_value: Maximum value for auto-generated bins
        soft_targets: Whether to use soft targets (probability distributions)
        sigma: Standard deviation for soft targets
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(
        self, 
        bins: Union[int, torch.Tensor] = 10, 
        min_value: float = 0.0, 
        max_value: float = 1.0,
        soft_targets: bool = True, 
        sigma: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        
        # Set up binning
        self.soft_targets = soft_targets
        self.sigma = sigma
        
        # Set up bins
        if isinstance(bins, int):
            self.n_bins = bins
            self.register_buffer('bin_edges', torch.linspace(min_value, max_value, bins + 1))
        else:
            self.n_bins = len(bins) - 1
            self.register_buffer('bin_edges', bins)
            
        # Calculate bin centers for soft targets
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.register_buffer('bin_centers', bin_centers)
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate regression-as-classification loss.
        
        Args:
            y_true: Ground truth values [batch_size, 1]
            y_pred: Predicted logits [batch_size, n_bins]
            mask: Optional mask [batch_size]
            weights: Optional weights [batch_size]
            
        Returns:
            Loss value after applying reduction
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Check input shapes
        if y_pred.shape[-1] != self.n_bins:
            raise ValueError(f"Expected y_pred last dimension to be {self.n_bins}, got {y_pred.shape[-1]}")
        
        # Convert targets to bin indices or soft targets
        if self.soft_targets:
            # Use soft targets as in HistogramLoss
            y_true_expanded = y_true.expand(-1, self.n_bins)
            distances = (y_true_expanded - self.bin_centers.unsqueeze(0)) ** 2
            
            target_probs = torch.exp(-distances / (2 * self.sigma**2))
            target_probs = target_probs / (torch.sum(target_probs, dim=1, keepdim=True) + 1e-10)
            
            # Calculate cross-entropy with soft targets
            log_probs = F.log_softmax(y_pred, dim=1)
            loss = -torch.sum(target_probs * log_probs, dim=1)
        else:
            # Hard binning - determine which bin each target falls into
            bin_indices = torch.bucketize(y_true.squeeze(), self.bin_edges) - 1
            
            # Handle edge case: values exactly equal to max_value
            bin_indices = torch.clamp(bin_indices, max=self.n_bins-1)
            
            # Calculate cross-entropy with hard targets
            loss = F.cross_entropy(y_pred, bin_indices.long(), reduction='none')
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)
    
    def decode_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Decode classification predictions back to continuous values.
        
        Args:
            y_pred: Predicted logits or probabilities [batch_size, n_bins]
            
        Returns:
            Regression predictions [batch_size, 1]
        """
        # Apply softmax if not already probabilities
        if torch.min(y_pred) < 0 or torch.max(y_pred) > 1 + 1e-4:
            y_pred = F.softmax(y_pred, dim=1)
            
        # Option 1: Take argmax and use bin centers (hard decoding)
        # bin_indices = torch.argmax(y_pred, dim=1)
        # return self.bin_centers[bin_indices].unsqueeze(1)
        
        # Option 2: Weighted average of bin centers (soft decoding)
        weighted_values = y_pred * self.bin_centers.unsqueeze(0)
        return torch.sum(weighted_values, dim=1, keepdim=True)
