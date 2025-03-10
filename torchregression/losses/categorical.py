"""
Loss functions for categorical outputs and classification-as-regression tasks.

This module provides loss functions that are useful for converting regression
tasks to classification tasks and vice versa, as well as handling categorical data.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Dict, Any

from .base import MaskedLoss, DistributionLoss

class MaskedSoftmaxCELoss(MaskedLoss):
    """
    Cross-entropy loss with softmax, supporting masked values and probabilistic targets.
    
    Args:
        reduction: Method for reducing the loss ('none', 'mean', 'sum')
        ignore_index: Specifies target value that is ignored
        label_smoothing: Float in [0, 1] for label smoothing
        temperature: Temperature for softmax scaling (higher values make distribution softer)
    """
    def __init__(self, 
                reduction: str = 'mean', 
                ignore_index: int = -100, 
                label_smoothing: float = 0.0,
                temperature: float = 1.0):
        super().__init__(reduction=reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        
    def forward(self, 
                y_true: torch.Tensor, 
                y_pred: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                weights: Optional[torch.Tensor] = None):
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
        
        # Apply temperature scaling if needed
        if self.temperature != 1.0:
            y_pred = y_pred / self.temperature
        
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

class HistogramLoss(DistributionLoss):
    """
    Histogram loss for regression-as-classification with binning.
    
    This loss converts regression to a soft histogram prediction,
    useful for handling complex or multi-modal output distributions.
    
    Args:
        bins: Number of bins or array of bin edges
        min_value: Minimum value for auto-generated bins
        max_value: Maximum value for auto-generated bins
        sigma: Standard deviation for soft binning
        reduction: Method for reducing the loss ('none', 'mean', 'sum')
        soft_targets: Whether to use soft targets (probability distributions)
        loss_type: Type of loss to use ('kl_div', 'cross_entropy', 'focal', 'nll')
        normalize_targets: Whether to normalize target distributions to sum to 1
        focal_gamma: Gamma parameter for focal loss (if loss_type='focal')
    """
    def __init__(
        self, 
        bins: Union[int, torch.Tensor] = 10, 
        min_value: float = 0.0, 
        max_value: float = 1.0,
        sigma: float = 0.1,
        reduction: str = 'mean',
        soft_targets: bool = True,
        loss_type: str = 'kl_div',
        normalize_targets: bool = True,
        focal_gamma: float = 2.0
    ):
        super().__init__(reduction=reduction)
        
        # Set up bins
        if isinstance(bins, int):
            self.n_bins = bins
            self.register_buffer('bin_edges', torch.linspace(min_value, max_value, bins + 1))
        else:
            self.n_bins = len(bins) - 1
            self.register_buffer('bin_edges', bins)
            
        # Calculate bin centers and widths
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        self.register_buffer('bin_centers', bin_centers)
        self.register_buffer('bin_widths', bin_widths)
        
        # Parameters for soft targets and loss type
        self.sigma = sigma
        self.soft_targets = soft_targets
        self.loss_type = loss_type
        self.normalize_targets = normalize_targets
        self.focal_gamma = focal_gamma
        
    def _extract_distribution_parameters(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract distribution parameters from model outputs.
        
        Args:
            y_pred: Predicted histogram logits [batch_size, n_bins]
            
        Returns:
            Dictionary of distribution parameters (bin_probs)
        """
        # Apply softmax to get probabilities
        bin_probs = F.softmax(y_pred, dim=1)
        
        return {
            'bin_probs': bin_probs, 
            'bin_centers': self.bin_centers, 
            'bin_widths': self.bin_widths
        }
    
    def _calculate_nll(self, y_true: torch.Tensor, params: Dict[str, torch.Tensor], 
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate negative log likelihood for the histogram distribution.
        
        Args:
            y_true: Ground truth values [batch_size, 1]
            params: Distribution parameters from _extract_distribution_parameters
            mask: Optional mask [batch_size]
            
        Returns:
            Negative log likelihood [batch_size]
        """
        bin_probs = params['bin_probs']
        
        # Convert targets to bin indices
        bin_indices = torch.bucketize(y_true.squeeze(), self.bin_edges) - 1
        bin_indices = torch.clamp(bin_indices, max=self.n_bins-1)
        
        # Get probability for the correct bin
        batch_indices = torch.arange(y_true.shape[0], device=y_true.device)
        correct_bin_probs = bin_probs[batch_indices, bin_indices]
        
        # Avoid log(0)
        eps = 1e-10
        nll = -torch.log(correct_bin_probs + eps)
        
        return nll
        
    def _get_target_distribution(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous targets to probability distributions over bins.
        
        Args:
            y_true: Ground truth values [batch_size, 1]
            
        Returns:
            Probability distributions [batch_size, n_bins]
        """
        if self.soft_targets:
            # Expand targets to compare with each bin center
            y_true_expanded = y_true.expand(-1, self.n_bins)
            
            # Calculate distances to bin centers using vectorized operations
            distances = (y_true_expanded - self.bin_centers.unsqueeze(0)) ** 2
            
            # Convert distances to probabilities using Gaussian kernel
            target_probs = torch.exp(-distances / (2 * self.sigma**2))
            
            # Normalize to ensure it's a proper PDF
            if self.normalize_targets:
                target_probs = target_probs / (torch.sum(target_probs, dim=1, keepdim=True) + 1e-10)
                
            return target_probs
        else:
            # Hard binning - determine which bin each target falls into
            bin_indices = torch.bucketize(y_true.squeeze(), self.bin_edges) - 1
            
            # Handle edge case: values exactly equal to max_value
            bin_indices = torch.clamp(bin_indices, max=self.n_bins-1)
            
            # Convert to one-hot encoding
            target_probs = torch.zeros(y_true.shape[0], self.n_bins, 
                                     device=y_true.device, dtype=y_true.dtype)
            batch_indices = torch.arange(y_true.shape[0], device=y_true.device)
            target_probs[batch_indices, bin_indices] = 1.0
            
            return target_probs
            
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                weights: Optional[torch.Tensor] = None):
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
            
        # Convert targets to probability distributions
        target_probs = self._get_target_distribution(y_true)
        
        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred)
        pred_probs = params['bin_probs']
        log_probs = F.log_softmax(y_pred, dim=1)
        
        # Calculate loss based on loss_type
        if self.loss_type == 'nll':
            # Use the _calculate_nll method from DistributionLoss
            loss = self._calculate_nll(y_true, params, mask)
            
        elif self.loss_type == 'kl_div':
            # KL divergence loss
            loss = F.kl_div(
                log_probs,
                target_probs,
                reduction='none',
                log_target=False
            )
            # Sum over bin dimension
            loss = torch.sum(loss, dim=1)
            
        elif self.loss_type == 'cross_entropy':
            # Cross-entropy loss
            loss = -torch.sum(target_probs * log_probs, dim=1)
            
        elif self.loss_type == 'focal':
            # Focal loss - focuses more on hard examples
            # Weight by (1-p_t)^gamma where p_t is the correct class probability
            p_t = torch.sum(target_probs * pred_probs, dim=1)
            focal_weight = (1 - p_t) ** self.focal_gamma
            ce_loss = -torch.sum(target_probs * log_probs, dim=1)
            loss = focal_weight * ce_loss
            
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
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
        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred) if torch.min(y_pred) < 0 else {'bin_probs': y_pred}
        pred_probs = params['bin_probs']
        
        # Weighted average of bin centers (expected value of the distribution)
        weighted_values = pred_probs * self.bin_centers.unsqueeze(0)
        return torch.sum(weighted_values, dim=1, keepdim=True)
    
    def get_distribution(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get the full distribution information from predictions.
        
        Args:
            y_pred: Predicted histogram logits [batch_size, n_bins]
            
        Returns:
            Dictionary with distribution parameters
        """
        return self._extract_distribution_parameters(y_pred)

class RegressionAsClassificationLoss(DistributionLoss):
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
        reduction: Method for reducing the loss ('none', 'mean', 'sum')
        label_smoothing: Float in [0, 1] for label smoothing with hard targets
        loss_type: Type of loss to use ('cross_entropy', 'kl_div', 'focal', 'nll')
        ordinal: Whether to use ordinal regression approach
        focal_gamma: Gamma parameter for focal loss (if loss_type='focal')
        normalize_targets: Whether to normalize target distributions to sum to 1
    """
    def __init__(
        self, 
        bins: Union[int, torch.Tensor] = 10, 
        min_value: float = 0.0, 
        max_value: float = 1.0,
        soft_targets: bool = True, 
        sigma: float = 0.1,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        loss_type: str = 'cross_entropy',
        ordinal: bool = False,
        focal_gamma: float = 2.0,
        normalize_targets: bool = True
    ):
        super().__init__(reduction=reduction)
        
        # Set up binning
        self.soft_targets = soft_targets
        self.sigma = sigma
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.ordinal = ordinal
        self.focal_gamma = focal_gamma
        self.normalize_targets = normalize_targets
        
        # Set up bins
        if isinstance(bins, int):
            self.n_bins = bins
            self.register_buffer('bin_edges', torch.linspace(min_value, max_value, bins + 1))
        else:
            self.n_bins = len(bins) - 1
            self.register_buffer('bin_edges', bins)
            
        # Calculate bin centers for soft targets and bin widths
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        self.register_buffer('bin_centers', bin_centers)
        self.register_buffer('bin_widths', bin_widths)
        
        # For ordinal regression
        if self.ordinal:
            # Create binary encoding matrix for ordinal regression
            encoding_matrix = torch.zeros(self.n_bins, self.n_bins-1)
            for i in range(self.n_bins):
                encoding_matrix[i, :i] = 1
            self.register_buffer('encoding_matrix', encoding_matrix)
    
    def _extract_distribution_parameters(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract distribution parameters from model outputs.
        
        Args:
            y_pred: For regular mode: Predicted logits [batch_size, n_bins]
                    For ordinal mode: Predicted binary logits [batch_size, n_bins-1]
            
        Returns:
            Dictionary of distribution parameters
        """
        if self.ordinal:
            # For ordinal regression, convert binary logits to class probabilities
            binary_probs = torch.sigmoid(y_pred)
            
            # Calculate class probabilities from binary probabilities
            probs = torch.zeros(y_pred.shape[0], self.n_bins, 
                               device=y_pred.device, dtype=y_pred.dtype)
            
            probs[:, 0] = 1 - binary_probs[:, 0]
            for k in range(1, self.n_bins-1):
                probs[:, k] = binary_probs[:, k-1] - binary_probs[:, k]
            probs[:, self.n_bins-1] = binary_probs[:, self.n_bins-2]
            
            return {
                'bin_probs': probs,
                'binary_probs': binary_probs,
                'bin_centers': self.bin_centers,
                'bin_widths': self.bin_widths
            }
        else:
            # Regular mode
            bin_probs = F.softmax(y_pred, dim=1)
            
            return {
                'bin_probs': bin_probs,
                'bin_centers': self.bin_centers,
                'bin_widths': self.bin_widths
            }
    
    def _calculate_nll(self, y_true: torch.Tensor, params: Dict[str, torch.Tensor], 
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate negative log likelihood for the distribution.
        
        Args:
            y_true: Ground truth values [batch_size, 1]
            params: Distribution parameters
            mask: Optional mask [batch_size]
            
        Returns:
            Negative log likelihood [batch_size]
        """
        bin_probs = params['bin_probs']
        
        # Convert targets to bin indices
        bin_indices = torch.bucketize(y_true.squeeze(), self.bin_edges) - 1
        bin_indices = torch.clamp(bin_indices, max=self.n_bins-1)
        
        # Get probability for the correct bin
        batch_indices = torch.arange(y_true.shape[0], device=y_true.device)
        correct_bin_probs = bin_probs[batch_indices, bin_indices]
        
        # Avoid log(0)
        eps = 1e-10
        nll = -torch.log(correct_bin_probs + eps)
        
        return nll
        
    def _get_target_distribution(self, y_true: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert continuous targets to probability distributions or class indices.
        
        Args:
            y_true: Ground truth values [batch_size, 1]
            
        Returns:
            Target probabilities or class indices
        """
        if self.soft_targets:
            # Use soft targets
            y_true_expanded = y_true.expand(-1, self.n_bins)
            distances = (y_true_expanded - self.bin_centers.unsqueeze(0)) ** 2
            
            # Vectorized Gaussian kernel calculation
            target_probs = torch.exp(-distances / (2 * self.sigma**2))
            
            # Normalize to ensure it's a proper PDF
            if self.normalize_targets:
                target_probs = target_probs / (torch.sum(target_probs, dim=1, keepdim=True) + 1e-10)
            
            if self.ordinal:
                # For ordinal mode, compute the binary targets
                batch_size = y_true.shape[0]
                ordinal_targets = torch.bmm(
                    target_probs.unsqueeze(1), 
                    self.encoding_matrix.unsqueeze(0).expand(batch_size, -1, -1)
                ).squeeze(1)
                return target_probs, ordinal_targets
            else:
                return target_probs
        else:
            # Hard binning
            bin_indices = torch.bucketize(y_true.squeeze(), self.bin_edges) - 1
            bin_indices = torch.clamp(bin_indices, max=self.n_bins-1)
            
            if self.ordinal:
                # For ordinal regression with hard targets, create binary targets
                batch_size = y_true.shape[0]
                ordinal_targets = torch.zeros(batch_size, self.n_bins-1, 
                                            device=y_true.device, dtype=torch.float32)
                for i in range(batch_size):
                    ordinal_targets[i, :bin_indices[i]] = 1.0
                return bin_indices, ordinal_targets
            else:
                return bin_indices
        
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                weights: Optional[torch.Tensor] = None):
        """
        Calculate regression-as-classification loss.
        
        Args:
            y_true: Ground truth values [batch_size, 1]
            y_pred: For regular mode: Predicted logits [batch_size, n_bins]
                    For ordinal mode: Predicted binary logits [batch_size, n_bins-1]
            mask: Optional mask [batch_size]
            weights: Optional weights [batch_size]
            
        Returns:
            Loss value after applying reduction
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Check input shapes
        if self.ordinal:
            expected_dim = self.n_bins - 1
            if y_pred.shape[-1] != expected_dim:
                raise ValueError(f"In ordinal mode, expected y_pred last dimension to be {expected_dim}, got {y_pred.shape[-1]}")
        elif y_pred.shape[-1] != self.n_bins:
            raise ValueError(f"Expected y_pred last dimension to be {self.n_bins}, got {y_pred.shape[-1]}")
        
        # Get the distribution parameters
        params = self._extract_distribution_parameters(y_pred)
        
        # Special case for NLL loss
        if self.loss_type == 'nll':
            return self._reduce(self._calculate_nll(y_true, params, mask), mask, weights)
        
        # Convert targets to appropriate format
        targets = self._get_target_distribution(y_true)
        
        # Calculate loss based on mode
        if self.ordinal:
            if self.soft_targets:
                # Unpack targets (class_probs and ordinal_targets)
                _, ordinal_targets = targets
                # Binary cross entropy for each threshold
                loss = F.binary_cross_entropy_with_logits(
                    y_pred, 
                    ordinal_targets,
                    reduction='none'
                )
                # Sum over thresholds dimension
                loss = loss.mean(dim=1)
            else:
                # Unpack targets (class_indices and ordinal_targets)
                _, ordinal_targets = targets
                # Binary cross entropy for ordinal regression
                loss = F.binary_cross_entropy_with_logits(
                    y_pred,
                    ordinal_targets,
                    reduction='none'
                )
                # Sum over thresholds dimension
                loss = loss.mean(dim=1)
        else:
            if self.soft_targets:
                # Get log probabilities
                log_probs = F.log_softmax(y_pred, dim=1)
                pred_probs = F.softmax(y_pred, dim=1)
                
                if self.loss_type == 'cross_entropy':
                    # Calculate cross-entropy with soft targets
                    loss = -torch.sum(targets * log_probs, dim=1)
                    
                elif self.loss_type == 'kl_div':
                    # Calculate KL divergence
                    loss = F.kl_div(
                        log_probs,
                        targets,
                        reduction='none',
                        log_target=False
                    )
                    # Sum over class dimension
                    loss = torch.sum(loss, dim=1)
                    
                elif self.loss_type == 'focal':
                    # Focal loss with soft targets
                    p_t = torch.sum(targets * pred_probs, dim=1)
                    focal_weight = (1 - p_t) ** self.focal_gamma
                    ce_loss = -torch.sum(targets * log_probs, dim=1)
                    loss = focal_weight * ce_loss
                
                else:
                    raise ValueError(f"Unknown loss_type: {self.loss_type}")
            else:
                if self.loss_type == 'focal':
                    # Focal loss for hard targets
                    # Convert targets to one-hot
                    one_hot = F.one_hot(targets.long(), num_classes=self.n_bins).float()
                    
                    # Get probabilities for predicted classes
                    pred_probs = F.softmax(y_pred, dim=1)
                    p_t = torch.sum(one_hot * pred_probs, dim=1)
                    
                    # Focal weight
                    focal_weight = (1 - p_t) ** self.focal_gamma
                    
                    # Cross entropy
                    ce_loss = F.cross_entropy(
                        y_pred, 
                        targets.long(), 
                        reduction='none', 
                        label_smoothing=self.label_smoothing
                    )
                    
                    # Apply focal weighting
                    loss = focal_weight * ce_loss
                else:
                    # Standard cross-entropy with hard targets
                    loss = F.cross_entropy(
                        y_pred, 
                        targets.long(), 
                        reduction='none',
                        label_smoothing=self.label_smoothing
                    )
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)
    
    def decode_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Decode classification predictions back to continuous values.
        
        Args:
            y_pred: For regular mode: Predicted logits or probs [batch_size, n_bins]
                    For ordinal mode: Predicted binary logits [batch_size, n_bins-1]
            
        Returns:
            Regression predictions [batch_size, 1]
        """
        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred)
        bin_probs = params['bin_probs']
            
        # Weighted average of bin centers (expected value)
        weighted_values = bin_probs * self.bin_centers.unsqueeze(0)
        return torch.sum(weighted_values, dim=1, keepdim=True)
    
    def get_distribution(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get the full distribution information from predictions.
        
        Args:
            y_pred: Predicted logits
            
        Returns:
            Dictionary with distribution parameters
        """
        return self._extract_distribution_parameters(y_pred)
