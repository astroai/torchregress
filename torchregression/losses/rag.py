"""
Regression-as-classification losses for converting continuous regression problems 
into classification problems with binning.

This module provides a framework for treating regression as a classification problem
by binning continuous values and predicting probability distributions. This approach
offers several advantages:
1. Handling multi-modal target distributions
2. Better uncertainty quantification
3. More flexible loss functions
4. Handling noisy labels
Some references for inspiration:
- https://arxiv.org/abs/1901.07884
- https://arxiv.org/abs/2402.13425
- https://arxiv.org/abs/2102.06164
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any

from .base import DistributionLoss
from ..utils.labels import (
    create_bin_edges, calculate_bin_properties, create_gaussian_target_distribution,
    create_ordinal_encoding_matrix, convert_to_ordinal_targets, decode_bin_probabilities
)

class BinnedRegressionLoss(DistributionLoss):
    """
    Base class for regression losses using binning.
    
    This class provides common functionality for converting regression
    problems to classification problems through binning.
    
    Args:
        bins (Union[int, torch.Tensor]): Number of bins or array of bin edges
            Default: 10
        min_value (float, optional): Minimum value for auto-generated bins
            Default: 0.0
        max_value (float, optional): Maximum value for auto-generated bins
            Default: 1.0
        soft_targets (bool): Whether to use soft targets (probability distributions)
            Default: True
        sigma (float): Standard deviation for soft targets
            Default: 0.1
        reduction (str): Method for reducing the loss ('none', 'mean', 'sum')
            Default: 'mean'
        extrapolate_beyond_bins (bool): Whether to extrapolate for values outside bin range
            Default: False
        noise_aware (bool): Whether to adjust for noisy labels
            Default: False
        adaptive_sigma (bool): Whether to adjust sigma based on bin widths
            Default: False
        normalize_targets (bool): Whether to normalize target distributions
            Default: True
            
    Mathematical Formulation:
        The binned regression approach maps continuous values y to discrete bins and
        either assigns them to a single bin (hard targets) or distributes them across 
        multiple bins (soft targets) using a probability distribution, typically:
        
        p(bin_i|y) = exp(-0.5 * ((y - bin_center_i)/sigma)²) / Z
        
        where Z is a normalization constant ensuring sum(p(bin_i|y)) = 1.
    """
    def __init__(
        self,
        bins: Union[int, torch.Tensor] = 10,
        min_value: Optional[float] = 0.0,
        max_value: Optional[float] = 1.0,
        soft_targets: bool = True,
        sigma: float = 0.1,
        reduction: str = 'mean',
        extrapolate_beyond_bins: bool = False,
        noise_aware: bool = False,
        adaptive_sigma: bool = False,
        normalize_targets: bool = True
    ):
        super().__init__(reduction=reduction)
        
        # Store configuration
        self.soft_targets = soft_targets
        self.sigma = sigma
        self.extrapolate_beyond_bins = extrapolate_beyond_bins
        self.noise_aware = noise_aware
        self.adaptive_sigma = adaptive_sigma
        self.normalize_targets = normalize_targets
        
        # Setup bins
        self.n_bins = bins if isinstance(bins, int) else len(bins) - 1
        self.register_buffer('bin_edges', create_bin_edges(bins, min_value, max_value))
        
        # Calculate bin centers and widths
        bin_centers, bin_widths = calculate_bin_properties(self.bin_edges)
        self.register_buffer('bin_centers', bin_centers)
        self.register_buffer('bin_widths', bin_widths)
    
    def _get_target_distribution(self, target: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convert continuous targets to probability distributions over bins.
        
        Args:
            target: Ground truth values [batch_size, 1]
            uncertainty: Optional uncertainty estimates for targets [batch_size, 1]
            
        Returns:
            Target distributions [batch_size, n_bins]
        """
        if self.soft_targets:
            # Determine sigma (for noise-aware case)
            if self.noise_aware and uncertainty is not None:
                # Use provided uncertainties to adjust sigma
                sigmas = uncertainty.squeeze(-1)
            elif self.adaptive_sigma:
                # Adaptive sigma based on bin widths (scaled by global sigma)
                avg_bin_width = torch.mean(self.bin_widths)
                sigmas = self.sigma * torch.ones_like(target.squeeze(-1)) * avg_bin_width
            else:
                # Fixed sigma
                sigmas = self.sigma * torch.ones_like(target.squeeze(-1))
                
            # Handle vectorized sigma
            if sigmas.ndim == 1:
                # Vectorized calculation with different sigma per sample
                batch_size = target.shape[0]
                target_probs = torch.zeros(batch_size, self.n_bins, 
                                         device=target.device, dtype=target.dtype)
                
                for i in range(batch_size):
                    # Use individual sigma for each sample
                    distances = (target[i] - self.bin_centers) ** 2
                    target_probs[i] = torch.exp(-distances / (2 * sigmas[i]**2))
                    
                    # Normalize
                    if self.normalize_targets:
                        target_probs[i] = target_probs[i] / (torch.sum(target_probs[i]) + 1e-10)
            else:
                # Use single sigma for all samples
                target_probs = create_gaussian_target_distribution(
                    target, self.bin_centers, self.sigma, self.normalize_targets
                )
                
            return target_probs
        else:
            # Hard binning - determine which bin each target falls into
            bin_indices = torch.bucketize(target.squeeze(), self.bin_edges) - 1
            
            # Handle edge case: values exactly equal to max_value
            bin_indices = torch.clamp(bin_indices, max=self.n_bins-1)
            
            return bin_indices
    
    def _handle_out_of_range(self, target: torch.Tensor) -> torch.Tensor:
        """
        Handle values outside the bin range.
        
        Args:
            target: Ground truth values [batch_size, 1]
            
        Returns:
            Adjusted values [batch_size, 1]
        """
        if self.extrapolate_beyond_bins:
            # Keep values as is, allowing extrapolation
            return target
        else:
            # Clamp values to the valid range
            min_val = self.bin_edges[0]
            max_val = self.bin_edges[-1]
            return torch.clamp(target, min=min_val, max=max_val)
    
    def _extract_distribution_parameters(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract distribution parameters from model outputs.
        This should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _extract_distribution_parameters")
    
    def _calculate_nll(self, y_true: torch.Tensor, params: Dict[str, torch.Tensor], 
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate negative log likelihood for the distribution.
        This should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _calculate_nll")
    
    def decode_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Decode classification predictions back to continuous values.
        
        Args:
            y_pred: Predicted outputs from model
            
        Returns:
            Continuous values [batch_size, 1]
        """
        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred)
        bin_probs = params['bin_probs']
        
        # Use common utility function for decoding
        return decode_bin_probabilities(bin_probs, self.bin_centers)
    
    def get_distribution(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get the full distribution information from predictions.
        
        Args:
            y_pred: Predicted outputs from model
            
        Returns:
            Dictionary with distribution parameters
        """
        return self._extract_distribution_parameters(y_pred)

class StandardClassificationRegressionLoss(BinnedRegressionLoss):
    """
    Standard classification-based regression loss.
    
    This loss treats regression as a classification problem by binning
    continuous values and applying classification loss functions.
    
    Args:
        bins (Union[int, torch.Tensor]): Number of bins or array of bin edges
            Default: 10
        min_value (float, optional): Minimum value for auto-generated bins
            Default: 0.0
        max_value (float, optional): Maximum value for auto-generated bins
            Default: 1.0
        soft_targets (bool): Whether to use soft targets (probability distributions)
            Default: True
        sigma (float): Standard deviation for soft targets
            Default: 0.1
        reduction (str): Method for reducing the loss ('none', 'mean', 'sum')
            Default: 'mean'
        label_smoothing (float): Smoothing factor in [0, 1] for hard targets
            Default: 0.0
        loss_type (str): Type of loss ('cross_entropy', 'kl_div', 'focal', 'nll')
            Default: 'cross_entropy'
        focal_gamma (float): Gamma parameter for focal loss
            Default: 2.0
            
    Mathematical Formulation:
        For soft targets with a Gaussian kernel distribution p(bin_i|y), and model
        predicted distribution q(bin_i), the loss is calculated as:
        
        Cross-entropy: -∑_i p(bin_i|y) * log(q(bin_i))
        KL-div: ∑_i p(bin_i|y) * log(p(bin_i|y)/q(bin_i))
        Focal: -∑_i p(bin_i|y) * (1-q(bin_i))^γ * log(q(bin_i))
    """
    def __init__(
        self,
        bins: Union[int, torch.Tensor] = 10,
        min_value: Optional[float] = 0.0,
        max_value: Optional[float] = 1.0,
        soft_targets: bool = True,
        sigma: float = 0.1,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        loss_type: str = 'cross_entropy',
        focal_gamma: float = 2.0,
        **kwargs
    ):
        super().__init__(
            bins=bins,
            min_value=min_value,
            max_value=max_value,
            soft_targets=soft_targets,
            sigma=sigma,
            reduction=reduction,
            **kwargs
        )
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
    
    def _extract_distribution_parameters(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract distribution parameters from predicted logits."""
        bin_probs = F.softmax(y_pred, dim=1)
        
        return {
            'logits': y_pred,
            'bin_probs': bin_probs,
            'bin_centers': self.bin_centers,
            'bin_widths': self.bin_widths
        }
    
    def _calculate_nll(self, y_true: torch.Tensor, params: Dict[str, torch.Tensor], 
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate loss based on the selected loss type."""
        logits = params['logits']
        bin_probs = params['bin_probs']
        
        # Handle targets based on soft_targets setting
        if self.soft_targets:
            # y_true is already target_distribution from _get_target_distribution
            target_distribution = y_true
            
            if self.loss_type == 'kl_div':
                # KL divergence for soft targets
                log_probs = F.log_softmax(logits, dim=1)
                loss = F.kl_div(log_probs, target_distribution, reduction='none')
                loss = loss.sum(dim=1)
                
            elif self.loss_type == 'cross_entropy':
                # Cross entropy with soft targets
                log_probs = F.log_softmax(logits, dim=1)
                loss = -torch.sum(target_distribution * log_probs, dim=1)
                
            elif self.loss_type == 'focal':
                # Focal loss for soft targets
                log_probs = F.log_softmax(logits, dim=1)
                focal_weight = (1 - bin_probs) ** self.focal_gamma
                loss = -torch.sum(target_distribution * focal_weight * log_probs, dim=1)
                
            elif self.loss_type == 'nll':
                # Direct NLL of target under predicted distribution
                # Avoid log(0) by adding small epsilon
                loss = -torch.sum(target_distribution * torch.log(bin_probs + 1e-10), dim=1)
                
            else:
                raise ValueError(f"Unsupported loss_type: {self.loss_type}")
                
        else:
            # Hard targets (bin indices)
            # y_true contains bin indices from _get_target_distribution
            target_indices = y_true.long()
            
            if self.loss_type == 'cross_entropy':
                # Standard cross-entropy with optional label smoothing
                loss = F.cross_entropy(
                    logits, 
                    target_indices, 
                    reduction='none',
                    label_smoothing=self.label_smoothing
                )
                
            elif self.loss_type == 'focal':
                # Focal loss for hard targets
                log_probs = F.log_softmax(logits, dim=1)
                probs = torch.exp(log_probs)
                
                # Get probabilities for target classes
                target_probs = probs.gather(1, target_indices.unsqueeze(1)).squeeze()
                focal_weight = (1 - target_probs) ** self.focal_gamma
                
                # Apply focal weight to cross entropy
                loss = -focal_weight * torch.log(target_probs + 1e-10)
                
            else:
                # Default to cross-entropy for other loss types with hard targets
                loss = F.cross_entropy(
                    logits, 
                    target_indices, 
                    reduction='none',
                    label_smoothing=self.label_smoothing
                )
                
        return loss
    
    def forward(self, y_pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                weights: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate loss using classification approach.
        
        Args:
            y_pred: Predicted logits [batch_size, n_bins]
            target: Ground truth values [batch_size, 1]
            mask: Optional mask [batch_size]
            weights: Optional sample weights [batch_size]
            uncertainty: Optional uncertainty values for targets [batch_size, 1]
            
        Returns:
            Loss value after applying reduction
        """
        # Handle values outside bin range
        target = self._handle_out_of_range(target)
        
        # Convert targets to distributions or indices
        target_distribution = self._get_target_distribution(target, uncertainty)
        
        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred)
        
        # Calculate loss
        loss = self._calculate_nll(target_distribution, params, mask)
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
            
        # Apply mask and reduction
        return self._reduce(loss, mask)


class OrdinalRegressionLoss(BinnedRegressionLoss):
    """
    Ordinal regression loss for handling ordered categories.
    
    This loss uses binary encoding to represent the ordinal relationship
    between bins, which can improve performance for regression tasks.
    
    Args:
        bins (Union[int, torch.Tensor]): Number of bins or array of bin edges
            Default: 10
        min_value (float, optional): Minimum value for auto-generated bins
            Default: 0.0
        max_value (float, optional): Maximum value for auto-generated bins
            Default: 1.0
        soft_targets (bool): Whether to use soft targets (probability distributions)
            Default: True
        sigma (float): Standard deviation for soft targets
            Default: 0.1
        reduction (str): Method for reducing the loss ('none', 'mean', 'sum')
            Default: 'mean'
        loss_type (str): Type of loss to use ('bce', 'focal')
            Default: 'bce'
        focal_gamma (float): Gamma parameter for focal loss
            Default: 2.0
            
    Mathematical Formulation:
        Ordinal regression converts the problem into a series of binary classifications:
        "Is the value greater than threshold_k?" for each threshold k.
        
        For n bins, we have n-1 thresholds. A value in bin i will have:
        - 1 for thresholds k < i
        - 0 for thresholds k >= i
        
        With soft targets, we calculate expected binary encoding based on the
        probability distribution over bins.
    """
    def __init__(
        self,
        bins: Union[int, torch.Tensor] = 10,
        min_value: Optional[float] = 0.0,
        max_value: Optional[float] = 1.0,
        soft_targets: bool = True,
        sigma: float = 0.1,
        reduction: str = 'mean',
        loss_type: str = 'bce',
        focal_gamma: float = 2.0,
        **kwargs
    ):
        super().__init__(
            bins=bins,
            min_value=min_value,
            max_value=max_value,
            soft_targets=soft_targets,
            sigma=sigma,
            reduction=reduction,
            **kwargs
        )
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        
        # Create encoding matrix for ordinal regression
        encoding_matrix = create_ordinal_encoding_matrix(self.n_bins)
        self.register_buffer('encoding_matrix', encoding_matrix)
    
    def _extract_distribution_parameters(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract distribution parameters from binary logits."""
        # y_pred should be [batch_size, n_bins-1] binary logits
        binary_probs = torch.sigmoid(y_pred)
        
        # Calculate class probabilities from binary probabilities
        batch_size = y_pred.shape[0]
        probs = torch.zeros(batch_size, self.n_bins, 
                          device=y_pred.device, dtype=y_pred.dtype)
        
        # Calculate probabilities for each bin
        probs[:, 0] = 1 - binary_probs[:, 0]
        
        for k in range(1, self.n_bins-1):
            probs[:, k] = binary_probs[:, k-1] - binary_probs[:, k]
            
        probs[:, self.n_bins-1] = binary_probs[:, self.n_bins-2]
        
        # Ensure non-negative probabilities (numerical stability)
        probs = F.relu(probs)
        
        # Normalize to ensure valid distribution
        probs = probs / (torch.sum(probs, dim=1, keepdim=True) + 1e-10)
        
        return {
            'binary_logits': y_pred,
            'binary_probs': binary_probs,
            'bin_probs': probs,
            'bin_centers': self.bin_centers,
            'bin_widths': self.bin_widths
        }
    
    def _calculate_nll(self, y_true: torch.Tensor, params: Dict[str, torch.Tensor], 
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate ordinal regression loss."""
        binary_logits = params['binary_logits']
        
        # Convert targets to ordinal encoding
        if self.soft_targets:
            # y_true is already a distribution over bins
            # Convert to binary targets for each threshold
            ordinal_targets = convert_to_ordinal_targets(
                y_true, self.n_bins, True, self.encoding_matrix
            )
        else:
            # y_true contains bin indices
            ordinal_targets = convert_to_ordinal_targets(
                y_true, self.n_bins, False, self.encoding_matrix
            )
            
        # Calculate loss based on loss type
        if self.loss_type == 'bce':
            # Binary cross-entropy for each threshold
            loss = F.binary_cross_entropy_with_logits(
                binary_logits, ordinal_targets, reduction='none'
            )
            
        elif self.loss_type == 'focal':
            # Focal loss for binary targets
            binary_probs = torch.sigmoid(binary_logits)
            
            # Calculate pt (probability of true class)
            pt = torch.where(ordinal_targets == 1, binary_probs, 1 - binary_probs)
            
            # Calculate focal weight
            focal_weight = (1 - pt) ** self.focal_gamma
            
            # Calculate binary cross entropy
            bce = F.binary_cross_entropy_with_logits(
                binary_logits, ordinal_targets, reduction='none'
            )
            
            # Apply focal weight
            loss = focal_weight * bce
            
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
            
        # Sum across thresholds
        return loss.mean(dim=1)
    
    def forward(self, y_pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                weights: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate loss using ordinal regression approach.
        
        Args:
            y_pred: Predicted binary logits [batch_size, n_bins-1]
            target: Ground truth values [batch_size, 1]
            mask: Optional mask [batch_size]
            weights: Optional sample weights [batch_size]
            uncertainty: Optional uncertainty values for targets [batch_size, 1]
            
        Returns:
            Loss value after applying reduction
        """
        # Check output size
        if y_pred.shape[1] != self.n_bins - 1:
            raise ValueError(
                f"Expected y_pred to have {self.n_bins - 1} outputs for ordinal regression, "
                f"but got {y_pred.shape[1]}"
            )
            
        # Handle values outside bin range
        target = self._handle_out_of_range(target)
        
        # Convert targets to distributions or indices
        target_distribution = self._get_target_distribution(target, uncertainty)
        
        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred)
        
        # Calculate loss
        loss = self._calculate_nll(target_distribution, params, mask)
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
            
        # Apply mask and reduction
        return self._reduce(loss, mask)


class HistogramRegressionLoss(BinnedRegressionLoss):
    """
    Histogram regression loss for flexibility in capturing output distributions.
    
    This loss treats the output as a histogram (probability distribution) over bins,
    which is particularly useful for multi-modal distributions and uncertainty estimation.
    
    Args:
        bins (Union[int, torch.Tensor]): Number of bins or array of bin edges
            Default: 10
        min_value (float, optional): Minimum value for auto-generated bins
            Default: 0.0
        max_value (float, optional): Maximum value for auto-generated bins
            Default: 1.0
        soft_targets (bool): Whether to use soft targets (probability distributions)
            Default: True
        sigma (float): Standard deviation for soft targets
            Default: 0.1
        reduction (str): Method for reducing the loss ('none', 'mean', 'sum')
            Default: 'mean'
        loss_type (str): Type of loss ('kl_div', 'cross_entropy', 'wasserstein')
            Default: 'kl_div'
        normalize_targets (bool): Whether to normalize target distributions
            Default: True
        wasserstein_p (int): P parameter for Wasserstein distance
            Default: 1
            
    Mathematical Formulation:
        For a histogram regression, the target is represented as a probability 
        distribution over bins. Common distance metrics between distributions include:
        
        KL-divergence: ∑_i p(bin_i) * log(p(bin_i)/q(bin_i))
        Cross-entropy: -∑_i p(bin_i) * log(q(bin_i))
        Wasserstein-1: ∑_i |CDF_p(bin_i) - CDF_q(bin_i)|
        
        Where p is the target distribution and q is the predicted distribution.
    """
    def __init__(
        self,
        bins: Union[int, torch.Tensor] = 10,
        min_value: Optional[float] = 0.0,
        max_value: Optional[float] = 1.0,
        soft_targets: bool = True,
        sigma: float = 0.1,
        reduction: str = 'mean',
        loss_type: str = 'kl_div',
        normalize_targets: bool = True,
        wasserstein_p: int = 1,
        **kwargs
    ):
        super().__init__(
            bins=bins,
            min_value=min_value,
            max_value=max_value,
            soft_targets=soft_targets,
            sigma=sigma,
            reduction=reduction,
            normalize_targets=normalize_targets,
            **kwargs
        )
        self.loss_type = loss_type
        self.wasserstein_p = wasserstein_p
        
    def _extract_distribution_parameters(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract distribution parameters from predicted logits or probabilities."""
        # Check if input is already probabilities
        if torch.all((y_pred >= 0) & (y_pred <= 1)):
            # Ensure distribution sums to 1
            bin_probs = y_pred / (torch.sum(y_pred, dim=1, keepdim=True) + 1e-10)
            logits = torch.log(bin_probs + 1e-10)
        else:
            # Convert logits to probabilities
            bin_probs = F.softmax(y_pred, dim=1)
            logits = y_pred
            
        return {
            'logits': logits,
            'bin_probs': bin_probs,
            'bin_centers': self.bin_centers,
            'bin_widths': self.bin_widths
        }
    
    def _calculate_nll(self, y_true: torch.Tensor, params: Dict[str, torch.Tensor], 
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate histogram loss based on the selected loss type."""
        logits = params['logits']
        bin_probs = params['bin_probs']
        
        # For histogram loss, target is always a distribution (whether soft or hard)
        target_distribution = y_true
        
        if self.loss_type == 'kl_div':
            # KL divergence
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, target_distribution, reduction='none')
            loss = loss.sum(dim=1)
            
        elif self.loss_type == 'cross_entropy':
            # Cross entropy
            log_probs = F.log_softmax(logits, dim=1)
            loss = -torch.sum(target_distribution * log_probs, dim=1)
            
        elif self.loss_type == 'wasserstein':
            # Approximate Wasserstein distance using histogram densities
            # This uses a simple EMD approximation for 1D histograms
            
            # Calculate cumulative distributions
            target_cdf = torch.cumsum(target_distribution, dim=1)
            pred_cdf = torch.cumsum(bin_probs, dim=1)
            
            # Calculate Wasserstein distance
            if self.wasserstein_p == 1:
                # W1 distance (absolute differences)
                loss = torch.sum(torch.abs(target_cdf - pred_cdf), dim=1) 
            else:
                # Wp distance
                loss = torch.sum(torch.abs(target_cdf - pred_cdf) ** self.wasserstein_p, dim=1) ** (1.0 / self.wasserstein_p)
                
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
            
        return loss
    
    def forward(self, y_pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                weights: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate histogram regression loss.
        
        Args:
            y_pred: Predicted logits or probabilities [batch_size, n_bins]
            target: Ground truth values [batch_size, 1]
            mask: Optional mask [batch_size]
            weights: Optional sample weights [batch_size]
            uncertainty: Optional uncertainty values for targets [batch_size, 1]
            
        Returns:
            Loss value after applying reduction
        """
        # Check output size
        if y_pred.shape[1] != self.n_bins:
            raise ValueError(
                f"Expected y_pred to have {self.n_bins} outputs for histogram regression, "
                f"but got {y_pred.shape[1]}"
            )
            
        # Handle values outside bin range
        target = self._handle_out_of_range(target)
        
        # For histogram loss, always get a distribution (even for hard targets)
        if not self.soft_targets:
            # Convert indices to one-hot
            bin_indices = torch.bucketize(target.squeeze(), self.bin_edges) - 1
            bin_indices = torch.clamp(bin_indices, max=self.n_bins-1)
            
            # Convert to one-hot
            target_distribution = F.one_hot(bin_indices, self.n_bins).float()
        else:
            # Get soft distribution
            target_distribution = self._get_target_distribution(target, uncertainty)
        
        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred)
        
        # Calculate loss
        loss = self._calculate_nll(target_distribution, params, mask)
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
            
        # Apply mask and reduction
        return self._reduce(loss, mask)


def create_binned_regression_loss(
    method: str = 'auto',
    bins: Union[int, torch.Tensor] = 10,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    data: Optional[torch.Tensor] = None,
    noise_aware: bool = False,
    ordinal_for_smooth: bool = True,
    **kwargs
) -> BinnedRegressionLoss:
    """
    Factory function to create the most appropriate binned regression loss.
    
    Args:
        method (str): Loss method ('auto', 'classification', 'ordinal', 'histogram')
            Default: 'auto'
        bins (Union[int, torch.Tensor]): Number of bins or array of bin edges
            Default: 10
        min_value (float, optional): Minimum value for auto-generated bins
            Default: None
        max_value (float, optional): Maximum value for auto-generated bins
            Default: None
        data (torch.Tensor, optional): Data for automatic bin range detection
            Default: None
        noise_aware (bool): Whether to adjust for noisy labels
            Default: False
        ordinal_for_smooth (bool): Use ordinal regression for smooth/continuous targets
            Default: True
        **kwargs: Additional arguments for the specific loss class
        
    Returns:
        Appropriate binned regression loss instance
        
    Examples:
        >>> # Create a binned regression loss with automatic bin detection
        >>> loss_fn = create_binned_regression_loss(
        ...     method='histogram',
        ...     bins=20,
        ...     data=training_data
        ... )
        >>> # Use the loss function
        >>> y_pred = model(x)
        >>> loss = loss_fn(y_pred, target)
    """
    # Initialize bins if min/max not provided
    if (min_value is None or max_value is None) and isinstance(bins, int):
        bin_edges = create_bin_edges(bins, min_value, max_value, data)
    else:
        bin_edges = bins
    
    # Determine the best method if 'auto' is specified
    if method == 'auto':
        if noise_aware:
            # For noisy labels, histogram regression is generally more robust
            method = 'histogram'
        elif ordinal_for_smooth and kwargs.get('soft_targets', True):
            # For smooth targets, ordinal regression is usually better
            method = 'ordinal'
        else:
            # Default to standard classification approach
            method = 'classification'
    
    # Create the requested loss function
    if method == 'classification':
        return StandardClassificationRegressionLoss(
            bins=bin_edges, 
            min_value=min_value, 
            max_value=max_value,
            noise_aware=noise_aware,
            **kwargs
        )
    elif method == 'ordinal':
        return OrdinalRegressionLoss(
            bins=bin_edges, 
            min_value=min_value, 
            max_value=max_value,
            noise_aware=noise_aware,
            **kwargs
        )
    elif method == 'histogram':
        return HistogramRegressionLoss(
            bins=bin_edges, 
            min_value=min_value, 
            max_value=max_value,
            noise_aware=noise_aware,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

def create_classification_regression_loss(
    bins: Union[int, torch.Tensor] = 10, 
    **kwargs
) -> StandardClassificationRegressionLoss:
    """Create a classification-based regression loss."""
    return StandardClassificationRegressionLoss(bins=bins, **kwargs)

def create_ordinal_regression_loss(
    bins: Union[int, torch.Tensor] = 10, 
    **kwargs
) -> OrdinalRegressionLoss:
    """Create an ordinal regression loss."""
    return OrdinalRegressionLoss(bins=bins, **kwargs)

def create_histogram_regression_loss(
    bins: Union[int, torch.Tensor] = 10, 
    **kwargs
) -> HistogramRegressionLoss:
    """Create a histogram regression loss."""
    return HistogramRegressionLoss(bins=bins, **kwargs)

def create_noise_aware_regression_loss(
    bins: Union[int, torch.Tensor] = 20,
    method: str = 'histogram',
    **kwargs
) -> BinnedRegressionLoss:
    """Create a regression loss designed for noisy labels."""
    return create_binned_regression_loss(
        method=method,
        bins=bins,
        noise_aware=True,
        adaptive_sigma=True,
        **kwargs
    )

def regression_as_classification(
    bins: Union[int, torch.Tensor] = 15,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    smooth_targets: bool = True,
    robust_to_noise: bool = False,
    auto_adapt: bool = True,
    **advanced_config
) -> BinnedRegressionLoss:
    """
    Convert a regression problem into classification for better uncertainty modeling.
    
    This creates a loss function that treats regression as a classification problem
    by binning continuous values and predicting probability distributions over those
    bins. This approach provides better uncertainty quantification, can model 
    multi-modal distributions, and is more robust to noisy labels.
    
    Args:
        bins (Union[int, torch.Tensor]): Number of bins or explicit bin edges
            Default: 15
        min_value (float, optional): Minimum value (auto-detected from data if None)
            Default: None
        max_value (float, optional): Maximum value (auto-detected from data if None)
            Default: None
        smooth_targets (bool): Use smooth distributions instead of hard bin assignments
            Default: True
        robust_to_noise (bool): Make the loss robust to noisy labels
            Default: False
        auto_adapt (bool): Automatically adjust parameters based on data characteristics
            Default: True
        **advanced_config: Additional options for fine-tuning
    
    Returns:
        A loss function that converts regression to classification
        
    Examples:
        >>> # Basic usage with default settings
        >>> loss_fn = regression_as_classification(bins=15)
        >>> 
        >>> # For noisy data with automatic parameter optimization
        >>> loss_fn = regression_as_classification(
        ...     bins=20,
        ...     robust_to_noise=True,
        ...     auto_adapt=True
        ... )
        >>> 
        >>> # With auto-detection of value range from training data
        >>> loss_fn = regression_as_classification(
        ...     bins=10, 
        ...     min_value=None, 
        ...     max_value=None,
        ...     data=training_targets
        ... )
    """
    # Set intelligent defaults
    sigma = advanced_config.pop('sigma', 0.1)
    loss_type = advanced_config.pop('loss_type', 'cross_entropy')
    order_aware = advanced_config.pop('order_aware', True)
    adaptive_sigma = advanced_config.pop('adaptive_sigma', auto_adapt)
    
    # Adjust sigma based on bin count (more bins = smaller sigma)
    if auto_adapt and isinstance(bins, int) and 'sigma' not in advanced_config:
        # For more bins, we want a narrower distribution
        if bins > 20:
            sigma = 0.07
        elif bins < 10:
            sigma = 0.15
    
    # For noisy data, use more robust settings
    if robust_to_noise:
        # Use a more robust loss type
        if loss_type == 'cross_entropy':
            loss_type = 'kl_div'  # KL divergence handles noise better
        
        # Increase sigma for smoother target distributions
        if 'sigma' not in advanced_config:
            sigma = max(sigma, 0.12)  # Wider sigma for noisy data
            
        # Enable adaptive sigma
        adaptive_sigma = True
    
    # Create the smart regression loss
    return RegressionAsClassificationLoss(
        bins=bins,
        min_value=min_value,
        max_value=max_value,
        order_aware=order_aware,
        smooth_targets=smooth_targets,
        sigma=sigma,
        loss_type=loss_type,
        adaptive_sigma=adaptive_sigma,
        **advanced_config
    )

def uncertainty_regression(
    bins: int = 20,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    **kwargs
) -> BinnedRegressionLoss:
    """
    Create a regression loss that captures prediction uncertainty.
    
    This is a specialized version of regression_as_classification optimized
    for uncertainty estimation in regression problems.
    
    Args:
        bins: Number of bins to divide the output range
        min_value: Minimum value (auto-detected from data if None)
        max_value: Maximum value (auto-detected from data if None)
        **kwargs: Additional configuration options
    
    Returns:
        A loss function for uncertainty-aware regression
    """
    return regression_as_classification(
        bins=bins,
        min_value=min_value,
        max_value=max_value,
        smooth_targets=True,
        robust_to_noise=True,
        **kwargs
    )

class RegressionAsClassificationLoss(BinnedRegressionLoss):
    """
    Unified regression-as-classification loss.
    
    This loss function combines the benefits of histogram binning, ordinal regression,
    and soft targets to create a powerful and flexible approach for regression problems.
    
    Args:
        bins (Union[int, torch.Tensor]): Number of bins or array of bin edges
            Default: 15
        min_value (float, optional): Minimum value for auto-generated bins
            Default: 0.0
        max_value (float, optional): Maximum value for auto-generated bins
            Default: 1.0
        order_aware (bool): Whether to use ordinal encoding to encode bin ordering
            Default: True
        smooth_targets (bool): Whether to use smooth target distributions
            Default: True
        sigma (float): Standard deviation for soft targets
            Default: 0.1
        reduction (str): Method for reducing the loss
            Default: 'mean'
        loss_type (str): Type of loss function ('cross_entropy', 'kl_div', 'focal', 'wasserstein')
            Default: 'cross_entropy'
        adaptive_sigma (bool): Whether to adjust sigma based on bin widths
            Default: True
        focal_gamma (float): Gamma parameter for focal loss
            Default: 2.0
            
    Mathematical Formulation:
        This unified approach supports both classification-style and ordinal-style regression:
        
        - For standard mode (n_bins outputs): Uses classification-style losses
        - For ordinal mode (n_bins-1 outputs): Uses ordinal binary encoding
        
        The ordinal encoding represents each bin i with a binary vector where
        the first i elements are 1 and the remaining are 0.
    """
    def __init__(
        self,
        bins: Union[int, torch.Tensor] = 15,
        min_value: Optional[float] = 0.0,
        max_value: Optional[float] = 1.0,
        order_aware: bool = True,
        smooth_targets: bool = True,
        sigma: float = 0.1,
        reduction: str = 'mean',
        loss_type: str = 'cross_entropy',
        adaptive_sigma: bool = True,
        focal_gamma: float = 2.0,
        **kwargs
    ):
        super().__init__(
            bins=bins,
            min_value=min_value,
            max_value=max_value,
            soft_targets=smooth_targets,
            sigma=sigma,
            reduction=reduction,
            adaptive_sigma=adaptive_sigma,
            **kwargs
        )
        self.order_aware = order_aware
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        
        # Create ordinal encoding matrix if needed
        if self.order_aware:
            encoding_matrix = create_ordinal_encoding_matrix(self.n_bins)
            self.register_buffer('encoding_matrix', encoding_matrix)
    
    def _extract_distribution_parameters(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract distribution parameters from model outputs."""
        # For order-aware mode with ordinal encoding
        if self.order_aware and y_pred.shape[1] == self.n_bins - 1:
            # Convert binary logits to class probabilities
            binary_probs = torch.sigmoid(y_pred)
            
            # Calculate class probabilities from binary probabilities
            probs = torch.zeros(y_pred.shape[0], self.n_bins, 
                              device=y_pred.device, dtype=y_pred.dtype)
            
            # First bin probability = 1 - prob of exceeding first threshold
            probs[:, 0] = 1 - binary_probs[:, 0]
            
            # Middle bins = prob of exceeding k-1 threshold minus prob of exceeding k threshold
            for k in range(1, self.n_bins-1):
                probs[:, k] = binary_probs[:, k-1] - binary_probs[:, k]
                
            # Last bin probability = prob of exceeding last threshold
            probs[:, self.n_bins-1] = binary_probs[:, self.n_bins-2]
            
            # Ensure non-negative probabilities (numerical stability)
            probs = F.relu(probs)
            
            # Normalize to ensure valid distribution
            probs = probs / (torch.sum(probs, dim=1, keepdim=True) + 1e-10)
            
            return {
                'bin_probs': probs,
                'binary_probs': binary_probs,
                'binary_logits': y_pred,
                'bin_centers': self.bin_centers,
                'bin_widths': self.bin_widths
            }
        else:
            # Standard classification mode
            if torch.all((y_pred >= 0) & (y_pred <= 1)):
                # Input is already probabilities
                bin_probs = y_pred / (torch.sum(y_pred, dim=1, keepdim=True) + 1e-10)
                logits = torch.log(bin_probs + 1e-10)
            else:
                # Input is logits
                logits = y_pred
                bin_probs = F.softmax(y_pred, dim=1)
                
            return {
                'bin_probs': bin_probs,
                'logits': logits,
                'bin_centers': self.bin_centers,
                'bin_widths': self.bin_widths
            }
    
    def forward(self, y_pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                weights: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate regression loss using classification approach.
        
        Args:
            y_pred: Predicted values [batch_size, n_bins] or [batch_size, n_bins-1]
                   for ordinal mode
            target: Ground truth values [batch_size, 1]
            mask: Optional mask [batch_size]
            weights: Optional sample weights [batch_size]
            uncertainty: Optional uncertainty values [batch_size, 1]
            
        Returns:
            Loss value after applying reduction
        """
        # Apply masking
        if mask is not None:
            # Create safe copies
            target = target.clone()
            y_pred = y_pred.clone()
            
            # Apply mask
            if mask.dim() < target.dim():
                mask = mask.unsqueeze(-1).expand_as(target)
            
            # Set masked values to a specific value that won't affect the loss
            # Since we'll exclude these later in reduction
            if not self.soft_targets:
                # For hard targets, just keep the mask
                pass
            else:
                # For soft targets, zero out masked regions
                target[~mask] = 0
        
        # Handle values outside bin range
        target = self._handle_out_of_range(target)
        
        # Check output size
        if self.order_aware and y_pred.shape[1] == self.n_bins - 1:
            # Ordinal mode
            expected_dim = self.n_bins - 1
            if y_pred.shape[1] != expected_dim:
                raise ValueError(
                    f"For ordinal mode, expected y_pred with {expected_dim} outputs, got {y_pred.shape[1]}"
                )
        elif y_pred.shape[1] != self.n_bins:
            raise ValueError(
                f"Expected y_pred with {self.n_bins} outputs, got {y_pred.shape[1]}"
            )
        
        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred)
        
        # Get target distribution
        target_distribution = self._get_target_distribution(target, uncertainty)
        
        # Calculate loss based on configuration
        if self.order_aware and y_pred.shape[1] == self.n_bins - 1:
            # Ordinal regression approach
            binary_logits = params['binary_logits']
            
            # Convert targets to ordinal encoding
            if self.soft_targets:
                # For soft targets, calculate expected ordinal encoding
                batch_size = target.shape[0]
                ordinal_targets = torch.bmm(
                    target_distribution.unsqueeze(1),
                    self.encoding_matrix.unsqueeze(0).expand(batch_size, -1, -1)
                ).squeeze(1)
            else:
                # For hard targets, directly create binary targets
                bin_indices = torch.bucketize(target.squeeze(), self.bin_edges) - 1
                bin_indices = torch.clamp(bin_indices, max=self.n_bins-1)
                
                batch_size = target.shape[0]
                ordinal_targets = torch.zeros(batch_size, self.n_bins-1, 
                                            device=target.device, dtype=torch.float32)
                for i in range(batch_size):
                    ordinal_targets[i, :bin_indices[i]] = 1.0
            
            # Binary cross-entropy for ordinal targets
            loss = F.binary_cross_entropy_with_logits(
                binary_logits,
                ordinal_targets,
                reduction='none'
            )
            
            # Average over thresholds dimension
            loss = loss.mean(dim=1)
        else:
            # Standard classification approach
            if self.soft_targets:
                # For soft targets, use distribution-based losses
                log_probs = F.log_softmax(params['logits'], dim=1)
                bin_probs = params['bin_probs']
                
                if self.loss_type == 'cross_entropy':
                    # Cross-entropy with soft targets
                    loss = -torch.sum(target_distribution * log_probs, dim=1)
                    
                elif self.loss_type == 'kl_div':
                    # KL divergence
                    loss = F.kl_div(log_probs, target_distribution, reduction='none')
                    loss = loss.sum(dim=1)
                    
                elif self.loss_type == 'focal':
                    # Focal loss for soft targets
                    focal_weight = (1 - bin_probs) ** self.focal_gamma
                    loss = -torch.sum(target_distribution * focal_weight * log_probs, dim=1)
                    
                elif self.loss_type == 'wasserstein':
                    # Approximate 1D Wasserstein distance
                    target_cdf = torch.cumsum(target_distribution, dim=1)
                    pred_cdf = torch.cumsum(bin_probs, dim=1)
                    loss = torch.sum(torch.abs(target_cdf - pred_cdf), dim=1)
                    
                else:
                    raise ValueError(f"Unsupported loss_type: {self.loss_type}")
            else:
                # For hard targets, use standard classification losses
                bin_indices = torch.bucketize(target.squeeze(), self.bin_edges) - 1
                bin_indices = torch.clamp(bin_indices, max=self.n_bins-1)
                
                if self.loss_type == 'cross_entropy':
                    # Standard cross-entropy for classification
                    loss = F.cross_entropy(
                        params['logits'],
                        bin_indices.long(),
                        reduction='none'
                    )
                elif self.loss_type == 'focal':
                    # Focal loss for hard targets
                    log_probs = F.log_softmax(params['logits'], dim=1)
                    
                    # Get target probabilities
                    target_probs = params['bin_probs'].gather(1, bin_indices.long().unsqueeze(1)).squeeze()
                    
                    # Apply focal weighting
                    focal_weight = (1 - target_probs) ** self.focal_gamma
                    ce_loss = F.cross_entropy(
                        params['logits'],
                        bin_indices.long(),
                        reduction='none'
                    )
                    loss = focal_weight * ce_loss
                else:
                    # Default to cross-entropy
                    loss = F.cross_entropy(
                        params['logits'],
                        bin_indices.long(),
                        reduction='none'
                    )
        
        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights
            
        # Apply mask and reduction
        return self._reduce(loss, mask)