"""
Normalizing Flow loss functions for regression tasks.

This module provides loss functions for regression models that use
normalizing flows to model complex output distributions.
Uses the zuko package for efficient implementation of various flows.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple, Dict, Any, List, Callable

try:
    import zuko
    from zuko.flows import Flow, MAF, RealNVP, NSF, IAF
    ZUKO_AVAILABLE = True
except ImportError:
    ZUKO_AVAILABLE = False

from .base import DistributionLoss
from ..utils.tensor_ops import apply_mask, masked_reduction
from ..utils.validation import validate_reduction


class NormalizingFlowLoss(DistributionLoss):
    """
    Negative Log-Likelihood loss for normalizing flow models using zuko.
    
    This loss allows modeling complex multi-dimensional target distributions
    for regression tasks using various normalizing flow architectures.
    
    Args:
        n_features (int): Number of output features (dimensions)
        flow_type (str): Type of flow to use ('realnvp', 'maf', 'nsf', 'iaf')
        n_blocks (int): Number of transformation blocks in the flow
        hidden_features (int): Size of hidden layers in coupling/autoregressive networks
        n_hidden_layers (int): Number of hidden layers in coupling/autoregressive networks
        base_distribution (str): Base distribution ('normal' or 'uniform')
        activation (str): Activation function for hidden layers
        dropout (float): Dropout rate for hidden layers
        batch_norm (bool): Whether to use batch normalization
        reduction (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'
    """
    def __init__(
        self,
        n_features: int,
        flow_type: str = 'realnvp',
        n_blocks: int = 3,
        hidden_features: int = 64,
        n_hidden_layers: int = 2,
        base_distribution: str = 'normal',
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = False,
        reduction: str = 'mean'
    ):
        if not ZUKO_AVAILABLE:
            raise ImportError("zuko package is required for NormalizingFlowLoss but not installed")
            
        super().__init__(reduction=reduction)
        self.n_features = n_features
        self.flow_type = flow_type.lower()
        self.n_blocks = n_blocks
        self.hidden_features = hidden_features
        self.n_hidden_layers = n_hidden_layers
        self.base_distribution = base_distribution.lower()
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        # Map of supported flow types
        self.FLOW_TYPES = {
            'realnvp': RealNVP,
            'maf': MAF,
            'nsf': NSF,
            'iaf': IAF
        }
        
        if self.flow_type not in self.FLOW_TYPES:
            raise ValueError(f"Unsupported flow_type: {flow_type}. Supported types: {list(self.FLOW_TYPES.keys())}")
        
        # Different flow types require different output parameters
        if self.flow_type == 'realnvp':
            # For RealNVP: scale and shift nets for each block, each d-dimensional
            self.params_per_block = 2 * n_features  # Scale and shift
        elif self.flow_type in ['maf', 'iaf']:
            # For MAF/IAF: autoregressive network parameters for each block
            self.params_per_block = n_features * (n_features + 1)  # Approx for AR networks
        elif self.flow_type == 'nsf':
            # For NSF: rational-quadratic spline parameters for each block
            # Bins per dimension + bin widths + heights + derivatives
            self.params_per_block = n_features * 3 * hidden_features  # Approximate
        else:
            # Default conservative estimate
            self.params_per_block = 2 * n_features
            
        self.expected_output_size = self.params_per_block * n_blocks
        
    def _create_flow(self, params_dict):
        """
        Create a zuko flow with the specified parameters.
        
        Args:
            params_dict: Dictionary of flow parameters extracted from model output
            
        Returns:
            zuko.flows.Flow: Configured normalizing flow
        """
        flow_class = self.FLOW_TYPES[self.flow_type]
        
        # Create the flow with extracted parameters
        flow = flow_class(
            features=self.n_features,
            transforms=self.n_blocks,
            hidden_features=self.hidden_features,
            hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            base=self.base_distribution
        )
        
        # Set the flow parameters from the model output
        with torch.no_grad():
            flow_state_dict = flow.state_dict()
            for name, param in flow_state_dict.items():
                if name in params_dict:
                    param.copy_(params_dict[name])
                    
        return flow
        
    def _extract_distribution_parameters(self, y_pred):
        """
        Extract normalizing flow parameters from model predictions.
        
        Args:
            y_pred: Model predictions containing flow parameters
        
        Returns:
            dict: Mapping of parameter names to tensors
        """
        batch_size = y_pred.shape[0]
        
        # Create a mock flow to get parameter shapes and names
        mock_flow = self.FLOW_TYPES[self.flow_type](
            features=self.n_features,
            transforms=self.n_blocks,
            hidden_features=self.hidden_features,
            hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            batch_norm=self.batch_norm
        )
        
        # Get parameter shapes from mock flow
        param_shapes = {name: p.shape for name, p in mock_flow.named_parameters()}
        params_dict = {}
        
        # Slice the output tensor into parameters for the flow
        start_idx = 0
        for name, shape in param_shapes.items():
            param_size = torch.prod(torch.tensor(shape)).item()
            
            # Extract and reshape parameter
            end_idx = start_idx + param_size
            param_flat = y_pred[:, start_idx:end_idx]
            param_reshaped = param_flat.reshape(batch_size, *shape)
            
            params_dict[name] = param_reshaped
            start_idx = end_idx
            
            if start_idx >= y_pred.shape[1]:
                break
                
        return params_dict
    
    def _calculate_nll(self, y_true, params, mask=None):
        """
        Calculate negative log likelihood using a zuko flow.
        
        Args:
            y_true: Target values
            params: Flow parameters dictionary
            mask: Optional mask for valid values
            
        Returns:
            Negative log likelihood values
        """
        batch_size = y_true.shape[0]
        nlls = []
        
        # Process in minibatches to avoid memory issues with large batch sizes
        for i in range(0, batch_size, 32):
            # Get batch slice
            batch_end = min(i + 32, batch_size)
            y_batch = y_true[i:batch_end]
            batch_params = {name: param[i:batch_end] for name, param in params.items()}
            batch_mask = None if mask is None else mask[i:batch_end]
            
            # Create flow for this minibatch
            flow = self._create_flow(batch_params)
            
            # Calculate NLL for the minibatch
            with torch.no_grad():
                if batch_mask is not None:
                    # Handle masking on sample level
                    valid_samples = batch_mask.all(dim=1) if batch_mask.dim() > 1 else batch_mask
                    if valid_samples.sum() > 0:
                        # Only calculate for valid samples
                        valid_y = y_batch[valid_samples]
                        batch_nll = -flow.log_prob(valid_y)
                        
                        # Create full batch result with zeros for masked samples
                        full_batch_nll = torch.zeros(batch_end - i, device=y_batch.device)
                        full_batch_nll[valid_samples] = batch_nll
                        nlls.append(full_batch_nll)
                    else:
                        # All samples masked
                        nlls.append(torch.zeros(batch_end - i, device=y_batch.device))
                else:
                    # No masking
                    batch_nll = -flow.log_prob(y_batch)
                    nlls.append(batch_nll)
        
        # Combine results from all minibatches
        return torch.cat(nlls, dim=0)
    
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate normalizing flow negative log-likelihood loss.
        
        Args:
            y_true: Ground truth values [..., n_features]
            y_pred: Model predictions [..., expected_output_size]
            mask: Optional boolean mask [..., n_features] or [...] for sample-level
            weights: Optional sample weights
            
        Returns:
            Negative log-likelihood loss
        """
        # Basic input validation
        if y_true.shape[-1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features in y_true, got {y_true.shape[-1]}")
        
        # Extract flow parameters
        flow_params = self._extract_distribution_parameters(y_pred)
        
        # Convert feature-level mask to sample-level mask if needed
        sample_mask = None
        if mask is not None:
            if mask.dim() > 1 and mask.shape[-1] > 1:
                # If any feature is masked, the whole sample is considered invalid
                sample_mask = mask.all(dim=-1)
            else:
                sample_mask = mask
        
        # Calculate negative log likelihood (per sample)
        nll = self._calculate_nll(y_true, flow_params, sample_mask)
        
        # Apply sample weights if provided
        if weights is not None:
            if weights.dim() > 1 and weights.shape[-1] > 1:
                # Convert feature-level weights to sample-level (average)
                sample_weights = weights.mean(dim=-1)
            else:
                sample_weights = weights
            nll = nll * sample_weights
        
        # Apply mask and reduction
        if sample_mask is not None:
            if self.reduction == 'none':
                return nll * sample_mask  # Zero out invalid samples
            elif self.reduction == 'mean':
                valid_count = sample_mask.sum().clamp(min=1)  # Avoid division by zero
                return (nll * sample_mask).sum() / valid_count
            else:  # 'sum'
                return (nll * sample_mask).sum()
        else:
            if self.reduction == 'none':
                return nll
            elif self.reduction == 'mean':
                return nll.mean()
            else:  # 'sum'
                return nll.sum()
    
    def sample(self, y_pred, n_samples=1):
        """
        Generate samples from the predicted distribution.
        
        Args:
            y_pred: Model predictions containing flow parameters
            n_samples: Number of samples to generate per prediction
            
        Returns:
            Samples from the predicted distribution
        """
        batch_size = y_pred.shape[0]
        samples_list = []
        
        # Extract flow parameters
        flow_params = self._extract_distribution_parameters(y_pred)
        
        # Process in minibatches to avoid memory issues with large batch sizes
        for i in range(0, batch_size, 32):
            # Get batch slice
            batch_end = min(i + 32, batch_size)
            batch_params = {name: param[i:batch_end] for name, param in flow_params.items()}
            
            # Create flow for this minibatch
            flow = self._create_flow(batch_params)
            
            # Sample from the flow
            with torch.no_grad():
                batch_samples = flow.sample((n_samples,))  # [n_samples, batch_size, n_features]
                # Reshape to [batch_size, n_samples, n_features]
                batch_samples = batch_samples.transpose(0, 1)
                
            samples_list.append(batch_samples)
        
        # Combine samples from all minibatches
        return torch.cat(samples_list, dim=0)


def create_flow_loss(
    n_features: int,
    flow_type: str = 'realnvp',
    n_blocks: int = 3,
    hidden_features: int = 64,
    n_hidden_layers: int = 2,
    base_distribution: str = 'normal',
    activation: str = 'relu',
    dropout: float = 0.0,
    batch_norm: bool = False,
    reduction: str = 'mean'
) -> DistributionLoss:
    """
    Factory function to create a normalizing flow loss using zuko.
    
    Args:
        n_features: Number of output features (dimensions)
        flow_type: Type of flow ('realnvp', 'maf', 'nsf', 'iaf')
        n_blocks: Number of transformation blocks
        hidden_features: Size of hidden layers in networks
        n_hidden_layers: Number of hidden layers
        base_distribution: Base distribution ('normal' or 'uniform')
        activation: Activation function for networks ('relu', 'tanh', etc.)
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization
        reduction: 'none' | 'mean' | 'sum'
    
    Returns:
        An appropriate normalizing flow loss object
    """
    if not ZUKO_AVAILABLE:
        raise ImportError(
            "zuko package is required for normalizing flow losses. "
            "Install it with: pip install zuko"
        )
        
    return NormalizingFlowLoss(
        n_features=n_features,
        flow_type=flow_type,
        n_blocks=n_blocks,
        hidden_features=hidden_features,
        n_hidden_layers=n_hidden_layers,
        base_distribution=base_distribution,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
        reduction=reduction
    )
