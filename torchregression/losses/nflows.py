"""
Normalizing Flow loss functions for regression tasks.

This module provides loss functions for regression models that use
normalizing flows to model complex output distributions.
Uses the zuko package for efficient implementation of various flows.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, List, Any

try:
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
            raise ImportError(
                "The zuko package is required for NormalizingFlowLoss. "
                "Install it with: pip install zuko"
            )
            
        super().__init__(reduction=reduction)
        self.n_features = n_features
        self.flow_type = flow_type.lower()
        self.n_blocks = n_blocks
        self.hidden_features = hidden_features
        self.n_hidden_layers = n_hidden_layers
        self.base_distribution = base_distribution
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        # Validate flow type
        self._validate_flow_type()
        
    def _validate_flow_type(self):
        """Validate the flow type selection."""
        valid_types = ['realnvp', 'maf', 'nsf', 'iaf']
        if self.flow_type not in valid_types:
            raise ValueError(
                f"Invalid flow_type: {self.flow_type}. "
                f"Must be one of: {', '.join(valid_types)}"
            )
        
    def _create_flow(self, params_dict):
        """
        Create a flow model from parameters dictionary.
        
        Args:
            params_dict: Dictionary of flow parameters from model output
            
        Returns:
            Flow model instance
        """
        # Common arguments for all flows
        common_args = {
            'features': self.n_features,
            'transforms': self.n_blocks,
            'hidden_features': self.hidden_features,
            'hidden_layers': self.n_hidden_layers,
            'activation': self.activation,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'base_dist': self.base_distribution
        }
        
        # Create flow based on type
        if self.flow_type == 'realnvp':
            flow = RealNVP(**common_args)
        elif self.flow_type == 'maf':
            flow = MAF(**common_args)
        elif self.flow_type == 'nsf':
            flow = NSF(**common_args)
        elif self.flow_type == 'iaf':
            flow = IAF(**common_args)
        
        # Set the flow parameters
        flow.set_parameters(params_dict)
        return flow
    
    def _extract_distribution_parameters(self, y_pred):
        """
        Extract flow parameters from model predictions.
        
        Args:
            y_pred: Model predictions containing flow parameters
            
        Returns:
            Dictionary of flow parameters
        """
        # If y_pred is already a dictionary of parameters, return it directly
        if isinstance(y_pred, dict):
            return y_pred
            
        # Otherwise, we assume the model outputs the serialized parameters directly
        # We'll create a valid parameter dictionary expected by zuko
        return {'params': y_pred}
    
    def _calculate_nll(self, y_true, params, mask=None):
        """
        Calculate negative log-likelihood for the flow model.
        
        Args:
            y_true: Target values [batch_size, n_features]
            params: Flow parameters
            mask: Optional mask [batch_size, n_features]
            
        Returns:
            Negative log-likelihood [batch_size]
        """
        # Create flow
        flow = self._create_flow(params)
        
        # Calculate log probability
        log_prob = flow.log_prob(y_true)
        
        # Return negative log-likelihood
        return -log_prob
    
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate normalizing flow negative log-likelihood loss.
        
        Args:
            y_true: Ground truth values [batch_size, n_features]
            y_pred: Flow parameters from model
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional sample weights [batch_size]
            
        Returns:
            Negative log-likelihood loss
        """
        # Apply mask if provided
        y_true = apply_mask(y_true, mask)
        
        # Extract parameters
        params = self._extract_distribution_parameters(y_pred)
        
        # Calculate negative log-likelihood
        nll = self._calculate_nll(y_true, params, mask)
        
        # Apply weights if provided
        if weights is not None:
            # Handle different weight shapes
            if weights.dim() > 1 and weights.shape[1] > 1:
                # Average across features if weights are per-feature
                weights = weights.mean(dim=1)
            nll = nll * weights
            
        # Apply reduction
        return masked_reduction(nll, mask, self.reduction)
    
    def sample(self, y_pred, n_samples=1):
        """
        Generate samples from the flow distribution.
        
        Args:
            y_pred: Flow parameters from model
            n_samples: Number of samples to generate per input
            
        Returns:
            Samples [batch_size, n_samples, n_features]
        """
        # Extract parameters
        params = self._extract_distribution_parameters(y_pred)
        
        # Create flow
        flow = self._create_flow(params)
        
        # Generate samples - shape depends on the flow implementation
        # Typically [batch_size, n_samples, n_features] or [batch_size * n_samples, n_features]
        samples = flow.sample(n_samples)
        
        # Reshape if needed to ensure [batch_size, n_samples, n_features]
        if samples.dim() == 2:
            batch_size = params['params'].shape[0] if 'params' in params else y_pred.shape[0]
            samples = samples.reshape(batch_size, n_samples, self.n_features)
            
        return samples


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
    Factory function to create a normalizing flow loss.
    
    Args:
        n_features: Number of features in the target
        flow_type: Type of flow ('realnvp', 'maf', 'nsf', 'iaf')
        n_blocks: Number of transformation blocks
        hidden_features: Size of hidden layers
        n_hidden_layers: Number of hidden layers
        base_distribution: Base distribution ('normal', 'uniform')
        activation: Activation function
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization
        reduction: Loss reduction method
        
    Returns:
        NormalizingFlowLoss instance
    """
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
