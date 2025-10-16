"""
Normalizing Flow loss functions for regression tasks.

This module provides loss functions for regression models that use
normalizing flows to model complex output distributions.
Uses the zuko package for efficient implementation of various flows.
"""

from typing import Optional

from torch import Tensor
from torch.nn import Module

# Direct import as per CLAUDE.md policy - users must install required dependencies
# Note: IAF is not available in current zuko version, removed from imports
from zuko.flows import MAF, NSF, Flow, RealNVP

from ..utils.tensor_ops import apply_mask, masked_reduction
from .base import DistributionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("nflow")
class NormalizingFlowLoss(DistributionLoss):
    """
    Negative Log-Likelihood loss for conditional normalizing flow models using zuko.

    This loss allows modeling complex multi-dimensional target distributions
    for regression tasks using various normalizing flow architectures. The flow
    is conditioned on the model's output, allowing it to learn target distributions
    that depend on the input.

    Args:
        flow (Flow): A zuko Flow instance (RealNVP, MAF, NSF, etc.)
            The flow must be created with context dimension matching the model output.
        reduction (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'
            Default: 'mean'

    Mathematical Formulation:
        Normalizing flows transform a simple base distribution into a complex target
        distribution through a series of invertible transformations. For conditional
        flows, the transformation depends on context c (model output):

        NLL = -log(p_X(x|c)) = -log(p_Z(f(x|c))) - log|det(df/dx)|

        where p_Z is the density of the base distribution, f is the invertible
        transformation conditioned on c, and |det(df/dx)| is the absolute determinant
        of the Jacobian.

    Notes:
        - Requires the 'zuko' package: `pip install zuko`
        - The flow must be a trainable nn.Module that will be part of your model
        - The model should output context vectors that condition the flow
        - Different flow types (RealNVP, MAF, NSF) have different modeling
          capacities and computational characteristics
        - The flow's parameters are trained alongside the model via backpropagation

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from zuko.flows import NSF
        >>>
        >>> # Create a conditional flow (2D targets conditioned on 10D context)
        >>> flow = NSF(features=2, context=10, transforms=3, hidden_features=[64, 64])
        >>>
        >>> # Create loss function with the flow
        >>> loss_fn = NormalizingFlowLoss(flow=flow)
        >>>
        >>> # Model outputs context vectors
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = nn.Linear(5, 10)  # outputs 10D context
        ...     def forward(self, x):
        ...         return self.net(x)
        >>>
        >>> model = MyModel()
        >>> x = torch.randn(32, 5)
        >>> context = model(x)  # [32, 10]
        >>> target = torch.randn(32, 2)  # [32, 2]
        >>>
        >>> # Compute loss - gradients flow through both model and flow
        >>> loss = loss_fn(context, target)
        >>> loss.backward()  # Updates both model and flow parameters
    """

    def __init__(
        self,
        flow: Flow,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)

        if not isinstance(flow, Module):
            raise TypeError(f"flow must be a torch.nn.Module (zuko Flow), got {type(flow)}")

        self.flow = flow

        # Extract flow configuration for validation
        # Zuko flows store dimensions in the base distribution
        try:
            base_dist = flow.base()
            self.n_features = base_dist.event_shape[0] if len(base_dist.event_shape) > 0 else 1
        except Exception as e:
            raise ValueError(f"Could not extract feature dimension from flow: {e}")

        # Try to get context dimension - check if it was added by create_flow_model
        self.context_dim = getattr(flow, "context", None)  # May be None if not set

    def _extract_distribution_parameters(self, y_pred: Tensor) -> Tensor:
        """
        Extract context from model predictions.

        Args:
            y_pred: Model predictions serving as context for the flow
                    Shape: [batch_size, context_dim]

        Returns:
            Context tensor for conditioning the flow
        """
        return y_pred

    def _calculate_nll(
        self,
        target: Tensor,
        context: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calculate negative log-likelihood for the conditional flow model.

        Args:
            target: Target values [batch_size, n_features]
            context: Context vectors from model [batch_size, context_dim]
            mask: Optional mask [batch_size, n_features]

        Returns:
            Negative log-likelihood [batch_size]
        """
        # Get conditional distribution from flow
        # zuko flows: flow(context) returns a Distribution
        if self.context_dim is not None and self.context_dim > 0:
            # Conditional flow
            dist = self.flow(context)
        else:
            # Unconditional flow (context_dim=0 or None with empty context)
            dist = self.flow()

        # Calculate log probability
        log_prob = dist.log_prob(target)

        # Return negative log-likelihood
        return -log_prob

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calculate normalizing flow negative log-likelihood loss.

        Args:
            y_pred: Context from model [batch_size, context_dim]
            target: Ground truth values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional sample weights [batch_size]

        Returns:
            Negative log-likelihood loss

        Raises:
            ValueError: If shapes don't match expected dimensions
        """
        # Validate target shape
        if target.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features in target, got {target.shape[-1]}"
            )

        # Infer and store context_dim on first forward pass
        if self.context_dim is None and y_pred.numel() > 0:
            self.context_dim = y_pred.shape[-1] if y_pred.dim() > 0 else 0

        # Apply mask to targets if provided
        if mask is not None:
            target_masked = apply_mask(target, mask)
        else:
            target_masked = target

        # Extract context
        context = self._extract_distribution_parameters(y_pred)

        # Calculate negative log-likelihood
        nll = self._calculate_nll(target_masked, context, mask)

        # Apply weights if provided
        if weights is not None:
            # Handle different weight shapes
            if weights.dim() > 1 and weights.shape[1] > 1:
                # Average across features if weights are per-feature
                weights = weights.mean(dim=1)
            nll = nll * weights

        # Handle mask shape - reduce to match nll shape if needed
        if mask is not None and mask.dim() > nll.dim():
            # If mask is per-feature, reduce to per-sample (all features must be valid)
            mask = mask.all(dim=1)

        # Apply reduction
        return masked_reduction(nll, mask, self.reduction)

    def sample(self, y_pred: Tensor, n_samples: int = 1) -> Tensor:
        """
        Generate samples from the conditional flow distribution.

        Args:
            y_pred: Context from model [batch_size, context_dim]
            n_samples: Number of samples to generate per input
                Default: 1

        Returns:
            Samples [batch_size, n_samples, n_features] or [batch_size, n_features] if n_samples=1
        """
        # Extract context
        context = self._extract_distribution_parameters(y_pred)

        # Get conditional distribution
        if self.context_dim is not None and self.context_dim > 0:
            dist = self.flow(context)
        else:
            dist = self.flow()

        # Generate samples
        if n_samples == 1:
            # Single sample per input
            samples = dist.sample()  # [batch_size, n_features]
        else:
            # Multiple samples per input
            samples = dist.sample((n_samples,))  # [n_samples, batch_size, n_features]
            # Transpose to [batch_size, n_samples, n_features]
            samples = samples.transpose(0, 1)

        return samples


def create_flow_model(
    n_features: int,
    context_dim: int,
    flow_type: str = "nsf",
    n_transforms: int = 3,
    hidden_features: int = 64,
    n_hidden_layers: int = 2,
    **kwargs,
) -> Flow:
    """
    Factory function to create a conditional normalizing flow model.

    This creates a zuko flow that can be used with NormalizingFlowLoss.
    The flow will be a trainable nn.Module.

    Args:
        n_features (int): Number of features in the target distribution
        context_dim (int): Dimension of conditioning context from model
            Set to 0 for unconditional flows
        flow_type (str): Type of flow ('realnvp', 'maf', 'nsf')
            Default: 'nsf' (Neural Spline Flow - most flexible)
        n_transforms (int): Number of transformation blocks
            Default: 3
        hidden_features (int): Size of hidden layers in transformations
            Default: 64
        n_hidden_layers (int): Number of hidden layers
            Default: 2
        **kwargs: Additional arguments passed to the flow constructor
            For NSF: bins, randperm, etc.
            For MAF/RealNVP: activation, dropout, etc.

    Returns:
        Flow: A trainable zuko Flow instance

    Raises:
        ValueError: If invalid flow_type is specified

    Examples:
        >>> # Create a conditional NSF for 2D targets with 10D context
        >>> flow = create_flow_model(
        ...     n_features=2,
        ...     context_dim=10,
        ...     flow_type='nsf',
        ...     n_transforms=5,
        ...     hidden_features=128
        ... )
        >>>
        >>> # Use with loss function
        >>> loss_fn = NormalizingFlowLoss(flow=flow)
        >>>
        >>> # Create unconditional flow
        >>> uncond_flow = create_flow_model(
        ...     n_features=3,
        ...     context_dim=0,
        ...     flow_type='realnvp'
        ... )
    """
    flow_type = flow_type.lower()

    # Validate flow type
    valid_types = ["realnvp", "maf", "nsf"]
    if flow_type not in valid_types:
        raise ValueError(
            f"Invalid flow_type: {flow_type}. Must be one of: {', '.join(valid_types)}"
        )

    # Common arguments for all flows
    common_args = {
        "features": n_features,
        "context": context_dim,
        "transforms": n_transforms,
    }

    # Flow-specific arguments
    if flow_type == "nsf":
        # NSF uses bins for spline transformations
        flow_args = {
            **common_args,
            "bins": kwargs.get("bins", 8),
            "hidden_features": [hidden_features] * n_hidden_layers,
        }
        flow = NSF(**flow_args)

    elif flow_type == "maf":
        # MAF (Masked Autoregressive Flow)
        flow_args = {
            **common_args,
            "hidden_features": [hidden_features] * n_hidden_layers,
        }
        # Add optional MAF-specific args
        if "randperm" in kwargs:
            flow_args["randperm"] = kwargs["randperm"]
        flow = MAF(**flow_args)

    elif flow_type == "realnvp":
        # RealNVP (Real-valued Non-Volume Preserving)
        flow_args = {
            **common_args,
            "hidden_features": [hidden_features] * n_hidden_layers,
        }
        flow = RealNVP(**flow_args)

    # Add metadata attributes for convenience (zuko flows don't have these by default)
    flow.features = n_features
    flow.context = context_dim

    return flow
