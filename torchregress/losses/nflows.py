"""
Normalizing Flow loss functions for regression tasks.

This module provides loss functions for regression models that use
normalizing flows to model complex output distributions.
Uses the zuko package for efficient implementation of various flows.
"""

from typing import Any, Callable, Optional, cast

from torch import Tensor
from torch.nn import Module

try:
    from zuko.flows import MAF, NSF, RealNVP  # type: ignore[import-untyped]

    HAS_ZUKO = True
except ImportError:
    HAS_ZUKO = False

    MAF = NSF = RealNVP = None

from ..utils.tensor_ops import apply_mask, masked_reduction
from .base import DistributionLoss
from .loss_registry import register_regression_loss


def create_flow_model(
    n_features: int,
    context_dim: int = 0,
    flow_type: str = "nsf",
    n_transforms: int = 5,
    hidden_features: Optional[list[int]] = None,
    **kwargs: Any,
) -> Module:
    """
    Create a zuko flow model with a torchregress-friendly interface.

    This helper exists primarily for examples/docs and keeps naming consistent with
    the rest of the library (`n_features`, `context_dim`, `n_transforms`).
    """
    if not HAS_ZUKO:
        raise ImportError("zuko is required for normalizing flows. Install torchregress[flows].")

    flow_type_key = flow_type.lower()
    flow_cls_map = {
        "nsf": NSF,
        "maf": MAF,
        "realnvp": RealNVP,
    }
    if flow_type_key not in flow_cls_map:
        raise ValueError(
            f"Unsupported flow_type {flow_type!r}. Expected one of {sorted(flow_cls_map)}."
        )

    hidden = [64, 64] if hidden_features is None else hidden_features
    flow_cls = flow_cls_map[flow_type_key]
    assert flow_cls is not None
    flow = flow_cls(
        features=n_features,
        context=context_dim,
        transforms=n_transforms,
        hidden_features=hidden,
        **kwargs,
    )
    # Store context dim in a stable attribute used by NormalizingFlowLoss.
    setattr(flow, "context", context_dim)
    return cast(Module, flow)


def create_flow_loss(*, reduction: str = "mean", **flow_kwargs: Any) -> "NormalizingFlowLoss":
    """Create a normalizing-flow loss from flow constructor arguments."""
    flow = create_flow_model(**flow_kwargs)
    return NormalizingFlowLoss(flow=flow, reduction=reduction)


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
        flow: Module,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)

        if not isinstance(flow, Module):
            raise TypeError(f"flow must be a torch.nn.Module (zuko Flow), got {type(flow)}")

        self.flow = flow

        # Extract flow configuration for validation
        # Zuko flows store dimensions in the base distribution
        try:
            base_fn = cast(Callable[[], Any], getattr(flow, "base"))
            base_dist = base_fn()
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
            dist = cast(Any, self.flow(context))
        else:
            # Unconditional flow (context_dim=0 or None with empty context)
            dist = cast(Any, self.flow())

        # Calculate log probability
        log_prob = cast(Tensor, dist.log_prob(target))

        # Return negative log-likelihood
        return cast(Tensor, -log_prob)

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
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
        target_masked = apply_mask(target, mask) if mask is not None else target

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
        batch_size = y_pred.shape[0]

        # Get conditional distribution
        if self.context_dim is not None and self.context_dim > 0:
            dist = cast(Any, self.flow(context))
            # Conditional flow: batch_shape is [batch_size]
            if n_samples == 1:
                # [batch_size, n_features]
                samples = cast(Tensor, dist.sample())
            else:
                # [n_samples, batch_size, n_features]
                samples = cast(Tensor, dist.sample((n_samples,)))
                # [batch_size, n_samples, n_features]
                samples = samples.transpose(0, 1)
        else:
            dist = cast(Any, self.flow())
            # Unconditional flow: batch_shape is []
            if n_samples == 1:
                # [batch_size, n_features]
                samples = cast(Tensor, dist.sample((batch_size,)))
            else:
                # [batch_size, n_samples, n_features]
                samples = cast(Tensor, dist.sample((batch_size, n_samples)))

        return cast(Tensor, samples)
