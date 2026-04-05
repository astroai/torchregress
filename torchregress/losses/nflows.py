"""
Normalizing Flow loss functions for regression tasks.

This module provides loss functions for regression models that use
normalizing flows to model complex output distributions.
Uses the zuko package for efficient implementation of various flows.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, cast

import torch
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
    hidden_features: Optional[int | Sequence[int]] = None,
    n_hidden_layers: int | None = None,
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

    hidden = _resolve_hidden_features(
        hidden_features=hidden_features,
        n_hidden_layers=n_hidden_layers,
    )
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


def _resolve_hidden_features(
    *,
    hidden_features: Optional[int | Sequence[int]],
    n_hidden_layers: int | None,
) -> list[int]:
    if hidden_features is None:
        width = 64
        depth = 2 if n_hidden_layers is None else int(n_hidden_layers)
        return [width] * max(depth, 1)

    if isinstance(hidden_features, Sequence) and not isinstance(hidden_features, (str, bytes)):
        resolved = [int(width) for width in hidden_features]
        if not resolved:
            raise ValueError("hidden_features sequence must contain at least one layer width")
        return resolved

    width = int(hidden_features)
    depth = 1 if n_hidden_layers is None else int(n_hidden_layers)
    return [width] * max(depth, 1)


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

    def distribution(self, y_pred: Tensor) -> Any:
        """Return the conditional distribution induced by ``y_pred`` context."""
        context = self._extract_distribution_parameters(y_pred)
        if self.context_dim is None and context.numel() > 0:
            self.context_dim = context.shape[-1] if context.dim() > 0 else 0
        if self.context_dim is not None and self.context_dim > 0:
            return cast(Any, self.flow(context))
        return cast(Any, self.flow())

    def log_prob(self, y_pred: Tensor, target: Tensor) -> Tensor:
        """Return per-sample log probability for ``target`` under the flow."""
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        if target.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features in target, got {target.shape[-1]}"
            )
        dist = self.distribution(y_pred)
        return cast(Tensor, dist.log_prob(target))

    def _sample_tensor(self, y_pred: Tensor, n_samples: int) -> Tensor:
        samples = self.sample(y_pred, n_samples=n_samples)
        if samples.dim() == 2:
            return samples.unsqueeze(1)
        return samples

    def quantile(
        self,
        y_pred: Tensor,
        levels: Tensor | list[float] | tuple[float, ...],
        *,
        n_samples: int = 512,
    ) -> Tensor:
        """Approximate scalar predictive quantiles using flow samples."""
        if self.n_features != 1:
            raise ValueError("quantile currently only supports scalar normalizing-flow targets")
        level_tensor = torch.as_tensor(levels, dtype=y_pred.dtype, device=y_pred.device).reshape(-1)
        level_tensor = level_tensor.clamp(0.0, 1.0)
        samples = self._sample_tensor(y_pred, n_samples=n_samples)[..., 0]
        sorted_samples = torch.sort(samples, dim=1).values
        n_draws = sorted_samples.shape[1]
        if n_draws <= 1:
            return sorted_samples.expand(-1, level_tensor.numel())
        positions = level_tensor * float(n_draws - 1)
        lower_idx = torch.floor(positions).long().clamp(0, n_draws - 1)
        upper_idx = torch.ceil(positions).long().clamp(0, n_draws - 1)
        weight = (positions - lower_idx.to(positions.dtype)).view(1, -1)
        lower = sorted_samples.index_select(1, lower_idx)
        upper = sorted_samples.index_select(1, upper_idx)
        return lower + (upper - lower) * weight

    def cdf(
        self,
        y_pred: Tensor,
        values: Tensor,
        *,
        n_samples: int = 512,
    ) -> Tensor:
        """Approximate scalar predictive CDF values using flow samples."""
        if self.n_features != 1:
            raise ValueError("cdf currently only supports scalar normalizing-flow targets")
        value_tensor = values
        squeeze_last = False
        if value_tensor.dim() == 1:
            value_tensor = value_tensor.unsqueeze(-1)
            squeeze_last = True
        elif value_tensor.dim() == 2 and value_tensor.shape[-1] == 1:
            squeeze_last = True
        samples = self._sample_tensor(y_pred, n_samples=n_samples)[..., 0]
        cdf = (samples.unsqueeze(-1) <= value_tensor.unsqueeze(1)).to(samples.dtype).mean(dim=1)
        return cdf.squeeze(-1) if squeeze_last else cdf

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
        del mask
        return cast(Tensor, -self.log_prob(context, target))

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
        batch_size = y_pred.shape[0]
        dist = self.distribution(y_pred)

        if self.context_dim is not None and self.context_dim > 0:
            if n_samples == 1:
                samples = cast(Tensor, dist.sample())
            else:
                samples = cast(Tensor, dist.sample((n_samples,))).transpose(0, 1)
        else:
            if n_samples == 1:
                samples = cast(Tensor, dist.sample((batch_size,)))
            else:
                samples = cast(Tensor, dist.sample((batch_size, n_samples)))

        return cast(Tensor, samples)
