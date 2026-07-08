"""
Base loss functions and abstract classes for regression tasks.

This module provides the foundation classes for all loss functions
in the torchregress library. Each base class handles specific aspects
of regression losses:

- BaseLoss: Root base class defining common interfaces
- RegressionLoss: Standard regression loss functions
- DistributionLoss: Losses for probabilistic/distributional regression

All losses follow PyTorch conventions with forward methods expecting
inputs in the form of (y_pred, target, ...).
"""

from typing import Any, Callable, Optional, Union, cast

import torch
import torch.nn as nn

from ..utils.reduction import REDUCERS
from ..utils.validation import validate_reduction, validate_weights
from .loss_registry import register_regression_loss


def _broadcast_weights(weights: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Right-pad ``weights`` with singleton dims so it broadcasts to ``target_dim``."""
    if weights.dim() < target_dim:
        return weights.reshape(weights.shape + (1,) * (target_dim - weights.dim()))
    return weights


class BaseLoss(nn.Module):
    """
    Base class for all regression losses.

    This defines the common interface that all losses should implement,
    including reduction handling and weight application.

    Args:
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'min' | 'max'. Default: 'mean'

    Example:
        >>> class MyLoss(BaseLoss):
        ...     def forward(self, y_pred, target, **kwargs):
        ...         return torch.abs(y_pred - target)
        >>> loss_fn = MyLoss(reduction='mean')
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([1.5, 1.5, 2.5])
        >>> loss_fn(y_pred, target)
        tensor(0.5000)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    @property
    def reduction(self) -> str:
        return self._reduction

    @reduction.setter
    def reduction(self, value: str) -> None:
        self._reduction = validate_reduction(value)
        self._reducer: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor] = REDUCERS[
            self._reduction
        ]

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            y_pred: Predicted values
            target: Target values
            mask: Optional boolean mask for indicating valid values
            weights: Optional sample weights
            **kwargs: Additional arguments

        Returns:
            Loss value
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def _reduce(
        self,
        loss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply reduction to the loss tensor with support for masking and weighting.
        """
        # Collapse extra dims on mask and weights when loss is lower-dimensional
        if mask is not None and mask.dim() > loss.dim():
            for _ in range(mask.dim() - loss.dim()):
                # ponytail: all() matches reduce_per_sample in utils/reduction.py
                mask = mask.all(dim=-1)
        if weights is not None and weights.dim() > loss.dim():
            for _ in range(weights.dim() - loss.dim()):
                weights = weights.mean(dim=-1)

        # Apply mask and/or weights
        if mask is not None:
            if weights is not None:
                weights = weights[mask]
            loss = loss[mask]
        elif weights is not None:
            weights = _broadcast_weights(weights, loss.dim())

        return self._reducer(loss, weights)

    def _validate_inputs(
        self, y_pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> None:
        """Validate that y_pred, target, and mask shapes are compatible."""
        if target.shape != y_pred.shape:
            raise ValueError(f"target shape {target.shape} must match y_pred shape {y_pred.shape}")
        if mask is not None and mask.shape != target.shape:
            raise ValueError(f"Mask shape {mask.shape} must match target shape {target.shape}")


class RegressionLoss(BaseLoss):
    """
    Base class for regression losses.

    This extends BaseLoss with additional functionality specific
    to regression tasks. All implementations should override forward()
    with the PyTorch standard parameter ordering (y_pred, target).

    Args:
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'min' | 'max'. Default: 'mean'

    Example:
        >>> class L1Loss(RegressionLoss):
        ...     def forward(self, y_pred, target, mask=None, weights=None):
        ...         self._validate_inputs(y_pred, target, mask)
        ...         loss = torch.abs(y_pred - target)
        ...         return self._reduce(loss, mask, weights)
        >>> loss_fn = L1Loss()
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([0.0, 2.0, 3.0])
        >>> loss_fn(y_pred, target)
        tensor(0.3333)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate regression loss.

        Args:
            y_pred: Predicted values
            target: Ground truth target values
            mask: Optional boolean mask for indicating valid values
            weights: Optional sample weights

        Returns:
            Loss value according to the specified reduction
        """
        raise NotImplementedError("Subclasses must implement this method")


class DistributionLoss(BaseLoss):
    """
    Base class for distributional losses.

    This class supports losses that estimate full probability
    distributions rather than just point predictions.

    Args:
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'min' | 'max'. Default: 'mean'

    Example:
        >>> class GaussianNLL(DistributionLoss):
        ...     def forward(self, y_pred, target, mask=None, weights=None):
        ...         # y_pred contains [mean, log_var]
        ...         mean, log_var = self._extract_distribution_parameters(y_pred)
        ...         loss = 0.5 * (log_var + (target - mean)**2 / torch.exp(log_var))
        ...         return self._reduce(loss, mask, weights)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)

    def _extract_distribution_parameters(self, y_pred: Any) -> Any:
        """
        Extract distribution parameters from model outputs.

        Args:
            y_pred: Model predictions containing distribution parameters

        Returns:
            Dictionary of parameter names and tensors

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _calculate_nll(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Calculate negative log likelihood.

        Args:
            y_pred: Model predictions containing distribution parameters
            target: Ground truth target values
            mask: Optional boolean mask

        Returns:
            Negative log likelihood tensor

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement this method")

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate distributional loss.

        Args:
            y_pred: Model predictions containing distribution parameters
            target: Ground truth target values
            mask: Optional boolean mask for indicating valid values
            weights: Optional sample weights

        Returns:
            Loss value according to the specified reduction

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement this method")


class WeightedLossWrapper(BaseLoss):
    """
    Wrapper for PyTorch loss functions to add masking and weighting support.

    This allows standard PyTorch losses to be used with the same interfaces
    as TorchRegression losses, including mask and weight support.

    Args:
        loss_fn: PyTorch loss function class (nn.Module) or instance
        reduction: Specifies the reduction to apply to the output. Pass ``None``
            (default) to inherit the wrapped loss's currently-configured
            reduction; pass an explicit value (``'mean'`` | ``'sum'`` |
            ``'none'`` | ``'min'`` | ``'max'``) to override it.  This restores
            previously-silently-dropped behavior for callers that constructed
            the torch loss with a non-default ``reduction`` (e.g.
            ``WeightedLossWrapper(nn.MSELoss(reduction='sum'))``).
        **kwargs: Additional arguments to pass to the loss function

    Example:
        >>> import torch.nn.functional as F
        >>> mse_loss = WeightedLossWrapper(nn.MSELoss)
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([0.0, 2.0, 4.0])
        >>> mse_loss(y_pred, target)
        tensor(0.6667)
    """

    def __init__(
        self,
        loss_fn: Union[Callable, nn.Module],
        reduction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(reduction="none")  # We'll handle reduction ourselves

        # Capture the wrapped loss's already-configured reduction before we
        # mutate the inner tensor to ``"none"`` (necessary so the wrapper can
        # compute per-element weighted reductions). When the user passes a
        # class, there is no prior setting and we fall back to ``"mean"``.
        if isinstance(loss_fn, type):
            inferred_reduction: Optional[str] = "mean"
        else:
            inferred_reduction = getattr(loss_fn, "reduction", None)
            if not isinstance(inferred_reduction, str):
                inferred_reduction = "mean"

        # Handle both class and instance cases
        if isinstance(loss_fn, type):
            # It's a class, instantiate it with reduction='none'
            kwargs["reduction"] = "none"
            self.torch_loss = loss_fn(**kwargs)
        else:
            # It's already an instance
            self.torch_loss = loss_fn
            if hasattr(self.torch_loss, "reduction"):
                self.torch_loss.reduction = "none"

        # Honour an explicit caller override; otherwise preserve the wrapped
        # loss's previously-configured reduction (e.g. ``nn.MSELoss(reduction='sum')``).
        # Previously this always fell back to ``"mean"``, silently dropping the
        # wrapper user's configured reduction.
        if reduction is None:
            self.reduction = validate_reduction(inferred_reduction or "mean")
        else:
            self.reduction = validate_reduction(reduction)

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate loss using the wrapped PyTorch loss function.

        Args:
            y_pred: Predicted values
            target: Ground truth target values
            mask: Optional boolean mask for indicating valid values
            weights: Optional sample weights

        Returns:
            Loss value according to the specified reduction
        """
        # Validate inputs
        self._validate_inputs(y_pred, target, mask)

        # Validate weights if provided
        if weights is not None:
            weights = validate_weights(weights, target.shape[0])

        # PyTorch losses already expect (y_pred, target) order
        loss = self.torch_loss(y_pred, target)

        # Handle weights and reduction with mask
        return self._reduce(loss, mask, weights)


class WeightedCrossEntropyLoss(BaseLoss):
    """
    Weighted wrapper for torch.nn.CrossEntropyLoss with mask support.

    This wrapper supports class-index targets and per-sample weights.
    """

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        super().__init__(reduction="none")
        base_kwargs = dict(kwargs)
        self.torch_loss = nn.CrossEntropyLoss(reduction="none", **base_kwargs)
        self._torch_loss_mean = nn.CrossEntropyLoss(reduction="mean", **base_kwargs)
        self._torch_loss_sum = nn.CrossEntropyLoss(reduction="sum", **base_kwargs)
        self.reduction = validate_reduction(reduction)

    def _validate_classification_inputs(
        self, y_pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> None:
        if y_pred.dim() < 2:
            raise ValueError("y_pred must have at least 2 dimensions for classification losses")

        expected_target_shape = (y_pred.shape[0], *y_pred.shape[2:])
        if target.shape != expected_target_shape:
            raise ValueError(
                f"target shape {target.shape} must match y_pred shape {expected_target_shape} "
                "for class-index targets"
            )

        if mask is not None and mask.shape != target.shape:
            raise ValueError(f"Mask shape {mask.shape} must match target shape {target.shape}")

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._validate_classification_inputs(y_pred, target, mask)

        if weights is not None:
            weights = validate_weights(weights, y_pred.shape[0])

        # Keep exact native semantics when no torchregress-specific mask/weight
        # behavior is requested.
        if mask is None and weights is None:
            if self.reduction == "mean":
                return cast(torch.Tensor, self._torch_loss_mean(y_pred, target))
            if self.reduction == "sum":
                return cast(torch.Tensor, self._torch_loss_sum(y_pred, target))

        loss = self.torch_loss(y_pred, target)
        return self._reduce(loss, mask, weights)


class WeightedNLLLoss(BaseLoss):
    """
    Weighted wrapper for torch.nn.NLLLoss with mask support.

    This wrapper supports class-index targets and per-sample weights.
    """

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        super().__init__(reduction="none")
        base_kwargs = dict(kwargs)
        self.torch_loss = nn.NLLLoss(reduction="none", **base_kwargs)
        self._torch_loss_mean = nn.NLLLoss(reduction="mean", **base_kwargs)
        self._torch_loss_sum = nn.NLLLoss(reduction="sum", **base_kwargs)
        self.reduction = validate_reduction(reduction)

    def _validate_classification_inputs(
        self, y_pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> None:
        if y_pred.dim() < 2:
            raise ValueError("y_pred must have at least 2 dimensions for classification losses")

        expected_target_shape = (y_pred.shape[0], *y_pred.shape[2:])
        if target.shape != expected_target_shape:
            raise ValueError(
                f"target shape {target.shape} must match y_pred shape {expected_target_shape} "
                "for class-index targets"
            )

        if mask is not None and mask.shape != target.shape:
            raise ValueError(f"Mask shape {mask.shape} must match target shape {target.shape}")

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._validate_classification_inputs(y_pred, target, mask)

        if weights is not None:
            weights = validate_weights(weights, y_pred.shape[0])

        # Keep exact native semantics when no torchregress-specific mask/weight
        # behavior is requested.
        if mask is None and weights is None:
            if self.reduction == "mean":
                return cast(torch.Tensor, self._torch_loss_mean(y_pred, target))
            if self.reduction == "sum":
                return cast(torch.Tensor, self._torch_loss_sum(y_pred, target))

        loss = self.torch_loss(y_pred, target)
        return self._reduce(loss, mask, weights)


# ponytail: thin wrappers kept for backward compat.
@register_regression_loss("mse")
class WeightedMSELoss(WeightedLossWrapper):
    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        super().__init__(nn.MSELoss, reduction=reduction, **kwargs)


@register_regression_loss("l1")
@register_regression_loss("mae")
class WeightedL1Loss(WeightedLossWrapper):
    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        super().__init__(nn.L1Loss, reduction=reduction, **kwargs)


@register_regression_loss("huber")
class WeightedHuberLoss(WeightedLossWrapper):
    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        super().__init__(nn.HuberLoss, reduction=reduction, **kwargs)
