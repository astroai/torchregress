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

from ..utils.validation import validate_reduction, validate_weights

# ---------------------------------------------------------------------------
# Reduction strategy helpers (cached at __init__ to remove branch overhead
# from the hot per-element reduction path).
# ---------------------------------------------------------------------------


def _broadcast_weights(weights: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Right-pad ``weights`` with singleton dims so it broadcasts to ``target_dim``."""
    if weights.dim() < target_dim:
        return weights.reshape(weights.shape + (1,) * (target_dim - weights.dim()))
    return weights


def _reduce_mean(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is not None:
        return torch.sum(values * weights) / torch.sum(weights).clamp(min=1.0)
    return torch.mean(values)


def _reduce_sum(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is not None:
        return torch.sum(values * weights)
    return torch.sum(values)


def _reduce_none(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is not None:
        return values * weights
    return values


def _reduce_max(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is not None:
        return torch.max(values * weights)
    return torch.max(values)


def _reduce_min(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is not None:
        return torch.min(values * weights)
    return torch.min(values)


_REDUCERS: dict[str, Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]] = {
    "mean": _reduce_mean,
    "sum": _reduce_sum,
    "none": _reduce_none,
    "max": _reduce_max,
    "min": _reduce_min,
}


def reduce_per_sample(
    nll: torch.Tensor,
    sample_mask: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
    reduction: str,
) -> torch.Tensor:
    """Apply the standard mask/weight/reduction contract to a per-sample NLL vector.

    This is shared by multivariate / low-rank / count losses whose ``forward``
    computes a 1-D ``[batch]`` NLL tensor and needs the *same* reduction
    semantics as the rest of the library.  Keeping the logic in one place
    avoids the four-way branch drift observed in earlier versions of
    :class:`MultivariateGaussianLoss` and :class:`LowRankGaussianLoss`.

    Args:
        nll: ``[batch]`` (or broadcastable to it) tensor of per-sample losses.
        sample_mask: Optional ``[batch]`` boolean mask of valid samples.
        weights: Optional ``[batch]`` non-negative sample weights.
        reduction: One of ``'none'``, ``'mean'``, ``'sum'``.

    Returns:
        Reduced loss tensor.
    """
    reducer = _REDUCERS[reduction]

    # Collapse per-feature masks to per-sample mask once, here, instead of
    # inside every subclass.
    if sample_mask is not None:
        if sample_mask.dtype != torch.bool:
            sample_mask = sample_mask > 0
        if sample_mask.dim() > 1:
            sample_mask = sample_mask.all(dim=-1)

    if sample_mask is not None and weights is not None:
        masked_weights = weights.to(device=nll.device, dtype=nll.dtype)
        if masked_weights.shape[0] != nll.shape[0]:
            raise ValueError("weights must match batch size")
        masked_nll = nll[sample_mask]
        masked_weights = masked_weights[sample_mask]
        if reduction == "none":
            # Preserve shape: scatter back into a zeros tensor so the caller
            # gets a length-``batch`` output.
            result = torch.zeros_like(nll)
            result[sample_mask] = masked_nll * masked_weights
            return result
        if reduction == "sum":
            return torch.sum(masked_nll * masked_weights)
        # mean
        return torch.sum(masked_nll * masked_weights) / torch.sum(masked_weights).clamp(min=1.0)

    if sample_mask is not None:
        masked_nll = nll[sample_mask]
        if reduction == "none":
            result = torch.zeros_like(nll)
            result[sample_mask] = masked_nll
            return result
        if reduction == "sum":
            return torch.sum(masked_nll)
        return torch.sum(masked_nll) / sample_mask.sum().clamp(min=1)

    return reducer(nll, weights)


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
        self._reducer: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor] = _REDUCERS[
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
        # First, apply mask if provided
        if mask is not None:
            # Handle cases where mask has more dimensions than loss (e.g. [B, Dy] for loss [B])
            if mask.dim() > loss.dim():
                for _ in range(mask.dim() - loss.dim()):
                    mask = mask.any(dim=-1)

            if weights is not None:
                # Broadcast weights to match loss shape if needed for masking
                weights = _broadcast_weights(weights, loss.dim())
                weights = weights.expand_as(loss)
                weights = weights[mask]
            loss = loss[mask]
        elif weights is not None:
            # Broadcast weights to match loss shape if needed
            weights = _broadcast_weights(weights, loss.dim())

        return self._reducer(loss, weights)

    def _apply_mask(self, tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Filter tensor elements by boolean mask."""
        if mask is None:
            return tensor
        return tensor[mask]

    def _validate_inputs(
        self, y_pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> None:
        """Validate that y_pred, target, and mask shapes are compatible."""
        if target.shape != y_pred.shape:
            raise ValueError(f"target shape {target.shape} must match y_pred shape {y_pred.shape}")
        if mask is not None and mask.shape != target.shape:
            raise ValueError(f"Mask shape {mask.shape} must match target shape {target.shape}")

    def _reduce_with_mask(
        self,
        loss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply mask to filter values and then apply reduction.

        Note: This is an alias for _reduce() that provides semantic clarity
        when used in loss implementations that explicitly handle masking.
        """
        return self._reduce(loss, mask, weights)


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
        ...         return self._reduce_with_mask(loss, mask, weights)
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
        ...         return self._reduce_with_mask(loss, mask, weights)
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
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
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
        self, loss_fn: Union[Callable, nn.Module], reduction: str = "mean", **kwargs: Any
    ) -> None:
        super().__init__(reduction="none")  # We'll handle reduction ourselves

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


class WeightedGaussianNLLLoss(BaseLoss):
    """
    Weighted wrapper for torch.nn.GaussianNLLLoss with mask support.

    Accepts (mean, var) or (mean, log_var) predictions. When log_variance=True,
    the second element is treated as log-variance and exponentiated.
    """

    def __init__(self, reduction: str = "mean", log_variance: bool = True, **kwargs: Any) -> None:
        super().__init__(reduction="none")
        kwargs["reduction"] = "none"
        self.torch_loss = nn.GaussianNLLLoss(**kwargs)
        self.reduction = validate_reduction(reduction)
        self.log_variance = log_variance

    def forward(
        self,
        y_pred: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        target: torch.Tensor,
        log_variance: Optional[bool] = None,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Backward compatibility for legacy positional ordering:
        # forward(y_pred, target, mask, weights, log_variance)
        if isinstance(log_variance, torch.Tensor):
            legacy_mask = log_variance
            legacy_weights = mask if isinstance(mask, torch.Tensor) else None
            legacy_log_variance = weights if isinstance(weights, bool) else None
            mask = legacy_mask
            weights = legacy_weights
            log_variance = legacy_log_variance

        if isinstance(y_pred, (tuple, list)):
            if len(y_pred) != 2:
                raise ValueError(
                    "y_pred must be a tuple of (mean, var) or (mean, log_var) for "
                    "WeightedGaussianNLLLoss"
                )
            mean, var_or_log = y_pred
        else:
            mean = y_pred
            if "var" not in kwargs:
                raise ValueError(
                    "WeightedGaussianNLLLoss requires (mean, var) or (mean, log_var) inputs"
                )
            var_or_log = kwargs["var"]

        variance_is_log = self.log_variance if log_variance is None else log_variance
        var = torch.exp(var_or_log) if variance_is_log else var_or_log

        self._validate_inputs(mean, target, mask)

        if weights is not None:
            weights = validate_weights(weights, target.shape[0])

        loss = self.torch_loss(mean, target, var)
        return self._reduce(loss, mask, weights)


class WeightedMSELoss(WeightedLossWrapper):
    """Masked/weighted wrapper around ``torch.nn.MSELoss``."""

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        super().__init__(nn.MSELoss, reduction=reduction, **kwargs)


class WeightedL1Loss(WeightedLossWrapper):
    """Masked/weighted wrapper around ``torch.nn.L1Loss``."""

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        super().__init__(nn.L1Loss, reduction=reduction, **kwargs)


class WeightedHuberLoss(WeightedLossWrapper):
    """Masked/weighted wrapper around ``torch.nn.HuberLoss``."""

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        super().__init__(nn.HuberLoss, reduction=reduction, **kwargs)


# Compatibility aliases used throughout the docs/examples. The weighted variants
# are the canonical wrappers because they preserve torchregress mask/weight support.
WeightedMAELoss = WeightedL1Loss
MSELoss = WeightedMSELoss
L1Loss = WeightedL1Loss
HuberLoss = WeightedHuberLoss
