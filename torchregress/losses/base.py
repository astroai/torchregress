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

from functools import partial
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from ..utils.validation import validate_reduction, validate_weights


class BaseLoss(nn.Module):
    """
    Base class for all regression losses.

    This defines the common interface that all losses should implement,
    including reduction handling and weight application.

    Args:
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'

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
            if weights is not None:
                # Broadcast weights to match loss shape if needed for masking
                if weights.dim() < loss.dim():
                    for _ in range(loss.dim() - weights.dim()):
                        weights = weights.unsqueeze(-1)
                weights = weights.expand_as(loss)
                weights = weights[mask]
            loss = loss[mask]
        elif weights is not None:
            # Broadcast weights to match loss shape if needed
            if weights.dim() < loss.dim():
                for _ in range(loss.dim() - weights.dim()):
                    weights = weights.unsqueeze(-1)

        # No reduction: return raw values (potentially weighted)
        if self.reduction == "none":
            return loss * weights if weights is not None else loss

        # Sum reduction
        if self.reduction == "sum":
            if weights is not None:
                return torch.sum(loss * weights)
            return torch.sum(loss)

        # Mean reduction
        if self.reduction == "mean":
            if weights is not None:
                vals = loss * weights
                return torch.sum(vals) / torch.sum(weights).clamp(min=1)
            return torch.mean(loss)

        # Max / Min reduction
        if self.reduction == "max":
            return torch.max(loss)
        if self.reduction == "min":
            return torch.min(loss)

        # Default case - raise an error for unknown reduction types
        raise ValueError(f"Unknown reduction type: {self.reduction}")

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
            'none' | 'mean' | 'sum'. Default: 'mean'

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
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
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
            'none' | 'mean' | 'sum'. Default: 'mean'

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

    def _extract_distribution_parameters(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
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

    def _calculate_nll(
        self, y_pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
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


# Create weighted versions of PyTorch losses using functools.partial
# Regression losses - core point prediction
WeightedMSELoss = partial(WeightedLossWrapper, nn.MSELoss)
WeightedL1Loss = partial(WeightedLossWrapper, nn.L1Loss)
WeightedMAELoss = WeightedL1Loss
WeightedSmoothL1Loss = partial(WeightedLossWrapper, nn.SmoothL1Loss)
WeightedHuberLoss = partial(WeightedLossWrapper, nn.HuberLoss)

# Probabilistic regression losses
WeightedPoissonNLLLoss = partial(WeightedLossWrapper, nn.PoissonNLLLoss)
WeightedGaussianNLLLoss = partial(WeightedLossWrapper, nn.GaussianNLLLoss)

# Regression-as-classification (ordinal, discretized regression)
WeightedCrossEntropyLoss = partial(WeightedLossWrapper, nn.CrossEntropyLoss)
WeightedNLLLoss = partial(WeightedLossWrapper, nn.NLLLoss)
WeightedKLDivLoss = partial(WeightedLossWrapper, nn.KLDivLoss)
