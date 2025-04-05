"""
Base loss functions and abstract classes for regression tasks.

This module provides the foundation classes for all loss functions
in the torchregression library. Each base class handles specific aspects
of regression losses:

- BaseLoss: Root base class defining common interfaces
- MaskedLoss: Adds support for masked operations
- RegressionLoss: Standard regression loss functions
- DistributionLoss: Losses for probabilistic/distributional regression

All losses follow PyTorch conventions with forward methods expecting
inputs in the form of (y_pred, target, ...).
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Callable, Dict, Any

from ..utils.tensor_ops import apply_mask, masked_reduction
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

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            y_pred: Predicted values
            target: Target values
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
        Apply reduction to the loss tensor.

        Args:
            loss: Loss tensor
            mask: Optional boolean mask
            weights: Optional sample weights

        Returns:
            Reduced loss according to the reduction method
        """
        if self.reduction == "none":
            if weights is not None:
                return loss * weights
            return loss

        # Apply weights if provided
        if weights is not None:
            loss = loss * weights

        return masked_reduction(loss, mask, self.reduction)


class MaskedLoss(BaseLoss):
    """
    Base class for losses that support optional masking.

    This class provides common functionality for handling masked losses,
        where certain elements in the inputs should be ignored.

    Args:
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> class MyMaskedLoss(MaskedLoss):
        ...     def forward(self, y_pred, target, mask=None, weights=None):
        ...         loss = torch.abs(y_pred - target)
        ...         return self._reduce_with_mask(loss, mask, weights)
        >>> loss_fn = MyMaskedLoss()
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([0.0, 0.0, 3.0])
        >>> mask = torch.tensor([False, False, True])
        >>> loss_fn(y_pred, target, mask=mask)
        tensor(0.)  # Only the masked value is considered
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)

    def _apply_mask(self, tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply an optional mask to a tensor.

        Args:
            tensor: Input tensor
            mask: Optional boolean mask with the same shape as tensor

        Returns:
            Tensor with mask applied or original tensor if mask is None
        """
        return apply_mask(tensor, mask)

    def _validate_inputs(
        self, y_pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> None:
        """
        Validate input tensors.

        Args:
            y_pred: Predicted values
            target: Target values
            mask: Optional boolean mask

        Raises:
            ValueError: If inputs have incompatible shapes
        """
        # Check shape compatibility
        if target.shape != y_pred.shape:
            raise ValueError(f"target shape {target.shape} must match y_pred shape {y_pred.shape}")

        # Check mask if provided
        if mask is not None and mask.shape != target.shape:
            raise ValueError(f"Mask shape {mask.shape} must match target shape {target.shape}")

    def _reduce_with_mask(
        self,
        loss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply reduction to the loss tensor, handling masks.

        Args:
            loss: Loss tensor
            mask: Optional boolean mask
            weights: Optional sample weights

        Returns:
            Reduced loss according to the reduction method
        """
        if mask is not None:
            loss = self._apply_mask(loss, mask)

        return self._reduce(loss, mask, weights)


class RegressionLoss(MaskedLoss):
    """
    Base class for regression losses.

    This extends MaskedLoss with additional functionality specific
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


class DistributionLoss(MaskedLoss):
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


class WeightedLossWrapper(MaskedLoss):
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

        # Skip masking if not needed
        if mask is None:
            # PyTorch losses already expect (y_pred, target) order
            loss = self.torch_loss(y_pred, target)
        else:
            # Apply mask if provided
            target_masked = self._apply_mask(target, mask)
            y_pred_masked = self._apply_mask(y_pred, mask)

            # PyTorch losses already expect (y_pred, target) order
            loss = self.torch_loss(y_pred_masked, target_masked)

        # Handle weights and reduction
        return self._reduce(loss, mask, weights)

# Create weighted versions of PyTorch losses
WeightedMSELoss = WeightedLossWrapper(nn.MSELoss)
WeightedL1Loss = WeightedLossWrapper(nn.L1Loss)
WeightedCrossEntropyLoss = WeightedLossWrapper(nn.CrossEntropyLoss)
WeightedBCELoss = WeightedLossWrapper(nn.BCELoss)
WeightedBCEWithLogitsLoss = WeightedLossWrapper(nn.BCEWithLogitsLoss)
WeightedKLDivLoss = WeightedLossWrapper(nn.KLDivLoss)
WeightedNLLLoss = WeightedLossWrapper(nn.NLLLoss)
WeightedSmoothL1Loss = WeightedLossWrapper(nn.SmoothL1Loss)
WeightedHuberLoss = WeightedLossWrapper(nn.HuberLoss)
WeightedPoissonNLLLoss = WeightedLossWrapper(nn.PoissonNLLLoss)
WeightedGaussianNLLLoss = WeightedLossWrapper(nn.GaussianNLLLoss)
WeightedCTCLoss = WeightedLossWrapper(nn.CTCLoss)
WeightedCosineEmbeddingLoss = WeightedLossWrapper(nn.CosineEmbeddingLoss)
WeightedHingeEmbeddingLoss = WeightedLossWrapper(nn.HingeEmbeddingLoss)
WeightedMarginRankingLoss = WeightedLossWrapper(nn.MarginRankingLoss)
WeightedMultiMarginLoss = WeightedLossWrapper(nn.MultiMarginLoss)
WeightedMultiLabelMarginLoss = WeightedLossWrapper(nn.MultiLabelMarginLoss)
WeightedSoftMarginLoss = WeightedLossWrapper(nn.SoftMarginLoss)
WeightedMultiLabelSoftMarginLoss = WeightedLossWrapper(nn.MultiLabelSoftMarginLoss)
WeightedTripletMarginLoss = WeightedLossWrapper(nn.TripletMarginLoss)
WeightedTripletMarginWithDistanceLoss = WeightedLossWrapper(nn.TripletMarginWithDistanceLoss)


def create_weighted_losses():
    """
    Factory function that returns all weighted versions of standard PyTorch losses.
    
    Returns:
        Dictionary mapping from loss name to weighted loss instance
    """
    weighted_losses = {
        "MSELoss": WeightedMSELoss,
        "L1Loss": WeightedL1Loss, 
        "CrossEntropyLoss": WeightedCrossEntropyLoss,
        "BCELoss": WeightedBCELoss,
        "BCEWithLogitsLoss": WeightedBCEWithLogitsLoss,
        "KLDivLoss": WeightedKLDivLoss,
        "NLLLoss": WeightedNLLLoss,
        "SmoothL1Loss": WeightedSmoothL1Loss,
        "HuberLoss": WeightedHuberLoss,
        "PoissonNLLLoss": WeightedPoissonNLLLoss,
        "GaussianNLLLoss": WeightedGaussianNLLLoss,
        "CTCLoss": WeightedCTCLoss,
        "CosineEmbeddingLoss": WeightedCosineEmbeddingLoss,
        "HingeEmbeddingLoss": WeightedHingeEmbeddingLoss,
        "MarginRankingLoss": WeightedMarginRankingLoss,
        "MultiMarginLoss": WeightedMultiMarginLoss,
        "MultiLabelMarginLoss": WeightedMultiLabelMarginLoss,
        "SoftMarginLoss": WeightedSoftMarginLoss,
        "MultiLabelSoftMarginLoss": WeightedMultiLabelSoftMarginLoss,
        "TripletMarginLoss": WeightedTripletMarginLoss,
        "TripletMarginWithDistanceLoss": WeightedTripletMarginWithDistanceLoss,
    }
    return weighted_losses
