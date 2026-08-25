"""
Input validation utilities for regression tasks.

This module provides functions to validate input parameters
and ensure they meet the required constraints for regression models.

These utilities are used throughout the library to provide consistent
error checking and helpful error messages.
"""

from typing import Optional, Union

import torch


def validate_reduction(reduction: str, valid_reductions: Optional[list] = None) -> str:
    """
    Validate reduction method.

    Args:
        reduction: Reduction method to validate
        valid_reductions: List of valid reductions (defaults to
            ['none', 'mean', 'sum'])

    Returns:
        Reduction method if valid

    Raises:
        ValueError: If reduction is not valid

    Examples:
        >>> validate_reduction('mean')
        'mean'
        >>> validate_reduction('median', ['mean', 'median', 'sum'])
        'median'
        >>> validate_reduction('unknown')
        Traceback (most recent call last):
            ...
        ValueError: reduction must be one of ['none', 'mean', 'sum'], got unknown
    """
    if valid_reductions is None:
        # A9: 'min'/'max' removed — no caller used them and REDUCERS never
        # implemented them (they raised KeyError at reduction time).
        valid_reductions = ["none", "mean", "sum"]

    if reduction not in valid_reductions:
        raise ValueError(f"reduction must be one of {valid_reductions}, got {reduction}")

    return reduction


def validate_positive(
    value: Union[float, torch.Tensor], param_name: str, allow_zero: bool = False
) -> Union[float, torch.Tensor]:
    """
    Validate that a value or tensor is positive (or non-negative).

    Args:
        value: Value or tensor to validate
        param_name: Name of parameter (for error messages)
        allow_zero: Whether zero is considered valid

    Returns:
        Input value if valid

    Raises:
        ValueError: If value is negative (or zero when not allowed)

    Examples:
        >>> validate_positive(5.0, "alpha")
        5.0
        >>> validate_positive(0.0, "alpha", allow_zero=True)
        0.0
        >>> validate_positive(-1.0, "alpha")
        Traceback (most recent call last):
            ...
        ValueError: alpha must be positive, got -1.0
        >>> validate_positive(torch.tensor([1.0, 2.0]), "weights")
        tensor([1., 2.])
    """
    if isinstance(value, torch.Tensor):
        if allow_zero:
            if torch.any(value < 0):
                raise ValueError(
                    f"{param_name} must be non-negative, got tensor with minimum value "
                    f"{value.min().item()}"
                )
        else:
            if torch.any(value <= 0):
                raise ValueError(
                    f"{param_name} must be positive, got tensor with minimum value "
                    f"{value.min().item()}"
                )
    else:
        if allow_zero:
            if value < 0:
                raise ValueError(f"{param_name} must be non-negative, got {value}")
        else:
            if value <= 0:
                raise ValueError(f"{param_name} must be positive, got {value}")
    return value


def validate_range(
    value: Union[float, torch.Tensor], min_value: float, max_value: float, param_name: str
) -> Union[float, torch.Tensor]:
    """
    Validate that a value or tensor is within specified range.

    Args:
        value: Value or tensor to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        param_name: Name of parameter (for error messages)

    Returns:
        Input value if valid

    Raises:
        ValueError: If value is outside valid range

    Examples:
        >>> validate_range(0.5, 0.0, 1.0, "probability")
        0.5
        >>> validate_range(torch.tensor([0.1, 0.9]), 0.0, 1.0, "probabilities")
        tensor([0.1000, 0.9000])
        >>> validate_range(1.5, 0.0, 1.0, "probability")
        Traceback (most recent call last):
            ...
        ValueError: probability must be between 0.0 and 1.0, got 1.5
    """
    if isinstance(value, torch.Tensor):
        if torch.any(value < min_value) or torch.any(value > max_value):
            raise ValueError(
                f"{param_name} must be between {min_value} and {max_value}, "
                f"got tensor with values outside range [{value.min().item()}, {value.max().item()}]"
            )
    else:
        if value < min_value or value > max_value:
            raise ValueError(
                f"{param_name} must be between {min_value} and {max_value}, got {value}"
            )
    return value


def validate_quantile(q: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Validate quantile level(s) and convert to tensor.
    """
    q = torch.as_tensor(q)
    q_min = q.min().item()
    q_max = q.max().item()
    if q_min < 0.0 or q_max > 1.0:
        raise ValueError(f"Quantile(s) must be in range [0, 1], got {q_min} to {q_max}")
    return q


def validate_weights(
    weights: Optional[torch.Tensor],
    batch_size: int,
    allow_none: bool = True,
    *,
    flatten: bool = False,
) -> Optional[torch.Tensor]:
    """
    Validates sample weights for loss functions.

    Args:
        weights: Sample weights tensor or None
        batch_size: Expected batch size
        allow_none: Whether None is accepted as a valid value
        flatten: When True, reshape ``(batch_size, 1)`` weights to 1D for metrics

    Returns:
        Validated weights or None

    Raises:
        ValueError: If weights have invalid shape or values

    Examples:
        >>> pred = torch.randn(5, 3)  # Batch size of 5
        >>> weights = torch.ones(5)
        >>> validate_weights(weights, pred.shape[0])
        tensor([1., 1., 1., 1., 1.])
        >>> validate_weights(None, pred.shape[0])
        >>> bad_weights = torch.tensor([-1.0, 1.0, 1.0, 1.0, 1.0])
        >>> validate_weights(bad_weights, pred.shape[0])
        Traceback (most recent call last):
            ...
        ValueError: weights must be non-negative, got tensor with minimum value -1.0
    """
    if weights is None:
        if allow_none:
            return None
        else:
            raise ValueError("weights cannot be None")

    # Check shape
    if weights.ndim > 2:
        raise ValueError(f"weights must have 1 or 2 dimensions, got {weights.ndim}")

    if weights.shape[0] != batch_size:
        raise ValueError(
            f"weights must have same batch size as inputs, got {weights.shape[0]}, "
            f"expected {batch_size}"
        )

    # Check values
    validate_positive(weights, "weights", allow_zero=True)

    if flatten:
        if weights.ndim > 1 and weights.shape[1] != 1:
            raise ValueError(
                f"Sample weights should be 1D or have shape (batch_size, 1). Got {weights.shape}"
            )
        return weights.reshape(-1)

    return weights


def validate_metric_inputs(y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
    """Validate metric tensor pairs with broadcast-tolerant shape checks."""
    if y_pred.dim() == 0 or y_true.dim() == 0:
        raise ValueError("Inputs cannot be scalars, must have at least one dimension")

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_pred and y_true must have same batch size. "
            f"Got y_pred: {y_pred.shape}, y_true: {y_true.shape}"
        )

    if y_pred.shape != y_true.shape:
        try:
            _ = y_pred + y_true
        except RuntimeError:
            raise ValueError(
                f"y_pred shape {y_pred.shape} and y_true shape {y_true.shape} are not compatible"
            ) from None

    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
        raise ValueError("y_pred contains NaN or infinite values")

    if torch.isnan(y_true).any() or torch.isinf(y_true).any():
        raise ValueError("y_true contains NaN or infinite values")


def validate_sample_weight(sample_weight: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Validate and flatten per-sample metric weights."""
    validated = validate_weights(sample_weight, batch_size, allow_none=False, flatten=True)
    assert validated is not None
    return validated


def check_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    max_elements: Optional[int] = int(2e8),  # ~800MB limit for float32
) -> torch.Tensor:
    """
    Validates a tensor for common issues - checks for NaNs, infs, max elements (DoS protection),
    and ensures it's a proper torch tensor.

    Args:
        tensor: The tensor to validate
        name: Name of the tensor for error messages
        max_elements: Maximum number of elements allowed (default 2e8). Set to None to disable.

    Returns:
        The validated tensor

    Raises:
        ValueError: If tensor contains NaN or inf values, or exceeds max_elements limit
        TypeError: If input is not a torch.Tensor

    Examples:
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> check_tensor(x)
        tensor([1., 2., 3.])
        >>> y = torch.tensor([1.0, float('nan'), 3.0])
        >>> check_tensor(y)
        Traceback (most recent call last):
            ...
        ValueError: tensor contains NaN values
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")

    # Defense against Denial of Service via Resource Exhaustion (OOM)
    if max_elements is not None and tensor.numel() > max_elements:
        raise ValueError(
            f"{name} contains {tensor.numel()} elements, "
            f"which exceeds the maximum allowed limit of {max_elements}."
        )

    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")

    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains infinite values")

    return tensor
