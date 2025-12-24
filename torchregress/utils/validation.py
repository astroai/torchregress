"""
Input validation utilities for regression tasks.

This module provides functions to validate input parameters
and ensure they meet the required constraints for regression models.

These utilities are used throughout the library to provide consistent
error checking and helpful error messages.
"""

from typing import List, Optional, Union

import torch


def validate_reduction(reduction: str, valid_reductions: Optional[list] = None) -> str:
    """
    Validate reduction method.

    Args:
        reduction: Reduction method to validate
        valid_reductions: List of valid reductions (defaults to ['none', 'mean', 'sum'])

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
        valid_reductions = ["none", "mean", "sum"]

    if reduction not in valid_reductions:
        raise ValueError(f"reduction must be one of {valid_reductions}, got {reduction}")

    return reduction


def validate_shape(
    tensor: torch.Tensor, expected_shape: tuple, param_name: str, allow_broadcast: bool = True
) -> torch.Tensor:
    """
    Validate that a tensor has the expected shape.

    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (can contain None for any size)
        param_name: Name of parameter (for error messages)
        allow_broadcast: Whether broadcasting dimensions are allowed

    Returns:
        Input tensor if valid

    Raises:
        ValueError: If tensor shape is invalid

    Examples:
        >>> x = torch.randn(3, 4)
        >>> validate_shape(x, (3, 4), "x")
        tensor(...)
        >>> validate_shape(x, (3, None), "x")
        tensor(...)
        >>> validate_shape(x, (2, 4), "x")
        Traceback (most recent call last):
            ...
        ValueError: x has shape torch.Size([3, 4]), expected (2, 4)
    """
    if len(tensor.shape) != len(expected_shape):
        raise ValueError(
            f"{param_name} has {len(tensor.shape)} dimensions, " f"expected {len(expected_shape)}"
        )

    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected is not None:
            if actual != expected:
                if not allow_broadcast or actual != 1:
                    raise ValueError(
                        f"{param_name} has shape {tensor.shape}, " f"expected {expected_shape}"
                    )
    return tensor


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
                f"{param_name} must be between {min_value} and {max_value}, " f"got {value}"
            )
    return value


def validate_integer(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """
    Validate that a tensor has an integer dtype.

    Args:
        tensor: Tensor to validate
        name: Name of the tensor for error messages

    Returns:
        The validated tensor

    Raises:
        ValueError: If tensor does not have an integer dtype
    """
    if not tensor.dtype.is_floating_point:
        return tensor

    if not torch.allclose(tensor, tensor.round()):
        raise ValueError(
            f"{name} must contain only integer values, got tensor with non-integer values"
        )

    return tensor.to(torch.int64)


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


def validate_batch_consistency(
    tensors: List[torch.Tensor], names: Optional[List[str]] = None
) -> None:
    """
    Validates that all tensors have the same batch dimension.

    Args:
        tensors: List of tensors to validate
        names: Names of the tensors for error messages

    Raises:
        ValueError: If tensors have inconsistent batch dimensions

    Examples:
        >>> a = torch.randn(3, 4)
        >>> b = torch.randn(3, 5)
        >>> validate_batch_consistency([a, b])  # No error, same batch size
        >>> c = torch.randn(5, 4)
        >>> validate_batch_consistency([a, c])
        Traceback (most recent call last):
            ...
        ValueError: Batch size mismatch: tensor_0 has batch size 3, but tensor_1 has batch size 5
    """
    if not tensors:
        return

    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]

    batch_size = tensors[0].shape[0]

    for tensor, name in zip(tensors[1:], names[1:]):
        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: {names[0]} has batch size {batch_size}, "
                f"but {name} has batch size {tensor.shape[0]}"
            )


def validate_same_device(
    tensors: List[torch.Tensor], names: Optional[List[str]] = None
) -> torch.device:
    """
    Validates that all tensors are on the same device.

    Args:
        tensors: List of tensors to validate
        names: Names of the tensors for error messages

    Returns:
        The common device of all tensors

    Raises:
        ValueError: If tensors are on different devices

    Examples:
        >>> a = torch.randn(3, 4)
        >>> b = torch.randn(2, 5)
        >>> validate_same_device([a, b])  # Both on CPU
        device(type='cpu')
        >>> if torch.cuda.is_available():
        ...     c = torch.randn(3, 4, device='cuda:0')
        ...     validate_same_device([a, c])  # Error: different devices
        Traceback (most recent call last):
            ...
        ValueError: Device mismatch: tensor_0 is on cpu, but tensor_1 is on cuda:0
    """
    if not tensors:
        raise ValueError("No tensors provided")

    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]

    device = tensors[0].device

    for tensor, name in zip(tensors[1:], names[1:]):
        if tensor.device != device:
            raise ValueError(
                f"Device mismatch: {names[0]} is on {device}, but {name} is on {tensor.device}"
            )

    return device


def validate_weights(
    weights: Optional[torch.Tensor], batch_size: int, allow_none: bool = True
) -> Optional[torch.Tensor]:
    """
    Validates sample weights for loss functions.

    Args:
        weights: Sample weights tensor or None
        batch_size: Expected batch size
        allow_none: Whether None is accepted as a valid value

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

    return weights


def check_tensor(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """
    Validates a tensor for common issues - checks for NaNs, infs, and ensures it's
    a proper torch tensor.

    Args:
        tensor: The tensor to validate
        name: Name of the tensor for error messages

    Returns:
        The validated tensor

    Raises:
        ValueError: If tensor contains NaN or inf values
        TypeError: If input is not a torch.Tensor

    Examples:
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> validate_tensor(x)
        tensor([1., 2., 3.])
        >>> y = torch.tensor([1.0, float('nan'), 3.0])
        >>> validate_tensor(y)
        Traceback (most recent call last):
            ...
        ValueError: tensor contains NaN values
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")

    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")

    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains infinite values")

    return tensor
