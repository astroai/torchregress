"""
Utility functions for metrics calculations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import numpy as np
import torch


def convert_to_tensor(x: Union[torch.Tensor, np.ndarray, List, float, int]) -> torch.Tensor:
    """Convert various input types to torch tensors."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, (float, int)):
        return torch.tensor([x], dtype=torch.float32)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise TypeError(f"Cannot convert {type(x)} to torch.Tensor")


def apply_reduction(values: torch.Tensor, reduction: str) -> Union[torch.Tensor, float]:
    """Apply reduction to tensor values."""
    if reduction == "none":
        return values
    elif reduction == "mean":
        return torch.mean(values)
    elif reduction == "sum":
        return torch.sum(values)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def ensure_batch_dim(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has a batch dimension."""
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def create_metric_result(result: Any, as_numpy: bool) -> Any:
    """Convert result to appropriate type based on input."""
    if isinstance(result, dict):
        converted: dict[str, Any] = {}
        for k, v in result.items():
            if isinstance(v, dict):
                converted[k] = create_metric_result(v, as_numpy=as_numpy)
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                converted[k] = float(v.item())
            elif isinstance(v, torch.Tensor) and as_numpy:
                converted[k] = v.cpu().numpy()
            else:
                converted[k] = v
        return converted
    elif isinstance(result, torch.Tensor):
        if result.numel() == 1:
            return float(result.item())
        return result.cpu().numpy() if as_numpy else result
    return result


T = TypeVar("T")


def metric_state_tensor(state: Any) -> torch.Tensor:
    """Cast a TorchMetrics state attribute to a tensor for mypy-friendly arithmetic."""
    return cast(torch.Tensor, state)


class _MetricStateListCaster:
    """Runtime-safe caster supporting plain and generic-style metric state list casting."""

    def __call__(self, state: Any) -> list[Any]:
        return cast(list[Any], state)

    def __getitem__(self, _item: Any) -> Callable[[Any], list[Any]]:
        # Enable generic-style runtime syntax used for typing readability.
        return self.__call__


metric_state_list = _MetricStateListCaster()


def validate_inputs(y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
    """Validate that inputs have compatible shapes."""
    if y_pred.dim() == 0 or y_true.dim() == 0:
        raise ValueError("Inputs cannot be scalars, must have at least one dimension")

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_pred and y_true must have same batch size. "
            f"Got y_pred: {y_pred.shape}, y_true: {y_true.shape}"
        )

    # For regression metrics, if shapes don't match exactly, try to broadcast
    if y_pred.shape != y_true.shape:
        try:
            # Test if broadcasting would work
            _ = y_pred + y_true
        except RuntimeError:
            raise ValueError(
                f"y_pred shape {y_pred.shape} and y_true shape {y_true.shape} are not compatible"
            )

    # Check for NaN or infinite values
    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
        raise ValueError("y_pred contains NaN or infinite values")

    if torch.isnan(y_true).any() or torch.isinf(y_true).any():
        raise ValueError("y_true contains NaN or infinite values")


def validate_sample_weight(sample_weight: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Validate sample weights have correct shape and are positive."""
    if sample_weight.dim() > 1 and sample_weight.shape[1] != 1:
        raise ValueError(
            f"Sample weights should be 1D or have shape (batch_size, 1). Got {sample_weight.shape}"
        )

    if sample_weight.shape[0] != batch_size:
        raise ValueError(
            f"Sample weights must have same first dimension as inputs. "
            f"Expected {batch_size}, got {sample_weight.shape[0]}"
        )

    if torch.any(sample_weight < 0):
        raise ValueError("Sample weights must be non-negative")

    return sample_weight.reshape(-1)


# TorchMetrics compatibility and metric composition utilities
def torchmetrics_available() -> bool:
    """Check if torchmetrics is installed."""
    return True


def compose_metrics(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    metrics: Dict[str, Callable],
    sample_weight: Optional[torch.Tensor] = None,
    as_numpy: bool = False,
) -> Dict[str, Any]:
    """
    Compose multiple metric functions into a single report.
    Args:
        preds: Predictions
        targets: Ground truth values
        metrics: Mapping from metric name to metric function
        sample_weight: Optional sample weights
        as_numpy: Whether to convert results to numpy arrays
    Returns:
        Dictionary of metric results
    """
    results: Dict[str, Any] = {}
    for name, fn in metrics.items():
        try:
            if sample_weight is not None:
                val = fn(preds, targets, sample_weight=sample_weight)
            else:
                val = fn(preds, targets)
        except TypeError:
            val = fn(preds, targets)
        results[name] = create_metric_result(val, as_numpy)
    return results
