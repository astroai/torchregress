"""
Utility functions for metrics calculations.
"""

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torchmetrics  # noqa: F401
from torchmetrics import Metric  # noqa: F401


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


def create_metric_result(
    result: Union[torch.Tensor, float, Dict], as_numpy: bool
) -> Union[torch.Tensor, float, np.ndarray, Dict]:
    """Convert result to appropriate type based on input."""
    if isinstance(result, Dict):
        # Convert dict values - always convert scalar tensors to float
        return {
            k: (
                float(v.item())
                if isinstance(v, torch.Tensor) and v.numel() == 1
                else v.cpu().numpy() if isinstance(v, torch.Tensor) and as_numpy else v
            )
            for k, v in result.items()
        }
    elif isinstance(result, torch.Tensor):
        if result.numel() == 1:
            return float(result.item())
        return result.cpu().numpy() if as_numpy else result
    return result


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
) -> Dict[str, Union[torch.Tensor, float, np.ndarray]]:
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
    results: Dict[str, Union[torch.Tensor, float, np.ndarray]] = {}
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
