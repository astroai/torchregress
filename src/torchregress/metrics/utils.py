"""
Utility functions for metrics calculations.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar, cast

import torch

from torchregress.utils.tensor_ops import convert_to_tensor, ensure_batch_dim
from torchregress.utils.validation import validate_metric_inputs as validate_inputs
from torchregress.utils.validation import validate_sample_weight

__all__ = [
    "convert_to_tensor",
    "create_metric_result",
    "ensure_batch_dim",
    "metric_state_list",
    "metric_state_tensor",
    "validate_inputs",
    "validate_sample_weight",
]


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
