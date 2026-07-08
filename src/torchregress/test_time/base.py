"""Interfaces and small shared structures for test-time tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import numpy as np
import torch

from torchregress.prediction import PredictiveBatch


@runtime_checkable
class SupportsPredictiveBatch(Protocol):
    """Protocol for objects that can generate predictive batches."""

    def predict_distribution(
        self,
        X: torch.Tensor | np.ndarray,
        **kwargs: object,
    ) -> PredictiveBatch: ...


@dataclass(frozen=True)
class AdaptationBatch:
    """Batch container for unlabeled target-time adaptation utilities."""

    x: np.ndarray | torch.Tensor
    predictions: PredictiveBatch | None = None
    representations: np.ndarray | torch.Tensor | None = None
    sigma_x: np.ndarray | torch.Tensor | None = None


def flatten_adaptation_parameters(
    groups: dict[str, Iterable[torch.nn.Parameter]],
) -> list[torch.nn.Parameter]:
    """
    Flatten a dictionary of parameter groups into a unique list of parameters.

    Parameters
    ----------
    groups : dict[str, Iterable[torch.nn.Parameter]]
        Dictionary mapping group names to iterables of PyTorch parameters.

    Returns
    -------
    list[torch.nn.Parameter]
        A flat list of unique parameters.
    """
    seen: set[int] = set()
    params: list[torch.nn.Parameter] = []
    for values in groups.values():
        for param in values:
            ident = id(param)
            if ident not in seen:
                seen.add(ident)
                params.append(param)
    return params


__all__ = [
    "AdaptationBatch",
    "flatten_adaptation_parameters",
]
