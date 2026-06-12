"""Interfaces and small shared structures for test-time tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import numpy as np
import torch

from torchregress.prediction import PredictiveBatch


@dataclass(frozen=True)
class AdaptationBatch:
    """Batch container for unlabeled target-time adaptation utilities."""

    x: np.ndarray | torch.Tensor
    predictions: PredictiveBatch | None = None
    representations: np.ndarray | torch.Tensor | None = None
    sigma_x: np.ndarray | torch.Tensor | None = None


@runtime_checkable
class SupportsPredictiveBatch(Protocol):
    """
    Protocol for models supporting batch-level distributional predictions.
    """

    def predict_distribution(self, X: np.ndarray, **kwargs: object) -> PredictiveBatch: ...


@runtime_checkable
class SupportsRepresentation(Protocol):
    """
    Protocol for models that can expose internal feature representations.
    """

    def representation_dict(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]: ...


@runtime_checkable
class SupportsAdaptationParameters(Protocol):
    """
    Protocol for models exposing parameter groups suitable for test-time adaptation.
    """

    def adaptation_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]: ...


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
    "SupportsAdaptationParameters",
    "SupportsPredictiveBatch",
    "SupportsRepresentation",
    "flatten_adaptation_parameters",
]
