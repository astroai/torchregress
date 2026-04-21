"""Constraint-aware output heads for regression models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NonNegativeHead(nn.Module):
    """Wrap a base head and enforce non-negative outputs via softplus."""

    def __init__(self, module: nn.Module, beta: float = 1.0) -> None:
        super().__init__()
        self.module = module
        self.beta = beta

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return F.softplus(self.module(x, *args, **kwargs), beta=self.beta)


class BoundedHead(nn.Module):
    """Wrap a base head and map outputs into [low, high] via sigmoid."""

    def __init__(self, module: nn.Module, low: float = 0.0, high: float = 1.0) -> None:
        super().__init__()
        if not low < high:
            raise ValueError("BoundedHead requires low < high")
        self.module = module
        self.low = low
        self.high = high

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        raw = self.module(x, *args, **kwargs)
        return self.low + (self.high - self.low) * torch.sigmoid(raw)


class SimplexHead(nn.Module):
    """Wrap a base head and enforce simplex outputs (non-negative, sum-to-one)."""

    def __init__(self, module: nn.Module, dim: int = -1) -> None:
        super().__init__()
        self.module = module
        self.dim = dim

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        raw = self.module(x, *args, **kwargs)
        return torch.softmax(raw, dim=self.dim)


class NonCrossingSort(nn.Module):
    """Sort outputs along a dimension to enforce non-crossing order."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.sort(x, dim=self.dim).values


class SpectralNormWrapper(nn.Module):
    """Apply spectral normalization to a module and expose a wrapper forward."""

    def __init__(self, module: nn.Module, name: str = "weight") -> None:
        super().__init__()
        self.module = nn.utils.parametrizations.spectral_norm(module, name=name)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        out = self.module(*args, **kwargs)
        if not isinstance(out, torch.Tensor):
            raise TypeError("SpectralNormWrapper expects wrapped module to return a Tensor")
        return out


__all__ = [
    "NonNegativeHead",
    "BoundedHead",
    "SimplexHead",
    "NonCrossingSort",
    "SpectralNormWrapper",
]
