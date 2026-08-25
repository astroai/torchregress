"""Shared robust elementwise losses.

Single source of truth for the Huber, log-cosh, and Tukey biweight
elementwise losses used by ``transforms.py``, ``uncertain_gt.py``, and
``robust.py`` (consolidated per A11).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["huber_elementwise", "log_cosh", "tukey_biweight"]


def huber_elementwise(residual: Tensor, delta: float = 1.0) -> Tensor:
    """Elementwise Huber loss with quadratic/linear crossover at ``delta``."""
    abs_residual = residual.abs()
    quadratic = torch.minimum(abs_residual, torch.full_like(abs_residual, float(delta)))
    linear = abs_residual - quadratic
    return 0.5 * quadratic.square() + float(delta) * linear


def log_cosh(residual: Tensor, scale: float = 1.0) -> Tensor:
    """Numerically stable ``log(cosh(u))`` computed as |u| + log1p(e^{-2|u|}) - log 2."""
    diff = float(scale) * residual
    abs_diff = torch.abs(diff)
    return abs_diff + torch.log1p(torch.exp(-2.0 * abs_diff)) - math.log(2.0)


def tukey_biweight(residual: Tensor, c: float = 4.685) -> Tensor:
    """Tukey biweight with constant saturation beyond ``c``."""
    c = float(c)
    c_squared_over_6 = c**2 / 6.0
    abs_residual = residual.abs()
    scaled = abs_residual / c
    squared = scaled**2
    mask_within = abs_residual <= c
    inlier = c_squared_over_6 * (1.0 - (1.0 - squared) ** 3)
    return torch.where(
        mask_within,
        inlier,
        torch.full_like(residual, float(c_squared_over_6)),
    )
