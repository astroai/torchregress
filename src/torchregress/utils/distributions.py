"""Shared distribution primitives for losses, metrics, and calibration."""

from __future__ import annotations

import torch
from torch import Tensor

_normal_dist = torch.distributions.Normal(0.0, 1.0, validate_args=False)


def normal_cdf(z: Tensor) -> Tensor:
    """Standard normal CDF."""
    return _normal_dist.cdf(z)
