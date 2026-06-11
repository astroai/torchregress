"""Shared distribution primitives for losses, metrics, and calibration."""

from __future__ import annotations

import torch
from torch import Tensor


def normal_cdf(z: Tensor) -> Tensor:
    """Standard normal CDF via ``torch.erf``."""
    sqrt_two = torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype))
    return 0.5 * (1.0 + torch.erf(z / sqrt_two))
