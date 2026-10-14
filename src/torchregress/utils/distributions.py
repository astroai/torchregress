"""Shared distribution primitives for losses, metrics, and calibration."""

from __future__ import annotations

import math

import torch
from torch import Tensor

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def normal_cdf(z: Tensor) -> Tensor:
    """Standard normal CDF ``Phi(z) = 0.5 * erfc(-z / sqrt(2))``.

    The ``erfc`` form keeps full relative precision in the lower tail (``0.5 * (1 +
    erf(z / sqrt(2)))`` cancels to 0 for ``z < -8`` in float64).
    """
    return 0.5 * torch.erfc(-z * _INV_SQRT2)
