"""
PyTorch compatibility utilities.

This module provides functions for handling compatibility across
different PyTorch versions.
"""

import gc
import random
from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """
    Set all seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device based on availability and request.
    """
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def _collect_transform_cycles() -> Generator[None]:
    """Free the reference cycles that inverting a ``torch.distributions`` transform leaves.

    ``Transform.inv`` caches its ``_InverseTransform`` with a strong reference
    (torch 2.14), and the inverse references the transform back, so every
    ``t.inv(y)`` call on a freshly built transform leaves a cycle holding that
    transform's parameter tensors.  zuko builds such transforms per call and
    per autoregressive pass (``meta(x).inv(y)``), so sampling an NSF with
    ``N`` draws strands ``O(N * features * transforms * bins)`` floats (about
    11 KiB per 6-D draw, 3 transforms, 8 bins) until Python's cyclic collector
    runs.  Its thresholds count Python objects, not tensor bytes, so on CUDA
    the garbage piles up across calls until the device runs out of memory.

    Automatic collection is paused inside the block so every object it creates
    stays in the youngest generation, then that generation alone is collected
    on exit: the cost scales with the block's own objects, not the heap.
    Results are unchanged; only the lifetime of unreachable tensors is.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        gc.collect(0)
