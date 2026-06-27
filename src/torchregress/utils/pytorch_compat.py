"""
PyTorch compatibility utilities.

This module provides functions for handling compatibility across
different PyTorch versions.
"""

import random
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
