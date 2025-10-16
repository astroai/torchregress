"""
PyTorch compatibility utilities.

This module provides functions for handling compatibility across
different PyTorch versions.
"""

import random
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def convert_reduction_type(reduction: str) -> str:
    """
    Convert between different reduction naming conventions.
    """
    reduction = reduction.lower()
    if reduction in ["none", "mean", "sum"]:
        return reduction
    if reduction in ["batch_mean", "average"]:
        return "mean"
    raise ValueError(f"Unsupported reduction type: {reduction}")


def convert_to_pytorch_loss(loss_fn: Union[nn.Module, Callable], **kwargs) -> nn.Module:
    """
    Convert a loss function (callable or module) to a PyTorch loss module.
    """
    if isinstance(loss_fn, nn.Module):
        return loss_fn

    class LossFunctionWrapper(nn.Module):
        def __init__(self, loss_fn, **kwargs):
            super().__init__()
            self.loss_fn = loss_fn
            self.kwargs = kwargs

        def forward(self, output, target):
            return self.loss_fn(target, output, **self.kwargs)

    return LossFunctionWrapper(loss_fn, **kwargs)


def extract_output_size(
    model: nn.Module, input_size: Union[int, List[int], Tuple[int, ...]]
) -> Tuple[int, ...]:
    """
    Determine the output size of a PyTorch model given input size.
    """
    if isinstance(input_size, int):
        input_size = (input_size,)

    dummy_input = torch.zeros(1, *input_size)
    with torch.no_grad():
        output = model(dummy_input)

    if isinstance(output, torch.Tensor):
        return tuple(output.shape[1:])
    if isinstance(output, tuple) and all(isinstance(o, torch.Tensor) for o in output):
        return tuple(o.shape[1:] for o in output)
    raise ValueError(f"Model output is not a tensor or tuple of tensors, got {type(output)}")


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