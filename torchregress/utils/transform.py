"""
Transform utilities for regression tasks.
"""

import torch
from torch import Tensor


def log_transform(x: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Apply log transform with epsilon stability.
    """
    return torch.log(x + eps)


def log_inverse(y: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Inverse of log transform.
    """
    return torch.clamp(torch.exp(y) - eps, min=0.0)


def boxcox_transform(x: Tensor, lam: float = 0.0, eps: float = 1e-6) -> Tensor:
    """
    Apply Box-Cox transform.
    """
    x_shift = x + eps
    if abs(lam) < 1e-8:
        return torch.log(x_shift)
    return (torch.pow(x_shift, lam) - 1.0) / lam


def boxcox_inverse(y: Tensor, lam: float = 0.0, eps: float = 1e-6) -> Tensor:
    """
    Inverse of Box-Cox transform.
    """
    if abs(lam) < 1e-8:
        return torch.clamp(torch.exp(y) - eps, min=0.0)
    return torch.clamp((y * lam + 1.0).pow(1.0 / lam) - eps, min=0.0)


def sqrt_transform(x: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Apply square-root transform.
    """
    return torch.sqrt(x + eps)


def sqrt_inverse(y: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Inverse of square-root transform.
    """
    return torch.clamp(y.pow(2) - eps, min=0.0)
