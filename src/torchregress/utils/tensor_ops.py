"""
Tensor operations for regression tasks.

This module provides functions for common tensor operations used in
regression, including tensor conversion and masked reductions.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


def convert_to_tensor(
    x: Union[torch.Tensor, np.ndarray, List, float, int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert common array-like inputs to tensors without silent downcasting.

    Policy (TR-MET-14):
    - numpy arrays are **copied** (never aliased) and keep their dtype;
    - floating-point inputs keep their precision (float64 stays float64);
      only integer/boolean inputs are promoted to float32;
    - python scalars become 0-dim tensors;
    - optional ``dtype``/``device`` are applied after conversion.
    """
    if isinstance(x, torch.Tensor):
        tensor = x
    elif isinstance(x, np.ndarray):
        tensor = torch.from_numpy(np.array(x, copy=True))
    elif isinstance(x, (list, tuple)):
        tensor = torch.as_tensor(x)
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.float32)
    elif isinstance(x, (float, int, bool)):
        tensor = torch.as_tensor(x).reshape(())
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.float32)
    else:
        raise TypeError(f"Cannot convert {type(x)} to torch.Tensor")

    if dtype is not None:
        tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def ensure_batch_dim(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has a batch dimension."""
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def apply_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Apply a mask to a tensor.

    Args:
        tensor: Input tensor of any shape
        mask: Boolean mask with compatible shape for broadcasting

    Returns:
        Original tensor if mask is None, otherwise a copy with masked values set to 0

    Examples:
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> mask = torch.tensor([True, False, True, False])
        >>> apply_mask(x, mask)
        tensor([1., 0., 3., 0.])
    """
    if mask is None:
        return tensor

    return torch.where(mask, tensor, torch.zeros_like(tensor))


def masked_reduction(
    tensor: torch.Tensor, mask: Optional[torch.Tensor], reduction: str = "mean"
) -> torch.Tensor:
    """
    Apply reduction to a tensor with optional masking.

    Args:
        tensor: Input tensor to reduce
        mask: Optional boolean mask indicating valid values (True for valid)
        reduction: Reduction method ('none', 'mean', 'sum', 'max', 'min')

    Returns:
        Reduced tensor based on specified reduction method

    Examples:
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> mask = torch.tensor([True, False, True, False])
        >>> masked_reduction(x, mask, reduction='mean')
        tensor(2.)  # (1 + 3) / 2
    """
    if reduction == "none":
        return tensor

    if mask is None:
        if reduction == "mean":
            return torch.mean(tensor)
        elif reduction == "sum":
            return torch.sum(tensor)
        elif reduction == "max":
            return torch.max(tensor)
        elif reduction == "min":
            return torch.min(tensor)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    if reduction == "mean":
        return masked_mean(tensor, mask)
    elif reduction == "sum":
        return masked_sum(tensor, mask)
    else:
        # For max/min, we need to handle masked values specially
        if reduction == "max":
            fill_value = torch.finfo(tensor.dtype).min
        elif reduction == "min":
            fill_value = torch.finfo(tensor.dtype).max
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        masked_tensor = torch.where(mask, tensor, fill_value)
        return masked_tensor.max() if reduction == "max" else masked_tensor.min()


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    Compute masked mean along specified dimension(s).
    """
    if mask is None:
        return torch.mean(tensor, dim=dim, keepdim=keepdim)

    valid_count = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1)
    return (tensor * mask).sum(dim=dim, keepdim=keepdim) / valid_count


def masked_sum(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    Compute masked sum along specified dimension(s).
    """
    if mask is None:
        return torch.sum(tensor, dim=dim, keepdim=keepdim)

    return (tensor * mask).sum(dim=dim, keepdim=keepdim)


def prepare_cross_covariance(
    cov_xy: torch.Tensor,
    n_dims_x: int,
    n_dims_y: int,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Prepare cross-covariance matrix for correlated input-output noise.
    """
    if cov_xy is None:
        return torch.zeros((n_dims_y, n_dims_x), device=device, dtype=dtype)

    if isinstance(cov_xy, torch.Tensor):
        cov_xy = cov_xy.to(device=device, dtype=dtype)
        if cov_xy.shape == (n_dims_y, n_dims_x):
            return cov_xy
        raise ValueError(
            f"Cross-covariance shape {cov_xy.shape} doesn't match required shape "
            f"({n_dims_y}, {n_dims_x})."
        )

    raise TypeError(f"Cross-covariance must be a tensor, got {type(cov_xy).__name__}")


def prepare_model_input_for_gradients(x: torch.Tensor) -> torch.Tensor:
    """
    Prepare model input for gradient calculation.
    """
    if x.requires_grad:
        return x
    return x.clone().requires_grad_(True)


def compute_model_gradients(
    y_pred: torch.Tensor, x: torch.Tensor, n_features_y: int
) -> torch.Tensor:
    """
    Compute gradients of model predictions with respect to inputs (Jacobian per sample).
    """
    batch_size = x.shape[0]
    n_features_x = x.shape[1]

    if n_features_y == 1:
        grads = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grads.view(batch_size, 1, n_features_x)
    else:
        # Multi-output case: need Jacobian per sample
        # We compute this efficiently using batched gradients over the output dimension
        grad_outputs = (
            torch.eye(n_features_y, device=x.device, dtype=x.dtype)
            .unsqueeze(1)
            .expand(-1, batch_size, -1)
        )
        grads = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
            is_grads_batched=True,
        )[0]
        return grads.transpose(0, 1)


def calculate_gaussian_nll(
    residuals: torch.Tensor, var: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Calculate Gaussian negative log-likelihood.
    """
    if residuals.numel() == 0:
        return torch.zeros(0, device=residuals.device, dtype=residuals.dtype)

    device = residuals.device
    dtype = residuals.dtype

    elementwise_diagonal = var.shape == residuals.shape or var.dim() == residuals.dim() - 1
    full_covariance = var.dim() == residuals.dim() + 1

    if elementwise_diagonal:
        # Diagonal covariance case
        nll = 0.5 * (torch.log(var + eps) + (residuals**2) / (var + eps))
        nll = torch.sum(nll, dim=1)
        nll = nll + 0.5 * residuals.shape[1] * math.log(2 * math.pi)
    elif full_covariance:
        # Full covariance case [batch, Dy, Dy]
        # Add jitter for stability
        Dy = residuals.shape[1]
        jitter = torch.eye(Dy, device=device, dtype=dtype) * eps
        var_stable = var + jitter

        mvn = torch.distributions.MultivariateNormal(
            torch.zeros_like(residuals), covariance_matrix=var_stable
        )
        nll = -mvn.log_prob(residuals)
    else:
        raise AssertionError(
            "calculate_gaussian_nll supports var with the same shape as residuals "
            f"{tuple(residuals.shape)} (elementwise), a broadcastable per-sample diagonal "
            f"with dim == residuals.dim() - 1, or a full covariance [batch, D, D]; "
            f"got {tuple(var.shape)}"
        )

    return nll


def calculate_propagated_variance(
    grad: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: Optional[torch.Tensor] = None,
    sigma_xy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate propagated variance from input to output.
    """
    propagated_var = grad @ sigma_x @ grad.transpose(-1, -2)

    if sigma_y is not None:
        propagated_var += sigma_y

    if sigma_xy is not None:
        propagated_var -= grad @ sigma_xy.transpose(-1, -2) + sigma_xy @ grad.transpose(-1, -2)

    return propagated_var
