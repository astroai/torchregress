"""
Tensor operations for regression tasks.

This module provides functions for common tensor operations used in
regression, including tensor conversion, standardization, and special
linear algebra operations.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


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

    return tensor * mask


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

        return torch.where(mask, tensor, fill_value).max() if reduction == "max" else torch.where(mask, tensor, fill_value).min()


def masked_mean(
    tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> torch.Tensor:
    """
    Compute masked mean along specified dimension(s).
    """
    if mask is None:
        return torch.mean(tensor, dim=dim, keepdim=keepdim)

    valid_count = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1)
    return (tensor * mask).sum(dim=dim, keepdim=keepdim) / valid_count


def masked_sum(
    tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> torch.Tensor:
    """
    Compute masked sum along specified dimension(s).
    """
    if mask is None:
        return torch.sum(tensor, dim=dim, keepdim=keepdim)

    return (tensor * mask).sum(dim=dim, keepdim=keepdim)


def prepare_param(
    param: Union[float, torch.Tensor], n_dims: int, device: torch.device, default_value: float = 1.0
) -> torch.Tensor:
    """
    Prepare a parameter tensor for use in models.
    """
    if param is None:
        return torch.full((n_dims,), default_value, device=device)

    if isinstance(param, (int, float)):
        return torch.full((n_dims,), param, device=device)

    if isinstance(param, torch.Tensor):
        param = param.to(device)
        if param.numel() == 1:
            return param.expand(n_dims)
        if param.numel() == n_dims:
            return param.reshape(n_dims)
        raise ValueError(f"Parameter shape {param.shape} doesn't match required size {n_dims}.")

    raise TypeError(f"Parameter must be float or tensor, got {type(param).__name__}")


def prepare_sigma(
    sigma: Union[float, torch.Tensor], n_dims: int, device: torch.device, default_zero: bool = True
) -> torch.Tensor:
    """
    Prepare sigma (standard deviation) parameter for noise models.
    """
    default = 0.0 if default_zero else 1.0
    return prepare_param(sigma, n_dims, device, default)


def prepare_covariance(
    cov: Union[float, torch.Tensor], n_dims: int, device: torch.device
) -> torch.Tensor:
    """
    Prepare covariance matrix or vector for multivariate models.
    """
    if cov is None:
        return torch.eye(n_dims, device=device)

    if isinstance(cov, (int, float)):
        return torch.eye(n_dims, device=device) * cov

    if isinstance(cov, torch.Tensor):
        cov = cov.to(device)
        if cov.numel() == 1:
            return torch.eye(n_dims, device=device) * cov.item()
        if cov.ndim == 1:
            if cov.shape[0] != n_dims:
                raise ValueError(f"Diagonal covariance shape {cov.shape} doesn't match required dimensions {n_dims}.")
            return torch.diag(cov)
        if cov.ndim == 2:
            if cov.shape != (n_dims, n_dims):
                raise ValueError(f"Covariance matrix shape {cov.shape} doesn't match required shape ({n_dims}, {n_dims}).")
            if not torch.allclose(cov, cov.t()):
                import warnings
                warnings.warn("Covariance matrix is not symmetric. Using (cov + cov.T) / 2.")
                cov = (cov + cov.t()) / 2
            return cov
        raise ValueError(f"Covariance must be scalar, vector or matrix, got tensor with {cov.ndim} dimensions")

    raise TypeError(f"Covariance must be float or tensor, got {type(cov).__name__}")


def prepare_cross_covariance(
    cov_xy: torch.Tensor, n_dims_x: int, n_dims_y: int, device: torch.device
) -> torch.Tensor:
    """
    Prepare cross-covariance matrix for correlated input-output noise.
    """
    if cov_xy is None:
        return torch.zeros((n_dims_y, n_dims_x), device=device)

    if isinstance(cov_xy, torch.Tensor):
        cov_xy = cov_xy.to(device)
        if cov_xy.shape == (n_dims_y, n_dims_x):
            return cov_xy
        raise ValueError(f"Cross-covariance shape {cov_xy.shape} doesn't match required shape ({n_dims_y}, {n_dims_x}).")

    raise TypeError(f"Cross-covariance must be a tensor, got {type(cov_xy).__name__}")


def prepare_model_input_for_gradients(x: torch.Tensor) -> torch.Tensor:
    """
    Prepare model input for gradient calculation.
    """
    return x.detach().clone().requires_grad_(True)


def batched_linalg_solve(
    A: torch.Tensor, b: torch.Tensor, ridge_factor: float = 1e-8
) -> torch.Tensor:
    """
    Solve multiple linear systems in batched mode.
    """
    try:
        return torch.linalg.solve(A, b)
    except RuntimeError:
        try:
            A_jitter = A + ridge_factor * torch.eye(A.shape[-1], device=A.device)
            return torch.linalg.solve(A_jitter, b)
        except RuntimeError:
            return torch.linalg.pinv(A) @ b


def standardize(
    x: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize data to have zero mean and unit variance.
    """
    if mean is None:
        mean = x.mean(dim=0)
    if std is None:
        std = x.std(dim=0)
    return (x - mean) / (std + eps), mean, std


def unstandardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Reverse standardization transform.
    """
    return x * std + mean


def compute_model_gradients(
    y_pred: torch.Tensor, x: torch.Tensor, n_features_y: int
) -> torch.Tensor:
    """
    Compute gradients of model predictions with respect to inputs.
    """
    grads = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return grads.view(-1, n_features_y, x.shape[-1])


def calculate_gaussian_nll(
    residuals: torch.Tensor, var: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Calculate Gaussian negative log-likelihood.
    """
    if var.dim() <= 2:
        # Diagonal covariance case
        nll = 0.5 * (torch.log(var + eps) + (residuals**2) / (var + eps))
        nll = torch.sum(nll, dim=1)
        nll = nll + 0.5 * residuals.shape[1] * math.log(2 * math.pi)
    else:
        # Full covariance case
        mvn = torch.distributions.MultivariateNormal(torch.zeros_like(residuals), covariance_matrix=var)
        nll = -mvn.log_prob(residuals)

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
        propagated_var += grad @ sigma_xy.transpose(-1, -2) + sigma_xy @ grad.transpose(-1, -2)

    return propagated_var