"""
Tensor operations for regression tasks.

This module provides functions for common tensor operations used in
regression, including tensor conversion, standardization, and special
linear algebra operations.
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple, List, Any

def apply_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Apply a boolean mask to a tensor.
    
    Args:
        x: Input tensor
        mask: Boolean mask of same shape as x or broadcastable to x
        
    Returns:
        Masked tensor with same values but zeros where mask is False
    """
    if mask is None:
        return x
    
    # Ensure mask is broadcastable to x
    if x.shape != mask.shape:
        try:
            return x * mask
        except RuntimeError:
            raise ValueError(f"Mask shape {mask.shape} cannot be broadcast to tensor shape {x.shape}")
    
    return x * mask

def masked_reduction(x: torch.Tensor, mask: Optional[torch.Tensor], 
                    reduction: str = 'mean') -> torch.Tensor:
    """
    Apply reduction to a tensor with optional masking.
    
    Args:
        x: Input tensor
        mask: Boolean mask of same shape as x or broadcastable to x
        reduction: 'none' | 'mean' | 'sum'
        
    Returns:
        Reduced tensor according to specified reduction type
    """
    if reduction == 'none':
        return x
        
    if mask is None:
        if reduction == 'mean':
            return torch.mean(x)
        else:  # 'sum'
            return torch.sum(x)
    else:
        # Apply mask
        masked_x = x * mask
        
        if reduction == 'mean':
            # Count non-masked values for mean
            valid_count = torch.sum(mask)
            if valid_count > 0:
                return torch.sum(masked_x) / valid_count
            else:
                return torch.tensor(0.0, device=x.device)
        else:  # 'sum'
            return torch.sum(masked_x)

def prepare_param(param: Union[float, torch.Tensor], n_dims: int, 
                 device: torch.device, default_value: float = 1.0) -> torch.Tensor:
    """
    Prepare a parameter tensor for use in models.
    
    Args:
        param: Input parameter (scalar or tensor)
        n_dims: Number of dimensions required
        device: Device to put the tensor on
        default_value: Default value if param is None
        
    Returns:
        Parameter tensor of appropriate shape
    """
    if param is None:
        return torch.tensor(default_value, device=device).expand(n_dims)
        
    if isinstance(param, (int, float)):
        return torch.tensor(param, device=device).expand(n_dims)
        
    if isinstance(param, torch.Tensor):
        param = param.to(device)
        
        if param.ndim == 0:
            # Scalar tensor
            return param.expand(n_dims)
        elif param.shape[0] == n_dims:
            return param
        else:
            raise ValueError(f"Parameter shape {param.shape} doesn't match required size {n_dims}")
    
    raise TypeError(f"Parameter must be float or tensor, got {type(param)}")

def prepare_sigma(sigma: Union[float, torch.Tensor], n_dims: int, 
                device: torch.device, default_zero: bool = True) -> torch.Tensor:
    """
    Prepare sigma (standard deviation) parameter for noise models.
    
    Args:
        sigma: Standard deviation (scalar or tensor)
        n_dims: Number of dimensions required
        device: Device to put the tensor on
        default_zero: Whether default value is 0.0 (if sigma is None)
        
    Returns:
        Sigma tensor of appropriate shape
    """
    default = 0.0 if default_zero else 1.0
    return prepare_param(sigma, n_dims, device, default)

def prepare_covariance(cov: Union[float, torch.Tensor], n_dims: int, 
                      device: torch.device) -> torch.Tensor:
    """
    Prepare covariance matrix or vector for multivariate models.
    
    Args:
        cov: Covariance (scalar, vector or matrix)
        n_dims: Dimensionality of feature space
        device: Device to put tensor on
        
    Returns:
        Covariance tensor of appropriate shape
    """
    if cov is None:
        # Default to identity matrix
        return torch.eye(n_dims, device=device)
        
    if isinstance(cov, (int, float)):
        # Scalar covariance - use as diagonal
        return torch.eye(n_dims, device=device) * cov
        
    if isinstance(cov, torch.Tensor):
        cov = cov.to(device)
        
        if cov.ndim == 0:
            # Scalar tensor
            return torch.eye(n_dims, device=device) * cov
        elif cov.ndim == 1:
            # Vector - interpret as diagonal entries
            if cov.shape[0] != n_dims:
                raise ValueError(f"Diagonal covariance shape {cov.shape} doesn't match " 
                               f"required dimensions {n_dims}")
            return torch.diag(cov)
        elif cov.ndim == 2:
            # Matrix - check shape
            if cov.shape != (n_dims, n_dims):
                raise ValueError(f"Covariance matrix shape {cov.shape} doesn't match "
                               f"required shape ({n_dims}, {n_dims})")
            return cov
        else:
            raise ValueError(f"Covariance must be scalar, vector or matrix, got tensor with {cov.ndim} dimensions")
    
    raise TypeError(f"Covariance must be float or tensor, got {type(cov)}")

def prepare_cross_covariance(cov_xy: torch.Tensor, n_dims_x: int, n_dims_y: int,
                           device: torch.device) -> torch.Tensor:
    """
    Prepare cross-covariance matrix for correlated input-output noise.
    
    Args:
        cov_xy: Cross-covariance tensor
        n_dims_x: Number of input dimensions
        n_dims_y: Number of output dimensions
        device: Device to put tensor on
        
    Returns:
        Cross-covariance tensor [n_dims_y, n_dims_x]
    """
    if cov_xy is None:
        # Default to zero cross-covariance
        return torch.zeros((n_dims_y, n_dims_x), device=device)
        
    if isinstance(cov_xy, torch.Tensor):
        cov_xy = cov_xy.to(device)
        
        if cov_xy.shape == (n_dims_y, n_dims_x):
            return cov_xy
        else:
            raise ValueError(f"Cross-covariance shape {cov_xy.shape} doesn't match "
                           f"required shape ({n_dims_y}, {n_dims_x})")
    
    raise TypeError(f"Cross-covariance must be a tensor, got {type(cov_xy)}")

def prepare_model_input_for_gradients(x: torch.Tensor) -> torch.Tensor:
    """
    Prepare model input for gradient calculation.
    
    Args:
        x: Input tensor
        
    Returns:
        Input tensor with requires_grad=True
    """
    x_grad = x.detach().clone()
    x_grad.requires_grad_(True)
    return x_grad

def batched_linalg_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve multiple linear systems in batched mode.
    
    This function solves Ax = b for multiple A, b pairs in a batch.
    
    Args:
        A: Batch of matrices [batch_size, n, n]
        b: Batch of vectors [batch_size, n, 1] or [batch_size, n]
        
    Returns:
        x: Solution of Ax = b, [batch_size, n, 1] or [batch_size, n]
    """
    # Ensure b is a 3D tensor with shape [batch_size, n, 1]
    squeeze_dim = False
    if b.dim() == 2:
        b = b.unsqueeze(-1)
        squeeze_dim = True
        
    # Try direct solve first
    try:
        x = torch.linalg.solve(A, b)
    except RuntimeError:
        # Fall back to batched inverse method
        # Add small jitter to diagonal for numerical stability
        batch_size, n, _ = A.shape
        device = A.device
        jitter = 1e-8 * torch.eye(n, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        try:
            # Try with jitter
            A_jitter = A + jitter
            x = torch.linalg.solve(A_jitter, b)
        except RuntimeError:
            # Ultimate fallback: explicit inverse
            A_inv = torch.inverse(A + jitter)
            x = torch.bmm(A_inv, b)
    
    # Return with original dimensions
    if squeeze_dim:
        return x.squeeze(-1)
    return x

def to_tensor(x: Union[np.ndarray, List, Tuple, torch.Tensor], 
            device: Optional[torch.device] = None, 
            dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Convert various data types to PyTorch tensor.
    
    Args:
        x: Input data (numpy array, list, tuple, or tensor)
        device: Device to put tensor on (if None, uses input tensor's device)
        dtype: Desired tensor dtype (if None, inferred from input)
        
    Returns:
        PyTorch tensor
    """
    if isinstance(x, torch.Tensor):
        tensor = x
    elif isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x)
    elif isinstance(x, (list, tuple)):
        tensor = torch.tensor(x)
    else:
        raise TypeError(f"Cannot convert {type(x)} to torch.Tensor")
        
    # Move to specified device if provided
    if device is not None:
        tensor = tensor.to(device)
        
    # Convert to specified dtype if provided
    if dtype is not None:
        tensor = tensor.to(dtype)
        
    return tensor

def standardize(x: torch.Tensor, mean: Optional[torch.Tensor] = None, 
               std: Optional[torch.Tensor] = None, 
               eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize data to have zero mean and unit variance.
    
    Args:
        x: Input tensor [batch_size, n_features]
        mean: Mean values to use (if None, computed from data)
        std: Standard deviation values to use (if None, computed from data)
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (standardized_x, mean, std)
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)
        
    if mean is None:
        # Compute mean along batch dimension
        mean = torch.mean(x, dim=0)
        
    if std is None:
        # Compute standard deviation along batch dimension
        std = torch.std(x, dim=0, unbiased=False)
        
    # Avoid division by zero
    std = torch.clamp(std, min=eps)
    
    # Standardize
    standardized_x = (x - mean) / std
    
    return standardized_x, mean, std

def unstandardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Reverse standardization transform.
    
    Args:
        x: Standardized tensor
        mean: Mean values used for standardization
        std: Standard deviation values used for standardization
        
    Returns:
        Original scale tensor
    """
    return x * std + mean
