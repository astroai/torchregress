"""
Tensor operations for regression tasks.

This module provides functions for common tensor operations used in
regression, including tensor conversion, standardization, and special
linear algebra operations.
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple, List

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
        
    # Handle broadcasting if needed
    if tensor.shape != mask.shape:
        # Try to broadcast mask to tensor shape using torch's broadcasting rules
        try:
            # Use proper broadcasting using * operator
            return tensor * mask
        except RuntimeError:
            # Provide more detailed error message
            raise ValueError(
                f"Mask shape {mask.shape} cannot be broadcast to tensor shape {tensor.shape}. "
                f"For broadcasting to work, mask dimensions must be 1 or match tensor dimensions."
            )
    else:
        return tensor * mask

def masked_reduction(tensor: torch.Tensor, mask: Optional[torch.Tensor], 
                    reduction: str = 'mean') -> torch.Tensor:
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
    valid_reductions = ['none', 'mean', 'sum', 'max', 'min']
    if reduction not in valid_reductions:
        raise ValueError(f"Reduction '{reduction}' not supported. Must be one of {valid_reductions}")
        
    if reduction == 'none':
        return tensor
    
    if mask is None:
        # No mask, straightforward reduction
        if reduction == 'mean':
            return torch.mean(tensor)
        elif reduction == 'sum':
            return torch.sum(tensor)
        elif reduction == 'max':
            return torch.max(tensor)
        elif reduction == 'min':
            return torch.min(tensor)
    else:
        # With mask, we need to handle the reduction carefully
        # First apply the mask to zero out invalid values
        masked_tensor = tensor * mask
        
        if reduction == 'mean':
            # Count valid elements for mean reduction
            valid_count = torch.sum(mask).clamp(min=1)  # Avoid division by zero
            masked_sum = torch.sum(masked_tensor)
            return masked_sum / valid_count
        elif reduction == 'sum':
            return torch.sum(masked_tensor)
        elif reduction == 'max' or reduction == 'min':
            # For max/min, we need to handle masked values specially
            # Fill masked values with appropriate extreme values
            if reduction == 'max':
                # Fill masked positions with negative infinity
                fill_value = torch.finfo(tensor.dtype).min
            else:  # 'min'
                # Fill masked positions with positive infinity
                fill_value = torch.finfo(tensor.dtype).max
                
            # Create a tensor with fill values at masked positions
            filled_tensor = tensor.clone()
            filled_tensor = filled_tensor.masked_fill(~mask, fill_value)
            
            # Apply the reduction
            if reduction == 'max':
                return torch.max(filled_tensor)
            else:  # 'min'
                return torch.min(filled_tensor)

def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[int] = None) -> torch.Tensor:
    """
    Compute masked mean along specified dimension(s).
    
    Args:
        tensor: Input tensor
        mask: Boolean mask (True for valid values)
        dim: Dimension(s) to reduce over. None for all dimensions.
        
    Returns:
        Mean of valid elements along specified dimension(s)
        
    Examples:
        >>> x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        >>> mask = torch.tensor([[True, True, False], [True, False, True]])
        >>> masked_mean(x, mask, dim=1)
        tensor([1.5000, 5.0000])
    """
    if mask is None:
        return torch.mean(tensor, dim=dim)
    
    if dim is None:
        # Reduce over all dimensions
        valid_count = torch.sum(mask).clamp(min=1)
        return torch.sum(tensor * mask) / valid_count
    
    # Handle reducing along specific dimension(s)
    masked_tensor = tensor * mask
    valid_count = torch.sum(mask, dim=dim).clamp(min=1)
    return torch.sum(masked_tensor, dim=dim) / valid_count

def masked_sum(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[int] = None) -> torch.Tensor:
    """
    Compute masked sum along specified dimension(s).
    
    Args:
        tensor: Input tensor
        mask: Boolean mask (True for valid values)
        dim: Dimension(s) to reduce over. None for all dimensions.
        
    Returns:
        Sum of valid elements along specified dimension(s)
        
    Examples:
        >>> x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        >>> mask = torch.tensor([[True, True, False], [True, False, True]])
        >>> masked_sum(x, mask, dim=1)
        tensor([3., 10.])
    """
    if mask is None:
        return torch.sum(tensor, dim=dim)
    
    masked_tensor = tensor * mask
    return torch.sum(masked_tensor, dim=dim)

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
        
    Examples:
        >>> device = torch.device('cpu')
        >>> prepare_param(0.5, 3, device)
        tensor([0.5000, 0.5000, 0.5000])
        >>> prepare_param(torch.tensor([0.1, 0.2, 0.3]), 3, device)
        tensor([0.1000, 0.2000, 0.3000])
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
            # Check if we need to reshape to match n_dims
            if param.ndim > 1 and param.numel() == n_dims:
                return param.reshape(n_dims)
            return param
        else:
            raise ValueError(
                f"Parameter shape {param.shape} doesn't match required size {n_dims}. "
                f"Expected a tensor with first dimension of size {n_dims}."
            )
    
    raise TypeError(f"Parameter must be float or tensor, got {type(param).__name__}")

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
        
    Examples:
        >>> device = torch.device('cpu')
        >>> prepare_sigma(0.5, 3, device)
        tensor([0.5000, 0.5000, 0.5000])
        >>> prepare_sigma(None, 3, device, default_zero=False)
        tensor([1., 1., 1.])
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
        Covariance tensor of appropriate shape (n_dims, n_dims)
        
    Examples:
        >>> device = torch.device('cpu')
        >>> prepare_covariance(1.0, 2, device)
        tensor([[1., 0.],
                [0., 1.]])
        >>> prepare_covariance(torch.tensor([2.0, 3.0]), 2, device)
        tensor([[2., 0.],
                [0., 3.]])
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
                raise ValueError(
                    f"Diagonal covariance shape {cov.shape} doesn't match required dimensions {n_dims}. "
                    f"Expected a tensor with {n_dims} elements."
                )
            return torch.diag(cov)
        elif cov.ndim == 2:
            # Matrix - check shape
            if cov.shape != (n_dims, n_dims):
                raise ValueError(
                    f"Covariance matrix shape {cov.shape} doesn't match required shape ({n_dims}, {n_dims}). "
                    f"Expected a square matrix with dimensions ({n_dims}, {n_dims})."
                )
            # Ensure the covariance matrix is symmetric
            if not torch.allclose(cov, cov.t(), rtol=1e-5, atol=1e-8):
                warnings_module = __import__('warnings')
                warnings_module.warn(
                    f"Covariance matrix is not symmetric. Using (cov + cov.T) / 2 to ensure symmetry."
                )
                # Make it symmetric
                cov = (cov + cov.t()) / 2
            return cov
        else:
            raise ValueError(f"Covariance must be scalar, vector or matrix, got tensor with {cov.ndim} dimensions")
    
    raise TypeError(f"Covariance must be float or tensor, got {type(cov).__name__}")

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
        
    Examples:
        >>> device = torch.device('cpu')
        >>> prepare_cross_covariance(None, 2, 3, device)
        tensor([[0., 0.],
                [0., 0.],
                [0., 0.]])
    """
    if cov_xy is None:
        # Default to zero cross-covariance
        return torch.zeros((n_dims_y, n_dims_x), device=device)
        
    if isinstance(cov_xy, torch.Tensor):
        cov_xy = cov_xy.to(device)
        
        if cov_xy.shape == (n_dims_y, n_dims_x):
            return cov_xy
        else:
            raise ValueError(
                f"Cross-covariance shape {cov_xy.shape} doesn't match required shape ({n_dims_y}, {n_dims_x}). "
                f"Expected a matrix with dimensions ({n_dims_y}, {n_dims_x})."
            )
    
    raise TypeError(f"Cross-covariance must be a tensor, got {type(cov_xy).__name__}")

def prepare_model_input_for_gradients(x: torch.Tensor) -> torch.Tensor:
    """
    Prepare model input for gradient calculation.
    
    Args:
        x: Input tensor
        
    Returns:
        Input tensor with requires_grad=True
        
    Examples:
        >>> x = torch.tensor([1.0, 2.0])
        >>> x_grad = prepare_model_input_for_gradients(x)
        >>> x_grad.requires_grad
        True
    """
    x_grad = x.detach().clone()
    x_grad.requires_grad_(True)
    return x_grad

def batched_linalg_solve(A: torch.Tensor, b: torch.Tensor,
                        ridge_factor: float = 1e-8) -> torch.Tensor:
    """
    Solve multiple linear systems in batched mode.
    
    This function solves Ax = b for multiple A, b pairs in a batch.
    Uses progressively more robust but potentially slower methods if simple approach fails.
    
    Args:
        A: Batch of matrices [batch_size, n, n]
        b: Batch of vectors [batch_size, n, 1] or [batch_size, n]
        ridge_factor: Small constant added to diagonal for numerical stability
        
    Returns:
        x: Solution of Ax = b, [batch_size, n, 1] or [batch_size, n]
        
    Examples:
        >>> A = torch.tensor([[[2.0, 1.0], [1.0, 2.0]], [[1.0, 0.5], [0.5, 1.0]]])
        >>> b = torch.tensor([[5.0, 5.0], [2.0, 2.0]])
        >>> x = batched_linalg_solve(A, b)
        >>> x.shape
        torch.Size([2, 2])
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
        jitter = ridge_factor * torch.eye(n, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        try:
            # Try with jitter
            A_jitter = A + jitter
            x = torch.linalg.solve(A_jitter, b)
        except RuntimeError:
            # Ultimate fallback: explicit inverse
            try:
                A_inv = torch.inverse(A + jitter)
                x = torch.bmm(A_inv, b)
            except RuntimeError:
                # If everything fails, try SVD-based pseudo-inverse
                A_flat = A.reshape(-1, n, n)
                b_flat = b.reshape(-1, n, 1)
                x_flat = []
                
                for i in range(A_flat.shape[0]):
                    # Compute SVD-based pseudo-inverse for each matrix in batch
                    U, S, Vh = torch.linalg.svd(A_flat[i])
                    # Apply threshold to singular values
                    threshold = ridge_factor * torch.max(S)
                    S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))
                    # Compute pseudo-inverse: A⁺ = V·S⁺·U^T
                    A_pinv = torch.mm(Vh.t(), torch.mm(torch.diag(S_inv), U.t()))
                    # Solve system
                    x_i = torch.mm(A_pinv, b_flat[i])
                    x_flat.append(x_i)
                
                x = torch.stack(x_flat).reshape(batch_size, n, 1)
    
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
        
    Examples:
        >>> to_tensor([1, 2, 3])
        tensor([1, 2, 3])
        >>> to_tensor(np.array([1.0, 2.0, 3.0]), dtype=torch.float32)
        tensor([1., 2., 3.], dtype=torch.float32)
    """
    if isinstance(x, torch.Tensor):
        tensor = x
    elif isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x)
    elif isinstance(x, (list, tuple)):
        tensor = torch.tensor(x)
    else:
        raise TypeError(f"Cannot convert {type(x).__name__} to torch.Tensor")
        
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
        
    Examples:
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> std_x, mean, std = standardize(x)
        >>> mean
        tensor([3., 4.])
        >>> std
        tensor([1.6330, 1.6330])
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
        
    Examples:
        >>> x_std = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        >>> mean = torch.tensor([2.0, 3.0])
        >>> std = torch.tensor([1.0, 2.0])
        >>> unstandardize(x_std, mean, std)
        tensor([[2., 3.],
                [3., 5.]])
    """
    return x * std + mean

def broadcast_shapes(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Compute the shape resulting from broadcasting the given shapes.
    
    Args:
        *shapes: Input shapes to broadcast
        
    Returns:
        Resulting shape after broadcasting
        
    Raises:
        ValueError: If shapes are not broadcastable
        
    Examples:
        >>> broadcast_shapes((3, 1), (1, 4))
        (3, 4)
        >>> broadcast_shapes((5, 1, 3), (1, 4, 3))
        (5, 4, 3)
    """
    if not shapes:
        return ()
    
    try:
        return torch.broadcast_shapes(*shapes)
    except RuntimeError as e:
        # Provide more helpful error message
        raise ValueError(
            f"Shapes {shapes} are not broadcastable. For broadcasting to work, "
            f"trailing dimensions must match or be 1. Error: {str(e)}"
        )

def broadcast_tensors(*tensors: torch.Tensor) -> List[torch.Tensor]:
    """
    Broadcast tensors to a common shape.
    
    Args:
        *tensors: Tensors to broadcast together
        
    Returns:
        List of broadcasted tensors
        
    Examples:
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([[4], [5]])
        >>> c_a, c_b = broadcast_tensors(a, b)
        >>> c_a.shape, c_b.shape
        (torch.Size([2, 3]), torch.Size([2, 3]))
    """
    try:
        return torch.broadcast_tensors(*tensors)
    except RuntimeError as e:
        raise ValueError(f"Cannot broadcast tensors: {str(e)}")

def batch_diag(x: torch.Tensor) -> torch.Tensor:
    """
    Create a batch of diagonal matrices from a batch of vectors.
    
    Args:
        x: Input tensor of shape [..., n]
        
    Returns:
        Batch of diagonal matrices with shape [..., n, n]
        
    Examples:
        >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> batch_diag(x).shape
        torch.Size([2, 3, 3])
    """
    # Get the shape of the input
    shape = x.shape
    n = shape[-1]
    
    # Create identity matrices and scale them
    # First create a tensor of shape [..., 1, n, n] with 1s on the diagonal
    # Then multiply by x.unsqueeze(-1).unsqueeze(-1) of shape [..., n, 1, 1]
    identity = torch.eye(n, device=x.device, dtype=x.dtype)
    batch_shape = shape[:-1]  # shape without the last dimension
    batch_identity = identity.expand(*batch_shape, n, n)
    
    # Reshape x to have the right shape for broadcasting
    # From [..., n] to [..., n, 1]
    x_diag = x.unsqueeze(-1)
    
    # Multiply to get [..., n, n] diagonal matrices
    return batch_identity * x_diag.unsqueeze(-2)
