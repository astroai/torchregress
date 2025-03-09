"""
Utility functions for Error-in-Variables (EIV) regression.

These utilities support implementing regression models that account for
measurement errors in both inputs and outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Dict, Callable

def prepare_sigma(
    sigma: Union[float, torch.Tensor, None], 
    n_features: int, 
    device: torch.device, 
    default_zero: bool = True
) -> Optional[torch.Tensor]:
    """
    Convert sigma parameter to tensor with appropriate shape.
    
    Args:
        sigma: Noise parameter (float, int, tensor, or None)
        n_features: Number of features
        device: Target device
        default_zero: If True, return 0.0 for None, otherwise return None
        
    Returns:
        Tensor of appropriate shape on the target device
    """
    if sigma is None:
        if default_zero:
            return torch.tensor(0.0, device=device)
        return None
        
    if isinstance(sigma, (int, float)):
        return torch.tensor(sigma, device=device)
        
    if isinstance(sigma, torch.Tensor):
        return sigma.to(device)
        
    raise TypeError(f"sigma must be float, int or tensor, got {type(sigma)}")


def prepare_param(
    param: Union[float, torch.Tensor, None], 
    size: int, 
    device: torch.device, 
    default_value: Optional[float] = None
) -> Optional[torch.Tensor]:
    """
    Convert parameter to appropriate tensor type and shape.
    
    Args:
        param: Parameter value (float, int, tensor, or None)
        size: Size of the output tensor
        device: Target device
        default_value: Value to use if param is None
        
    Returns:
        Tensor of appropriate shape on the target device
    """
    if param is None:
        if default_value is None:
            return None
        return torch.full((size,), default_value, device=device)
    
    if isinstance(param, (int, float)):
        return torch.full((size,), param, device=device)
    
    if isinstance(param, torch.Tensor):
        if param.ndim == 0:  # Scalar tensor
            return torch.full((size,), param.item(), device=device)
        return param.to(device)
    
    raise TypeError(f"Parameter must be float, int or tensor, got {type(param)}")


def prepare_covariance(
    cov: Union[float, torch.Tensor, None], 
    size: int, 
    device: torch.device
) -> torch.Tensor:
    """
    Convert covariance parameter to appropriate tensor form.
    
    Args:
        cov: Covariance parameter (float, int, tensor, or None)
        size: Size of the output matrix
        device: Target device
        
    Returns:
        Covariance tensor of shape [size, size] on the target device
    """
    if cov is None:
        return torch.zeros((size, size), device=device)
    
    if isinstance(cov, (int, float)):
        # Scalar variance becomes identity matrix scaled by the value
        return cov * torch.eye(size, device=device)
    
    if isinstance(cov, torch.Tensor):
        if cov.ndim == 0:  # Scalar tensor
            return cov.item() * torch.eye(size, device=device)
        elif cov.ndim == 1:  # Diagonal covariance
            return torch.diag(cov.to(device))
        elif cov.ndim == 2:  # Full covariance matrix
            return cov.to(device)
    
    raise TypeError(f"Covariance must be float, int or tensor, got {type(cov)}")


def prepare_cross_covariance(
    cross_cov: Union[torch.Tensor, None], 
    size_x: int, 
    size_y: int, 
    device: torch.device
) -> torch.Tensor:
    """
    Convert cross-covariance parameter to appropriate tensor form.
    
    Args:
        cross_cov: Cross-covariance parameter (tensor or None)
        size_x: Size of the first dimension
        size_y: Size of the second dimension
        device: Target device
        
    Returns:
        Cross-covariance tensor of shape [size_x, size_y] on the target device
    """
    if cross_cov is None:
        return torch.zeros((size_x, size_y), device=device)
    
    if isinstance(cross_cov, torch.Tensor):
        return cross_cov.to(device)
    
    raise TypeError(f"Cross-covariance must be tensor, got {type(cross_cov)}")


def prepare_model_input_for_gradients(x: torch.Tensor) -> torch.Tensor:
    """
    Prepare model input for gradient computations.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with requires_grad=True for gradient computation
    """
    x_grad = x.detach().clone()
    x_grad.requires_grad_(True)
    return x_grad


def compute_model_gradients(y_pred: torch.Tensor, x: torch.Tensor, n_outputs: int) -> torch.Tensor:
    """
    Compute gradients of model predictions with respect to inputs.
    
    Args:
        y_pred: Predicted values [batch_size, n_outputs]
        x: Input tensor with requires_grad=True [batch_size, n_inputs]
        n_outputs: Number of output dimensions
        
    Returns:
        Gradients tensor [batch_size, n_outputs, n_inputs]
    """
    batch_size, n_inputs = x.shape
    
    # Initialize gradient tensor
    grads = torch.zeros((batch_size, n_outputs, n_inputs), device=x.device)
    
    # For each output dimension, compute gradient across all batch elements
    for i in range(n_outputs):
        # Get the i-th output for all batch elements
        y_i = y_pred[:, i]
        
        # Create gradient tensor of ones (for batch computation)
        ones = torch.ones_like(y_i)
        
        # Compute gradients
        x_grad = torch.autograd.grad(
            outputs=y_i,
            inputs=x,
            grad_outputs=ones,
            create_graph=False,
            retain_graph=(i < n_outputs - 1),  # Retain graph except for last iteration
            only_inputs=True
        )[0]
        
        # Store gradients
        grads[:, i, :] = x_grad
    
    return grads


def calculate_propagated_variance(
    gradients: torch.Tensor,
    sigma_x: Union[float, torch.Tensor],
    per_sample: bool = True
) -> torch.Tensor:
    """
    Calculate variance propagated from inputs to outputs using gradients.
    
    Args:
        gradients: Gradients tensor [batch_size, n_outputs, n_inputs]
        sigma_x: Input noise standard deviation (scalar or tensor)
        per_sample: Whether to return per-sample variances or average
        
    Returns:
        Propagated variance [batch_size, n_outputs] or [n_outputs]
    """
    batch_size, n_outputs, n_inputs = gradients.shape
    device = gradients.device
    
    # Handle different types of sigma_x
    if isinstance(sigma_x, (int, float)):
        # Scalar variance
        sigma_x_tensor = sigma_x**2 * torch.ones(n_inputs, device=device)
    elif isinstance(sigma_x, torch.Tensor):
        if sigma_x.ndim == 0:
            # Scalar tensor
            sigma_x_tensor = sigma_x.item()**2 * torch.ones(n_inputs, device=device)
        elif sigma_x.ndim == 1:
            # Vector variance (diagonal covariance)
            sigma_x_tensor = sigma_x**2
        else:
            # Full covariance matrix not supported in this function
            raise ValueError("Full covariance matrices not supported in calculate_propagated_variance")
    else:
        raise TypeError(f"sigma_x must be float or tensor, got {type(sigma_x)}")
    
    # Calculate propagated variance for each output dimension
    # Formula: var(y_i) = sum_j (∂y_i/∂x_j)^2 * var(x_j)
    variances = torch.zeros(batch_size, n_outputs, device=device)
    
    for b in range(batch_size):
        for i in range(n_outputs):
            # Square the gradients and multiply by input variances
            variances[b, i] = torch.sum(gradients[b, i]**2 * sigma_x_tensor)
    
    # Average across batch if not per-sample
    if not per_sample:
        variances = torch.mean(variances, dim=0)
        
    return variances


def calculate_gaussian_nll(
    residuals: torch.Tensor,
    covariance_matrices: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Calculate negative log-likelihood for Gaussian distribution.
    
    Args:
        residuals: Residuals tensor [batch_size, n_features]
        covariance_matrices: Covariance matrices [batch_size, n_features, n_features]
        eps: Small constant for numerical stability
        
    Returns:
        NLL values [batch_size]
    """
    batch_size, n_features = residuals.shape
    device = residuals.device
    
    # Ensure the matrices are well-conditioned
    eye = torch.eye(n_features, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    cov_stable = covariance_matrices + eps * eye
    
    # Calculate log determinant and quadratic form for each sample
    nll = torch.zeros(batch_size, device=device)
    
    # Try more efficient batch processing first
    try:
        # Use Cholesky decomposition for numerical stability
        L = torch.linalg.cholesky(cov_stable)
        
        # Calculate log determinant = 2 * sum(log(diag(L)))
        log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2) + eps), dim=1)
        
        # Calculate quadratic form: (y-μ)ᵀΣ⁻¹(y-μ)
        # First solve the linear system Lx = residuals
        residuals_reshaped = residuals.unsqueeze(-1)  # [batch_size, n_features, 1]
        
        # Solve LLᵀx = residuals
        temp = torch.linalg.solve_triangular(L, residuals_reshaped, upper=False)
        quad_form = torch.sum(temp**2, dim=1).squeeze(-1)
        
        # Calculate NLL: 0.5 * (log|Σ| + (y-μ)ᵀΣ⁻¹(y-μ) + n*log(2π))
        nll = 0.5 * (log_det + quad_form + n_features * torch.log(torch.tensor(2 * torch.pi, device=device)))
        
    except RuntimeError:
        # Fall back to per-sample processing
        for i in range(batch_size):
            # Get covariance matrix and residual for this sample
            cov_i = cov_stable[i]
            res_i = residuals[i]
            
            try:
                # Try Cholesky decomposition
                L = torch.linalg.cholesky(cov_i)
                log_det = 2 * torch.sum(torch.log(torch.diagonal(L) + eps))
                
                # Solve the linear system
                temp = torch.triangular_solve(res_i.unsqueeze(-1), L, upper=False)[0]
                quad_form = torch.sum(temp**2)
                
            except RuntimeError:
                # Fall back to eigenvalue decomposition
                eigvals = torch.linalg.eigvalsh(cov_i)
                log_det = torch.sum(torch.log(eigvals + eps))
                
                # Calculate inverse and quadratic form
                cov_inv = torch.linalg.pinv(cov_i)
                quad_form = torch.matmul(res_i.unsqueeze(0), torch.matmul(cov_inv, res_i.unsqueeze(-1))).item()
            
            # Calculate NLL for this sample
            nll[i] = 0.5 * (log_det + quad_form + n_features * torch.log(torch.tensor(2 * torch.pi, device=device)))
    
    return nll
