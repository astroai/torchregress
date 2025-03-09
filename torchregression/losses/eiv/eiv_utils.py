"""
Utility functions for Error-in-Variables (EIV) regression.

This module provides helper functions for working with measurement errors
in both dependent and independent variables.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, List, Tuple, Callable

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

def compute_model_gradients(y_pred: torch.Tensor, x_grad: torch.Tensor, 
                           n_features_y: int) -> torch.Tensor:
    """
    Compute gradients of model predictions with respect to inputs.
    
    Args:
        y_pred: Model predictions [batch_size, n_features_y]
        x_grad: Input tensor with requires_grad=True [batch_size, n_features_x]
        n_features_y: Number of output features
        
    Returns:
        Gradient tensor [batch_size, n_features_y, n_features_x]
    """
    batch_size = y_pred.shape[0]
    n_features_x = x_grad.shape[1]
    device = y_pred.device
    
    # Create tensor to hold gradients for each batch element
    grads = torch.zeros(batch_size, n_features_y, n_features_x, device=device)
    
    # For each output dimension, compute gradients
    for i in range(n_features_y):
        # Sum outputs for this dimension across batch
        # We need to handle the batch dimension without for loops for efficiency
        if y_pred.shape[1] > 1:
            outputs_i = y_pred[:, i]
        else:
            outputs_i = y_pred.squeeze(1)
        
        # Compute gradients for this output dimension
        grad_outputs = torch.ones_like(outputs_i)
        
        # Get gradients of outputs_i with respect to x_grad
        batch_grad = torch.autograd.grad(
            outputs=outputs_i,
            inputs=x_grad,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]
        
        # Store gradients for this output dimension
        grads[:, i, :] = batch_grad
    
    return grads

def calculate_propagated_variance(grad: torch.Tensor, 
                                  sigma_x: Union[float, torch.Tensor],
                                  sigma_xy: Optional[torch.Tensor] = None,
                                  sigma_y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate variance propagated from input to output through model gradients.
    
    Args:
        grad: Gradient tensor [batch_size, n_features_y, n_features_x]
        sigma_x: Covariance of feature noise (scalar, vector or matrix)
        sigma_xy: Optional cross-covariance between inputs and outputs
        sigma_y: Optional intrinsic variance of outputs
        
    Returns:
        Propagated variance tensor [batch_size, n_features_y, n_features_y]
    """
    batch_size, n_features_y, n_features_x = grad.shape
    device = grad.device
    
    # Handle different types of sigma_x
    if isinstance(sigma_x, float):
        # Scalar case: isotropic noise
        propagated_var = sigma_x * torch.bmm(grad, grad.transpose(1, 2))
        
    elif isinstance(sigma_x, torch.Tensor):
        if sigma_x.ndim == 0:
            # Scalar tensor case
            propagated_var = sigma_x * torch.bmm(grad, grad.transpose(1, 2))
            
        elif sigma_x.ndim == 1:
            # Vector case: diagonal covariance
            # Scale each column of gradient by corresponding sigma
            scaled_grad = grad * sigma_x.unsqueeze(0).unsqueeze(0)
            propagated_var = torch.bmm(scaled_grad, grad.transpose(1, 2))
            
        else:
            # Full matrix case
            # We need to compute grad * sigma_x * grad^T for each batch element
            propagated_var = torch.zeros(batch_size, n_features_y, n_features_y, device=device)
            
            # Efficient batched matrix multiplication
            # First compute scaled_grad = grad * sigma_x
            scaled_grad = torch.bmm(grad, sigma_x.expand(batch_size, -1, -1))
            
            # Then compute grad * sigma_x * grad^T
            propagated_var = torch.bmm(scaled_grad, grad.transpose(1, 2))
    else:
        raise TypeError(f"sigma_x must be float or tensor, got {type(sigma_x)}")
    
    # Add cross-covariance contribution if provided
    if sigma_xy is not None:
        if sigma_xy.shape != (n_features_y, n_features_x):
            raise ValueError(f"sigma_xy shape {sigma_xy.shape} doesn't match "
                          f"expected shape ({n_features_y}, {n_features_x})")
            
        # For each batch element, add grad * sigma_xy + sigma_xy^T * grad^T
        sigma_xy_expanded = sigma_xy.expand(batch_size, -1, -1)
        
        # Compute grad * sigma_xy
        term1 = torch.bmm(grad, sigma_xy_expanded.transpose(1, 2))
        
        # Add it and its transpose
        propagated_var = propagated_var + term1 + term1.transpose(1, 2)
    
    # Add intrinsic output variance if provided
    if sigma_y is not None:
        if isinstance(sigma_y, float):
            # Add scalar variance to diagonal
            diag = torch.eye(n_features_y, device=device).expand(batch_size, -1, -1)
            propagated_var = propagated_var + sigma_y * diag
            
        elif isinstance(sigma_y, torch.Tensor):
            if sigma_y.ndim == 0:
                # Scalar tensor
                diag = torch.eye(n_features_y, device=device).expand(batch_size, -1, -1)
                propagated_var = propagated_var + sigma_y * diag
                
            elif sigma_y.ndim == 1:
                # Vector case: diagonal intrinsic variance
                if sigma_y.shape[0] != n_features_y:
                    raise ValueError(f"sigma_y shape {sigma_y.shape} doesn't match "
                                  f"n_features_y = {n_features_y}")
                
                # Create diagonal matrix for each batch element
                diag = torch.diag_embed(sigma_y.expand(batch_size, -1))
                propagated_var = propagated_var + diag
                
            else:
                # Full matrix case
                if sigma_y.shape != (n_features_y, n_features_y):
                    raise ValueError(f"sigma_y shape {sigma_y.shape} doesn't match "
                                  f"expected shape ({n_features_y}, {n_features_y})")
                
                # Add sigma_y to each batch element
                propagated_var = propagated_var + sigma_y.expand(batch_size, -1, -1)
                
    return propagated_var

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
    
    if sigma is None:
        return torch.tensor(default, device=device).expand(n_dims)
        
    if isinstance(sigma, (int, float)):
        return torch.tensor(sigma, device=device).expand(n_dims)
        
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(device)
        
        if sigma.ndim == 0:
            # Scalar tensor
            return sigma.expand(n_dims)
        elif sigma.shape[0] == n_dims:
            return sigma
        else:
            raise ValueError(f"Sigma shape {sigma.shape} doesn't match required size {n_dims}")
    
    raise TypeError(f"Sigma must be float or tensor, got {type(sigma)}")

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

def calculate_gaussian_nll(residuals: torch.Tensor, cov: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate the negative log-likelihood of a multivariate Gaussian distribution.
    
    Args:
        residuals: Residual tensor [batch_size, n_dims] or [batch_size, 1, n_dims]
        cov: Covariance tensor [batch_size, n_dims, n_dims]
        eps: Small constant for numerical stability
        
    Returns:
        Negative log-likelihood [batch_size]
    """
    batch_size = residuals.shape[0]
    n_dims = residuals.shape[-1]
    device = residuals.device
    
    # Ensure residuals have shape [batch_size, 1, n_dims]
    if residuals.ndim == 2:
        residuals = residuals.unsqueeze(1)
    
    # Add jitter to diagonal for numerical stability
    jitter = eps * torch.eye(n_dims, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    cov_jitter = cov + jitter
    
    try:
        # Try Cholesky decomposition for stability
        L = torch.linalg.cholesky(cov_jitter)
        
        # Calculate log determinant using Cholesky: log|Σ| = 2 * sum(log(diag(L)))
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2) + eps), dim=1)
        
        # Solve linear system (L * L^T)^(-1) * residuals = L^(-T) * (L^(-1) * residuals)
        residuals_col = residuals.transpose(1, 2)  # [batch_size, n_dims, 1]
        temp = torch.linalg.solve_triangular(L, residuals_col, upper=False)
        quad_form = torch.sum(temp**2, dim=1).squeeze(-1)  # [batch_size]
    except RuntimeError:
        # Fall back to direct calculation
        # Calculate log determinant
        log_det = torch.logdet(cov_jitter)
        
        # Calculate quadratic form: r^T * Σ^(-1) * r
        quad_form = torch.bmm(
            torch.bmm(residuals, torch.linalg.inv(cov_jitter)),
            residuals.transpose(1, 2)
        ).squeeze(1).squeeze(1)
    
    # Calculate full NLL: 0.5 * (log|Σ| + r^T * Σ^(-1) * r + n * log(2π))
    nll = 0.5 * (log_det + quad_form + n_dims * torch.log(torch.tensor(2.0 * torch.pi, device=device)))
    
    return nll
