"""
Utility functions for Error-in-Variables (EIV) regression.

This module provides utility functions for handling EIV regression tasks,
including preparation of parameters, covariance matrices, and gradient calculations.
"""
import torch

def prepare_param(param, size, device):
    """
    Prepare a parameter tensor.
    
    Args:
        param: Parameter value (scalar or tensor)
        size: Desired size of the parameter tensor
        device: Device to place the tensor on
        
    Returns:
        Prepared parameter tensor
    """
    if isinstance(param, torch.Tensor):
        return param.to(device)
    else:
        return torch.tensor(param, device=device).expand(size)

def prepare_sigma(sigma, size, device):
    """
    Prepare a sigma tensor.
    
    Args:
        sigma: Sigma value (scalar or tensor)
        size: Desired size of the sigma tensor
        device: Device to place the tensor on
        
    Returns:
        Prepared sigma tensor
    """
    return prepare_param(sigma, size, device)

def prepare_covariance(cov, size, device):
    """
    Prepare a covariance matrix.
    
    Args:
        cov: Covariance value (scalar, vector, or matrix)
        size: Desired size of the covariance matrix
        device: Device to place the tensor on
        
    Returns:
        Prepared covariance matrix
    """
    if isinstance(cov, torch.Tensor):
        return cov.to(device)
    else:
        return torch.diag(torch.tensor(cov, device=device).expand(size))

def prepare_cross_covariance(cross_cov, size_x, size_y, device):
    """
    Prepare a cross-covariance matrix.
    
    Args:
        cross_cov: Cross-covariance value (scalar or matrix)
        size_x: Size of the x dimension
        size_y: Size of the y dimension
        device: Device to place the tensor on
        
    Returns:
        Prepared cross-covariance matrix
    """
    if isinstance(cross_cov, torch.Tensor):
        return cross_cov.to(device)
    else:
        return torch.tensor(cross_cov, device=device).expand(size_x, size_y)

def compute_model_gradients(y_pred, x, n_features_y):
    """
    Compute gradients of model predictions with respect to inputs.
    
    Args:
        y_pred: Model predictions
        x: Input features
        n_features_y: Number of output features
        
    Returns:
        Gradients of model predictions with respect to inputs
    """
    grads = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    return grads.view(-1, n_features_y, x.shape[-1])

def calculate_gaussian_nll(residuals, var, eps=1e-8):
    """
    Calculate Gaussian negative log-likelihood.
    
    Args:
        residuals: Residuals between true and predicted values
        var: Variance of the Gaussian distribution
        eps: Small constant for numerical stability
        
    Returns:
        Gaussian negative log-likelihood
    """
    nll = 0.5 * (torch.log(var + eps) + residuals**2 / (var + eps) + torch.log(torch.tensor(2 * torch.pi)))
    return nll

def prepare_model_input_for_gradients(x):
    """
    Prepare model input for gradient computation.
    
    Args:
        x: Input features
        
    Returns:
        Prepared input features with requires_grad=True
    """
    return x.detach().clone().requires_grad_(True)

def calculate_propagated_variance(grad, sigma_x, sigma_y=None, sigma_xy=None):
    """
    Calculate propagated variance from input to output.
    
    Args:
        grad: Gradients of model predictions with respect to inputs
        sigma_x: Covariance of input features
        sigma_y: Covariance of output features (optional)
        sigma_xy: Cross-covariance between input and output features (optional)
        
    Returns:
        Propagated variance
    """
    propagated_var = torch.matmul(grad, torch.matmul(sigma_x, grad.transpose(-1, -2)))
    
    if sigma_y is not None:
        propagated_var += sigma_y
        
    if sigma_xy is not None:
        propagated_var += 2 * torch.matmul(grad, sigma_xy)
        
    return propagated_var
