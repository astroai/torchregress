import torch
from typing import Callable, Optional, Union, Dict, Tuple, List

def prepare_sigma(sigma, n_features, device, default_zero=True):
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


def prepare_param(param, size, device, default_value=None):
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
        return torch.tensor(default_value, device=device)
    
    if isinstance(param, (int, float)):
        return torch.full((size,), param, device=device)
    
    if isinstance(param, torch.Tensor):
        return param.to(device)
    
    raise TypeError(f"Parameter must be float, int or tensor, got {type(param)}")


def prepare_covariance(cov, size, device):
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


def prepare_cross_covariance(cross_cov, size_x, size_y, device):
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


def compute_model_gradients(model_output, x_input, n_features_y):
    """
    Compute gradients of model outputs with respect to inputs.
    
    Args:
        model_output: Output tensor from the model
        x_input: Input tensor with requires_grad=True
        n_features_y: Number of output features
        
    Returns:
        Tensor of gradients with shape [batch_size, n_features_y, n_features_x]
    """
    grads = []
    for i in range(n_features_y):
        grad = torch.autograd.grad(
            outputs=model_output[:, i].sum(),
            inputs=x_input,
            create_graph=True,
            retain_graph=True
        )[0]  # Shape: (batch_size, n_features_x)
        grads.append(grad)
    
    # Stack gradients along a new dimension
    return torch.stack(grads, dim=1)  # Shape: (batch_size, n_features_y, n_features_x)


def calculate_propagated_variance(grads, sigma_x, batch_size, n_features_y, device):
    """
    Calculate the propagated variance term ∇f(x)^T Σx ∇f(x).
    
    Args:
        grads: Gradients from compute_model_gradients
        sigma_x: Covariance matrix or scalar of feature noise
        batch_size: Batch size
        n_features_y: Number of output features
        device: Target device
        
    Returns:
        Tensor of propagated variance with shape [batch_size, n_features_y]
    """
    if sigma_x.ndim == 0:  # Scalar case
        # Compute squared gradient norm for each output dimension
        grad_norm_squared = torch.sum(grads**2, dim=2)  # Shape: (batch_size, n_features_y)
        # Multiply by scalar sigma_x^2
        return (sigma_x**2) * grad_norm_squared  # Shape: (batch_size, n_features_y)
        
    elif sigma_x.ndim == 1:  # Vector case (per-feature variance)
        # Weighted sum of squared gradients
        return torch.sum((grads**2) * (sigma_x**2).unsqueeze(0).unsqueeze(0), dim=2)
        
    else:  # Full covariance matrix case
        propagated_var = torch.zeros(batch_size, n_features_y, device=device)
        for i in range(batch_size):
            for j in range(n_features_y):
                # Compute grad^T * Sigma_x * grad
                propagated_var[i, j] = torch.matmul(
                    torch.matmul(grads[i, j], sigma_x), 
                    grads[i, j]
                )
        return propagated_var


def calculate_gaussian_nll(diff, cov_or_precision_matrix, is_precision=False):
    """
    Calculate the Gaussian negative log-likelihood.
    
    Args:
        diff: Difference between true and predicted values
        cov_or_precision_matrix: Covariance or precision matrix
        is_precision: Whether the provided matrix is a precision matrix
        
    Returns:
        Tensor of negative log-likelihood values
    """
    batch_size, n_features = diff.shape
    device = diff.device

    if is_precision:
        quadratic_term = torch.einsum('bi,bij,bj->b', diff, cov_or_precision_matrix, diff)
        logdet = torch.logdet(cov_or_precision_matrix)
        return 0.5 * (quadratic_term - logdet + n_features * torch.log(torch.tensor(2 * torch.pi, device=device)))
    else:
        L = torch.linalg.cholesky(cov_or_precision_matrix)
        z = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
        quadratic_term = torch.sum(z**2, dim=(-2, -1))
        logdet = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
        return 0.5 * (quadratic_term + logdet + n_features * torch.log(torch.tensor(2 * torch.pi, device=device)))


def prepare_model_input_for_gradients(x_obs):
    """
    Prepare model input tensor for gradient computation.
    
    Args:
        x_obs: Input tensor
        
    Returns:
        Clone of input tensor with requires_grad=True
    """
    x_grad = x_obs.detach().clone()
    x_grad.requires_grad_(True)
    return x_grad
