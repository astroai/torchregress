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

def prepare_sigma(sigma, size, device, default_zero=True):
    """
    Prepare a sigma tensor.
    
    Args:
        sigma: Sigma value (scalar or tensor)
        size: Desired size of the sigma tensor
        device: Device to place the tensor on
        default_zero: If True and sigma is None, returns zeros. Otherwise, returns None.
        
    Returns:
        Prepared sigma tensor
    """
    if sigma is None:
        return torch.zeros(size, device=device) if default_zero else None
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

def generate_perturbed_samples(x_obs, sigma_x_tensor, n_samples, perturb_method='gaussian'):
    """
    Generate perturbed samples around observed features.
    
    Args:
        x_obs: Observed features [batch_size, n_features_x]
        sigma_x_tensor: Standard deviation of feature noise
        n_samples: Number of samples to generate
        perturb_method: Method for perturbing inputs ('gaussian', 'uniform')
        
    Returns:
        List of perturbed samples
    """
    batch_size, n_features_x = x_obs.shape
    device = x_obs.device
    
    perturbed_samples = []
    for _ in range(n_samples):
        if perturb_method == 'gaussian':
            if sigma_x_tensor.ndim <= 1:
                # Diagonal covariance
                noise = torch.randn_like(x_obs) * sigma_x_tensor.view(1, -1)
            else:
                # Full covariance - use multivariate normal
                noise = torch.distributions.MultivariateNormal(
                    torch.zeros(n_features_x, device=device),
                    sigma_x_tensor
                ).sample((batch_size,))
        elif perturb_method == 'uniform':
            # Scale factor to match standard deviation between uniform and normal distributions
            scale_factor = 1.732  # sqrt(3)
            if sigma_x_tensor.ndim <= 1:
                half_range = sigma_x_tensor.view(1, -1) * scale_factor
                noise = (torch.rand_like(x_obs) * 2 - 1) * half_range
            else:
                # Use diagonal approximation for uniform with full covariance
                half_range = torch.sqrt(torch.diag(sigma_x_tensor)).view(1, -1) * scale_factor
                noise = (torch.rand_like(x_obs) * 2 - 1) * half_range
        else:
            raise ValueError(f"Unknown perturbation method: {perturb_method}")
            
        perturbed_samples.append(x_obs + noise)
        
    return perturbed_samples
