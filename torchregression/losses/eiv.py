"""
Error-in-Variables (EIV) loss functions for regression with uncertain inputs.

This module provides implementations of various loss functions that account for
measurement error in both features and targets.
"""
import torch
import torch.nn as nn
from typing import Callable, Optional, Union, Dict, Any

from .base import MaskedLoss
from ..utils.tensor_ops import prepare_param, prepare_covariance
from ..eiv.eiv_utils import (
    prepare_cross_covariance,
    compute_model_gradients, 
    calculate_gaussian_nll,
    prepare_model_input_for_gradients
)
from ..eiv.eiv_chamfer import ChamferEIVLoss, HybridEIVChamferLoss
from ..eiv.eiv_mdn import MDNEIVLoss
from ..eiv.eiv_rfit import RobustEIVLoss, gaussian_variation, uniform_variation, bootstrap_variation

class TotalLeastSquaresLoss(MaskedLoss):
    """
    Total Least Squares loss for linear Error-in-Variables regression.
    
    This implements the classic TLS method where errors in both x and y are considered.
    The solution minimizes the perpendicular distances from data points to the model.
    
    Args:
        sigma_x: Standard deviation of feature noise (scalar, vector or matrix)
        sigma_y: Standard deviation of target noise (scalar, vector or matrix)
        lambd: Regularization parameter for ridge-like regularization
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(
        self,
        sigma_x: Union[float, torch.Tensor],
        sigma_y: Union[float, torch.Tensor],
        lambd: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.lambd = lambd
        
    def forward(self, x_obs, y_true, weights=None):
        """
        Calculate Total Least Squares loss.
        
        Args:
            x_obs: Observed features with noise [batch_size, n_features_x]
            y_true: Observed targets with noise [batch_size, 1]
            weights: Optional sample weights [batch_size]
            
        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        batch_size, n_features_x = x_obs.shape
        device = x_obs.device
        
        # Center the data (equivalent to including an intercept)
        x_mean = torch.mean(x_obs, dim=0, keepdim=True)
        y_mean = torch.mean(y_true, dim=0, keepdim=True)
        
        x_centered = x_obs - x_mean
        y_centered = y_true - y_mean
        
        # Prepare covariance matrices
        sigma_x = prepare_covariance(self.sigma_x, n_features_x, device)
        sigma_y = prepare_param(self.sigma_y, 1, device)
        
        # Combine into augmented matrices for TLS
        if sigma_x.ndim <= 1 and sigma_y.ndim <= 1:
            # Diagonal case - simpler computation
            # Form augmented data matrix [X y]
            data = torch.cat([x_centered, y_centered], dim=1)
            
            # Compute covariance matrix of augmented data
            cov = torch.matmul(data.t(), data) / batch_size
            
            # Add error covariance to diagonal
            error_cov = torch.zeros(n_features_x + 1, device=device)
            error_cov[:n_features_x] = sigma_x if sigma_x.ndim == 1 else sigma_x.diagonal()
            error_cov[-1] = sigma_y.item() if hasattr(sigma_y, 'item') else sigma_y
            
            # Regularize if needed
            if self.lambd > 0:
                error_cov[:n_features_x] += self.lambd
                
            # Get eigendecomposition of adjusted covariance
            adjusted_cov = cov - torch.diag(error_cov)
            
            # Find the eigenvector corresponding to the smallest eigenvalue
            try:
                eigenvals, eigenvecs = torch.linalg.eigh(adjusted_cov)
            except:
                # Fallback for numerical issues
                jitter = torch.eye(adjusted_cov.shape[0], device=device) * 1e-6
                eigenvals, eigenvecs = torch.linalg.eigh(adjusted_cov + jitter)
                
            # Last eigenvector gives the normal to the best-fit hyperplane
            a = -eigenvecs[:-1, 0]  # Coefficients for features
            b = eigenvecs[-1, 0]    # Coefficient for target
            
            # Calculate distances to the hyperplane
            if torch.abs(b) < 1e-10:
                # Handling degenerate case where hyperplane is parallel to y-axis
                dists = torch.sum((x_centered @ a.unsqueeze(1))**2, dim=1)
            else:
                # Calculate orthogonal distances to the hyperplane
                dists = torch.abs(torch.sum(x_centered * a, dim=1) - y_centered.squeeze(1) * b)**2 / (torch.norm(a)**2 + b**2)
                
            # Apply weights if provided
            if weights is not None:
                dists = dists * weights
                
            # Return weighted loss
            return self._reduce(dists, None)
        else:
            # Full covariance case - more complex
            # Form augmented data matrix [X y]
            data = torch.cat([x_centered, y_centered], dim=1)
            
            # Create full error covariance block matrix [[Σx, 0], [0, Σy]]
            error_cov = torch.zeros(n_features_x + 1, n_features_x + 1, device=device)
            error_cov[:n_features_x, :n_features_x] = sigma_x
            error_cov[n_features_x, n_features_x] = sigma_y.item() if hasattr(sigma_y, 'item') else sigma_y
            
            # Add regularization if needed
            if self.lambd > 0:
                reg = torch.eye(n_features_x + 1, device=device) * self.lambd
                error_cov[:n_features_x, :n_features_x] += reg[:n_features_x, :n_features_x]
            
            # Compute sample covariance of augmented data
            cov = torch.matmul(data.t(), data) / batch_size
            
            # Adjust covariance by error covariance
            adjusted_cov = cov - error_cov
            
            # Get generalized eigendecomposition
            try:
                eigenvals, eigenvecs = torch.linalg.eigh(adjusted_cov, error_cov)
            except:
                # Fallback for numerical issues
                jitter = torch.eye(adjusted_cov.shape[0], device=device) * 1e-6
                eigenvals, eigenvecs = torch.linalg.eigh(adjusted_cov, error_cov + jitter)
            
            # Use eigenvector with smallest eigenvalue
            params = eigenvecs[:, 0]
            a = -params[:n_features_x]  # Feature coefficients
            b = params[n_features_x]    # Target coefficient
            
            # Calculate Mahalanobis distances to the hyperplane
            if torch.abs(b) < 1e-10:
                # Handle degenerate case
                mahal_dists = torch.zeros(batch_size, device=device)
                for i in range(batch_size):
                    v = x_centered[i].unsqueeze(0).t()
                    mahal_dists[i] = torch.matmul(torch.matmul(v.t(), torch.linalg.inv(sigma_x)), v)
            else:
                # Calculate distances using error covariance
                mahal_dists = torch.zeros(batch_size, device=device)
                for i in range(batch_size):
                    # Calculate residual vector
                    residual = torch.cat([
                        -x_centered[i] * b,
                        y_centered[i] * a.sum()
                    ]).unsqueeze(1)
                    
                    # Calculate Mahalanobis distance
                    mahal_dists[i] = torch.matmul(
                        torch.matmul(residual.t(), torch.linalg.inv(error_cov)),
                        residual
                    )
                    
            # Apply weights if provided
            if weights is not None:
                mahal_dists = mahal_dists * weights
                
            # Return weighted loss
            return self._reduce(mahal_dists, None)
        
class GeneralErrorInVariablesLoss(MaskedLoss):
    """
    General Error-in-Variables loss for nonlinear models.
    
    This loss propagates uncertainty from the inputs to the outputs
    by using the model's gradient to calculate the propagated variance.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise (scalar, vector or matrix)
        sigma_y: Standard deviation of target noise (scalar, vector or matrix)
        monte_carlo: Whether to use Monte Carlo sampling for gradient estimation
        n_samples: Number of MC samples if monte_carlo=True
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability
    """
    def __init__(
        self,
        model: Callable,
        sigma_x: Union[float, torch.Tensor],
        sigma_y: Optional[Union[float, torch.Tensor]] = None,
        monte_carlo: bool = False,
        n_samples: int = 20,
        reduction: str = 'mean',
        eps: float = 1e-8
    ):
        super().__init__(reduction=reduction)
        self.model = model
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.monte_carlo = monte_carlo
        self.n_samples = n_samples
        self.eps = eps
        
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate General EIV loss.
        
        Args:
            x_obs: Observed features with noise [batch_size, n_features_x]
            y_true: Observed targets with noise [batch_size, n_features_y]
            mask: Optional boolean mask [batch_size, n_features_y]
            
        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        # Apply mask to targets if provided
        y_true = self._apply_mask(y_true, mask)
        
        batch_size, n_features_x = x_obs.shape
        n_features_y = y_true.shape[1]
        device = x_obs.device
        
        # Prepare input for gradient computation (needs requires_grad=True)
        x_grad = prepare_model_input_for_gradients(x_obs)
        
        # Forward pass through the model
        y_pred = self.model(x_grad)
        
        # Apply mask to predictions if provided
        if mask is not None:
            y_pred = self._apply_mask(y_pred, mask)
            
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Prepare noise parameters
        sigma_x_tensor = prepare_covariance(self.sigma_x, n_features_x, device)
        
        if self.sigma_y is not None:
            sigma_y_tensor = prepare_covariance(self.sigma_y, n_features_y, device)
        else:
            sigma_y_tensor = None
        
        # Calculate propagated variance if using analytical approach
        if not self.monte_carlo:
            # Calculate gradients of predictions with respect to inputs
            grad = compute_model_gradients(y_pred, x_grad, n_features_y)
            
            # Propagate variance from inputs to outputs
            propagated_var = calculate_propagated_variance(
                grad, sigma_x_tensor, sigma_y=sigma_y_tensor
            )
                
            # Calculate negative log-likelihood given residuals and propagated variance
            nll = calculate_gaussian_nll(residuals, propagated_var, eps=self.eps)
        else:
            # Monte Carlo approach: sample multiple inputs around observed values
            samples = []
            for _ in range(self.n_samples):
                # Generate input sample
                if sigma_x_tensor.ndim <= 1:
                    # Diagonal covariance - simple sampling
                    noise = torch.randn_like(x_obs) * sigma_x_tensor.view(1, -1)
                else:
                    # Full covariance - need to use multivariate normal
                    noise = torch.distributions.MultivariateNormal(
                        torch.zeros(n_features_x, device=device),
                        sigma_x_tensor
                    ).sample((batch_size,))
                
                x_sample = x_obs + noise
                samples.append(x_sample)
            
            # Stack samples [n_samples, batch_size, n_features_x]
            x_samples = torch.stack(samples)
            
            # Reshape for batch processing
            x_flat = x_samples.reshape(-1, n_features_x)
            
            # Forward pass for all samples
            with torch.no_grad():
                y_preds_flat = self.model(x_flat)
                
                # Reshape predictions [n_samples, batch_size, n_features_y]
                if y_preds_flat.shape[-1] == n_features_y:
                    y_preds = y_preds_flat.reshape(self.n_samples, batch_size, n_features_y)
                else:
                    # Handle case where model output doesn't match expected shape
                    y_preds = y_preds_flat.reshape(self.n_samples, batch_size, -1)
                    y_preds = y_preds[..., :n_features_y]
            
            # Calculate mean prediction across samples
            mean_pred = torch.mean(y_preds, dim=0)
            
            # Calculate sample covariance across predictions
            y_centered = y_preds - mean_pred.unsqueeze(0)
            
            # [batch_size, n_features_y, n_features_y]
            pred_cov = torch.zeros(batch_size, n_features_y, n_features_y, device=device)
            
            for i in range(batch_size):
                # Calculate sample covariance for this batch element
                batch_centered = y_centered[:, i, :]  # [n_samples, n_features_y]
                batch_cov = torch.matmul(batch_centered.t(), batch_centered) / (self.n_samples - 1)
                pred_cov[i] = batch_cov
            
            # Add intrinsic output noise if provided
            if sigma_y_tensor is not None:
                if sigma_y_tensor.ndim <= 1:
                    # Diagonal case
                    for i in range(batch_size):
                        pred_cov[i] = pred_cov[i] + torch.diag(sigma_y_tensor)
                else:
                    # Full covariance case
                    for i in range(batch_size):
                        pred_cov[i] = pred_cov[i] + sigma_y_tensor
            
            # Calculate residuals from mean prediction
            residuals_mc = y_true - mean_pred
            
            # Calculate negative log-likelihood
            nll = calculate_gaussian_nll(residuals_mc, pred_cov, eps=self.eps)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)
        else:  # 'none'
            return nll


class CorrelatedEIVLoss(MaskedLoss):
    """
    EIV loss that handles correlated errors in features and targets.
    
    This extends the GeneralErrorInVariablesLoss to account for correlations
    between errors in x and y through a cross-covariance matrix.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Covariance of feature noise
        sigma_y: Covariance of target noise
        sigma_xy: Cross-covariance between feature and target noise
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability
    """
    def __init__(
        self,
        model: Callable,
        sigma_x: Union[float, torch.Tensor],
        sigma_y: Union[float, torch.Tensor],
        sigma_xy: torch.Tensor,
        reduction: str = 'mean',
        eps: float = 1e-8
    ):
        super().__init__(reduction=reduction)
        self.model = model
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_xy = sigma_xy
        self.eps = eps
        
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate Correlated EIV loss.
        
        Args:
            x_obs: Observed features with noise [batch_size, n_features_x]
            y_true: Observed targets [batch_size, n_features_y]
            mask: Optional boolean mask [batch_size, n_features_y]
            
        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        # Apply mask to targets if provided
        y_true = self._apply_mask(y_true, mask)
        
        batch_size, n_features_x = x_obs.shape
        n_features_y = y_true.shape[1]
        device = x_obs.device
        
        # Prepare input for gradient computation
        x_grad = prepare_model_input_for_gradients(x_obs)
        
        # Forward pass through the model
        y_pred = self.model(x_grad)
        
        # Apply mask to predictions if provided
        if mask is not None:
            y_pred = self._apply_mask(y_pred, mask)
            
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Prepare covariance matrices
        sigma_x_tensor = prepare_covariance(self.sigma_x, n_features_x, device)
        sigma_y_tensor = prepare_covariance(self.sigma_y, n_features_y, device)
        sigma_xy_tensor = prepare_cross_covariance(self.sigma_xy, n_features_x, n_features_y, device)
        
        # Calculate gradients of predictions with respect to inputs
        grad = compute_model_gradients(y_pred, x_grad, n_features_y)
        
        # Propagate input variance to output variance
        propagated_var = calculate_propagated_variance(
            grad, sigma_x_tensor, sigma_xy=sigma_xy_tensor, sigma_y=sigma_y_tensor
        )
        
        # Calculate negative log-likelihood
        nll = calculate_gaussian_nll(residuals, propagated_var, eps=self.eps)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)
        else:  # 'none'
            return nll


# Factory functions for easier creation of EIV losses
def create_eiv_loss(
    model: Callable,
    sigma_x: Union[float, torch.Tensor],
    sigma_y: Optional[Union[float, torch.Tensor]] = None,
    reduction: str = 'mean',
    monte_carlo: bool = False,
    n_samples: int = 20,
    eps: float = 1e-8,
    **kwargs
) -> GeneralErrorInVariablesLoss:
    """
    Create a general Error-in-Variables loss function.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise
        sigma_y: Standard deviation of target noise (optional)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        monte_carlo: Whether to use Monte Carlo sampling
        n_samples: Number of MC samples if monte_carlo=True
        eps: Small value for numerical stability
        **kwargs: Additional parameters for specific loss types
        
    Returns:
        GeneralErrorInVariablesLoss instance
    """
    return GeneralErrorInVariablesLoss(
        model=model,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        monte_carlo=monte_carlo,
        n_samples=n_samples,
        reduction=reduction,
        eps=eps
    )


def create_correlated_eiv_loss(
    model: Callable,
    sigma_x: Union[float, torch.Tensor],
    sigma_y: Union[float, torch.Tensor],
    sigma_xy: torch.Tensor,
    reduction: str = 'mean',
    eps: float = 1e-8,
    **kwargs
) -> CorrelatedEIVLoss:
    """
    Create an EIV loss that handles correlated errors.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Covariance of feature noise
        sigma_y: Covariance of target noise
        sigma_xy: Cross-covariance between feature and target noise
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability
        **kwargs: Additional parameters for specific loss types
        
    Returns:
        CorrelatedEIVLoss instance
    """
    return CorrelatedEIVLoss(
        model=model,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        sigma_xy=sigma_xy,
        reduction=reduction,
        eps=eps
    )


def create_chamfer_eiv_loss(
    model: Callable,
    sigma_x: Optional[Union[float, torch.Tensor]] = None,
    method: str = 'monte_carlo',
    n_samples: int = 100,
    optim_steps: int = 50,
    optim_lr: float = 0.01,
    reduction: str = 'mean',
    **kwargs
) -> ChamferEIVLoss:
    """
    Create a Chamfer distance-based EIV loss.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise (for sampling)
        method: Method for finding closest point ('monte_carlo', 'optimization')
        n_samples: Number of Monte Carlo samples (for 'monte_carlo' method)
        optim_steps: Number of optimization steps (for 'optimization' method)
        optim_lr: Learning rate for optimization (for 'optimization' method)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        **kwargs: Additional parameters for specific loss types
        
    Returns:
        ChamferEIVLoss instance
    """
    return ChamferEIVLoss(
        model=model,
        sigma_x=sigma_x,
        method=method,
        n_samples=n_samples,
        optim_steps=optim_steps,
        optim_lr=optim_lr,
        reduction=reduction
    )


def create_hybrid_eiv_loss(
    model: Callable,
    sigma_x: Union[float, torch.Tensor],
    sigma_y: Optional[Union[float, torch.Tensor]] = None,
    alpha: float = 0.5,
    method: str = 'monte_carlo',
    n_samples: int = 50,
    reduction: str = 'mean',
    **kwargs
) -> HybridEIVChamferLoss:
    """
    Create a hybrid EIV-Chamfer loss.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise
        sigma_y: Standard deviation of target noise (optional)
        alpha: Weight for the EIV component (0-1)
        method: Method for Chamfer loss ('monte_carlo', 'optimization')
        n_samples: Number of Monte Carlo samples (for 'monte_carlo' method)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        **kwargs: Additional parameters for specific loss types
        
    Returns:
        HybridEIVChamferLoss instance
    """
    eiv_loss = GeneralErrorInVariablesLoss(
        model=model,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        reduction='none'
    )
    
    chamfer_loss = ChamferEIVLoss(
        model=model,
        sigma_x=sigma_x,
        method=method,
        n_samples=n_samples,
        reduction='none'
    )
    
    return HybridEIVChamferLoss(
        eiv_loss=eiv_loss,
        chamfer_loss=chamfer_loss,
        alpha=alpha,
        reduction=reduction
    )


def create_mdn_eiv_loss(
    num_components: int,
    n_features_y: int,
    sigma_x: Union[float, torch.Tensor],
    sigma_y: Optional[Union[float, torch.Tensor]] = None,
    min_sigma: float = 1e-4,
    eps: float = 1e-8,
    reduction: str = 'mean',
    **kwargs
) -> MDNEIVLoss:
    """
    Create an EIV loss for Mixture Density Networks.
    
    Args:
        num_components: Number of mixture components
        n_features_y: Dimensionality of the target
        sigma_x: Standard deviation of feature noise
        sigma_y: Standard deviation of target noise (optional)
        min_sigma: Minimum value for standard deviation
        eps: Small value for numerical stability
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        **kwargs: Additional parameters for specific loss types
        
    Returns:
        MDNEIVLoss instance
    """
    return MDNEIVLoss(
        num_components=num_components,
        n_features=n_features_y,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        min_sigma=min_sigma,
        eps=eps,
        reduction=reduction
    )


def create_robust_eiv_loss(
    model: Callable,
    base_loss: str = 'huber',
    delta: float = 1.0,
    variation_fn: Optional[Callable] = None,
    sigma_x: Union[float, torch.Tensor] = 1.0,
    n_samples: int = 10,
    aggregation: str = 'median',
    reduction: str = 'mean',
    **kwargs
) -> RobustEIVLoss:
    """
    Create a robust EIV loss that uses multiple forward passes.
    
    Args:
        model: Model function f(x) that predicts y
        base_loss: Base loss function ('huber', 'l1', 'mse')
        delta: Delta parameter for Huber loss
        variation_fn: Function to generate input variations
        sigma_x: Standard deviation for input variations
        n_samples: Number of variations to generate
        aggregation: How to aggregate losses: 'mean', 'median', 'max', 'quantile'
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        **kwargs: Additional parameters for specific loss types
        
    Returns:
        RobustEIVLoss instance
    """
    if variation_fn is None:
        variation_fn = gaussian_variation
        
    return RobustEIVLoss(
        model=model,
        base_loss=base_loss,
        delta=delta,
        variation_fn=variation_fn,
        sigma_x=sigma_x,
        n_samples=n_samples,
        variation_params=kwargs.get('variation_params', {}),
        aggregation=aggregation,
        quantile=kwargs.get('quantile', 0.95),
        reduction=reduction
    )
