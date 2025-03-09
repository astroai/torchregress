"""
Gaussian loss functions for regression tasks.

This module provides various Gaussian-based loss functions,
including:
- Weighted Mean Squared Error
- Gaussian Negative Log-Likelihood with diagonal covariance
- Gaussian Negative Log-Likelihood with full covariance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal, MultivariateNormal
from typing import Optional, Union, Tuple, Dict

from .base import MaskedLoss, RegressionLoss, DistributionLoss

class WeightedMSELoss(RegressionLoss):
    """
    Weighted Mean Squared Error Loss.
    
    Calculates weighted MSE between predictions and targets,
    with optional masking for missing values.
    
    Args:
        reduction (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'.
                         Default: 'mean'
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction=reduction)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, mask: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate weighted MSE loss.
        
        Args:
            y_true (tensor): Target values (batch_size, n_features)
            y_pred (tensor): Predicted values (batch_size, n_features)
            mask (tensor, optional): Optional mask (batch_size, n_features)
            weights (tensor, optional): Weights for each feature (batch_size, n_features)
            
        Returns:
            tensor: The loss value
        """
        self._validate_inputs(y_true, y_pred, mask)
        
        # Apply mask if provided
        y_true = apply_mask(y_true, mask)
        y_pred = apply_mask(y_pred, mask) 
        
        # Calculate MSE
        mse = F.mse_loss(y_pred, y_true, reduction='none')
        
        # Apply weights if provided
        if weights is not None:
            weights = apply_mask(weights, mask)
            mse = mse * weights
        
        # Apply reduction
        return masked_reduction(mse, mask, self.reduction)


class DiagonalGaussianNLL(DistributionLoss):
    """
    Negative Log-Likelihood loss for diagonal Gaussian distributions.
    
    This loss models each output dimension with an independent Gaussian distribution
    where the diagonal covariance matrix can be learned or fixed.
    
    Args:
        n_features (int, optional): Number of output features (required when learnable_variance=True)
        learnable_variance (bool): Whether to use learnable variance parameters
        fixed_variance (float or tensor): Fixed variance value when learnable_variance=False
        min_variance (float): Minimum variance for numerical stability
        eps (float): Small constant for numerical stability
        reduction (str): 'none' | 'mean' | 'sum'
    """
    def __init__(self, n_features: Optional[int] = None, learnable_variance: bool = True,
                 fixed_variance: float = 1.0, min_variance: float = 1e-6, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.n_features = n_features
        self.min_variance = min_variance
        self.eps = eps
        self.learnable_variance = learnable_variance
        self.log_2pi = math.log(2 * math.pi)
        
        if learnable_variance:
            # Initialize learnable log-variance parameters
            if n_features is None:
                raise ValueError("n_features must be specified when learnable_variance=True")
            self.log_variances = nn.Parameter(torch.zeros(n_features))
        else:
            # Use fixed variance value
            self.register_buffer('fixed_variance', torch.tensor(fixed_variance))
    
    def _extract_distribution_parameters(self, y_pred):
        """
        Extract mean and variance from model predictions.
        
        Args:
            y_pred: Model predictions, can be either:
                   - Mean values (when learnable_variance=True)
                   - Tuple of (mean, log_var)
                   - Tensor with shape [..., 2*n_features] containing concatenated [mean, log_var]
                   
        Returns:
            tuple: (mean, variance)
        """
        if self.learnable_variance:
            # Model only predicts mean, use learned variance parameters
            mean = y_pred
            log_var = self.log_variances
            
            # Broadcast log_var to match mean's shape
            for _ in range(mean.dim() - log_var.dim()):
                log_var = log_var.unsqueeze(0)
            
            var = torch.exp(log_var).expand_as(mean).clamp(min=self.min_variance)
        else:
            # Model might predict both mean and log variance
            if isinstance(y_pred, tuple) and len(y_pred) == 2:
                mean, log_var = y_pred
                var = torch.exp(log_var).clamp(min=self.min_variance)
            elif y_pred.shape[-1] == 2 * (y_pred.shape[-1] // 2):
                # Assume concatenated [mean, log_var]
                n_features = y_pred.shape[-1] // 2
                mean, log_var = y_pred[..., :n_features], y_pred[..., n_features:]
                var = torch.exp(log_var).clamp(min=self.min_variance)
            else:
                # Just mean predictions, use fixed variance
                mean = y_pred
                var = self.fixed_variance * torch.ones_like(mean)
                
        return mean, var
    
    def _calculate_nll(self, y_true, params, mask=None):
        """
        Calculate negative log likelihood for diagonal Gaussian.
        
        Args:
            y_true: Target values
            params: Tuple of (mean, variance)
            mask: Optional mask for valid values
            
        Returns:
            tensor: Negative log likelihood values
        """
        mean, var = params
        
        # Apply mask if provided
        if mask is not None:
            y_true = apply_mask(y_true, mask)
            mean = apply_mask(mean, mask)
            var = apply_mask(var, mask)
            
        # Calculate NLL: 0.5 * (log(var) + (y-μ)²/var + log(2π))
        squared_error = (y_true - mean) ** 2
        nll = 0.5 * (torch.log(var + self.eps) + squared_error / (var + self.eps) + self.log_2pi)
        
        return nll
    
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Gaussian negative log-likelihood loss.
        
        Args:
            y_true: Ground truth values [batch_size, n_features]
            y_pred: Predictions - see _extract_distribution_parameters for formats
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional sample weights [batch_size, n_features] or [batch_size]
        
        Returns:
            Negative log-likelihood loss
        """
        self._validate_inputs(y_true, y_pred if not isinstance(y_pred, tuple) else y_pred[0], mask)
        
        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred)
        
        # Calculate NLL
        nll = self._calculate_nll(y_true, params, mask)
        
        # Apply weights if provided
        if weights is not None:
            weights = apply_mask(weights, mask)
            nll = nll * weights
        
        # Apply reduction
        return masked_reduction(nll, mask, self.reduction)


class GaussianNLLWithCovariance(DistributionLoss):
    """
    Negative Log-Likelihood loss for multivariate Gaussian with full covariance matrices.
    
    Models the output with a multivariate Gaussian distribution with a full covariance matrix,
    which can capture correlations between features.
    
    Args:
        n_features (int, optional): Number of features (required for learnable_adjustment)
        learnable_adjustment (bool): Whether to learn feature-specific variance adjustments
        jitter (float): Small value added to diagonal for numerical stability
        eps (float): Small constant for numerical stability in log calculations
        reduction (str): 'none' | 'mean' | 'sum'
    """
    def __init__(
        self, 
        n_features=None,
        learnable_adjustment=False,
        jitter=1e-6, 
        eps=1e-8,
        reduction='mean'
    ):
        super().__init__(reduction=reduction)
        self.n_features = n_features
        self.jitter = jitter
        self.eps = eps
        self.learnable_adjustment = learnable_adjustment
        self.log_2pi = math.log(2 * math.pi)
        
        if learnable_adjustment:
            if n_features is None:
                raise ValueError("n_features must be specified when learnable_adjustment=True")
            self.log_variance_adjustment = nn.Parameter(torch.zeros(n_features))
    
    def _extract_distribution_parameters(self, y_pred, covariance_matrices):
        """
        Extract mean and covariance from model predictions.
        
        Args:
            y_pred: Predicted mean values [batch_size, n_features]
            covariance_matrices: Covariance matrices [batch_size, n_features, n_features] 
                                 or [n_features, n_features]
                   
        Returns:
            tuple: (mean, covariance)
        """
        # Handle single covariance matrix case
        if covariance_matrices.ndim == 2:
            batch_size = y_pred.shape[0]
            covariance_matrices = covariance_matrices.unsqueeze(0).expand(batch_size, -1, -1)
            
        # Apply variance adjustment if needed
        adjusted_cov = covariance_matrices
        if self.learnable_adjustment:
            # Get variance adjustment factors
            variance_adjustment = torch.exp(self.log_variance_adjustment)
            n_expand_dims = y_pred.dim() - 1
            for _ in range(n_expand_dims):
                variance_adjustment = variance_adjustment.unsqueeze(0)
            variance_adjustment = variance_adjustment.expand(*y_pred.shape)
            
            # Add to diagonal of covariance matrices
            adjusted_cov = covariance_matrices + torch.diag_embed(variance_adjustment)
            
        # Add small jitter to diagonal for numerical stability
        n_features = adjusted_cov.shape[-1]
        device = y_pred.device
        jitter_matrix = self.jitter * torch.eye(n_features, device=device)
        jitter_matrix = jitter_matrix.unsqueeze(0).expand_as(adjusted_cov)
        cov_matrices = adjusted_cov + jitter_matrix
        
        return y_pred, cov_matrices
    
    def _calculate_nll(self, y_true, params, mask=None):
        """
        Calculate negative log likelihood for multivariate Gaussian.
        
        Args:
            y_true: Target values [batch_size, n_features]
            params: Tuple of (mean, covariance)
            mask: Optional mask for valid values
            
        Returns:
            tensor: Negative log likelihood values [batch_size]
        """
        mean, cov = params
        batch_size, n_features = mean.shape
        
        # We can't directly apply masks to multivariate distributions
        # If masks are provided, we should handle this separately
        # For this implementation, we assume no masking or complete masking of samples
        
        try:
            # Try using torch distribution (most stable)
            mvn = MultivariateNormal(mean, cov)
            nll = -mvn.log_prob(y_true)
        except Exception:
            # Fallback with Cholesky decomposition
            try:
                L = torch.linalg.cholesky(cov)
                
                # log(det(Σ)) = 2 * sum(log(diag(L)))
                log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + self.eps), dim=1)
                
                # (y-μ)ᵀΣ⁻¹(y-μ) = ‖L⁻¹(y-μ)‖²
                residuals = y_true - mean
                z = torch.linalg.solve_triangular(L, residuals.unsqueeze(-1), upper=False).squeeze(-1)
                quadratic = torch.sum(z**2, dim=1)
                
                # -log p(y) = 0.5 * (log(det(Σ)) + (y-μ)ᵀΣ⁻¹(y-μ) + n*log(2π))
                nll = 0.5 * (log_det + quadratic + n_features * self.log_2pi)
            except Exception:
                # Ultimate fallback with eigendecomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                eigenvalues = torch.clamp(eigenvalues, min=self.eps)
                
                # log(det(Σ)) = sum(log(λ_i))
                log_det = torch.sum(torch.log(eigenvalues), dim=1)
                
                # Compute (y-μ)ᵀΣ⁻¹(y-μ) using eigendecomposition
                residuals = y_true - mean
                whitened = torch.bmm(eigenvectors.transpose(-1, -2), residuals.unsqueeze(-1)).squeeze(-1)
                quadratic = torch.sum(whitened**2 / eigenvalues, dim=1)
                
                nll = 0.5 * (log_det + quadratic + n_features * self.log_2pi)
                
        return nll
    
    def forward(self, y_true, y_pred, covariance_matrices, mask=None, weights=None):
        """
        Calculate multivariate Gaussian negative log-likelihood loss.
        
        Args:
            y_true: Ground truth values [batch_size, n_features]
            y_pred: Predicted mean values [batch_size, n_features]
            covariance_matrices: Predicted covariance matrices [batch_size, n_features, n_features] or
                                [n_features, n_features] for a shared covariance
            mask: Optional boolean mask for complete samples [batch_size] 
            weights: Optional sample weights [batch_size]
            
        Returns:
            Negative log-likelihood loss
        """
        self._validate_inputs(y_true, y_pred, None)  # Special handling for mask with multivariate
        
        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred, covariance_matrices)
        
        # Calculate NLL - returns per-sample NLL [batch_size]
        nll = self._calculate_nll(y_true, params, None)  # Handle mask separately
        
        # Apply sample mask if provided (can only mask whole samples)
        if mask is not None:
            if mask.dim() > 1:
                # Convert feature-level mask to sample-level mask
                sample_mask = mask.all(dim=1) if mask.dim() == 2 else mask
            else:
                sample_mask = mask
            nll = nll * sample_mask
        
        # Apply weights if provided
        if weights is not None:
            if weights.dim() > 1:
                # Convert feature-level weights to sample-level weights (average)
                sample_weights = weights.mean(dim=1) if weights.dim() == 2 else weights
            else:
                sample_weights = weights
            nll = nll * sample_weights
        
        # Apply reduction
        if self.reduction == 'none':
            return nll
        elif self.reduction == 'mean':
            return nll.mean()
        else:  # 'sum'
            return nll.sum()


def create_gaussian_nll(n_features: int, covariance_type: str = 'diagonal', learnable_variance: bool = True,
                        fixed_variance: float = 1.0, jitter: float = 1e-6, reduction: str = 'mean', **kwargs) -> DistributionLoss:
    """
    Factory function to create an appropriate Gaussian NLL loss.
    
    Args:
        n_features (int): Number of features
        covariance_type (str): One of 'diagonal', 'full'
        learnable_variance (bool): Whether to learn variance parameters
        fixed_variance (float): Fixed variance value when not learning
        jitter (float): Regularization strength for numerical stability
        reduction (str): 'none' | 'mean' | 'sum'
        **kwargs: Additional arguments for specific loss types
    
    Returns:
        An appropriate Gaussian NLL loss object
    """
    if covariance_type == 'diagonal':
        return DiagonalGaussianNLL(n_features=n_features, learnable_variance=learnable_variance,
                                   fixed_variance=fixed_variance, reduction=reduction, **kwargs)
    elif covariance_type == 'full':
        return GaussianNLLWithCovariance(n_features=n_features, learnable_adjustment=learnable_variance,
                                         jitter=jitter, reduction=reduction, **kwargs)
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type}")