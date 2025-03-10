import torch
from typing import Callable, Optional, Union

from ..base import RegressionLoss
from .eiv_utils import (
    prepare_covariance,
    prepare_cross_covariance,
    compute_model_gradients, 
    calculate_gaussian_nll,
    prepare_model_input_for_gradients,
    calculate_propagated_variance,
    generate_perturbed_samples
)

class BaseEIVLoss(RegressionLoss):
    """
    Base class for Errors-In-Variables regression loss functions.
    
    This provides common functionality for all EIV loss variants.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise (scalar, vector or matrix)
        sigma_y: Standard deviation of target noise (scalar, vector or matrix)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability
    """
    def __init__(self, model: Callable, sigma_x: Union[float, torch.Tensor], 
                 sigma_y: Optional[Union[float, torch.Tensor]] = None,
                 reduction: str = 'mean', eps: float = 1e-8):
        super().__init__(reduction=reduction)
        self.model = model
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.eps = eps
        
    def _prepare_covariances(self, n_features_x, n_features_y, device):
        """
        Prepare covariance matrices for features and targets.
        
        Args:
            n_features_x: Number of features in input
            n_features_y: Number of features in output
            device: Device to create tensors on
            
        Returns:
            Tuple of (sigma_x_tensor, sigma_y_tensor)
        """
        sigma_x_tensor = prepare_covariance(self.sigma_x, n_features_x, device)
        sigma_y_tensor = None if self.sigma_y is None else prepare_covariance(
            self.sigma_y, n_features_y, device
        )
        return sigma_x_tensor, sigma_y_tensor
        
    def _prepare_inverse_covariances(self, sigma_x_tensor, sigma_y_tensor, n_features_x, n_features_y, device):
        """
        Calculate inverse covariance matrices for Mahalanobis distances.
        
        Args:
            sigma_x_tensor: Covariance matrix for features
            sigma_y_tensor: Covariance matrix for targets
            n_features_x: Number of features in input
            n_features_y: Number of features in output
            device: Device to create tensors on
            
        Returns:
            Tuple of (sigma_x_inv, sigma_y_inv)
        """
        if sigma_x_tensor.ndim <= 1:
            sigma_x_inv = 1.0 / (sigma_x_tensor + self.eps)
        else:
            sigma_x_inv = torch.inverse(sigma_x_tensor + torch.eye(n_features_x, device=device) * self.eps)
            
        if sigma_y_tensor.ndim <= 1:
            sigma_y_inv = 1.0 / (sigma_y_tensor + self.eps)
        else:
            sigma_y_inv = torch.inverse(sigma_y_tensor + torch.eye(n_features_y, device=device) * self.eps)
            
        return sigma_x_inv, sigma_y_inv
    
    def _calculate_mahalanobis_distance(self, diff, sigma_inv):
        """
        Calculate weighted Mahalanobis distance.
        
        Args:
            diff: Difference vector or matrix [batch_size, n_features]
            sigma_inv: Inverse covariance matrix or diagonal vector
            
        Returns:
            Distance per sample [batch_size]
        """
        if sigma_inv.ndim <= 1:
            return torch.sum(diff**2 * sigma_inv, dim=1)
        else:
            return torch.sum(diff * torch.matmul(diff, sigma_inv), dim=1)
    
    def _apply_model_with_mask(self, x, mask=None):
        """Apply model to input and apply mask if needed"""
        y_pred = self.model(x)
        if mask is not None:
            y_pred = self._apply_mask(y_pred, mask)
        return y_pred


class FunctionalEIVLoss(BaseEIVLoss):
    """
    Functional Errors-In-Variables Loss (previously GeneralErrorInVariablesLoss).
    
    This loss implements the functional approach to errors-in-variables modeling,
    where the true values are treated as fixed but unknown parameters.
    It propagates uncertainty from the inputs to the outputs using a 
    first-order Taylor approximation through model gradients.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise (scalar, vector or matrix)
        sigma_y: Standard deviation of target noise (scalar, vector or matrix)
        monte_carlo: Whether to use Monte Carlo sampling for gradient estimation
        n_samples: Number of MC samples if monte_carlo=True
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability
    """
    def __init__(self, model: Callable, sigma_x: Union[float, torch.Tensor], 
                 sigma_y: Optional[Union[float, torch.Tensor]] = None,
                 monte_carlo: bool = False, n_samples: int = 20, 
                 reduction: str = 'mean', eps: float = 1e-8):
        super().__init__(model, sigma_x, sigma_y, reduction, eps)
        self.monte_carlo = monte_carlo
        self.n_samples = n_samples
        
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate Functional EIV loss.
        
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
        
        # Prepare noise parameters
        sigma_x_tensor, sigma_y_tensor = self._prepare_covariances(n_features_x, n_features_y, device)
        
        if not self.monte_carlo:
            # Analytical approach: use gradients to propagate uncertainty
            x_grad = prepare_model_input_for_gradients(x_obs)
            y_pred = self._apply_model_with_mask(x_grad, mask)
            residuals = y_true - y_pred
            
            # Calculate gradients and propagate variance
            grad = compute_model_gradients(y_pred, x_grad, n_features_y)
            
            # Propagate variance from inputs to outputs
            propagated_var = calculate_propagated_variance(
                grad, sigma_x_tensor, sigma_y=sigma_y_tensor
            )
                
            # Calculate negative log-likelihood
            nll = calculate_gaussian_nll(residuals, propagated_var, eps=self.eps)
        else:
            # Monte Carlo approach
            nll = self._monte_carlo_forward(x_obs, y_true, sigma_x_tensor, sigma_y_tensor, 
                                           batch_size, n_features_x, n_features_y, device, mask)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)
        else:  # 'none'
            return nll
            
    def _monte_carlo_forward(self, x_obs, y_true, sigma_x_tensor, sigma_y_tensor, 
                            batch_size, n_features_x, n_features_y, device, mask=None):
        """Monte Carlo implementation of variance propagation"""
        # Sample multiple inputs around observed values
        samples = generate_perturbed_samples(
            x_obs, sigma_x_tensor, self.n_samples, perturb_method='gaussian'
        )
        
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
            
            # Apply mask if provided
            if mask is not None:
                mask_expanded = mask.unsqueeze(0).expand(self.n_samples, -1, -1)
                y_preds = torch.where(mask_expanded, y_preds, torch.zeros_like(y_preds))
        
        # Calculate mean prediction across samples
        mean_pred = torch.mean(y_preds, dim=0)
        
        # Calculate covariance more efficiently
        y_centered = y_preds - mean_pred.unsqueeze(0)
        batch_centered = y_centered.permute(1, 0, 2)  # [batch_size, n_samples, n_features_y]
        
        # Batch matrix multiplication for all batch elements at once
        batch_cov = torch.bmm(
            batch_centered.transpose(1, 2), 
            batch_centered
        ) / (self.n_samples - 1)  # [batch_size, n_features_y, n_features_y]
        
        # Add intrinsic output noise if provided
        if sigma_y_tensor is not None:
            if sigma_y_tensor.ndim <= 1:
                # Diagonal case
                diag_indices = torch.arange(n_features_y, device=device)
                batch_cov[:, diag_indices, diag_indices] += sigma_y_tensor
            else:
                # Full covariance case
                batch_cov += sigma_y_tensor
        
        # Calculate residuals from mean prediction
        residuals_mc = y_true - mean_pred
        
        # Calculate negative log-likelihood
        return calculate_gaussian_nll(residuals_mc, batch_cov, eps=self.eps)


class StructuralEIVLoss(BaseEIVLoss):
    """
    Structural Errors-In-Variables Loss (previously CorrelatedEIVLoss).
    
    This implements the structural approach to errors-in-variables modeling,
    which accounts for correlations between errors in x and y through 
    a cross-covariance matrix.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Covariance of feature noise
        sigma_y: Covariance of target noise
        sigma_xy: Cross-covariance between feature and target noise
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability
    """
    def __init__(self, model: Callable, sigma_x: Union[float, torch.Tensor], 
                 sigma_y: Union[float, torch.Tensor], sigma_xy: torch.Tensor, 
                 reduction: str = 'mean', eps: float = 1e-8):
        super().__init__(model, sigma_x, sigma_y, reduction, eps)
        self.sigma_xy = sigma_xy
        
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate Structural EIV loss.
        
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
        
        # Forward pass and apply mask if needed
        y_pred = self._apply_model_with_mask(x_grad, mask)
            
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Prepare covariance matrices
        sigma_x_tensor, sigma_y_tensor = self._prepare_covariances(n_features_x, n_features_y, device)
        sigma_xy_tensor = prepare_cross_covariance(self.sigma_xy, n_features_x, n_features_y, device)
        
        # Calculate gradients of predictions with respect to inputs
        grad = compute_model_gradients(y_pred, x_grad, n_features_y)
        
        # Propagate input variance to output variance with cross-covariance
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


class OrthogonalDistanceRegressionLoss(BaseEIVLoss):
    """
    Orthogonal Distance Regression (ODR) loss.
    
    This loss minimizes the orthogonal (perpendicular) distances from data points
    to the model curve by optimizing latent true x values during the forward pass.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise (scalar, vector, or matrix)
        sigma_y: Standard deviation of target noise (scalar, vector, or matrix)
        learning_rate: Learning rate for the latent x optimization
        max_iterations: Maximum iterations for latent x optimization
        tolerance: Convergence criterion for optimization
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability
    """
    def __init__(self, model: Callable, sigma_x: Union[float, torch.Tensor], 
                 sigma_y: Union[float, torch.Tensor], learning_rate: float = 0.01, 
                 max_iterations: int = 10, tolerance: float = 1e-6,
                 reduction: str = 'mean', eps: float = 1e-8):
        super().__init__(model, sigma_x, sigma_y, reduction, eps)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate the ODR loss by optimizing latent true x values.
        
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
        
        # Prepare covariance matrices
        sigma_x_tensor, sigma_y_tensor = self._prepare_covariances(n_features_x, n_features_y, device)
        
        # Prepare inverse covariance matrices for Mahalanobis distance
        sigma_x_inv, sigma_y_inv = self._prepare_inverse_covariances(
            sigma_x_tensor, sigma_y_tensor, n_features_x, n_features_y, device)
        
        # Initialize latent true x as observed x with gradient tracking enabled
        x_latent = x_obs.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_latent], lr=self.learning_rate)
        
        # Optimize latent true x values
        prev_loss = float('inf')
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Forward pass with current latent x
            y_pred = self._apply_model_with_mask(x_latent, mask)
            
            # Calculate x distance (between observed and latent x)
            x_diff = x_obs - x_latent
            x_dist = self._calculate_mahalanobis_distance(x_diff, sigma_x_inv)
            
            # Calculate y distance (between observed y and predicted y)
            y_diff = y_true - y_pred
            y_dist = self._calculate_mahalanobis_distance(y_diff, sigma_y_inv)
            
            # Total ODR objective: minimize weighted sum of distances
            total_dist = x_dist + y_dist
            odr_objective = torch.mean(total_dist)
            
            # Backward pass and update
            odr_objective.backward()
            optimizer.step()
            
            # Check for convergence
            if abs(prev_loss - odr_objective.item()) < self.tolerance:
                break
                
            prev_loss = odr_objective.item()
        
        # Final forward pass with optimized latent x (detached to avoid gradient tracking)
        x_latent_final = x_latent.detach()
        y_pred_final = self._apply_model_with_mask(x_latent_final, mask)
            
        # Calculate final orthogonal distances
        x_diff_final = x_obs - x_latent_final
        final_x_dist = self._calculate_mahalanobis_distance(x_diff_final, sigma_x_inv)
        
        y_diff_final = y_true - y_pred_final
        final_y_dist = self._calculate_mahalanobis_distance(y_diff_final, sigma_y_inv)
            
        # Total loss is the weighted sum of squared orthogonal distances
        loss = final_x_dist + final_y_dist
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss


class EnsembleEIVLoss(BaseEIVLoss):
    """
    Simple Ensemble Errors-in-Variables Loss.
    
    This loss implements a straightforward approach to handling uncertainty in inputs
    by generating multiple perturbed versions, running the model on each, and
    averaging the predictions before calculating the loss.
    
    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise (scalar, vector, or matrix)
        n_samples: Number of perturbed samples to generate
        perturb_method: Method for perturbing inputs ('gaussian', 'uniform')
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability
    """
    def __init__(self, model: Callable, sigma_x: Union[float, torch.Tensor], 
                 n_samples: int = 20, perturb_method: str = 'gaussian',
                 reduction: str = 'mean', eps: float = 1e-8):
        super().__init__(model, sigma_x, None, reduction, eps)
        self.n_samples = n_samples
        self.perturb_method = perturb_method
        
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate Ensemble EIV loss.
        
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
        device = x_obs.device
        
        # Prepare noise parameters
        sigma_x_tensor = prepare_covariance(self.sigma_x, n_features_x, device)
        
        # Generate perturbed samples
        perturbed_samples = generate_perturbed_samples(
            x_obs, sigma_x_tensor, self.n_samples, perturb_method=self.perturb_method
        )
            
        # Stack perturbed samples and reshape for batch processing
        x_perturbed = torch.stack(perturbed_samples)  # [n_samples, batch_size, n_features_x]
        x_flat = x_perturbed.reshape(-1, n_features_x)  # [n_samples * batch_size, n_features_x]
        
        # Forward pass for all samples
        with torch.no_grad():
            y_preds_flat = self.model(x_flat)
            
            # Get output feature dimension
            n_features_y = y_preds_flat.shape[1] if y_preds_flat.dim() > 1 else 1
            
            # Reshape predictions 
            if y_preds_flat.dim() == 1:
                # Handle scalar output case
                y_preds = y_preds_flat.reshape(self.n_samples, batch_size, 1)
            else:
                y_preds = y_preds_flat.reshape(self.n_samples, batch_size, n_features_y)
                
            # Apply mask if provided
            if mask is not None:
                mask_expanded = mask.unsqueeze(0).expand(self.n_samples, -1, -1)
                y_preds = torch.where(mask_expanded, y_preds, torch.zeros_like(y_preds))
        
        # Average predictions across samples
        mean_pred = torch.mean(y_preds, dim=0)  # [batch_size, n_features_y]
        
        # Calculate loss between averaged prediction and target
        loss = torch.sum((mean_pred - y_true)**2, dim=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss


# Aliases for backward compatibility
GeneralErrorInVariablesLoss = FunctionalEIVLoss
CorrelatedEIVLoss = StructuralEIVLoss