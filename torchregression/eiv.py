import torch

from typing import Callable, Optional, Union, Dict

from .base import MaskedLoss
from .eiv_utils import (
    prepare_param, prepare_covariance, prepare_cross_covariance,
    compute_model_gradients, calculate_gaussian_nll,
    prepare_model_input_for_gradients
)
from .eiv_chamfer import ChamferEIVLoss, HybridEIVChamferLoss
from .eiv_mdn import MDNEIVLoss
from .eiv_rfit import RobustEIVLoss, gaussian_variation, uniform_variation, bootstrap_variation

class TotalLeastSquaresLoss(MaskedLoss):
    """
    Total Least Squares (TLS) loss for *linear* regression, accounting for
    errors in both independent (x) and dependent (y) variables.

    Args:
        reduction (str): 'none' | 'mean' | 'sum'.  Default: 'mean'.
        use_svd (bool): use Singular Value Decomposition. Default: True
    """

    def __init__(self, reduction='mean', use_svd = True):
        super().__init__()
        self.reduction = reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.use_svd = use_svd

    def forward(self, x_true, y_true, y_pred, mask=None):
        """
        Calculates the Total Least Squares loss.

        Args:
            x_true: Observed independent variables. Shape: (batch_size, n_features_x)
            y_true: Observed dependent variables. Shape: (batch_size, n_features_y)
            y_pred: Predicted values for the *dependent* variables.  Shape: (batch_size, n_features_y)
                    This should be the output of a *linear* model:  y_pred = model(x_true)
            mask: (Optional) Mask. Shape: (batch_size, n_features_y) – Mask is applied to y only.

        Returns:
            loss: The TLS loss.
        """

        # Apply masking to y_true and y_pred (errors are in y)
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true and y_pred shapes must match. Got {y_true.shape} and {y_pred.shape}")
        if x_true.shape[0] != y_true.shape[0]:  # Check batch sizes
            raise ValueError("Batch sizes of x_true and y_true must match.")

        batch_size = x_true.shape[0]
        n_features_x = x_true.shape[1]
        n_features_y = y_true.shape[1]


        # Center the data
        x_true_centered = x_true - torch.mean(x_true, dim=0)
        y_true_centered = y_true - torch.mean(y_true, dim=0)
        y_pred_centered = y_pred - torch.mean(y_pred, dim=0) #center prediction also


        # Stack the *centered* x and *centered* y_true (NOT y_pred)
        combined = torch.cat((x_true_centered, y_true_centered), dim=-1)  # (batch_size, n_features_x + n_features_y)

        # --- Compute Loss ---
        if self.use_svd:
          # Singular Value Decomposition
          _, s, _ = torch.linalg.svd(combined)
          # Sum of the squares of the *smallest* singular values, corresponding to y_true.
          loss = torch.sum(s[:, -n_features_y:] ** 2, dim=-1)
        else:
            # Compute the covariance matrix
            covariance_matrix = torch.matmul(combined.transpose(0, 1), combined) # (n_features_x + n_features_y, n_features_x + n_features_y)
            # Eigenvalue decomposition
            eigenvalues, _ = torch.linalg.eig(covariance_matrix)  # Use linalg.eig for complex numbers
            # Sort eigenvalues (in case they are not sorted)
            eigenvalues = torch.real(eigenvalues)  # We only care about the real part
            eigenvalues, _ = torch.sort(eigenvalues, descending=False)  # Sort in ascending order
             # Sum of the smallest n_features eigenvalues
            loss = torch.sum(eigenvalues[:n_features_y], dim=-1)


        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        
class GeneralErrorInVariablesLoss(MaskedLoss):
    """
    General Error-in-Variables loss, assuming Gaussian errors.

    Handles both uncorrelated (diagonal covariance) and correlated (full covariance)
    errors for both independent (x) and dependent (y) variables.

    Args:
        model (Callable): The model function, that must provide derivative wrt to the parameters AND x.
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'.
        regularization_strength (float): Strength of regularization added to covariance matrices.
    """
    def __init__(self, model: Callable, reduction='mean', regularization_strength=1e-6):
        super().__init__()
        self.model = model
        self.reduction = reduction
        self.regularization_strength = regularization_strength
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction: {reduction}")

    def forward(self, x_true, y_true, x_error=None, y_error=None, mask=None):
        """
        Calculates the Error-in-Variables loss.

        Args:
            x_true: Observed independent variables (batch_size, n_features_x).
            y_true: Observed dependent variables (batch_size, n_features_y).
            x_error: Covariance matrix of errors in x_true.
                    Shape: (batch_size, n_features_x, n_features_x) or (n_features_x, n_features_x) or None
            y_error: Covariance matrix of errors in y_true.
                    Shape: (batch_size, n_features_y, n_features_y) or (n_features_y, n_features_y) or None
            mask: (Optional) Mask (batch_size, n_features_y)

        Returns:
            loss: The Error-in-Variables loss.
        """
        y_true = self._apply_mask(y_true, mask)  # Apply mask to y_true only
        y_pred = self.model(x_true)
        y_pred = self._apply_mask(y_pred, mask)

        batch_size = y_true.shape[0]
        n_features_x = x_true.shape[1]
        n_features_y = y_true.shape[1]
        device = y_true.device

        # Handle Covariance Matrices using utility function
        x_error_reg = None
        y_error_reg = None
        
        if x_error is not None:
            x_error_reg = prepare_covariance(x_error, n_features_x, device)
            if x_error_reg.ndim == 2:
                x_error_reg = x_error_reg.unsqueeze(0).expand(batch_size, -1, -1)
            x_error_reg = x_error_reg + self.regularization_strength * torch.eye(n_features_x, device=device).expand(batch_size, -1, -1)
            
        if y_error is not None:
            y_error_reg = prepare_covariance(y_error, n_features_y, device)
            if y_error_reg.ndim == 2:
                y_error_reg = y_error_reg.unsqueeze(0).expand(batch_size, -1, -1)
            y_error_reg = y_error_reg + self.regularization_strength * torch.eye(n_features_y, device=device).expand(batch_size, -1, -1)

        # Compute Jacobian
        jac = torch.autograd.functional.jacobian(self.model, x_true, create_graph=True) # (B, Ny, Nx)
        jac = torch.diagonal(jac, dim1=0, dim2=1).T #shape is (B, Nx, Ny)

        # Calculate the combined precision matrix
        precision_matrices = []
        for i in range(batch_size):
            j = jac[i]  # (n_features_x, n_features_y)
            sigma_x = x_error_reg[i] if x_error_reg is not None else torch.zeros((n_features_x, n_features_x), device=device)
            sigma_y = y_error_reg[i] if y_error_reg is not None else torch.zeros((n_features_y, n_features_y), device=device)
            sigma_total = sigma_y + j.T @ sigma_x @ j
            
            # Invert (using Cholesky for stability)
            L = torch.linalg.cholesky(sigma_total)
            precision_matrix = torch.cholesky_inverse(L)
            precision_matrices.append(precision_matrix)
        
        precision_matrices = torch.stack(precision_matrices)

        # Calculate the NLL using utility function
        diff = y_true - y_pred
        nll = calculate_gaussian_nll(diff, precision_matrices, is_precision=True)

        # Apply reduction
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll

class CorrelatedEIVLoss(MaskedLoss):
    """
    Generalized Error-in-Variables loss with correlated errors between features and labels.
    
    Mathematical formulation:
    L = 0.5 * (y - f(x) - μy + ∇f(x)^T μx)^T V^-1 (y - f(x) - μy + ∇f(x)^T μx) + 0.5*log|V|
    
    where V = σy^2 + ∇f(x)^T Σx ∇f(x) + 2∇f(x)^T Σxy
    
    Args:
        model: The model function f(x) that predicts y
        sigma_y: Variance of label noise (scalar or tensor)
        sigma_x: Covariance matrix of feature noise
        sigma_xy: Cross-covariance between feature and label noise
        mu_x: Mean/bias of feature errors (default: None)
        mu_y: Mean/bias of label errors (default: None)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small constant for numerical stability
        regularization: Optional dict with regularization strengths for 'mu_x', 'mu_y', 'sigma_xy'
    """
    def __init__(
        self,
        model: Callable,
        sigma_y: Union[float, torch.Tensor],
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        sigma_xy: Optional[Union[float, torch.Tensor]] = None,
        mu_x: Optional[Union[float, torch.Tensor]] = None,
        mu_y: Optional[Union[float, torch.Tensor]] = None,
        reduction: str = 'mean',
        eps: float = 1e-8,
        regularization: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.model = model
        self.sigma_y = sigma_y
        self.sigma_x = sigma_x if sigma_x is not None else 0.0
        self.sigma_xy = sigma_xy
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.eps = eps
        self.reduction = reduction
        self.regularization = regularization or {}
        
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction: {reduction}")
    
    def forward(self, x_obs, y_true, weights=None, mask=None):
        """
        Calculate the correlated EIV loss.
        
        Args:
            x_obs: Observed features with noise. Shape: (batch_size, n_features_x)
            y_true: Observed targets with noise. Shape: (batch_size, n_features_y)
            weights: Optional sample weights. Shape: (batch_size, n_features_y)
            mask: Optional mask. Shape: (batch_size, n_features_y)
            
        Returns:
            loss: The correlated EIV loss
        """
        y_true = self._apply_mask(y_true, mask)
        batch_size, n_features_y = y_true.shape
        n_features_x = x_obs.shape[1]
        device = x_obs.device
        
        # Convert parameters to appropriate tensors using utility functions
        sigma_y = prepare_param(self.sigma_y, n_features_y, device)
        sigma_x = prepare_covariance(self.sigma_x, n_features_x, device)
        sigma_xy = prepare_cross_covariance(self.sigma_xy, n_features_x, n_features_y, device)
        mu_x = prepare_param(self.mu_x, n_features_x, device, default_value=0.0)
        mu_y = prepare_param(self.mu_y, n_features_y, device, default_value=0.0)
        
        # Use utility function to prepare model input for gradients
        x_grad = prepare_model_input_for_gradients(x_obs)
        
        # Forward pass
        y_pred = self.model(x_grad)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Use utility function to compute model gradients
        grads = compute_model_gradients(y_pred, x_grad, n_features_y)
        
        # Compute the bias correction term: ∇f(x)^T μx
        bias_correction = torch.zeros(batch_size, n_features_y, device=device)
        if torch.any(mu_x != 0):
            for i in range(batch_size):
                for j in range(n_features_y):
                    bias_correction[i, j] = torch.dot(grads[i, j], mu_x)
        
        # Compute adjusted residuals: (y - f(x) - μy + ∇f(x)^T μx)
        adjusted_residuals = y_true - y_pred - mu_y + bias_correction
        
        # Compute adjusted variance for each data point and output dimension
        total_var = torch.zeros(batch_size, n_features_y, device=device)
        for i in range(batch_size):
            for j in range(n_features_y):
                # Feature noise propagation: ∇f(x)^T Σx ∇f(x)
                propagated_var = torch.matmul(
                    torch.matmul(grads[i, j], sigma_x), 
                    grads[i, j]
                )
                
                # Cross-covariance term: 2∇f(x)^T Σxy
                cross_cov_term = 0
                if sigma_xy is not None:
                    if sigma_xy.ndim == 2:
                        cross_cov_term = 2 * torch.matmul(grads[i, j], sigma_xy[:, j])
                    else:
                        cross_cov_term = 2 * torch.matmul(grads[i, j], sigma_xy[i, :, j])
                
                # Total variance: σy^2 + ∇f(x)^T Σx ∇f(x) + 2∇f(x)^T Σxy
                total_var[i, j] = sigma_y[j]**2 + propagated_var + cross_cov_term
        
        # Add epsilon for numerical stability
        total_var = total_var + self.eps
        
        # Compute the loss: 0.5 * [(y-f(x)-μy+∇f^Tμx)^2/V + log(V)]
        squared_error_term = (adjusted_residuals**2) / total_var
        log_term = torch.log(total_var)
        loss = 0.5 * (squared_error_term + log_term)
        
        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights
        
        # Add regularization terms if specified
        reg_loss = 0
        if self.regularization:
            if 'mu_x' in self.regularization and self.mu_x is not None:
                reg_loss += self.regularization['mu_x'] * torch.sum(mu_x**2)
            if 'mu_y' in self.regularization and self.mu_y is not None:
                reg_loss += self.regularization['mu_y'] * torch.sum(mu_y**2)
            if 'sigma_xy' in self.regularization and self.sigma_xy is not None:
                reg_loss += self.regularization['sigma_xy'] * torch.norm(sigma_xy, p='fro')**2
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss) + reg_loss
        elif self.reduction == 'sum':
            return torch.sum(loss) + reg_loss
        else:  # 'none'
            return loss + reg_loss/batch_size if reg_loss > 0 else loss


# Factory functions for easier creation of EIV losses
def create_eiv_loss(
    model: Callable,
    sigma_y: Union[float, torch.Tensor],
    sigma_x: Union[float, torch.Tensor],
    reduction: str = 'mean',
    eps: float = 1e-8,
    **kwargs
):
    """
    Factory function to create a Generalized MLE Error-in-Variables loss.
    
    Args:
        model: The model function f(x) that predicts y
        sigma_y: Standard deviation of noise in the labels
        sigma_x: Standard deviation of noise in the features
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small constant for numerical stability
        **kwargs: Additional arguments passed to the loss constructor
        
    Returns:
        An EIV loss instance
    """
    return GeneralizedMLEIVLoss(
        model=model,
        sigma_y=sigma_y,
        sigma_x=sigma_x,
        reduction=reduction,
        eps=eps,
        **kwargs
    )


def create_correlated_eiv_loss(
    model: Callable,
    sigma_y: Union[float, torch.Tensor],
    sigma_x: Optional[Union[float, torch.Tensor]] = None,
    sigma_xy: Optional[Union[float, torch.Tensor]] = None,
    mu_x: Optional[Union[float, torch.Tensor]] = None,
    mu_y: Optional[Union[float, torch.Tensor]] = None,
    reduction: str = 'mean',
    eps: float = 1e-8,
    regularization: Optional[Dict[str, float]] = None,
    **kwargs
):
    """
    Factory function to create a Correlated Error-in-Variables loss.
    
    Args:
        model: The model function f(x) that predicts y
        sigma_y: Standard deviation of noise in the labels
        sigma_x: Standard deviation of noise in the features
        sigma_xy: Cross-covariance between feature and label noise
        mu_x: Mean/bias of feature errors
        mu_y: Mean/bias of label errors
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small constant for numerical stability
        regularization: Optional dict with regularization strengths
        **kwargs: Additional arguments passed to the loss constructor
        
    Returns:
        A CorrelatedEIVLoss instance
    """
    return CorrelatedEIVLoss(
        model=model,
        sigma_y=sigma_y,
        sigma_x=sigma_x,
        sigma_xy=sigma_xy,
        mu_x=mu_x,
        mu_y=mu_y,
        reduction=reduction,
        eps=eps,
        regularization=regularization,
        **kwargs
    )


def create_chamfer_eiv_loss(
    model: Callable,
    method: str = 'monte_carlo',
    n_samples: int = 100,
    optim_steps: int = 50,
    optim_lr: float = 0.01,
    sigma_x: Optional[Union[float, torch.Tensor]] = None,
    reduction: str = 'mean',
    **kwargs
):
    """
    Factory function to create a Chamfer Error-in-Variables loss.
    
    Args:
        model: The model function f(x) that predicts y
        method: Method for finding closest point ('monte_carlo', 'optimization')
        n_samples: Number of Monte Carlo samples
        optim_steps: Number of optimization steps
        optim_lr: Learning rate for optimization
        sigma_x: Standard deviation of feature noise (for sampling)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        **kwargs: Additional arguments passed to the loss constructor
        
    Returns:
        A ChamferEIVLoss instance
    """
    return ChamferEIVLoss(
        model=model,
        method=method,
        n_samples=n_samples,
        optim_steps=optim_steps,
        optim_lr=optim_lr,
        sigma_x=sigma_x,
        reduction=reduction,
        **kwargs
    )


def create_hybrid_eiv_loss(
    eiv_loss: MaskedLoss,
    chamfer_loss: ChamferEIVLoss,
    alpha: float = 0.5,
    reduction: str = 'mean',
    **kwargs
):
    """
    Factory function to create a Hybrid EIV-Chamfer loss.
    
    Args:
        eiv_loss: An instance of an EIV loss
        chamfer_loss: An instance of ChamferEIVLoss
        alpha: Weight for the EIV component (0-1)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        **kwargs: Additional arguments passed to the loss constructor
        
    Returns:
        A HybridEIVChamferLoss instance
    """
    return HybridEIVChamferLoss(
        eiv_loss=eiv_loss,
        chamfer_loss=chamfer_loss,
        alpha=alpha,
        reduction=reduction,
        **kwargs
    )


def create_mdn_eiv_loss(
    num_components: int,
    n_features: int,
    sigma_x: Union[float, torch.Tensor],
    sigma_y: Optional[Union[float, torch.Tensor]] = None,
    reduction: str = 'mean',
    eps: float = 1e-8,
    **kwargs
):
    """
    Factory function to create an MDN Error-in-Variables loss.
    
    Args:
        num_components: Number of mixture components
        n_features: Dimensionality of target variables
        sigma_x: Standard deviation of noise in the features
        sigma_y: Standard deviation of noise in the labels (optional if included in MDN)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small constant for numerical stability
        **kwargs: Additional arguments passed to the loss constructor
        
    Returns:
        An MDN EIV loss instance
    """
    return MDNEIVLoss(
        num_components=num_components,
        n_features=n_features,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        reduction=reduction,
        eps=eps,
        **kwargs
    )


def create_robust_eiv_loss(
    model: Callable,
    base_loss: str = 'huber',
    delta: float = 1.0,
    variation_type: str = 'gaussian',
    sigma_x: Union[float, torch.Tensor] = 1.0,
    n_samples: int = 10,
    aggregation: str = 'median',
    reduction: str = 'mean',
    **kwargs
):
    """
    Factory function to create a Robust Error-in-Variables loss.
    
    Args:
        model: The model function f(x) that predicts y
        base_loss: Base loss function ('huber', 'l1', 'mse'). Default: 'huber'
        delta: Delta parameter for Huber loss. Default: 1.0
        variation_type: Type of variation ('gaussian', 'uniform', 'bootstrap'). Default: 'gaussian'
        sigma_x: Standard deviation for input variations. Default: 1.0
        n_samples: Number of samples to generate. Default: 10
        aggregation: Aggregation method ('mean', 'median', 'max', 'quantile'). Default: 'median'
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        **kwargs: Additional arguments passed to the loss constructor or variation function
    
    Returns:
        A RobustEIVLoss instance
    """
    # Select variation function based on type
    if variation_type.lower() == 'gaussian':
        variation_fn = gaussian_variation
    elif variation_type.lower() == 'uniform':
        variation_fn = uniform_variation
    elif variation_type.lower() == 'bootstrap':
        variation_fn = bootstrap_variation
    elif callable(variation_type):
        # Allow passing a custom variation function directly
        variation_fn = variation_type
    else:
        raise ValueError(f"Unknown variation type: {variation_type}")
    
    # Extract variation params from kwargs
    variation_params = kwargs.pop('variation_params', {})
    
    # Extract quantile parameter for quantile aggregation
    quantile = kwargs.pop('quantile', 0.95)
    
    return RobustEIVLoss(
        model=model,
        base_loss=base_loss,
        delta=delta,
        variation_fn=variation_fn,
        sigma_x=sigma_x,
        n_samples=n_samples,
        variation_params=variation_params,
        aggregation=aggregation,
        quantile=quantile,
        reduction=reduction,
        **kwargs
    )
