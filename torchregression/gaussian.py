import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.distributions import Normal, MultivariateNormal

from .base import MaskedLoss

class WeightedMSELoss(MaskedLoss):
    """
    Weighted Mean Squared Error Loss.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
                         Default: 'mean'
    """
    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)

    def forward(self, y_true, y_pred, weights, mask=None):
        """
        Calculate weighted MSE loss.
        
        Args:
            y_true (tensor): Target values (batch_size, n_features)
            y_pred (tensor): Predicted values (batch_size, n_features)
            weights (tensor): Weights for each feature (batch_size, n_features)
            mask (tensor, optional): Optional mask (batch_size, n_features)
            
        Returns:
            tensor: The loss value
        """
        # Apply mask to inputs
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        weights = self._apply_mask(weights, mask)
        
        # Calculate weighted MSE
        mse = F.mse_loss(y_true, y_pred, reduction='none')
        weighted_mse = mse * weights
        
        return self._reduce(weighted_mse, mask)

class DistributionBasedLoss(MaskedLoss):
    """Base class for losses based on torch.distributions."""
    
    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        
    def get_distribution(self, y_pred, **kwargs):
        """Return a torch.distributions object based on parameters."""
        raise NotImplementedError
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate negative log likelihood loss.
        
        Args:
            y_true (tensor): Target values (batch_size, n_features)
            y_pred (tensor): Predicted distribution parameters
            mask (tensor, optional): Optional mask (batch_size, n_features)
            weights (tensor, optional): Optional sample weights
            
        Returns:
            tensor: The loss value
        """
        # Apply mask to inputs
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Get distribution object
        distribution = self.get_distribution(y_pred)
        
        # Calculate negative log likelihood
        nll = -distribution.log_prob(y_true)
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            nll = nll * weights
            
        return self._reduce(nll, mask)


class DiagonalGaussianNLL(DistributionBasedLoss):
    """
    Gaussian NLL with diagonal covariance matrix (independent features).
    
    Args:
        n_features (int): Number of features in the input.
        reduction (str): Specifies the reduction to apply to the output.
    """
    def __init__(self, n_features, reduction='mean'):
        super().__init__(reduction=reduction)
        self.log_variances = nn.Parameter(torch.zeros(n_features))
        self.n_features = n_features

    def get_distribution(self, y_pred):
        batch_size = y_pred.shape[0]
        variances = torch.exp(self.log_variances).unsqueeze(0).expand(batch_size, -1)
        return Normal(y_pred, torch.sqrt(variances))
    
    def forward(self, y_true, y_pred, mask=None):
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        distribution = self.get_distribution(y_pred)
        
        # Sum log_prob across features (for independent dimensions)
        nll = -distribution.log_prob(y_true).sum(dim=-1)
        
        return self._reduce(nll, mask)


class GaussianNLLWithCovariance(DistributionBasedLoss):
    """
    Gaussian NLL loss with a provided covariance matrix.
    
    Args:
        regularization_strength (float): Small value added to diagonal for numerical stability
        reduction (str): Specifies the reduction to apply to the output
    """
    def __init__(self, regularization_strength=1e-6, reduction='mean'):
        super().__init__(reduction=reduction)
        self.regularization_strength = regularization_strength

    def get_distribution(self, y_pred, covariance_matrices=None):
        if covariance_matrices is None:
            raise ValueError("covariance_matrices must be provided")
            
        batch_size = y_pred.shape[0]
        device = y_pred.device
        n_features = y_pred.shape[1]
        
        # Add regularization for numerical stability
        reg_matrix = self.regularization_strength * torch.eye(n_features, device=device).unsqueeze(0)
        cov_matrices_reg = covariance_matrices + reg_matrix
        
        return MultivariateNormal(y_pred, cov_matrices_reg)
    
    def forward(self, y_true, y_pred, covariance_matrices, mask=None, weights=None):
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        distribution = self.get_distribution(y_pred, covariance_matrices)
        nll = -distribution.log_prob(y_true)
        
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            nll = nll * weights
            
        return self._reduce(nll, mask)


class AdjustedGaussianNLL(DistributionBasedLoss):
    """
    Gaussian NLL with a base covariance matrix and learned per-feature variance adjustments.
    
    Args:
        n_features (int): Number of features in the input.
        regularization_strength (float): Small value added to diagonal for numerical stability.
        reduction (str): Specifies the reduction to apply to the output.
    """
    def __init__(self, n_features, regularization_strength=1e-6, reduction='mean'):
        super().__init__(reduction=reduction)
        self.n_features = n_features
        self.regularization_strength = regularization_strength
        self.log_variance_adjustment = nn.Parameter(torch.zeros(n_features))

    def get_distribution(self, y_pred, covariance_matrix=None):
        if covariance_matrix is None:
            raise ValueError("covariance_matrix must be provided")
            
        batch_size = y_pred.shape[0]
        device = y_pred.device
        
        # Ensure covariance matrix is expanded if needed
        if covariance_matrix.ndim == 2:
            covariance_matrix = covariance_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply variance adjustment
        variance_adjustment = torch.exp(self.log_variance_adjustment).expand(batch_size, -1)
        adjusted_covariance_matrix = covariance_matrix + torch.diag_embed(variance_adjustment)
        
        # Add regularization
        reg_matrix = self.regularization_strength * torch.eye(self.n_features, device=device).unsqueeze(0)
        cov_matrices_reg = adjusted_covariance_matrix + reg_matrix
        
        return MultivariateNormal(y_pred, cov_matrices_reg)

    def forward(self, y_true, y_pred, covariance_matrix, mask=None):
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        distribution = self.get_distribution(y_pred, covariance_matrix)
        nll = -distribution.log_prob(y_true)
        
        return self._reduce(nll, mask)


class HeteroscedasticGaussianNLL(nn.Module):
    """
    Heteroscedastic Gaussian NLL with two components: a full covariance part and a diagonal part.
    
    Args:
        n_features_cov (int): Number of features in the covariance component
        n_features_diag (int): Number of features in the diagonal component
        use_fixed_diag_for_other (bool): Whether to use fixed diagonal for other component
        regularization_strength (float): Small value added to diagonal for numerical stability
    """
    def __init__(self, n_features_cov, n_features_diag, use_fixed_diag_for_other=False, regularization_strength=1e-6):
        super().__init__()        
        self.n_features_cov = n_features_cov
        self.n_features_diag = n_features_diag
        self.use_fixed_diag_for_other = use_fixed_diag_for_other
        self.regularization_strength = regularization_strength

        # Using our new distribution-based classes
        self.cov_loss = AdjustedGaussianNLL(n_features_cov, regularization_strength)
        if use_fixed_diag_for_other:
            self.diag_loss = WeightedMSELoss()
        else:
            self.diag_loss = DiagonalGaussianNLL(n_features_diag)

    def forward(self, x_cov, x_cov_reconstructed, covariance_matrices, mask_cov=None,
                x_diag=None, x_diag_reconstructed=None, mask_diag=None, other_variances=None):
        """
        Calculate heteroscedastic Gaussian NLL loss.
        
        Args:
            x_cov (torch.Tensor): Target values for covariance component
            x_cov_reconstructed (torch.Tensor): Predicted values for covariance component
            covariance_matrices (torch.Tensor): Covariance matrices for covariance component
            mask_cov (torch.Tensor, optional): Mask for covariance component
            x_diag (torch.Tensor, optional): Target values for diagonal component
            x_diag_reconstructed (torch.Tensor, optional): Predicted values for diagonal component
            mask_diag (torch.Tensor, optional): Mask for diagonal component
            other_variances (torch.Tensor, optional): Variances for diagonal component
                                                     (required if use_fixed_diag_for_other=True)
            
        Returns:
            torch.Tensor: The loss value
        """
        # Validate inputs
        if self.use_fixed_diag_for_other and other_variances is None:
            raise ValueError("other_variances must be provided when use_fixed_diag_for_other=True")
        
        if x_diag is None or x_diag_reconstructed is None:
            raise ValueError("x_diag and x_diag_reconstructed must be provided")

        cov_loss = self.cov_loss(x_cov, x_cov_reconstructed, covariance_matrices, mask_cov)

        if self.use_fixed_diag_for_other:  # When diagonal component is diagonal
            diag_loss = self.diag_loss(x_diag, x_diag_reconstructed, 1.0/other_variances, mask_diag)
        else:
            diag_loss = self.diag_loss(x_diag, x_diag_reconstructed, mask_diag)

        combined_loss = cov_loss + diag_loss
        return combined_loss


class LearnedGaussianNLL(DistributionBasedLoss):
    """
    Gaussian NLL with a *fully learned* covariance matrix.
    
    Args:
        n_features: The number of features.
        init_covariance: (Optional) Initial covariance matrix
        regularization_strength: Strength of regularization
        reduction (str): Specifies the reduction to apply to the output
    """
    def __init__(self, n_features, init_covariance=None, regularization_strength=1e-6, reduction='mean'):
        super().__init__(reduction=reduction)
        self.n_features = n_features
        self.regularization_strength = regularization_strength
        
        # Initialize covariance/cholesky parameters
        if init_covariance is None:
            init_covariance = torch.eye(n_features)
        else:
            if not isinstance(init_covariance, torch.Tensor):
                init_covariance = torch.tensor(init_covariance, dtype=torch.float)
            
            # Make symmetric and add jitter
            init_covariance = 0.5 * (init_covariance + init_covariance.T)
            jitter = 1e-5 * torch.eye(n_features, device=init_covariance.device)
            init_covariance = init_covariance + jitter
        
        # Get initial Cholesky factor safely
        try:
            chol_factor = torch.linalg.cholesky(init_covariance)
        except RuntimeError:
            # Fallback using eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(init_covariance)
            eigenvalues = torch.clamp(eigenvalues, min=1e-5)
            init_covariance = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
            chol_factor = torch.linalg.cholesky(init_covariance)
            
        # Learn the Cholesky factor
        self.cholesky_factor = nn.Parameter(chol_factor)
    
    def get_distribution(self, y_pred):
        batch_size = y_pred.shape[0]
        device = y_pred.device
        
        # Ensure lower triangular with positive diagonal
        L = torch.tril(self.cholesky_factor)
        diag_indices = torch.arange(self.n_features, device=device)
        diag_values = L[diag_indices, diag_indices]
        L = L.clone()
        L[diag_indices, diag_indices] = torch.exp(diag_values)
        
        # Construct covariance matrix and expand to batch
        covariance_matrix = L @ L.transpose(-1, -2)
        covariance_matrix = covariance_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add regularization
        reg_matrix = self.regularization_strength * torch.eye(self.n_features, device=device).unsqueeze(0)
        covariance_matrix = covariance_matrix + reg_matrix
        
        return MultivariateNormal(y_pred, covariance_matrix)
    
    def forward(self, y_true, y_pred, mask=None):
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        distribution = self.get_distribution(y_pred)
        nll = -distribution.log_prob(y_true)
        
        return self._reduce(nll, mask)


class LowRankGaussianNLL(DistributionBasedLoss):
    """
    Gaussian NLL with a low-rank plus diagonal covariance matrix.
    
    Args:
        n_features: The number of features
        rank: The rank of the low-rank approximation
        init_scale: Initial scale for factors and variances
        min_variance: Minimum allowed variance
        reduction (str): Specifies the reduction to apply to the output
    """
    def __init__(self, n_features, rank, init_scale=0.1, min_variance=1e-5, reduction='mean'):
        super().__init__(reduction=reduction)
        self.n_features = n_features
        self.rank = rank
        self.min_variance = min_variance
        
        # Validate inputs
        if rank <= 0 or rank >= n_features:
            raise ValueError(f"Rank should be between 1 and {n_features-1}, got {rank}")
        if init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {init_scale}")
        
        # Initialize parameters
        self.U = nn.Parameter(torch.randn(n_features, rank) * init_scale / np.sqrt(rank))
        self.log_variances = nn.Parameter(torch.zeros(n_features))
    
    def get_distribution(self, y_pred):
        batch_size = y_pred.shape[0]
        device = y_pred.device
        
        # Construct covariance matrix: Σ ≈ U @ U.T + diag(v)
        variances = F.softplus(self.log_variances) + self.min_variance
        covariance_matrix = self.U @ self.U.T + torch.diag_embed(variances)
        covariance_matrix = covariance_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        return MultivariateNormal(y_pred, covariance_matrix)
    
    def forward(self, y_true, y_pred, mask=None):
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        distribution = self.get_distribution(y_pred)
        nll = -distribution.log_prob(y_true)
        
        return self._reduce(nll, mask)


# Factory functions for easy creation of loss objects
def create_gaussian_nll(n_features, covariance_type='diagonal', **kwargs):
    """
    Factory function to create an appropriate Gaussian NLL loss.
    
    Args:
        n_features (int): Number of features
        covariance_type (str): One of 'diagonal', 'full', 'low_rank'
        **kwargs: Additional arguments for the specific loss type
    
    Returns:
        An appropriate Gaussian NLL loss object
    """
    if covariance_type == 'diagonal':
        return DiagonalGaussianNLL(n_features=n_features, **kwargs)
    elif covariance_type == 'full':
        return LearnedGaussianNLL(n_features=n_features, **kwargs)
    elif covariance_type == 'low_rank':
        # Default to rank n_features//2 if not specified
        rank = kwargs.pop('rank', n_features // 2)
        return LowRankGaussianNLL(n_features=n_features, rank=rank, **kwargs)
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type}")