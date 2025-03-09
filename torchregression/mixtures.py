import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MixtureSameFamily, Categorical, StudentT, Distribution
from torch.distributions.utils import broadcast_all

from typing import List, Dict, Tuple

from .poisson import PoissonNLL
from .base import MaskedLoss

class DistributionModule:
    """Base class for distribution modules in a mixture model."""
    
    def __init__(self, eps=1e-6):
        self.eps = eps
    
    def get_params_size(self, n_features: int) -> int:
        """Return number of parameters needed per component."""
        raise NotImplementedError
    
    def transform_params(self, params: torch.Tensor, batch_size: int, 
                        n_components: int, n_features: int) -> Dict[str, torch.Tensor]:
        """Transform raw parameters into distribution parameters."""
        raise NotImplementedError
    
    def create_component_distribution(self, params: Dict[str, torch.Tensor]) -> Distribution:
        """Create the PyTorch distribution using transformed parameters."""
        raise NotImplementedError
    
    def get_distribution(self, params: torch.Tensor, batch_size: int, 
                        n_components: int, n_features: int) -> Tuple:
        """
        Process network output into distribution parameters and return them.
        Each distribution module will return its specific parameters.
        """
        transformed_params = self.transform_params(params, batch_size, n_components, n_features)
        return tuple(transformed_params.values())
    
    def create_mixture_distribution(self, params: torch.Tensor, mixture_weights: torch.Tensor,
                                  batch_size: int, n_components: int, n_features: int) -> Distribution:
        """
        Create a mixture distribution from parameters and weights.
        
        Args:
            params: Tensor of distribution parameters
            mixture_weights: Tensor of shape (batch_size, n_components)
            batch_size: Batch size
            n_components: Number of mixture components
            n_features: Number of features
            
        Returns:
            A MixtureSameFamily distribution
        """
        transformed_params = self.transform_params(params, batch_size, n_components, n_features)
        component_distribution = self.create_component_distribution(transformed_params)
        categorical = Categorical(probs=mixture_weights)
        return MixtureSameFamily(categorical, component_distribution)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self.eps})"

class GaussianModule(DistributionModule):
    """
    Gaussian (Normal) distribution module.
    
    Args:
        eps: Small value for numerical stability
        initial_scale: Optional initial scale values
            - If None: scale = exp(log_scale_params)
            - If float: scale = initial_scale * exp(log_scale_params)
            - If tensor: Will be broadcast according to its shape
    """
    
    def __init__(self, eps=1e-6, initial_scale=None):
        super().__init__(eps=eps)
        self.initial_scale = initial_scale
    
    def get_params_size(self, n_features: int) -> int:
        # Mean and log scale for each feature
        return 2 * n_features
    
    def transform_params(self, params: torch.Tensor, batch_size: int, 
                        n_components: int, n_features: int) -> Dict[str, torch.Tensor]:
        """Transform raw parameters into loc and scale."""
        # Extract location parameters
        loc = params[..., :n_features * n_components].reshape(batch_size, n_components, n_features)
        
        # Extract and transform scale parameters
        log_scale_offset = params[..., n_features * n_components:2 * n_features * n_components]
        log_scale_offset = log_scale_offset.reshape(batch_size, n_components, n_features)
        
        # Apply initial scale if provided
        scale = self._get_scale(log_scale_offset, batch_size, n_components, n_features, params.device)
        
        return {'loc': loc, 'scale': scale}
    
    def _get_scale(self, log_scale_offset, batch_size, n_components, n_features, device):
        """Process scale parameters with optional initial_scale."""
        if self.initial_scale is not None:
            if isinstance(self.initial_scale, torch.Tensor):
                initial_scale = self.initial_scale.to(device)
                
                if initial_scale.ndim == 0:  # Scalar tensor
                    scale = float(initial_scale) * torch.exp(log_scale_offset)
                elif initial_scale.ndim == 1:  # Per-feature
                    if initial_scale.size(0) != n_features:
                        raise ValueError(f"initial_scale must have size {n_features}, got {initial_scale.size(0)}")
                    initial_scale = initial_scale.view(1, 1, n_features)
                    scale = initial_scale * torch.exp(log_scale_offset)
                elif initial_scale.ndim == 2:  # Per-component, per-feature
                    if initial_scale.size() != (n_components, n_features):
                        raise ValueError(f"initial_scale must have size ({n_components}, {n_features}), got {initial_scale.size()}")
                    initial_scale = initial_scale.unsqueeze(0)
                    scale = initial_scale * torch.exp(log_scale_offset)
                elif initial_scale.ndim == 3:  # Batch-specific
                    if initial_scale.size() != (batch_size, n_components, n_features):
                        raise ValueError(f"initial_scale must have size ({batch_size}, {n_components}, {n_features}), got {initial_scale.size()}")
                    scale = initial_scale * torch.exp(log_scale_offset)
                else:
                    raise ValueError(f"initial_scale must have 0-3 dimensions, got {initial_scale.ndim}")
            else:  # Scalar
                scale = self.initial_scale * torch.exp(log_scale_offset)
        else:
            scale = torch.exp(log_scale_offset)
        
        # Ensure numerical stability
        return scale + self.eps
    
    def create_component_distribution(self, params: Dict[str, torch.Tensor]) -> Distribution:
        """Create a Normal distribution from parameters."""
        return Normal(params['loc'], params['scale'])
    
    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self.eps}, initial_scale={self.initial_scale})"


class StudentTModule(DistributionModule):
    """
    Student's t-distribution module for heavier tails.
    
    Args:
        eps: Small value for numerical stability
        initial_scale: Optional initial scale values (similar to GaussianModule)
        min_df: Minimum degrees of freedom (>2 ensures finite variance)
        fixed_df: Optional fixed degrees of freedom
    """
    
    def __init__(self, eps=1e-6, initial_scale=None, min_df=2.1, fixed_df=None):
        super().__init__(eps=eps)
        self.initial_scale = initial_scale
        self.min_df = min_df
        self.fixed_df = fixed_df
    
    def get_params_size(self, n_features: int) -> int:
        # Location and scale for each feature, plus df (unless fixed)
        return 2 * n_features + (0 if self.fixed_df is not None else 1)
    
    def transform_params(self, params: torch.Tensor, batch_size: int, 
                        n_components: int, n_features: int) -> Dict[str, torch.Tensor]:
        """Transform raw parameters into loc, scale and df."""
        # Extract location parameters
        loc = params[..., :n_features * n_components].reshape(batch_size, n_components, n_features)
        
        # Extract and transform scale parameters (similar to Gaussian)
        log_scale_offset = params[..., n_features * n_components:2 * n_features * n_components]
        log_scale_offset = log_scale_offset.reshape(batch_size, n_components, n_features)
        
        # Apply initial scale if provided
        scale = self._get_scale(log_scale_offset, batch_size, n_components, n_features, params.device)
        
        # Process degrees of freedom
        df = self._get_df(params, batch_size, n_components, n_features, params.device)
        
        return {'loc': loc, 'scale': scale, 'df': df}
    
    def _get_scale(self, log_scale_offset, batch_size, n_components, n_features, device):
        """Process scale parameters with optional initial_scale."""
        # Reusing the same logic as GaussianModule
        if self.initial_scale is not None:
            if isinstance(self.initial_scale, torch.Tensor):
                initial_scale = self.initial_scale.to(device)
                
                if initial_scale.ndim == 0:  # Scalar tensor
                    scale = float(initial_scale) * torch.exp(log_scale_offset)
                elif initial_scale.ndim == 1:  # Per-feature
                    if initial_scale.size(0) != n_features:
                        raise ValueError(f"initial_scale must have size {n_features}, got {initial_scale.size(0)}")
                    initial_scale = initial_scale.view(1, 1, n_features)
                    scale = initial_scale * torch.exp(log_scale_offset)
                elif initial_scale.ndim == 2:  # Per-component, per-feature
                    if initial_scale.size() != (n_components, n_features):
                        raise ValueError(f"initial_scale must have size ({n_components}, {n_features}), got {initial_scale.size()}")
                    initial_scale = initial_scale.unsqueeze(0)
                    scale = initial_scale * torch.exp(log_scale_offset)
                elif initial_scale.ndim == 3:  # Batch-specific
                    if initial_scale.size() != (batch_size, n_components, n_features):
                        raise ValueError(f"initial_scale must have size ({batch_size}, {n_components}, {n_features}), got {initial_scale.size()}")
                    scale = initial_scale * torch.exp(log_scale_offset)
                else:
                    raise ValueError(f"initial_scale must have 0-3 dimensions, got {initial_scale.ndim}")
            else:  # Scalar
                scale = self.initial_scale * torch.exp(log_scale_offset)
        else:
            scale = torch.exp(log_scale_offset)
        
        # Ensure numerical stability
        return scale + self.eps
    
    def _get_df(self, params, batch_size, n_components, n_features, device):
        """Process degrees of freedom parameter."""
        if self.fixed_df is not None:
            # Use fixed df if provided
            if isinstance(self.fixed_df, torch.Tensor):
                df = self.fixed_df.to(device)
                if df.ndim == 0:  # Scalar
                    df = df.expand(batch_size, n_components, n_features)
                elif df.ndim == 1:  # Per component
                    if df.size(0) != n_components:
                        raise ValueError(f"fixed_df must have size {n_components}, got {df.size(0)}")
                    df = df.view(1, n_components, 1).expand(batch_size, -1, n_features)
                else:
                    raise ValueError(f"fixed_df must be scalar or 1D tensor, got {df.ndim}D")
            else:  # Scalar case
                df = torch.full((batch_size, n_components, n_features), 
                               self.fixed_df, device=device)
        else:
            # Get df from parameters
            df_offset = params[..., 2 * n_features * n_components:2 * n_features * n_components + n_components]
            df = self.min_df + F.softplus(df_offset).view(batch_size, n_components, 1).expand(-1, -1, n_features)
        
        return df
    
    def create_component_distribution(self, params: Dict[str, torch.Tensor]) -> Distribution:
        """Create a StudentT distribution from parameters."""
        # StudentT in PyTorch doesn't natively support batched parameters the same way
        # as Normal, so we'll need to use a custom approach for the mixture
        return CustomBatchedStudentT(params['df'], params['loc'], params['scale'])
    
    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self.eps}, initial_scale={self.initial_scale}, min_df={self.min_df}, fixed_df={self.fixed_df})"


class CustomBatchedStudentT(Distribution):
    """
    Custom implementation of StudentT that correctly handles batched parameters
    for use in MixtureSameFamily.
    """
    arg_constraints = {}
    
    def __init__(self, df, loc, scale):
        self.df, self.loc, self.scale = broadcast_all(df, loc, scale)
        batch_shape = self.loc.shape[:-1]
        event_shape = self.loc.shape[-1:]
        super().__init__(batch_shape, event_shape)
    
    def log_prob(self, value):
        # Reshape value for broadcasting correctly with parameters
        if value.dim() < len(self.batch_shape) + len(self.event_shape):
            value = value.expand(self.batch_shape + self.event_shape)
        
        # Calculate StudentT log probability
        n = self.df
        k = value.size(-1)  # dimensionality
        
        t = torch.lgamma((n + k) / 2) - torch.lgamma(n / 2) - (k/2) * torch.log(n * torch.pi)
        t = t - torch.sum(torch.log(self.scale), dim=-1)
        t = t - ((n + k) / 2) * torch.log(1 + torch.sum(
            ((value - self.loc) / self.scale) ** 2, dim=-1) / n)
        return t
    
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        # Generate samples using reparameterization trick
        z = torch.randn(shape, device=self.loc.device)
        u = torch.rand(shape[:-1], device=self.loc.device)
        v = self.df * torch.pow(u, -2.0/self.df)
        samples = self.loc + self.scale * z * torch.sqrt(v.unsqueeze(-1))
        return samples


# Registry for easy access to distribution modules
DISTRIBUTION_REGISTRY = {
    'gaussian': GaussianModule,
    'normal': GaussianModule,  # Alias
    'student-t': StudentTModule,
    'student_t': StudentTModule,  # Alias with underscore
}

def get_distribution_module(distribution_name: str, **kwargs) -> DistributionModule:
    """
    Factory function to get a distribution module by name.
    
    Args:
        distribution_name: Name of the distribution (case-insensitive)
        **kwargs: Parameters to pass to the distribution module constructor
        
    Returns:
        An instance of the requested distribution module
    """
    dist_name = distribution_name.lower()
    if dist_name not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution_name}. "
                        f"Available distributions: {list(DISTRIBUTION_REGISTRY.keys())}")
    
    return DISTRIBUTION_REGISTRY[dist_name](**kwargs)


class GaussianPoissonMixtureNLL(MaskedLoss):
    """
    Negative log-likelihood loss for a mixture of Gaussian (readout noise)
    and Poisson (count) noise, common in imaging.
    
    Args:
        eps (float): Small constant for numerical stability
        learn_gaussian_variance (bool): Whether to learn the Gaussian variance
        gaussian_variance (float): Initial variance of the Gaussian component
        log_input (bool): Whether y_pred is provided as log(lambda)
        reduction (str): Specifies the reduction to apply to the output
    """
    def __init__(self, eps=1e-8, learn_gaussian_variance=False, gaussian_variance=1.0, log_input=False, reduction='mean'):
        super().__init__(reduction=reduction)
        self.eps = eps
        self.learn_gaussian_variance = learn_gaussian_variance
        self.log_input = log_input

        if self.learn_gaussian_variance:
            self.log_gaussian_variance = nn.Parameter(torch.tensor(np.log(gaussian_variance)))
        else:
            self.register_buffer('fixed_gaussian_variance', torch.tensor(gaussian_variance, dtype=torch.float32))

        self.poisson_nll = PoissonNLL(eps=self.eps, log_input=self.log_input)

    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculates the Gaussian-Poisson mixture NLL.
        
        Args:
            y_true: Ground truth values (batch_size, n_features).
            y_pred: Predicted Poisson rate (lambda) (batch_size, n_features). 
                    If log_input=True this is log(lambda).
            mask: (Optional) Mask (batch_size, n_features).
            weights: (Optional) Sample weights (batch_size, 1) or (batch_size, n_features)
            
        Returns:
            loss: The mixture NLL loss (scalar).
        """
        # Apply mask to inputs
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)

        # Handle weights
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            if weights.ndim == 1:
                weights = weights.unsqueeze(-1)  # Make weights (batch_size, 1)
        else:
            weights = 1.0

        # Poisson component
        poisson_loss = self.poisson_nll(y_true, y_pred, mask=None)  # Mask already applied

        # Convert predictions to linear space if needed
        if self.log_input:
            y_pred_linear = torch.exp(y_pred)
        else:
            y_pred_linear = y_pred

        # Calculate Gaussian variance with numerical stability
        if self.learn_gaussian_variance:
            gaussian_variance = torch.exp(self.log_gaussian_variance) + self.eps
        else:
            gaussian_variance = self.fixed_gaussian_variance.to(y_true.device) + self.eps

        # Gaussian component - using standard normal NLL formula
        gaussian_loss = 0.5 * (
            (y_true - y_pred_linear)**2 / gaussian_variance + 
            torch.log(gaussian_variance) + 
            torch.log(torch.tensor(2 * torch.pi, device=y_true.device))
        )

        # Combine losses
        loss = poisson_loss + gaussian_loss
        loss = loss * weights

        return self._reduce(loss, mask)


class MixtureDensityNetworkLoss(MaskedLoss):
    """
    Mixture Density Network (MDN) loss for regression.
    Calculates the negative log-likelihood of targets under a mixture of distributions.

    Args:
        num_components (int): Number of mixture components.
        n_features (int): Dimensionality of y_true.
        distribution (str or DistributionModule): Distribution type or module instance.
        distribution_params (dict, optional): Parameters for the distribution module.
        eps (float): Small constant for numerical stability. Default: 1e-6.
    """

    def __init__(self, num_components, n_features, distribution='gaussian', 
                    distribution_params=None, eps=1e-6):
        super().__init__()
        self.num_components = num_components
        self.n_features = n_features
        self.eps = eps
        
        # Handle distribution parameter dictionary
        distribution_params = distribution_params or {}
        
        # Initialize distribution module based on type
        if isinstance(distribution, str):
            self.dist_module = get_distribution_module(distribution, eps=eps, **distribution_params)
        elif isinstance(distribution, DistributionModule):
            self.dist_module = distribution
        else:
            raise TypeError("distribution must be a string or DistributionModule instance")
        
        # Calculate total parameters needed
        self.dist_params_size = self.dist_module.get_params_size(n_features)
        self.total_params_size = self.dist_params_size * num_components + num_components

    def forward(self, y_true, y_pred, mask=None):
        """
        Calculate MDN loss.
        
        Args:
            y_true: Target values of shape (batch_size, n_features)
            y_pred: Predicted distribution parameters
            mask: Optional mask for masked loss calculation
            
        Returns:
            Negative log-likelihood loss
        """
        y_true = self._apply_mask(y_true, mask)
        batch_size, _ = y_true.shape
        device = y_true.device
        
        # Validate input shape
        expected_size = self.total_params_size
        if y_pred.size(-1) != expected_size:
            raise ValueError(f"Expected y_pred to have {expected_size} features, got {y_pred.size(-1)}")
            
        # Extract distribution parameters
        dist_params = y_pred[..., :-self.num_components]
        
        # Extract and process mixture weights
        logits = y_pred[..., -self.num_components:]
        mix_weights = F.softmax(logits, dim=-1)
        
        # Create mixture distribution
        mixture = self.dist_module.create_mixture_distribution(
            dist_params, mix_weights, batch_size, self.num_components, self.n_features)
            
        # Calculate negative log likelihood
        log_prob = mixture.log_prob(y_true)
        loss = -log_prob
            
        return self._reduce(loss, mask)


class CombinedMDNFixedErrorLoss(MaskedLoss):
    """
    Combines a Mixture Density Network loss with a fixed-error Gaussian loss.

    This loss models the target variable as a sum of two components:
        1. A mixture of distributions (learned by the MDN).
        2. A single Gaussian with a fixed, pre-specified covariance matrix.

    Args:
        num_components (int): Number of mixture components for the MDN.
        n_features (int): Dimensionality of y_true.
        fixed_covariance (torch.Tensor or str): The fixed covariance matrix or specification.
            - If tensor shape (n_features, n_features): Full covariance matrix
            - If tensor shape (n_features,): Diagonal covariance matrix
            - If 'eye': Identity matrix
            - If 'spherical': Spherical variance computed from data
        fixed_error_weight (float): Weight for the fixed-error component.
        distribution (str or DistributionModule): Distribution type or module instance.
        distribution_params (dict, optional): Parameters for the distribution module.
        eps (float): Small constant for numerical stability. Default: 1e-6.
    """

    def __init__(self, num_components, n_features, fixed_covariance,
                    fixed_error_weight=1.0, distribution='gaussian', 
                    distribution_params=None, eps=1e-6):
        super().__init__()
        self.num_components = num_components
        self.n_features = n_features
        self.fixed_error_weight = fixed_error_weight
        self.eps = eps
        self.is_spherical = isinstance(fixed_covariance, str) and fixed_covariance.lower() == 'spherical'
        self.device = None  # Will be set when data is seen

        # Initialize the base MDN loss with distribution parameters
        distribution_params = distribution_params or {}
        self.mdn_loss = MixtureDensityNetworkLoss(
            num_components, n_features, distribution=distribution,
            distribution_params=distribution_params, eps=eps)
        
        # Process fixed covariance specification
        self._setup_fixed_covariance(fixed_covariance, n_features)

    def _setup_fixed_covariance(self, fixed_covariance, n_features):
        """Set up the fixed covariance matrix."""
        if isinstance(fixed_covariance, str):
            if fixed_covariance.lower() == 'eye':
                self.fixed_cov_inv = torch.eye(n_features)
                self.is_str = False
            elif fixed_covariance.lower() == 'spherical':
                self.fixed_cov_inv = 1.0  # Placeholder, will be updated with set_spherical_variance
                self.is_str = True
            else:
                raise ValueError(f"Invalid covariance specification: {fixed_covariance}. "
                                "Use 'eye', 'spherical', or a tensor.")
        elif isinstance(fixed_covariance, torch.Tensor):
            if fixed_covariance.ndim == 1:
                # Diagonal case
                if fixed_covariance.shape[0] != n_features:
                    raise ValueError(f"Fixed covariance must have shape ({n_features},) "
                                    f"or ({n_features}, {n_features}), got {fixed_covariance.shape}")
                self.fixed_cov_inv = torch.diag(1.0 / (fixed_covariance + self.eps))
                self.is_str = False
                self.device = fixed_covariance.device  # Record the device
            elif fixed_covariance.ndim == 2:
                # Full matrix case
                if fixed_covariance.shape != (n_features, n_features):
                    raise ValueError(f"Fixed covariance must have shape ({n_features}, {n_features}) "
                                    f"or ({n_features},), got {fixed_covariance.shape}")
                try:
                    jitter = self.eps * torch.eye(n_features, device=fixed_covariance.device)
                    self.fixed_cov_inv = torch.linalg.inv(fixed_covariance + jitter)
                except torch.linalg.LinAlgError:
                    # Use pseudo-inverse for numerical stability
                    self.fixed_cov_inv = torch.linalg.pinv(fixed_covariance + jitter)
                self.is_str = False
                self.device = fixed_covariance.device  # Record the device
            else:
                raise ValueError(f"Fixed covariance must be 1D or 2D tensor, got {fixed_covariance.ndim}D")
        else:
            raise TypeError(f"Fixed covariance must be string or tensor, got {type(fixed_covariance)}")

    def set_spherical_variance(self, data):
        """
        Set the spherical variance from data.
        
        Args:
            data: Tensor of shape (n_samples, n_features) or a function that yields batches
        """
        if not self.is_spherical:
            raise ValueError("This method should only be called when fixed_covariance='spherical'")
        
        if callable(data):
            # If data is a function that yields batches (e.g., DataLoader)
            all_data = []
            for batch in data:
                if isinstance(batch, (tuple, list)):
                    # Assuming the target is the second element in the batch
                    all_data.append(batch[1])
                else:
                    all_data.append(batch)
            data = torch.cat(all_data, dim=0)
            
        # Compute variance across all samples
        variance = torch.var(data, dim=0, unbiased=False)
        
        # Create diagonal precision matrix (inverse covariance)
        self.fixed_cov_inv = torch.diag(1.0 / (variance + self.eps))
        self.device = data.device
        self.is_str = False  # No longer using a string representation
        return variance

    def forward(self, y_true, y_pred, mask=None):
        """
        Calculate combined MDN and fixed-error loss.

        Args:
            y_true: Target values of shape (batch_size, n_features)
            y_pred: Predicted distribution parameters
            mask: Optional mask for masked loss calculation
        
        Returns:
            Combined loss value
        """
        y_true = self._apply_mask(y_true, mask)
        batch_size, _ = y_true.shape
        device = y_true.device

        # For spherical covariance with automatic computation
        if self.is_str and self.is_spherical:
            # Compute variance across batch dimension
            variance = torch.var(y_true, dim=0, unbiased=False)
            # Create diagonal precision matrix (inverse covariance)
            self.fixed_cov_inv = torch.diag(1.0 / (variance + self.eps))
            self.device = device

        # Ensure fixed_cov_inv is on the correct device
        if isinstance(self.fixed_cov_inv, torch.Tensor) and self.fixed_cov_inv.device != device:
            self.fixed_cov_inv = self.fixed_cov_inv.to(device)

        # Calculate MDN loss using the base class
        mdn_loss = self.mdn_loss(y_true, y_pred, mask)
        
        # For fixed error component, extract means from MDN
        dist_params = y_pred[..., :-self.num_components]
        transformed_params = self.mdn_loss.dist_module.transform_params(
            dist_params, batch_size, self.num_components, self.n_features)
        
        # Use mean of component means as the prediction
        pred_mean = torch.mean(transformed_params['loc'], dim=1)  # shape: (batch_size, n_features)
        
        # Calculate quadratic form for fixed-error component: (y - μ)^T Σ^(-1) (y - μ)
        diff = y_true - pred_mean
        if isinstance(self.fixed_cov_inv, torch.Tensor) and self.fixed_cov_inv.dim() == 2:
            # Full matrix case
            fixed_error_loss = 0.5 * torch.sum(
                diff * torch.matmul(diff, self.fixed_cov_inv.transpose(-2, -1)), dim=1)
        else:
            # Diagonal case (simplified computation)
            fixed_error_loss = 0.5 * torch.sum(
                diff**2 * self.fixed_cov_inv.view(1, -1), dim=1)
        
        # Combine losses
        loss = mdn_loss + self.fixed_error_weight * fixed_error_loss
        
        return loss


class NMDRLoss(MaskedLoss):
    """
    Neural Mixture Distributional Regression (NMDR) loss based on Rugamer et al.
    
    This implements a mixture model with:
    1. Structured additive predictors for location parameters
    2. Support for different component distributions
    3. Improved regularization for mixture weights
    
    Args:
        num_components (int): Number of mixture components.
        n_features (int): Dimensionality of y_true.
        distribution (str or DistributionModule): Distribution type or module instance.
        distribution_params (dict, optional): Parameters for the distribution module.
        structured_bias (bool): Whether to include structured bias terms. Default: False.
        weight_concentration (float): Concentration parameter for Dirichlet prior on weights. Default: 1.0.
        eps (float): Small constant for numerical stability. Default: 1e-6.
    """
    
    def __init__(self, num_components, n_features, distribution='gaussian', 
                 distribution_params=None, structured_bias=False,
                 weight_concentration=1.0, eps=1e-6):
        super().__init__()
        self.num_components = num_components
        self.n_features = n_features
        self.eps = eps
        self.weight_concentration = weight_concentration
        self.structured_bias = structured_bias
        
        # Handle distribution parameter dictionary
        distribution_params = distribution_params or {}
        
        # Initialize distribution module based on type
        if isinstance(distribution, str):
            self.dist_module = get_distribution_module(distribution, eps=eps, **distribution_params)
        elif isinstance(distribution, DistributionModule):
            self.dist_module = distribution
        else:
            raise TypeError("distribution must be a string or DistributionModule instance")
        
        # Calculate total parameters needed
        self.dist_params_size = self.dist_module.get_params_size(n_features)
        self.total_params_size = self.dist_params_size * num_components + num_components
        
        # For structured bias if enabled
        if structured_bias:
            self.structured_bias_net = nn.Linear(n_features, n_features)
    
    def forward(self, y_true, y_pred, structured_input=None, mask=None):
        """
        Calculate NMDR loss with optional structured additive predictor.
        
        Args:
            y_true: Target values of shape (batch_size, n_features)
            y_pred: Predicted distribution parameters
            structured_input: Optional structured covariates for additive predictor
            mask: Optional mask for masked loss calculation
        
        Returns:
            NMDR loss value
        """
        y_true = self._apply_mask(y_true, mask)
        batch_size, _ = y_true.shape
        device = y_true.device
        
        # Validate input shape
        expected_size = self.total_params_size
        if y_pred.size(-1) != expected_size:
            raise ValueError(f"Expected y_pred to have {expected_size} features, got {y_pred.size(-1)}")
        
        # Check if structured_input is provided when required
        if self.structured_bias and structured_input is None:
            raise ValueError("structured_input is required when structured_bias is True")
            
        # Extract distribution parameters and mixture weights
        dist_params = y_pred[..., :-self.num_components]
        logits = y_pred[..., -self.num_components:]
        
        # Apply Dirichlet prior-inspired regularization to mixture weights
        # This is a simplified version using concentration parameters
        if self.weight_concentration != 1.0:
            # Add (concentration-1) to encourage more uniform weights when >1, or more sparse when <1
            logits = logits + (self.weight_concentration - 1.0)
            
        mix_weights = F.softmax(logits, dim=-1)
        
        # Transform parameters using the distribution module
        transformed_params = self.dist_module.transform_params(
            dist_params, batch_size, self.num_components, self.n_features)
        
        # Apply structured bias if enabled and input provided
        if self.structured_bias:
            structured_effect = self.structured_bias_net(structured_input).unsqueeze(1)
            if 'loc' in transformed_params:
                transformed_params['loc'] = transformed_params['loc'] + structured_effect
        
        # Create mixture distribution
        component_distribution = self.dist_module.create_component_distribution(transformed_params)
        categorical = Categorical(probs=mix_weights)
        mixture = MixtureSameFamily(categorical, component_distribution)
            
        # Calculate negative log likelihood
        log_prob = mixture.log_prob(y_true)
        loss = -log_prob
        
        return self._reduce(loss, mask)


class MDNEnsembleModel(nn.Module):
    """
    A model that outputs MDN parameters for mixture density network.
    This can be used as a component in a deep ensemble.
    """
    def __init__(self, 
                in_features: int, 
                out_features: int, 
                hidden_sizes: List[int] = [64, 64],
                activation: nn.Module = nn.ReLU(),
                num_components: int = 5,
                distribution: str = 'gaussian'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.num_components = num_components
        
        # Get distribution module to determine output size
        self.dist_module = get_distribution_module(distribution)
        self.dist_params_size = self.dist_module.get_params_size(out_features)
        self.total_params_size = self.dist_params_size * num_components + num_components
        
        self._build()
        
    def _build(self):
        layers = []
        layer_sizes = [self.in_features] + self.hidden_sizes
        
        # Create hidden layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(self.activation)
            
        self.hidden_net = nn.Sequential(*layers)
        
        # Output layer for MDN parameters
        self.output_layer = nn.Linear(self.hidden_sizes[-1], self.total_params_size)
        
    def forward(self, x):
        x = self.hidden_net(x)
        return self.output_layer(x)


class MDNEnsemble(nn.Module):
    """
    Specialized ensemble for Mixture Density Networks.
    
    This provides MDN-specific functionality for ensembling mixture density networks,
    including specialized prediction methods that correctly handle mixture parameters.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        ensemble_size: Number of models in the ensemble
        hidden_sizes: List of hidden layer sizes for each model
        activation: Activation function to use
        num_components: Number of mixture components for each MDN
        distribution: Distribution type ('gaussian', 'student-t', etc.)
    """
    def __init__(self, 
                in_features: int, 
                out_features: int, 
                ensemble_size: int = 5, 
                hidden_sizes: List[int] = [64, 64],
                activation: nn.Module = nn.ReLU(),
                num_components: int = 5,
                distribution: str = 'gaussian'):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.num_components = num_components
        self.distribution = distribution
        
        # Get distribution module
        self.dist_module = get_distribution_module(distribution)
        
        # Create ensemble of MDN models
        self.models = nn.ModuleList([
            MDNEnsembleModel(
                in_features, 
                out_features, 
                hidden_sizes=hidden_sizes,
                activation=activation, 
                num_components=num_components,
                distribution=distribution
            ) for _ in range(ensemble_size)
        ])
        
    def forward(self, x):
        """
        Forward pass through each model in the ensemble.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            List of raw MDN parameter outputs from each model
        """
        return [model(x) for model in self.models]
    
    def predict(self, x):
        """
        Make a prediction with the MDN ensemble, combining outputs properly.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Tuple containing:
            - mean: Mean prediction
            - variance: Predictive variance (aleatoric + epistemic)
        """
        with torch.no_grad():
            batch_size = x.shape[0]
            device = x.device
            
            # Get outputs from all models
            all_outputs = self(x)
            all_means = []
            all_variances = []
            
            # Process each model's output
            for output in all_outputs:
                # Extract distribution parameters
                dist_params = output[..., :-self.num_components]
                mixture_weights = F.softmax(output[..., -self.num_components:], dim=-1)
                
                # Transform parameters using distribution module
                transformed_params = self.dist_module.transform_params(
                    dist_params, batch_size, self.num_components, self.out_features)
                
                # Get component means
                component_means = transformed_params['loc']  # shape: [batch, components, features]
                
                # Calculate model mean (weighted average of component means)
                model_mean = torch.sum(mixture_weights.unsqueeze(-1) * component_means, dim=1)
                all_means.append(model_mean)
                
                # Get component scales and calculate variance
                component_vars = transformed_params['scale'] ** 2
                
                # Aleatoric uncertainty (weighted average of component variances)
                weighted_vars = torch.sum(mixture_weights.unsqueeze(-1) * component_vars, dim=1)
                
                # Epistemic uncertainty from mixture (variance of component means)
                mean_diffs = component_means - model_mean.unsqueeze(1)
                mixture_epistemic = torch.sum(mixture_weights.unsqueeze(-1) * (mean_diffs ** 2), dim=1)
                
                # Total model variance
                model_variance = weighted_vars + mixture_epistemic
                all_variances.append(model_variance)
            
            # Calculate ensemble mean and variance
            ensemble_mean = torch.stack(all_means).mean(dim=0)
            
            # Within-model variance (average of model variances)
            avg_model_var = torch.stack(all_variances).mean(dim=0)
            
            # Between-model variance (variance of model means)
            model_means_tensor = torch.stack(all_means)
            between_model_var = torch.var(model_means_tensor, dim=0, unbiased=False)
            
            # Total predictive variance
            predictive_variance = avg_model_var + between_model_var
            
            return ensemble_mean, predictive_variance
    
    def sample(self, x, num_samples=1):
        """
        Draw samples from the predictive distribution.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            num_samples: Number of samples to draw for each input
            
        Returns:
            Tensor of samples with shape [num_samples, batch_size, out_features]
        """
        with torch.no_grad():
            batch_size = x.shape[0]
            
            # Randomly select models for each sample
            model_indices = torch.randint(0, self.ensemble_size, (num_samples,))
            
            samples = []
            for i in range(num_samples):
                # Select a random model
                model = self.models[model_indices[i]]
                output = model(x)
                
                # Extract distribution parameters
                dist_params = output[..., :-self.num_components]
                mixture_weights = F.softmax(output[..., -self.num_components:], dim=-1)
                
                # Transform parameters using distribution module
                transformed_params = self.dist_module.transform_params(
                    dist_params, batch_size, self.num_components, self.out_features)
                
                # Create mixture distribution
                component_distribution = self.dist_module.create_component_distribution(transformed_params)
                categorical = Categorical(probs=mixture_weights)
                mixture = MixtureSameFamily(categorical, component_distribution)
                
                # Sample from the distribution
                samples.append(mixture.sample())
            
            return torch.stack(samples)


def mdn_loss(num_components, n_features, distribution='gaussian', **kwargs):
    """
    Creates a Mixture Density Network (MDN) loss function.
    
    Args:
        num_components: Number of mixture components
        n_features: Dimensionality of target variables
        distribution: Distribution type ('gaussian', 'student-t', etc.)
        **kwargs: Additional parameters for the loss function or distribution
    
    Returns:
        MixtureDensityNetworkLoss instance
    """
    distribution_params = {}
    if 'initial_scale' in kwargs:
        distribution_params['initial_scale'] = kwargs.pop('initial_scale')
    
    return MixtureDensityNetworkLoss(
        num_components=num_components,
        n_features=n_features,
        distribution=distribution,
        distribution_params=distribution_params,
        **kwargs
    )


def nmdr_loss(num_components, n_features, distribution='gaussian', 
                structured_bias=False, weight_concentration=1.0, **kwargs):
    """
    Creates a Neural Mixture Distributional Regression (NMDR) loss.
    
    Args:
        num_components: Number of mixture components
        n_features: Dimensionality of target variables
        distribution: Distribution type ('gaussian', 'student-t', etc.)
        structured_bias: Whether to include structured bias terms
        weight_concentration: Concentration parameter for mixture weights
        **kwargs: Additional parameters for the loss function or distribution
    
    Returns:
        NMDRLoss instance
    """
    distribution_params = {}
    if 'initial_scale' in kwargs:
        distribution_params['initial_scale'] = kwargs.pop('initial_scale')
    
    return NMDRLoss(
        num_components=num_components,
        n_features=n_features,
        distribution=distribution,
        distribution_params=distribution_params,
        structured_bias=structured_bias,
        weight_concentration=weight_concentration,
        **kwargs
    )


def combined_mdn_loss(num_components, n_features, fixed_covariance, 
                        fixed_error_weight=1.0, distribution='gaussian', **kwargs):
    """
    Creates a combined MDN with fixed-error loss function.
    
    Args:
        num_components: Number of mixture components
        n_features: Dimensionality of target variables
        fixed_covariance: Fixed covariance matrix or specification
        fixed_error_weight: Weight for the fixed error component
        distribution: Distribution type ('gaussian', 'student-t', etc.)
        **kwargs: Additional parameters for the loss function or distribution
    
    Returns:
        CombinedMDNFixedErrorLoss instance
    """
    distribution_params = {}
    if 'initial_scale' in kwargs:
        distribution_params['initial_scale'] = kwargs.pop('initial_scale')
    
    return CombinedMDNFixedErrorLoss(
        num_components=num_components,
        n_features=n_features,
        fixed_covariance=fixed_covariance,
        fixed_error_weight=fixed_error_weight,
        distribution=distribution,
        distribution_params=distribution_params,
        **kwargs
    )

def mdn_ensemble(in_features, out_features, ensemble_size=5, 
                num_components=5, hidden_sizes=None, distribution='gaussian', **kwargs):
    """
    Factory function to create an MDN ensemble model.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        ensemble_size: Number of models in the ensemble
        num_components: Number of mixture components
        hidden_sizes: List of hidden layer sizes (defaults to [64, 64])
        distribution: Distribution type ('gaussian', 'student-t', etc.)
        **kwargs: Additional arguments passed to MDNEnsemble constructor
        
    Returns:
        An MDNEnsemble instance
    """
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
        
    return MDNEnsemble(
        in_features=in_features,
        out_features=out_features,
        ensemble_size=ensemble_size,
        hidden_sizes=hidden_sizes,
        num_components=num_components,
        distribution=distribution,
        **kwargs
    )
