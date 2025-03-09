import torch
from typing import Callable, Optional, Union
from .base import MaskedLoss
from .eiv_utils import prepare_sigma, prepare_covariance

class ChamferEIVLoss(MaskedLoss):
    """
    Chamfer distance-based Error-in-Variables loss.
    
    This approach finds the closest point on the model manifold to each observation,
    which can provide unbiased estimates especially for highly nonlinear models.
    
    Args:
        model: The model function f(x) that predicts y
        method: Method for finding closest point ('monte_carlo', 'optimization')
        n_samples: Number of Monte Carlo samples (for 'monte_carlo' method)
        optim_steps: Number of optimization steps (for 'optimization' method)
        optim_lr: Learning rate for optimization (for 'optimization' method)
        sigma_x: Standard deviation of feature noise (for sampling)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(
        self,
        model: Callable,
        method: str = 'monte_carlo',
        n_samples: int = 100,
        optim_steps: int = 50,
        optim_lr: float = 0.01,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.model = model
        self.method = method
        self.n_samples = n_samples
        self.optim_steps = optim_steps
        self.optim_lr = optim_lr
        self.sigma_x = sigma_x if sigma_x is not None else 1.0
        self.reduction = reduction
        
        if method not in ('monte_carlo', 'optimization'):
            raise ValueError(f"Invalid method: {method}. Must be 'monte_carlo' or 'optimization'")
        
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction: {reduction}")
    
    def forward(self, x_obs, y_true, mask=None):
        """
        Calculate the Chamfer EIV loss.
        
        Args:
            x_obs: Observed features with noise. Shape: (batch_size, n_features_x)
            y_true: Observed targets with noise. Shape: (batch_size, n_features_y)
            mask: Optional mask. Shape: (batch_size, n_features_y)
            
        Returns:
            loss: The Chamfer EIV loss
        """
        y_true = self._apply_mask(y_true, mask)
        batch_size, n_features_y = y_true.shape
        n_features_x = x_obs.shape[1]
        device = x_obs.device
        
        if self.method == 'monte_carlo':
            # Monte Carlo approach: generate samples around each observation
            # and find the closest one on the model manifold
            losses = self._monte_carlo_chamfer(x_obs, y_true, mask)
        else:  # 'optimization'
            # Optimization approach: directly optimize for the closest point
            # on the model manifold for each observation
            losses = self._optimization_chamfer(x_obs, y_true, mask)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(losses)
        elif self.reduction == 'sum':
            return torch.sum(losses)
        else:  # 'none'
            return losses
    
    def _monte_carlo_chamfer(self, x_obs, y_true, mask=None):
        """Monte Carlo approach to Chamfer distance."""
        batch_size, n_features_x = x_obs.shape
        device = x_obs.device
        
        # Prepare sigma_x as covariance matrix if it's not already
        if isinstance(self.sigma_x, (float, int)) or (isinstance(self.sigma_x, torch.Tensor) and self.sigma_x.ndim == 0):
            sigma_x = (self.sigma_x**2) * torch.eye(n_features_x, device=device)
        elif isinstance(self.sigma_x, torch.Tensor) and self.sigma_x.ndim == 1:
            sigma_x = torch.diag(self.sigma_x**2).to(device)
        else:
            sigma_x = self.sigma_x.to(device)
        
        # Initialize loss tensor
        losses = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            # Generate Monte Carlo samples around the observed x
            if sigma_x.ndim == 2:  # Full covariance matrix
                mvn = torch.distributions.MultivariateNormal(x_obs[i], sigma_x)
                x_samples = mvn.sample((self.n_samples,))
            else:  # Batch of covariance matrices
                mvn = torch.distributions.MultivariateNormal(x_obs[i], sigma_x[i])
                x_samples = mvn.sample((self.n_samples,))
            
            # Predict y for all samples
            with torch.no_grad():
                y_samples = self.model(x_samples)
            
            # Calculate distances for each sample
            x_dists = torch.sum((x_samples - x_obs[i].unsqueeze(0))**2, dim=1)
            y_dists = torch.sum((y_samples - y_true[i].unsqueeze(0))**2, dim=1)
            total_dists = x_dists + y_dists
            
            # Find the minimum distance
            losses[i] = torch.min(total_dists)
        
        return losses
    
    def _optimization_chamfer(self, x_obs, y_true, mask=None):
        """Optimization approach to Chamfer distance."""
        batch_size = x_obs.shape[0]
        device = x_obs.device
        
        # Initialize loss tensor
        losses = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            # Initialize x_prime as the observed x
            x_prime = x_obs[i].clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([x_prime], lr=self.optim_lr)
            
            for _ in range(self.optim_steps):
                # Clear gradients
                optimizer.zero_grad()
                
                # Predict y_prime
                y_prime = self.model(x_prime.unsqueeze(0)).squeeze(0)
                
                # Calculate Chamfer loss for this point
                x_dist = torch.sum((x_prime - x_obs[i])**2)
                y_dist = torch.sum((y_prime - y_true[i])**2)
                point_loss = x_dist + y_dist
                
                # Backward and optimize
                point_loss.backward()
                optimizer.step()
            
            # Final prediction with optimized x_prime
            with torch.no_grad():
                y_prime = self.model(x_prime.unsqueeze(0)).squeeze(0)
                x_dist = torch.sum((x_prime - x_obs[i])**2)
                y_dist = torch.sum((y_prime - y_true[i])**2)
                losses[i] = x_dist + y_dist
        
        return losses


class HybridEIVChamferLoss(MaskedLoss):
    """
    Hybrid loss combining statistical EIV and geometric Chamfer approaches.
    
    This loss combines the statistical rigor of EIV approaches with the geometric 
    intuition of Chamfer distance, which can be particularly useful for handling 
    highly nonlinear models where Jensen's inequality becomes problematic.
    
    Args:
        eiv_loss: An instance of an EIV loss (CorrelatedEIVLoss, GeneralizedMLEIVLoss, etc.)
        chamfer_loss: An instance of ChamferEIVLoss
        alpha: Weight for the EIV loss component (1-alpha is weight for Chamfer)
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(
        self,
        eiv_loss: MaskedLoss,
        chamfer_loss: ChamferEIVLoss,
        alpha: float = 0.5,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.eiv_loss = eiv_loss
        self.chamfer_loss = chamfer_loss
        self.alpha = alpha
        self.reduction = reduction
        
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction: {reduction}")
    
    def forward(self, x_obs, y_true, **kwargs):
        """
        Calculate the hybrid EIV-Chamfer loss.
        
        Args:
            x_obs: Observed features with noise. Shape: (batch_size, n_features_x)
            y_true: Observed targets with noise. Shape: (batch_size, n_features_y)
            **kwargs: Additional arguments passed to the component losses
            
        Returns:
            loss: The hybrid loss
        """
        # Calculate EIV loss - this might use y_pred, weights, etc. from kwargs
        eiv_loss_val = self.eiv_loss(x_obs, y_true, **kwargs)
        
        # Calculate Chamfer loss - this typically only needs x_obs, y_true and mask
        chamfer_loss_val = self.chamfer_loss(x_obs, y_true, kwargs.get('mask', None))
        
        # Combine losses with weighting factor
        combined_loss = self.alpha * eiv_loss_val + (1 - self.alpha) * chamfer_loss_val
        
        return combined_loss
    
    def set_alpha(self, alpha: float):
        """
        Adjust the weighting between EIV and Chamfer components.
        
        Args:
            alpha: New weight for EIV component (0-1)
        """
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha
