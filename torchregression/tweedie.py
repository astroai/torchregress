import torch
import torch.nn as nn
from torch.distributions import Distribution

from .base import MaskedLoss
from .poisson import DistributionBasedLoss


class TweedieDistribution(Distribution):
    """
    Tweedie distribution for torch.distributions.
    
    The Tweedie distribution is a flexible distribution that can model
    various types of data, including zero-inflated, positive continuous,
    and count data. It is parameterized by mean (mu), dispersion (phi), and
    a power parameter (p) which determines the specific distribution in the family.
    
    Args:
        mu (Tensor): Mean of the distribution (μ)
        phi (Tensor): Dispersion parameter (φ)
        p (float): Power parameter determining the distribution type
            - p=0: Gaussian distribution
            - p=1: Poisson distribution
            - p=2: Gamma distribution
            - 1<p<2: Compound Poisson-Gamma distribution
        eps (float): Small value for numerical stability
        validate_args (bool): Whether to validate input arguments
    """
    def __init__(self, mu, phi=1.0, p=1.5, eps=1e-8, validate_args=None):
        self.mu = mu
        self.phi = phi
        self.p = p
        self.eps = eps
        batch_shape = self.mu.shape
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)
        
    def log_prob(self, value):
        """Calculate log probability of the Tweedie distribution."""
        safe_mu = torch.clamp(self.mu, min=self.eps)
        
        if self.p == 0:  # Gaussian
            return -0.5 * ((value - self.mu)**2 / self.phi + torch.log(2 * torch.pi * self.phi))
        elif self.p == 1:  # Poisson
            safe_value = torch.clamp(value, min=0)
            return safe_value * torch.log(safe_mu) - safe_mu - torch.lgamma(safe_value + 1)
        elif self.p == 2:  # Gamma
            safe_value = torch.clamp(value, min=self.eps)
            return -torch.log(safe_value) - safe_value / safe_mu - torch.log(safe_mu)
        elif 1 < self.p < 2:  # Compound Poisson-Gamma
            safe_value = torch.clamp(value, min=0)
            
            # Tweedie deviance function
            if torch.is_tensor(value) and value.dim() > 0:
                deviance = torch.where(
                    value == 0,
                    safe_mu**(2-self.p) / ((1-self.p)*(2-self.p)),
                    safe_value**(2-self.p) / ((1-self.p)*(2-self.p)) - 
                    (safe_value * safe_mu**(1-self.p)) / (1-self.p) + 
                    safe_mu**(2-self.p) / (2-self.p)
                )
            else:
                if value == 0:
                    deviance = safe_mu**(2-self.p) / ((1-self.p)*(2-self.p))
                else:
                    safe_value = torch.clamp(torch.tensor(value), min=self.eps)
                    deviance = safe_value**(2-self.p) / ((1-self.p)*(2-self-p)) - \
                              (safe_value * safe_mu**(1-self.p)) / (1-self.p) + \
                              safe_mu**(2-self.p) / (2-self.p)
            
            # Log probability is proportional to negative deviance
            log_prob = -deviance / self.phi
            
            # This is an approximation - the full form would include normalizing constants
            return log_prob
        else:
            raise ValueError(f"Unsupported power parameter p={self.p}")


class TweedieNLL(DistributionBasedLoss):
    """
    Tweedie Negative Log-Likelihood Loss.
    
    The Tweedie distribution is a flexible distribution that can model
    various types of data, including zero-inflated, positive continuous,
    and count data. The Tweedie loss is parameterized by a power parameter
    'p', which determines the shape of the distribution.
    
    Args:
        p (float): Tweedie power parameter.
            - p=0: Gaussian distribution
            - p=1: Poisson distribution
            - p=2: Gamma distribution
            - 1<p<2: Compound Poisson-Gamma distribution
        eps (float): Small value for numerical stability. Default: 1e-8
        learn_phi (bool): Whether to learn the dispersion parameter. Default: False
        reduction (str): Specifies the reduction to apply to the output. 
                         One of 'none', 'mean', 'sum'. Default: 'mean'
                         
    Shape:
        - y_true: (N, *) where * means any number of additional dimensions
        - y_pred: (N, *) same shape as y_true
        - mask: (N, *) optional mask tensor with same shape as inputs
        - weights: (N, *) optional weights tensor with same shape as inputs
        - Output: scalar unless reduction is 'none', then (N, *)
                
    Example::
        >>> loss = TweedieNLL(p=1.5)
        >>> y_true = torch.tensor([1.0, 2.0, 3.0])
        >>> y_pred = torch.tensor([1.5, 2.2, 2.8])
        >>> loss(y_true, y_pred)

    References:
        https://en.wikipedia.org/wiki/Tweedie_distribution
    """
    def __init__(self, p=1.5, eps=1e-8, learn_phi=False, reduction='mean'):
        super().__init__(reduction=reduction)
        if not ((1 < p < 2) or (p==0) or (p==1) or (p==2)):
            raise ValueError("p must be in the range (1, 2), equal to 0, 1 or 2")
        self.p = p
        self.eps = eps
        self.learn_phi = learn_phi
        
        if self.learn_phi:
            self.log_phi = nn.Parameter(torch.tensor(0.0))  # Initialize phi to 1
            
    def get_distribution(self, y_pred, phi=None):
        """Return a TweedieDistribution based on parameters."""
        if self.learn_phi:
            phi = torch.exp(self.log_phi)
        elif phi is not None:
            if isinstance(phi, (float, int)):
                phi = torch.tensor(phi, device=y_pred.device, dtype=y_pred.dtype)
            elif isinstance(phi, torch.Tensor) and phi.device != y_pred.device:
                phi = phi.to(y_pred.device)
        else:
            phi = torch.ones_like(y_pred)  # Default phi is 1
            
        return TweedieDistribution(y_pred, phi, p=self.p, eps=self.eps)
    
    def forward(self, y_true, y_pred, mask=None, weights=None, phi=None):
        """
        Calculate Tweedie NLL loss.
        
        Args:
            y_true (torch.Tensor): Ground truth values
            y_pred (torch.Tensor): Predicted mean values
            mask (torch.Tensor, optional): Optional mask
            weights (torch.Tensor, optional): Optional weights
            phi (float or torch.Tensor): Dispersion parameter (not used if learn_phi=True)
            
        Returns:
            torch.Tensor: Tweedie NLL loss
        """
        # Apply mask to inputs
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Get distribution object
        distribution = self.get_distribution(y_pred, phi)
        
        # Calculate negative log likelihood
        nll = -distribution.log_prob(y_true)
        
        if self.learn_phi:
            # Add regularization when learning phi
            nll = nll + 0.5 * self.log_phi
            
        return self._reduce(nll, mask, weights)


def create_tweedie_loss(p=1.5, learn_phi=False, eps=1e-8, reduction='mean'):
    """
    Factory function to create a Tweedie loss.
    
    Args:
        p (float): Tweedie power parameter.
            - p=0: Gaussian distribution
            - p=1: Poisson distribution
            - p=2: Gamma distribution
            - 1<p<2: Compound Poisson-Gamma distribution
        learn_phi (bool): Whether to learn the dispersion parameter. Default: False
        eps (float): Small value for numerical stability. Default: 1e-8
        reduction (str): Specifies the reduction to apply to the output. 
                         One of 'none', 'mean', 'sum'. Default: 'mean'
        
    Returns:
        TweedieNLL: Configured Tweedie negative log-likelihood loss
    """
    return TweedieNLL(p=p, eps=eps, learn_phi=learn_phi, reduction=reduction)