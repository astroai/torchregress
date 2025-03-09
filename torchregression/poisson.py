import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Poisson, NegativeBinomial, Distribution

from .base import MaskedLoss

class DistributionBasedLoss(MaskedLoss):
    """Base class for losses based on torch.distributions."""
    
    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        
    def get_distribution(self, y_pred, **kwargs):
        """Return a torch.distributions object based on parameters."""
        raise NotImplementedError
        
    def forward(self, y_true, y_pred, mask=None, weights=None, **kwargs):
        """
        Calculate negative log likelihood loss.
        
        Args:
            y_true (tensor): Target values (batch_size, n_features)
            y_pred (tensor): Predicted distribution parameters
            mask (tensor, optional): Optional mask (batch_size, n_features)
            weights (tensor, optional): Optional sample weights
            **kwargs: Additional parameters for get_distribution
            
        Returns:
            tensor: The loss value
        """
        # Apply mask to inputs
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Get distribution object
        distribution = self.get_distribution(y_pred, **kwargs)
        
        # Calculate negative log likelihood
        nll = -distribution.log_prob(y_true)
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            nll = nll * weights
            
        return self._reduce(nll, mask)


class PoissonNLL(DistributionBasedLoss):
    """
    Poisson Negative Log-Likelihood Loss, with optional weights and learned variance.
    
    Args:
        eps (float): Small constant for numerical stability. Default: 1e-8.
        learn_variance (bool): Whether to learn a single scaling factor for the variance. Default: False.
        log_input (bool): Whether the input is already in log space. Default: False.
            If True, y_pred is assumed to be log(lambda) instead of lambda.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
            
    References:
        - McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models.
    """
    def __init__(self, eps=1e-8, learn_variance=False, log_input=False, reduction='mean'):
        super().__init__(reduction=reduction)
        self.eps = eps
        self.learn_variance = learn_variance
        self.log_input = log_input
        if self.learn_variance:
            self.log_variance = nn.Parameter(torch.tensor(0.0))  # Initialize variance to 1

    def get_distribution(self, y_pred):
        if self.log_input:
            rate = torch.exp(y_pred)
        else:
            rate = y_pred
            
        # Add small epsilon for numerical stability
        return Poisson(rate + self.eps)
    
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Poisson NLL loss.
        
        Args:
            y_true (torch.Tensor): Ground truth counts
            y_pred (torch.Tensor): Predicted mean (lambda)
            mask (torch.Tensor, optional): Optional mask
            weights (torch.Tensor, optional): Optional weights
            
        Returns:
            torch.Tensor: Poisson NLL loss
        """
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        distribution = self.get_distribution(y_pred)
        nll = -distribution.log_prob(y_true)
        
        if self.learn_variance:
            variance = torch.exp(self.log_variance)
            nll = (nll / variance) + 0.5 * self.log_variance
        
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            if weights.ndim == 1:
                weights = weights.unsqueeze(-1)
            nll = nll * weights
            
        return self._reduce(nll, mask)


class ModifiedPoissonNLL(MaskedLoss):
    """
    Modified Poisson Negative Log-Likelihood Loss (Baker-Cousins Loss)
       with optional weights and learned variance.

    Args:
        eps (float): Small constant for numerical stability. Default: 1e-8
        learn_variance (bool): Whether to learn a single scaling factor for the variance. Default: False
        reduction (str): Specifies the reduction to apply to the output:
             'none' | 'mean' | 'sum'. Default: 'mean'

    References:
        - Baker, S., & Cousins, R. D. (1984).  Clarification of the use of
          chi-square and likelihood functions in fits to histograms.
          *Nuclear Instruments and Methods in Physics Research*, *221*(2), 437-442.
    """
    def __init__(self, eps=1e-8, learn_variance=False, reduction='mean'):
        super().__init__(reduction=reduction)
        self.eps = eps
        self.learn_variance = learn_variance
        if self.learn_variance:
             self.log_variance = nn.Parameter(torch.tensor(0.0))

    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Modified Poisson NLL loss.
        
        Args:
            y_true (torch.Tensor): Ground truth counts
            y_pred (torch.Tensor): Predicted mean (lambda)
            mask (torch.Tensor, optional): Optional mask
            weights (torch.Tensor, optional): Optional weights
            
        Returns:
            torch.Tensor: Modified Poisson NLL loss
        """
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)

        loss = (y_pred + self.eps) - y_true + y_true * (torch.log(y_true + self.eps) - torch.log(y_pred + self.eps))

        if self.learn_variance:
            variance = torch.exp(self.log_variance)
            loss = (loss / variance) + 0.5 * self.log_variance

        return self._reduce(loss, mask, weights)


class NegativeBinomialDistribution(Distribution):
    """
    Negative Binomial distribution parameterized by mean and dispersion.
    
    This is a wrapper around PyTorch's NegativeBinomial that uses a more intuitive
    parameterization with mean (mu) and dispersion parameter (theta).
    
    Args:
        mean (Tensor): Mean of the distribution (μ)
        theta (Tensor): Dispersion parameter (θ)
        validate_args (bool): Whether to validate input arguments
    """
    def __init__(self, mean, theta, validate_args=None):
        self.mean = mean
        self.theta = theta
        
        # Convert from (μ, θ) to PyTorch's (total_count, probs) parameterization
        # For NB: mean = r(1-p)/p, variance = r(1-p)/p²
        # With dispersion θ: variance = μ + μ²/θ
        # This gives: r = θ, p = θ/(θ + μ)
        total_count = theta
        probs = theta / (theta + mean)
        
        self.nb = NegativeBinomial(total_count=total_count, probs=probs, validate_args=validate_args)
        super().__init__(batch_shape=self.nb.batch_shape, validate_args=validate_args)
        
    def log_prob(self, value):
        return self.nb.log_prob(value)
        
    def sample(self, sample_shape=torch.Size([])):
        return self.nb.sample(sample_shape)


class NegativeBinomialNLL(DistributionBasedLoss):
    """
    Negative Binomial Negative Log-Likelihood Loss.

    The negative binomial distribution models count data with overdispersion
    (variance > mean), parameterized by mean μ and dispersion parameter θ.

    Args:
        eps (float): Small constant for numerical stability. Default: 1e-8.
        learn_theta (bool): Whether to learn the dispersion parameter (theta). Default: False.
                            If False, 'theta' must be provided in the forward pass.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
                            
    References:
        - Hilbe, J. M. (2011). Negative Binomial Regression. Cambridge University Press.
    """
    def __init__(self, eps=1e-8, learn_theta=False, reduction='mean'):
        super().__init__(reduction=reduction)
        self.eps = eps
        self.learn_theta = learn_theta
        if self.learn_theta:
            self.log_theta = nn.Parameter(torch.tensor(0.0))  # Learn log(theta)

    def get_distribution(self, y_pred, theta=None):
        if self.learn_theta:
            theta = torch.exp(self.log_theta)
        elif theta is not None:
            if isinstance(theta, (float, int)):
                theta = torch.tensor(theta, device=y_pred.device, dtype=y_pred.dtype)
            elif isinstance(theta, torch.Tensor) and theta.device != y_pred.device:
                theta = theta.to(y_pred.device)
        else:
            raise ValueError("theta must be provided if not learned")
        
        return NegativeBinomialDistribution(y_pred + self.eps, theta + self.eps)
    
    def forward(self, y_true, y_pred, mask=None, weights=None, theta=None):
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        distribution = self.get_distribution(y_pred, theta)
        nll = -distribution.log_prob(y_true)
        
        if self.learn_theta:
            # Add regularization when learning theta
            nll = nll + 0.5 * self.log_theta  
            
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            if weights.ndim == 1:
                weights = weights.unsqueeze(-1)
            nll = nll * weights
            
        return self._reduce(nll, mask)


class ZeroInflatedPoisson(Distribution):
    """
    Custom zero-inflated Poisson distribution for torch.distributions.
    
    Args:
        rate (tensor): The rate parameter for the Poisson component
        gate (tensor): The mixture probability (logits) for excess zeros
        validate_args (bool): Whether to validate input arguments
    """
    def __init__(self, rate, gate, validate_args=None):
        self.rate = rate
        self.gate = gate
        self.poisson = Poisson(rate, validate_args=validate_args)
        batch_shape = self.poisson.batch_shape
        super().__init__(batch_shape, validate_args=validate_args)
        
    def log_prob(self, value):
        # Calculate log probability for the zero-inflated model
        poisson_logp = self.poisson.log_prob(value)
        
        # Handle the zero inflation component
        is_zero = (value == 0)
        log_mix_prob = torch.where(
            is_zero,
            # For zeros: log(sigmoid(gate) + (1-sigmoid(gate)) * exp(poisson.log_prob(0)))
            torch.logaddexp(
                F.logsigmoid(self.gate),
                F.logsigmoid(-self.gate) + self.poisson.log_prob(torch.zeros_like(value))
            ),
            # For non-zeros: log((1-sigmoid(gate)) * exp(poisson.log_prob(value)))
            F.logsigmoid(-self.gate) + poisson_logp
        )
        
        return log_mix_prob
    
    def sample(self, sample_shape=torch.Size([])):
        # First sample whether each point comes from gate or poisson
        shape = sample_shape + self.batch_shape
        gate_sample = torch.distributions.Bernoulli(logits=self.gate).sample(sample_shape)
        poisson_sample = self.poisson.sample(sample_shape)
        
        # Zero inflation: where gate_sample is 1, return 0; otherwise return poisson sample
        samples = torch.where(gate_sample == 1, torch.zeros_like(poisson_sample), poisson_sample)
        return samples


class ZeroInflatedPoissonNLL(DistributionBasedLoss):
    """
    Zero-Inflated Poisson (ZIP) Negative Log-Likelihood Loss.
    
    Combines a Poisson distribution with a point mass at zero to model excess zeros
    in count data.

    Args:
        eps (float): Small constant for numerical stability. Default: 1e-8.
        learn_variance (bool): Whether to learn an overall variance scaling factor. Default: False.
        log_input (bool): Whether 'y_pred' is log(lambda). Default: False.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        
    References:
        - Lambert, D. (1992). Zero-inflated Poisson regression, with an application
          to defects in manufacturing. Technometrics, 34(1), 1-14.
    """
    def __init__(self, eps=1e-8, learn_variance=False, log_input=False, reduction='mean'):
        super().__init__(reduction=reduction)
        self.eps = eps
        self.learn_variance = learn_variance
        self.log_input = log_input
        
        if self.learn_variance:
            self.log_variance = nn.Parameter(torch.tensor(0.0))  # Initialize variance to 1

    def get_distribution(self, y_pred, pi_logits=None):
        if pi_logits is None:
            raise ValueError("pi_logits must be provided")
            
        if self.log_input:
            rate = torch.exp(y_pred)
        else:
            rate = y_pred
            
        # Add small epsilon for numerical stability
        return ZeroInflatedPoisson(rate + self.eps, pi_logits)
    
    def forward(self, y_true, y_pred, pi_logits, mask=None, weights=None):
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        pi_logits = self._apply_mask(pi_logits, mask)
        
        distribution = self.get_distribution(y_pred, pi_logits)
        nll = -distribution.log_prob(y_true)
        
        if self.learn_variance:
            variance = torch.exp(self.log_variance)
            nll = (nll / variance) + 0.5 * self.log_variance
        
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            if weights.ndim == 1:
                weights = weights.unsqueeze(-1)
            nll = nll * weights
            
        return self._reduce(nll, mask)


# Factory functions for easy creation of loss objects
def create_count_loss(loss_type='poisson', **kwargs):
    """
    Factory function to create an appropriate count regression loss.
    
    Args:
        loss_type (str): One of 'poisson', 'modified_poisson', 'negative_binomial',
                         'zero_inflated_poisson'
        **kwargs: Additional arguments for the specific loss type
    
    Returns:
        An appropriate count regression loss object
    """
    if loss_type.lower() == 'poisson':
        return PoissonNLL(**kwargs)
    elif loss_type.lower() == 'modified_poisson' or loss_type.lower() == 'baker_cousins':
        return ModifiedPoissonNLL(**kwargs)
    elif loss_type.lower() == 'negative_binomial' or loss_type.lower() == 'nb':
        return NegativeBinomialNLL(**kwargs)
    elif loss_type.lower() == 'zero_inflated_poisson' or loss_type.lower() == 'zip':
        return ZeroInflatedPoissonNLL(**kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")