"""
Mixture Density Network implementation for Errors-in-Variables regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Tuple, List, Callable

from ..losses.base import MaskedLoss
from ..utils.tensor_ops import prepare_sigma, prepare_param

class MDNEIVLoss(MaskedLoss):
    """
    Errors-in-Variables loss for Mixture Density Networks.
    
    This loss accounts for input uncertainty in MDN predictions by incorporating 
    feature noise into the mixture components. It's especially useful for 
    modeling multimodal output distributions in the presence of input noise.
    
    Args:
        num_components: Number of mixture components in the MDN
        n_features: Dimensionality of the target variable
        sigma_x: Standard deviation of noise in the features
        sigma_y: Standard deviation of noise in the labels (optional)
        min_sigma: Minimum value for standard deviation (for numerical stability)
        eps: Small constant for numerical stability
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(
        self,
        num_components: int,
        n_features: int,
        sigma_x: Union[float, torch.Tensor],
        sigma_y: Optional[Union[float, torch.Tensor]] = None,
        min_sigma: float = 1e-4,
        eps: float = 1e-8,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        self.num_components = num_components
        self.n_features = n_features
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.min_sigma = min_sigma
        self.eps = eps
        
        # Calculate param sizes for verification
        self.params_per_component = 2 * n_features + 1  # mean, sigma, and weight
        self.total_params = num_components * self.params_per_component
    
    def forward(self, x_obs, y_true, y_pred, mask=None):
        """
        Calculate MDN-EIV negative log-likelihood loss.
        
        Args:
            x_obs: Observed features with noise [batch_size, n_features_x]
            y_true: Observed targets [batch_size, n_features_y]
            y_pred: MDN parameters [batch_size, total_params]
            mask: Optional boolean mask [batch_size, n_features_y]
            
        Returns:
            Negative log-likelihood loss (scalar if reduction is applied)
        """
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Check input shapes
        batch_size = y_true.shape[0]
        device = y_true.device
        
        # Verify MDN output size
        if y_pred.size(-1) != self.total_params:
            raise ValueError(f"Expected output with {self.total_params} parameters, "
                           f"got {y_pred.size(-1)}")
        
        # Prepare sigma parameters
        sigma_x = prepare_sigma(self.sigma_x, x_obs.shape[1], device)
        sigma_y = prepare_sigma(self.sigma_y, self.n_features, device, default_zero=False)
        
        # Split MDN output into components
        mixture_params = self._extract_mixture_params(y_pred)
        logits = mixture_params['logits']
        means = mixture_params['means']
        log_sigmas = mixture_params['log_sigmas']
        
        # Ensure minimum sigma value for numerical stability
        sigmas = torch.exp(log_sigmas).clamp(min=self.min_sigma)
        
        # Calculate EIV-adjusted sigma for each component
        adjusted_sigmas = sigmas.clone()
        
        # For each component, propagate the input uncertainty
        for k in range(self.num_components):
            # Propagate input uncertainty using gradient
            # We approximate var(y) ≈ var(y|x) + (∂f/∂x)^T * var(x) * (∂f/∂x)
            
            # Since we don't have analytical derivatives, we'll use a simplified approach
            # that adds additional uncertainty to the output distribution
            if sigma_x is not None:
                # Simple diagonal approximation for propagated variance
                # This is a simplification and could be improved
                input_var_contribution = torch.sum(sigma_x**2) * torch.ones_like(adjusted_sigmas[:, k])
                adjusted_sigmas[:, k] = torch.sqrt(sigmas[:, k]**2 + input_var_contribution)
            
            # Add intrinsic noise in y, if specified
            if sigma_y is not None:
                adjusted_sigmas[:, k] = torch.sqrt(adjusted_sigmas[:, k]**2 + sigma_y**2)
        
        # Calculate mixture component log probabilities
        log_probs = torch.zeros(batch_size, self.num_components, device=device)
        for k in range(self.num_components):
            # Gaussian log probability
            # log(p) = -0.5 * (y - μ)^2 / σ^2 - 0.5 * log(2πσ^2)
            diff = y_true - means[:, k]
            var = adjusted_sigmas[:, k]**2
            
            # Sum across features for multivariate case with diagonal covariance
            log_prob_k = -0.5 * torch.sum(diff**2 / (var + self.eps), dim=1)
            log_prob_k -= 0.5 * torch.sum(torch.log(2 * torch.pi * var + self.eps), dim=1)
            log_probs[:, k] = log_prob_k
        
        # Calculate mixture log probability
        # log(p(y|x)) = log(∑_k π_k * p(y|x,k))
        log_pi = F.log_softmax(logits, dim=1)  # [batch_size, num_components]
        
        # Add log mixture weights to log probabilities
        log_probs += log_pi
        
        # Use the log-sum-exp trick for numeric stability
        max_log_probs = torch.max(log_probs, dim=1, keepdim=True)[0]
        log_probs_stable = log_probs - max_log_probs
        likelihood = torch.exp(log_probs_stable).sum(dim=1)
        log_likelihood = torch.log(likelihood + self.eps) + max_log_probs.squeeze(1)
        
        # Convert to negative log-likelihood
        nll = -log_likelihood
        
        # Reduce and return
        if self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)
        else:  # 'none'
            return nll
    
    def _extract_mixture_params(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract mixture parameters from model output.
        
        Args:
            y_pred: Model output tensor [batch_size, total_params]
            
        Returns:
            Dictionary with 'logits', 'means', and 'log_sigmas'
        """
        batch_size = y_pred.shape[0]
        
        # Calculate indices for splitting
        means_size = self.num_components * self.n_features
        sigmas_size = self.num_components * self.n_features
        
        # Split into components
        params = torch.split(y_pred, [self.num_components, means_size, sigmas_size], dim=1)
        
        # Extract logits, means, and log_sigmas
        logits = params[0]  # [batch_size, num_components]
        
        # Reshape means and sigmas
        means = params[1].view(batch_size, self.num_components, self.n_features)
        log_sigmas = params[2].view(batch_size, self.num_components, self.n_features)
        
        return {
            'logits': logits,
            'means': means,
            'log_sigmas': log_sigmas
        }
