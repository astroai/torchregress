"""
Mixture Density Network implementation for Errors-in-Variables regression.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, List

from ..base import MaskedLoss
from .eiv_utils import prepare_sigma

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
                # For simplicity, we'll add a constant amount of uncertainty
                # In a more sophisticated implementation, we would compute actual gradients
                # and propagate uncertainty properly
                sigma_x_component = torch.mean(sigma_x)  # Use mean as a simplification
                additional_variance = torch.ones_like(adjusted_sigmas[:, k]) * sigma_x_component**2
                adjusted_sigmas[:, k] = torch.sqrt(adjusted_sigmas[:, k]**2 + additional_variance)
            
            # Add intrinsic noise in y, if specified
            if sigma_y is not None:
                adjusted_sigmas[:, k] = torch.sqrt(adjusted_sigmas[:, k]**2 + sigma_y**2)
        
        # Calculate mixture component log probabilities
        log_probs = torch.zeros(batch_size, self.num_components, device=device)
        for k in range(self.num_components):
            # Calculate Gaussian log-likelihood for this component
            diff = y_true - means[:, k]
            mahalanobis_dist = -0.5 * torch.sum(
                (diff / adjusted_sigmas[:, k])**2, dim=1
            )
            log_det = -torch.sum(torch.log(adjusted_sigmas[:, k]), dim=1)
            const = -0.5 * self.n_features * torch.log(torch.tensor(2 * torch.pi, device=device))
            
            log_probs[:, k] = mahalanobis_dist + log_det + const
        
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
        n_features = self.n_features
        
        # Extract mixture weights
        logits = y_pred[:, :self.num_components]
        
        # Extract means and log_sigmas for each component
        means = torch.zeros(batch_size, self.num_components, n_features, device=y_pred.device)
        log_sigmas = torch.zeros_like(means)
        
        for k in range(self.num_components):
            # Each component has n_features means and n_features log_sigmas
            start_idx = self.num_components + k * 2 * n_features
            means[:, k] = y_pred[:, start_idx:start_idx + n_features]
            log_sigmas[:, k] = y_pred[:, start_idx + n_features:start_idx + 2*n_features]
            
        return {
            'logits': logits,
            'means': means,
            'log_sigmas': log_sigmas
        }
    
    def mdn_nll(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate standard MDN negative log-likelihood (without EIV adjustment).
        
        Args:
            y_true: Target values [batch_size, n_features]
            y_pred: MDN parameters [batch_size, total_params]
            
        Returns:
            NLL loss
        """
        batch_size = y_true.shape[0]
        device = y_true.device
        
        # Extract parameters
        mixture_params = self._extract_mixture_params(y_pred)
        logits = mixture_params['logits']
        means = mixture_params['means']
        log_sigmas = mixture_params['log_sigmas']
        
        # Calculate component probabilities
        sigmas = torch.exp(log_sigmas).clamp(min=self.min_sigma)
        log_pi = F.log_softmax(logits, dim=1)
        
        # Calculate log probabilities for each component
        log_probs = torch.zeros(batch_size, self.num_components, device=device)
        
        for k in range(self.num_components):
            # Standard Gaussian log likelihood
            diff = y_true - means[:, k]
            mahalanobis_dist = -0.5 * torch.sum(
                (diff / sigmas[:, k])**2, dim=1
            )
            log_det = -torch.sum(torch.log(sigmas[:, k]), dim=1)
            const = -0.5 * self.n_features * torch.log(torch.tensor(2 * torch.pi, device=device))
            
            log_probs[:, k] = mahalanobis_dist + log_det + const + log_pi[:, k]
        
        # Log-sum-exp for numerical stability
        max_log_probs = torch.max(log_probs, dim=1, keepdim=True)[0]
        log_probs_stable = log_probs - max_log_probs
        log_likelihood = max_log_probs.squeeze(1) + torch.log(torch.sum(torch.exp(log_probs_stable), dim=1) + self.eps)
        
        return -log_likelihood


class MDNEIVModel(nn.Module):
    """
    Mixture Density Network model with Error-in-Variables capabilities.
    
    This model outputs mixture parameters and can be trained with the MDNEIVLoss.
    
    Args:
        input_size: Input feature dimension
        hidden_layers: List of hidden layer sizes
        output_size: Dimensionality of output variable
        num_components: Number of mixture components
        activation: Activation function for hidden layers
    """
    def __init__(
        self,
        input_size: int, 
        hidden_layers: List[int], 
        output_size: int = 1,
        num_components: int = 5,
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_components = num_components
        
        # Calculate total parameters needed
        self.params_per_component = 2 * output_size  # mean and log_sigma for each feature
        self.total_params = num_components + num_components * self.params_per_component
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation)
            prev_size = size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer produces all MDN parameters
        self.output_layer = nn.Linear(prev_size, self.total_params)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            MDN parameters [batch_size, total_params]
        """
        features = self.feature_extractor(x)
        return self.output_layer(features)
    
    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Generate samples from the model's predictive distribution.
        
        Args:
            x: Input tensor [batch_size, input_size]
            n_samples: Number of samples to generate per input
            
        Returns:
            Samples [batch_size, n_samples, output_size]
        """
        batch_size = x.shape[0]
        device = x.device
        
        with torch.no_grad():
            # Get MDN parameters
            y_pred = self(x)
            
            # Extract component parameters
            params = self._extract_params(y_pred)
            pi = F.softmax(params['logits'], dim=1)  # [batch_size, num_components]
            mu = params['means']  # [batch_size, num_components, output_size]
            sigma = torch.exp(params['log_sigmas'])  # [batch_size, num_components, output_size]
            
            # Generate samples
            samples = torch.zeros(batch_size, n_samples, self.output_size, device=device)
            
            for i in range(batch_size):
                # For each input, sample component indices based on mixture weights
                components = torch.multinomial(pi[i], n_samples, replacement=True)
                
                # For each sample, generate from the selected component
                for j in range(n_samples):
                    k = components[j]
                    samples[i, j] = mu[i, k] + sigma[i, k] * torch.randn_like(sigma[i, k])
            
            return samples
    
    def _extract_params(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract mixture parameters from model output.
        
        Args:
            y_pred: Model output tensor [batch_size, total_params]
            
        Returns:
            Dictionary with 'logits', 'means', and 'log_sigmas'
        """
        batch_size = y_pred.shape[0]
        
        # Extract mixture weights
        logits = y_pred[:, :self.num_components]
        
        # Extract means and log_sigmas for each component
        means = torch.zeros(batch_size, self.num_components, self.output_size, device=y_pred.device)
        log_sigmas = torch.zeros_like(means)
        
        for k in range(self.num_components):
            # Each component has output_size means and output_size log_sigmas
            start_idx = self.num_components + k * 2 * self.output_size
            means[:, k] = y_pred[:, start_idx:start_idx + self.output_size]
            log_sigmas[:, k] = y_pred[:, start_idx + self.output_size:start_idx + 2*self.output_size]
            
        return {
            'logits': logits,
            'means': means,
            'log_sigmas': log_sigmas
        }
