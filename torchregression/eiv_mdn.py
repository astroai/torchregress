import torch
from typing import Callable, Optional, Union
from .base import MaskedLoss
from .eiv_utils import prepare_sigma, compute_model_gradients, calculate_propagated_variance

class MDNEIVLoss(MaskedLoss):
    """
    Error-in-Variables loss for Mixture Density Networks.
    
    Extends the EIV approach to MDNs by adjusting the component variances
    based on input noise propagation through the model gradients.
    
    Args:
        num_components: Number of mixture components
        n_features: Dimensionality of target variables
        sigma_y: Standard deviation of noise in the labels (can be None if included in MDN)
        sigma_x: Standard deviation of noise in the features
        eps: Small constant for numerical stability
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(
        self,
        num_components: int,
        n_features: int,
        sigma_x: Union[float, torch.Tensor],
        sigma_y: Optional[Union[float, torch.Tensor]] = None,
        eps: float = 1e-8,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.num_components = num_components
        self.n_features = n_features
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.eps = eps
        self.reduction = reduction
        
        # Calculate size of parameters needed
        self.dist_params_size = 2 * n_features  # mean and log_scale for each feature
        self.total_params_size = self.dist_params_size * num_components + num_components
    
    def forward(self, x_obs, y_true, y_pred, mask=None):
        """
        Calculate the MDN EIV loss.
        
        Args:
            x_obs: Observed features (batch_size, n_features_x)
            y_true: Observed targets (batch_size, n_features_y)
            y_pred: MDN parameters output (batch_size, total_params_size)
            mask: Optional mask (batch_size, n_features_y)
            
        Returns:
            loss: The MDN EIV loss
        """
        y_true = self._apply_mask(y_true, mask)
        batch_size = y_true.shape[0]
        n_features_x = x_obs.shape[1]
        device = x_obs.device
        
        # Validate input shape
        expected_size = self.total_params_size
        if y_pred.size(-1) != expected_size:
            raise ValueError(f"Expected y_pred to have {expected_size} features, got {y_pred.size(-1)}")
        
        # Prepare sigma parameters
        sigma_y = prepare_sigma(self.sigma_y, self.n_features, device, default_zero=False)
        sigma_x = prepare_sigma(self.sigma_x, n_features_x, device)
            
        # Ensure x_obs requires gradients
        x_obs_grad = x_obs.detach().clone()
        x_obs_grad.requires_grad_(True)
        
        # Forward pass through the model (we need gradients for each component mean)
        y_pred = self._recompute_with_gradients(x_obs_grad, y_pred)
        
        # Extract mixture parameters
        log_mix_weights = y_pred[..., -self.num_components:]
        mix_weights = torch.softmax(log_mix_weights, dim=-1)
        
        # Extract distribution parameters
        dist_params = y_pred[..., :-self.num_components]
        
        # Extract means and scales
        means = dist_params[..., :self.n_features * self.num_components].reshape(
            batch_size, self.num_components, self.n_features)
        log_scales = dist_params[..., self.n_features * self.num_components:].reshape(
            batch_size, self.num_components, self.n_features)
        scales = torch.exp(log_scales)
        
        # Compute gradients of each component mean with respect to inputs
        # This is more complex for MDN as we have multiple component means
        all_grads = []
        for k in range(self.num_components):
            component_grads = []
            for j in range(self.n_features):
                grad = torch.autograd.grad(
                    outputs=means[:, k, j].sum(),
                    inputs=x_obs_grad,
                    create_graph=True,
                    retain_graph=True
                )[0]  # Shape: (batch_size, n_features_x)
                component_grads.append(grad)
            # Stack gradients for this component
            component_grads = torch.stack(component_grads, dim=1)  # (batch_size, n_features_y, n_features_x)
            all_grads.append(component_grads)
        
        # Compute component-wise log probabilities
        log_probs = []
        
        for k in range(self.num_components):
            # Get component parameters
            mu_k = means[:, k, :]
            sigma_k = scales[:, k, :]
            
            # Compute variance propagation for this component
            grads_k = all_grads[k]  # (batch_size, n_features_y, n_features_x)
            
            # Calculate propagated variance
            propagated_var = calculate_propagated_variance(
                grads_k, sigma_x, batch_size, self.n_features, device
            )
            
            # Total variance includes model variance, label noise (if provided), and propagated feature noise
            total_var = sigma_k**2
            
            # Add label noise if provided
            if sigma_y is not None:
                if sigma_y.ndim == 0:
                    total_var = total_var + sigma_y**2
                else:
                    total_var = total_var + sigma_y**2
            
            # Add propagated feature noise
            total_var = total_var + propagated_var + self.eps
            
            # Compute negative log likelihood for this component
            diff = y_true - mu_k  # (batch_size, n_features_y)
            exponent = -0.5 * (diff**2) / total_var
            log_coeff = -0.5 * torch.log(2 * torch.pi * total_var)
            log_prob_k = log_coeff + exponent  # (batch_size, n_features_y)
            
            # Sum over features
            log_prob_k = torch.sum(log_prob_k, dim=1)  # (batch_size,)
            log_probs.append(log_prob_k)
        
        # Stack component log probabilities and apply mixture weights
        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_components)
        weighted_log_probs = log_probs + torch.log(mix_weights + self.eps)
        
        # LogSumExp trick for numerical stability
        max_log_probs = torch.max(weighted_log_probs, dim=1, keepdim=True)[0]
        log_sum_exp = max_log_probs + torch.log(
            torch.sum(torch.exp(weighted_log_probs - max_log_probs), dim=1, keepdim=True) + self.eps
        )
        log_likelihood = log_sum_exp.squeeze(1)  # (batch_size,)
        
        # Negative log likelihood
        nll = -log_likelihood
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)
        else:  # 'none'
            return nll
    
    def _recompute_with_gradients(self, x_obs_grad, y_pred):
        """
        Helper function to handle cases where y_pred is provided separately or from a model.
        Ensures we have the correct computational graph for gradient computation.
        """
        # If y_pred is already correctly connected to x_obs_grad, return it
        # Otherwise, we'd need to recompute using the actual model
        return y_pred


# Specialized MDN Chamfer EIV loss could be added here if needed
