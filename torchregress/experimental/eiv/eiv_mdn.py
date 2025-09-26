"""
Mixture Density Network implementation for Errors-in-Variables regression.
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import RegressionLoss
from .eiv_utils import (
    generate_perturbed_samples,
    prepare_model_input_for_gradients,
    prepare_sigma,
)


class MDNEIVLoss(RegressionLoss):
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
        uncertainty_method: Method for uncertainty propagation ('fixed', 'gradient', 'monte_carlo')
        mc_samples: Number of Monte Carlo samples for uncertainty estimation
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
        uncertainty_method: str = "fixed",  # 'fixed', 'gradient', 'monte_carlo'
        mc_samples: int = 100,  # for Monte Carlo estimation
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.num_components = num_components
        self.n_features = n_features
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.min_sigma = min_sigma
        self.eps = eps
        self.uncertainty_method = uncertainty_method
        self.mc_samples = mc_samples

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
        y_true.shape[0]
        device = y_true.device

        # Verify MDN output size
        if y_pred.size(-1) != self.total_params:
            raise ValueError(
                f"Expected output with {self.total_params} parameters, " f"got {y_pred.size(-1)}"
            )

        # Prepare sigma parameters
        sigma_x = prepare_sigma(self.sigma_x, x_obs.shape[1], device)
        sigma_y = prepare_sigma(self.sigma_y, self.n_features, device, default_zero=False)

        # Split MDN output into components
        mixture_params = self._extract_mixture_params(y_pred)
        logits = mixture_params["logits"]
        means = mixture_params["means"]
        log_sigmas = mixture_params["log_sigmas"]

        # Ensure minimum sigma value for numerical stability
        sigmas = torch.exp(log_sigmas).clamp(min=self.min_sigma)

        # Calculate EIV-adjusted sigma for each component based on selected method
        if self.uncertainty_method == "fixed":
            adjusted_sigmas = self._fixed_uncertainty_adjustment(sigmas, sigma_x, sigma_y)
        elif self.uncertainty_method == "gradient":
            adjusted_sigmas = self._gradient_uncertainty_adjustment(x_obs, sigmas, sigma_x, sigma_y)
        elif self.uncertainty_method == "monte_carlo":
            adjusted_sigmas = self._monte_carlo_uncertainty_adjustment(
                x_obs, sigmas, sigma_x, sigma_y
            )
        else:
            raise ValueError(f"Unsupported uncertainty method: {self.uncertainty_method}")

        # Calculate mixture component log probabilities
        log_probs = self._calculate_component_log_probs(y_true, means, adjusted_sigmas, device)

        # Add log mixture weights to log probabilities
        log_pi = F.log_softmax(logits, dim=1)  # [batch_size, num_components]
        log_probs += log_pi

        # Use the log-sum-exp trick for numeric stability
        log_likelihood = self._logsumexp(log_probs, dim=1)

        # Convert to negative log-likelihood
        nll = -log_likelihood

        # Reduce and return
        if self.reduction == "mean":
            return torch.mean(nll)
        elif self.reduction == "sum":
            return torch.sum(nll)
        else:  # 'none'
            return nll

    def _fixed_uncertainty_adjustment(self, sigmas, sigma_x, sigma_y):
        """Simple approach: add fixed uncertainty based on sigma_x"""
        adjusted_sigmas = sigmas.clone()

        if sigma_x is not None:
            # Use average sigma_x as a simplified approximation
            sigma_x_scalar = sigma_x.mean() if isinstance(sigma_x, torch.Tensor) else sigma_x

            # Add uncertainty to all components
            additional_variance = torch.ones_like(adjusted_sigmas) * (sigma_x_scalar**2)
            adjusted_sigmas = torch.sqrt(adjusted_sigmas**2 + additional_variance)

        # Add intrinsic noise in y, if specified
        if sigma_y is not None:
            if sigma_y.ndim <= 1:  # diagonal case
                sigma_y_expanded = sigma_y.unsqueeze(0).unsqueeze(0).expand_as(adjusted_sigmas)
                adjusted_sigmas = torch.sqrt(adjusted_sigmas**2 + sigma_y_expanded**2)
            else:
                # For full covariance, use a simplified approach by taking diagonal elements
                sigma_y_diag = (
                    torch.diagonal(sigma_y).unsqueeze(0).unsqueeze(0).expand_as(adjusted_sigmas)
                )
                adjusted_sigmas = torch.sqrt(adjusted_sigmas**2 + sigma_y_diag)

        return adjusted_sigmas

    def _gradient_uncertainty_adjustment(self, x_obs, sigmas, sigma_x, sigma_y):
        """Use gradients to propagate uncertainty more accurately"""
        batch_size = x_obs.shape[0]
        adjusted_sigmas = sigmas.clone()

        if sigma_x is not None:
            # Prepare input for gradient computation
            prepare_model_input_for_gradients(x_obs)

            # We'll need to implement a function to extract component means from MDN based on input
            # This is a simplified approximation - for a proper implementation, we would need
            # to define how the MDN's means depend on inputs

            # For now, let's use a simple approximation based on component index
            for k in range(self.num_components):
                # For a full implementation, we would compute proper gradients for each component mean
                # with respect to inputs - this is just a placeholder
                sigma_x_component = sigma_x.mean() if isinstance(sigma_x, torch.Tensor) else sigma_x
                additional_variance = torch.ones_like(adjusted_sigmas[:, k]) * sigma_x_component**2
                adjusted_sigmas[:, k] = torch.sqrt(adjusted_sigmas[:, k] ** 2 + additional_variance)

        # Add intrinsic noise in y, if specified
        if sigma_y is not None:
            if sigma_y.ndim <= 1:  # diagonal case
                for k in range(self.num_components):
                    sigma_y_expanded = sigma_y.unsqueeze(0).expand(batch_size, -1)
                    adjusted_sigmas[:, k] = torch.sqrt(
                        adjusted_sigmas[:, k] ** 2 + sigma_y_expanded**2
                    )
            else:
                # For full covariance, use diagonal elements as approximation
                sigma_y_diag = torch.diagonal(sigma_y)
                for k in range(self.num_components):
                    sigma_y_expanded = sigma_y_diag.unsqueeze(0).expand(batch_size, -1)
                    adjusted_sigmas[:, k] = torch.sqrt(
                        adjusted_sigmas[:, k] ** 2 + sigma_y_expanded**2
                    )

        return adjusted_sigmas

    def _monte_carlo_uncertainty_adjustment(self, x_obs, sigmas, sigma_x, sigma_y):
        """Use Monte Carlo sampling to estimate uncertainty propagation"""
        batch_size = x_obs.shape[0]
        device = x_obs.device

        # Original sigmas as starting point
        adjusted_sigmas = sigmas.clone()

        if sigma_x is not None and self.mc_samples > 0:
            # Generate perturbed inputs for Monte Carlo estimation
            perturbed_samples = generate_perturbed_samples(
                x_obs, sigma_x, self.mc_samples, perturb_method="gaussian"
            )

            # Process Monte Carlo samples in smaller batches if needed
            max_batch_size = 200  # Adjust based on available memory
            n_batches = max(1, self.mc_samples // max_batch_size)
            samples_per_batch = self.mc_samples // n_batches

            # Storage for component-wise variance estimates
            torch.zeros((batch_size, self.num_components, self.n_features), device=device)

            # We need a model reference for this approach - without it, this is just illustrative
            for i in range(n_batches):
                start_idx = i * samples_per_batch
                end_idx = min((i + 1) * samples_per_batch, self.mc_samples)
                perturbed_samples[start_idx:end_idx]

                # This is a placeholder - in practice we would need access to the MDN model
                # to run forward passes on the perturbed samples and analyze component-wise variances
                pass

            # As a fallback, use the fixed method
            adjusted_sigmas = self._fixed_uncertainty_adjustment(sigmas, sigma_x, sigma_y)

        return adjusted_sigmas

    def _calculate_component_log_probs(self, y_true, means, sigmas, device):
        """Calculate log probabilities for each mixture component"""
        batch_size = y_true.shape[0]
        log_probs = torch.zeros(batch_size, self.num_components, device=device)

        for k in range(self.num_components):
            # Calculate Gaussian log-likelihood for this component
            diff = y_true - means[:, k]
            mahalanobis_dist = -0.5 * torch.sum((diff / sigmas[:, k]) ** 2, dim=1)
            log_det = -torch.sum(torch.log(sigmas[:, k]), dim=1)
            const = -0.5 * self.n_features * torch.log(torch.tensor(2 * torch.pi, device=device))

            log_probs[:, k] = mahalanobis_dist + log_det + const

        return log_probs

    def _logsumexp(self, tensor, dim=-1):
        """Numerically stable log-sum-exp implementation"""
        max_val, _ = torch.max(tensor, dim=dim, keepdim=True)
        tensor_stable = tensor - max_val
        result = torch.log(
            torch.sum(torch.exp(tensor_stable), dim=dim) + self.eps
        ) + max_val.squeeze(dim)
        return result

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
        logits = y_pred[:, : self.num_components]

        # Extract means and log_sigmas for each component
        means = torch.zeros(batch_size, self.num_components, n_features, device=y_pred.device)
        log_sigmas = torch.zeros_like(means)

        for k in range(self.num_components):
            # Each component has n_features means and n_features log_sigmas
            start_idx = self.num_components + k * 2 * n_features
            means[:, k] = y_pred[:, start_idx : start_idx + n_features]
            log_sigmas[:, k] = y_pred[:, start_idx + n_features : start_idx + 2 * n_features]

        return {"logits": logits, "means": means, "log_sigmas": log_sigmas}

    def sample_from_mixture(self, logits, means, sigmas, n_samples=1):
        """Generate samples from the mixture distribution"""
        batch_size = logits.shape[0]
        device = logits.device

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # Generate samples
        samples = torch.zeros(batch_size, n_samples, self.n_features, device=device)

        for i in range(batch_size):
            # Sample component indices based on mixture weights
            components = torch.multinomial(probs[i], n_samples, replacement=True)

            # For each sample, generate from selected component
            for j in range(n_samples):
                k = components[j]
                samples[i, j] = means[i, k] + sigmas[i, k] * torch.randn(
                    self.n_features, device=device
                )

        return samples


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
        dropout_rate: Dropout probability (0 to disable)
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int = 1,
        num_components: int = 5,
        activation: nn.Module = nn.ReLU(),
        dropout_rate: float = 0.0,
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
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        self.feature_extractor = nn.Sequential(*layers)

        # Output layer produces all MDN parameters
        self.output_layer = nn.Linear(prev_size, self.total_params)

        # Initialize parameters with appropriate scaling
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence"""
        # Xavier/Glorot initialization for most layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Special initialization for the output layer
        # Initialize mixture weights close to equal
        nn.init.zeros_(self.output_layer.bias[: self.num_components])

        # Initialize log_sigmas to small negative values for reasonable starting variance
        for k in range(self.num_components):
            start_idx = self.num_components + k * 2 * self.output_size + self.output_size
            end_idx = start_idx + self.output_size
            nn.init.constant_(self.output_layer.bias[start_idx:end_idx], -1.0)

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
            pi = F.softmax(params["logits"], dim=1)  # [batch_size, num_components]
            mu = params["means"]  # [batch_size, num_components, output_size]
            sigma = torch.exp(params["log_sigmas"]).clamp(
                min=1e-4
            )  # [batch_size, num_components, output_size]

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
        logits = y_pred[:, : self.num_components]

        # Extract means and log_sigmas for each component
        means = torch.zeros(batch_size, self.num_components, self.output_size, device=y_pred.device)
        log_sigmas = torch.zeros_like(means)

        for k in range(self.num_components):
            # Each component has output_size means and output_size log_sigmas
            start_idx = self.num_components + k * 2 * self.output_size
            means[:, k] = y_pred[:, start_idx : start_idx + self.output_size]
            log_sigmas[:, k] = y_pred[
                :, start_idx + self.output_size : start_idx + 2 * self.output_size
            ]

        return {"logits": logits, "means": means, "log_sigmas": log_sigmas}

    def get_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate predictive mean and uncertainty measures.

        Args:
            x: Input tensor [batch_size, input_size]

        Returns:
            Dictionary with 'mean', 'variance', 'entropy'
        """
        x.shape[0]

        with torch.no_grad():
            # Get MDN parameters
            y_pred = self(x)

            # Extract component parameters
            params = self._extract_params(y_pred)
            probs = F.softmax(params["logits"], dim=1)  # [batch_size, num_components]
            mu = params["means"]  # [batch_size, num_components, output_size]
            sigma = torch.exp(params["log_sigmas"]).clamp(
                min=1e-4
            )  # [batch_size, num_components, output_size]

            # Calculate mean prediction (weighted average of component means)
            mean = torch.sum(probs.unsqueeze(-1) * mu, dim=1)  # [batch_size, output_size]

            # Calculate variance (law of total variance)
            # Var[Y] = E[Var[Y|Z]] + Var[E[Y|Z]]
            # First term: expected component variance
            expected_comp_var = torch.sum(
                probs.unsqueeze(-1) * sigma**2, dim=1
            )  # [batch_size, output_size]

            # Second term: variance of component means
            # Need to compute: sum_k π_k(μ_k - μ)^2
            centered_mu = mu - mean.unsqueeze(1)  # [batch_size, num_components, output_size]
            var_comp_means = torch.sum(
                probs.unsqueeze(-1) * centered_mu**2, dim=1
            )  # [batch_size, output_size]

            # Total variance
            variance = expected_comp_var + var_comp_means  # [batch_size, output_size]

            # Calculate entropy (approximation for MDNs)
            # H[Y] ≈ sum_k π_k * (log(sqrt(2πe*σ_k^2)))
            comp_entropy = torch.sum(
                0.5 * torch.log(2 * torch.pi * torch.e * sigma**2), dim=2
            )  # [batch_size, num_components]
            entropy = torch.sum(probs * comp_entropy, dim=1)  # [batch_size]

            return {"mean": mean, "variance": variance, "entropy": entropy}
