"""
Mixture Density Network (MDN) loss functions.

This module provides loss functions for Mixture Density Networks,
which model outputs as mixtures of Gaussian distributions.
"""

import math

import torch
import torch.nn.functional as F

from .base import DistributionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("mdn")
class MixtureDensityLoss(DistributionLoss):
    """
    Negative Log-Likelihood loss for Mixture Density Networks.

    Models the output as a mixture of Gaussian distributions with either
    diagonal or full covariance matrices.

    Args:
        n_components (int): Number of mixture components
        n_features (int): Number of output features
        covariance_type (str): Type of covariance matrix ('diagonal' or 'full')
        min_std (float): Minimum standard deviation for numerical stability
        eps (float): Small constant for numerical stability in calculations
        reduction (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'
            Default: 'mean'

    Mathematical Formulation:
        For a mixture density network with K components, the negative log
        likelihood is given by:

        NLL = -log(∑(w_k * p_k(y|μ_k, Σ_k)))

        where w_k are mixture weights, and p_k is the probability density
        function of the k-th Gaussian component with mean μ_k and covariance Σ_k.

    Notes:
        - The model output is expected to contain all distribution parameters
          concatenated in a single tensor.
        - For 'diagonal' covariance: output size = n_components + 2*n_components*n_features
        - For 'full' covariance: output size = n_components + n_components*n_features +
          n_components*n_features*(n_features+1)/2

    Examples:
        >>> import torch
        >>> loss_fn = MixtureDensityLoss(n_components=2, n_features=3, covariance_type='diagonal')
        >>> # Create model predictions with mixture weights, means, and stds
        >>> y_pred = torch.randn(10, 2 + 2*2*3)  # Batch of 10, 2 components, 3 features
        >>> target = torch.randn(10, 3)  # Ground truth values
        >>> loss = loss_fn(y_pred, target)
    """

    def __init__(
        self,
        n_components: int,
        n_features: int,
        covariance_type: str = "diagonal",
        min_std: float = 1e-3,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.n_components = n_components
        self.n_features = n_features
        self.covariance_type = covariance_type.lower()
        self.min_std = min_std
        self.eps = eps
        self.log_2pi = math.log(2 * math.pi)

        if self.covariance_type not in ["diagonal", "full"]:
            raise ValueError(
                f"Unsupported covariance_type: {covariance_type}. Expected 'diagonal' or 'full'"
            )

        # Calculate expected output size for validation
        if self.covariance_type == "diagonal":
            # n_components (mixture weights) + n_components * n_features (means) + n_components * n_features (stds)
            self.expected_output_size = n_components + 2 * n_components * n_features
        else:  # 'full'
            # n_components + n_components * n_features (means) + n_components * n_features * (n_features + 1) / 2 (covs)
            # We use triangular parameterization for covariance matrices
            self.expected_output_size = (
                n_components
                + n_components * n_features
                + n_components * n_features * (n_features + 1) // 2
            )

    def _extract_distribution_parameters(self, y_pred):
        """
        Extract mixture parameters from model predictions.

        Args:
            y_pred (torch.Tensor): Model predictions. For diagonal covariance, shape should be
                   [..., n_components + n_components*n_features + n_components*n_features]
                   For full covariance, shape depends on the parameterization.

        Returns:
            tuple: (mixture_weights, means, stds or cov_factors)
        """
        # Validate output size
        if y_pred.shape[-1] != self.expected_output_size:
            raise ValueError(
                f"Model output size {y_pred.shape[-1]} doesn't match expected size {self.expected_output_size}"
            )

        batch_shape = y_pred.shape[:-1]

        # Extract mixture weights - always the first n_components elements
        logits = y_pred[..., : self.n_components]
        # Use softmax to ensure weights sum to 1
        mixture_weights = F.softmax(logits, dim=-1)

        # Extract means - always after the mixture weights
        means_start = self.n_components
        means_end = means_start + self.n_components * self.n_features
        means = y_pred[..., means_start:means_end].reshape(
            *batch_shape, self.n_components, self.n_features
        )

        # Extract covariance parameters - depends on covariance_type
        if self.covariance_type == "diagonal":
            # For diagonal, we extract log_stds and convert to stds
            log_stds_start = means_end
            log_stds = y_pred[..., log_stds_start:].reshape(
                *batch_shape, self.n_components, self.n_features
            )

            # Apply softplus with shift for better numerical stability
            # This ensures stds are always positive and not too close to zero
            stds = F.softplus(log_stds) + self.min_std

            return mixture_weights, means, stds
        else:  # 'full'
            # For full covariance, we extract parameters for lower triangular matrices
            # using Cholesky decomposition parameterization
            tril_indices = torch.tril_indices(self.n_features, self.n_features)
            n_tril_elements = len(tril_indices[0])

            tril_start = means_end
            tril_values = y_pred[..., tril_start:].reshape(
                *batch_shape, self.n_components, n_tril_elements
            )

            # Create full batch of lower triangular matrices
            L_matrices = []
            for c in range(self.n_components):
                # Start with zeros for each component
                L = torch.zeros(
                    *batch_shape, self.n_features, self.n_features, device=y_pred.device
                )

                # Fill lower triangular part
                L[..., tril_indices[0], tril_indices[1]] = tril_values[..., c, :]

                # Ensure positive diagonal (for valid Cholesky decomposition)
                diag_indices = torch.arange(self.n_features)
                L[..., diag_indices, diag_indices] = (
                    F.softplus(L[..., diag_indices, diag_indices]) + self.min_std
                )

                L_matrices.append(L)

            # Stack along component dimension
            L_matrices = torch.stack(L_matrices, dim=-3)

            return mixture_weights, means, L_matrices

    def _log_prob_diagonal(self, target, means, stds):
        """
        Calculate log probability for diagonal covariance components.

        Args:
            target (torch.Tensor): Ground truth values [..., n_features]
            means (torch.Tensor): Component means [..., n_components, n_features]
            stds (torch.Tensor): Component standard deviations [..., n_components, n_features]

        Returns:
            torch.Tensor: Log probabilities [..., n_components]
        """
        # Expand target for broadcasting with components
        target_expanded = target.unsqueeze(-2)  # [..., 1, n_features]

        # Calculate normalized distances: (y - μ)/σ
        normalized_dist = (target_expanded - means) / (stds + self.eps)

        # Calculate log probability: -0.5 * (Σ(z²) + Σ(log(σ²)) + n*log(2π))
        # Separated for numerical stability
        exponent_term = -0.5 * torch.sum(normalized_dist**2, dim=-1)
        log_det_term = -torch.sum(torch.log(stds + self.eps), dim=-1)
        const_term = -0.5 * self.n_features * self.log_2pi

        log_probs = exponent_term + log_det_term + const_term

        return log_probs

    def _log_prob_full(self, target, means, L_matrices):
        """
        Calculate log probability for full covariance components using Cholesky factors.

        Args:
            target (torch.Tensor): Ground truth values [..., n_features]
            means (torch.Tensor): Component means [..., n_components, n_features]
            L_matrices (torch.Tensor): Lower triangular Cholesky factors
                                      [..., n_components, n_features, n_features]

        Returns:
            torch.Tensor: Log probabilities [..., n_components]
        """
        # Expand target for broadcasting with components
        target_expanded = target.unsqueeze(-2)  # [..., 1, n_features]

        # Calculate residuals: (y - μ)
        residuals = target_expanded - means  # [..., n_components, n_features]

        residuals.dim() - 2  # Exclude component and feature dimensions
        residuals.shape[:-2]

        log_probs = []

        # Process each component separately for better memory efficiency
        for c in range(self.n_components):
            # Extract component residuals and Cholesky factors
            comp_residuals = residuals[..., c, :]  # [..., n_features]
            L = L_matrices[..., c, :, :]  # [..., n_features, n_features]

            try:
                # Solve triangular system: z = L⁻¹(y-μ)
                # Using batch-friendly implementation
                z = torch.linalg.solve_triangular(
                    L, comp_residuals.unsqueeze(-1), upper=False
                ).squeeze(-1)

                # Calculate quadratic term: ‖z‖²
                quadratic_term = torch.sum(z**2, dim=-1)

                # Calculate log determinant: 2*Σlog(L_ii)
                # Equivalent to log(det(Σ)) since det(Σ) = det(LLᵀ) = det(L)²
                diag_L = torch.diagonal(L, dim1=-2, dim2=-1)  # [..., n_features]
                log_det = 2 * torch.sum(torch.log(diag_L + self.eps), dim=-1)

                # Calculate log probability
                comp_log_prob = -0.5 * (quadratic_term + log_det + self.n_features * self.log_2pi)

            except RuntimeError:
                # Fallback for numerical issues - use eigendecomposition
                # This is computationally more expensive but more stable
                cov = torch.bmm(L, L.transpose(-1, -2))  # Σ = LLᵀ

                # Add small regularization to diagonal for stability
                diag_indices = torch.arange(self.n_features, device=L.device)
                cov[..., diag_indices, diag_indices] += self.eps

                # Eigendecomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)

                # Ensure eigenvalues are positive
                eigenvalues = torch.clamp(eigenvalues, min=self.eps)

                # Log determinant: sum(log(λ_i))
                log_det = torch.sum(torch.log(eigenvalues), dim=-1)

                # Whiten the residuals: (y-μ)ᵀΣ⁻¹(y-μ) = Σ[(y-μ)ᵀv_i]²/λ_i
                # where v_i, λ_i are eigenvectors and eigenvalues
                whitened = torch.bmm(
                    eigenvectors.transpose(-1, -2), comp_residuals.unsqueeze(-1)
                ).squeeze(-1)
                quadratic_term = torch.sum(whitened**2 / eigenvalues, dim=-1)

                # Calculate log probability
                comp_log_prob = -0.5 * (quadratic_term + log_det + self.n_features * self.log_2pi)

            log_probs.append(comp_log_prob)

        # Stack component log probabilities
        return torch.stack(log_probs, dim=-1)  # [..., n_components]

    def _calculate_nll(self, target, params, mask=None):
        """
        Calculate negative log likelihood for mixture model.

        Args:
            target (torch.Tensor): Target values [..., n_features]
            params (tuple): Tuple of (weights, means, stds_or_L_matrices)
            mask (torch.Tensor, optional): Optional mask for valid values

        Returns:
            torch.Tensor: Negative log likelihood values
        """
        weights, means, stds_or_L = params

        # Calculate log probabilities for each component based on covariance type
        if self.covariance_type == "diagonal":
            log_probs = self._log_prob_diagonal(target, means, stds_or_L)
        else:  # 'full'
            log_probs = self._log_prob_full(target, means, stds_or_L)

        # Apply log weights: log(w_i) + log(p_i)
        log_weights = torch.log(weights + self.eps)
        log_probs_weighted = log_weights + log_probs

        # Calculate mixture log probability using logsumexp for numerical stability
        # log(Σw_i*p_i) = log(Σexp(log(w_i*p_i))) = logsumexp(log(w_i)+log(p_i))
        mixture_log_prob = torch.logsumexp(log_probs_weighted, dim=-1)

        # Return negative log likelihood
        return -mixture_log_prob

    def forward(self, y_pred, target, mask=None, weights=None):
        """
        Calculate mixture density negative log-likelihood loss.

        Args:
            y_pred (torch.Tensor): Model predictions (format depends on covariance_type)
            target (torch.Tensor): Ground truth values [..., n_features]
            mask (torch.Tensor, optional): Optional boolean mask [..., n_features]
            weights (torch.Tensor, optional): Optional sample weights

        Returns:
            torch.Tensor: Negative log-likelihood loss

        Raises:
            ValueError: If target shape doesn't match expected features
        """
        # Basic input validation
        if target.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features in target, got {target.shape[-1]}"
            )

        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred)

        # For mixture models, we need to handle each sample as a whole
        # So we convert feature-level mask to sample-level mask
        sample_mask = None
        if mask is not None:
            if mask.dim() > 1 and mask.shape[-1] > 1:
                # If any feature is masked, the whole sample is considered invalid
                # This is a common approach for MDNs since components span all dimensions
                sample_mask = mask.all(dim=-1)
            else:
                sample_mask = mask

        # Calculate negative log likelihood (per sample)
        nll = self._calculate_nll(target, params, sample_mask)

        # Apply sample weights if provided
        if weights is not None:
            if weights.dim() > 1 and weights.shape[-1] > 1:
                # Convert feature-level weights to sample-level (average)
                sample_weights = weights.mean(dim=-1)
            else:
                sample_weights = weights
            nll = nll * sample_weights

        # Apply mask and reduction
        if sample_mask is not None:
            # For MDNs, we use specialized masked reduction since we've already
            # converted to sample-level
            if self.reduction == "none":
                return nll * sample_mask  # Zero out invalid samples
            elif self.reduction == "mean":
                valid_count = sample_mask.sum().clamp(min=1)  # Avoid division by zero
                return (nll * sample_mask).sum() / valid_count
            else:  # 'sum'
                return (nll * sample_mask).sum()
        else:
            # Standard reduction with no masking
            if self.reduction == "none":
                return nll
            elif self.reduction == "mean":
                return nll.mean()
            else:  # 'sum'
                return nll.sum()
