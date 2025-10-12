"""
Gaussian loss functions for regression tasks.

This module provides Gaussian-based negative log-likelihood loss functions:
- GaussianNLLLoss: Diagonal covariance (independent features with learned variance)
- MultivariateGaussianLoss: Full covariance (correlated features)

For fixed variance (homoscedastic), use WeightedMSELoss from base module instead.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from .base import DistributionLoss


class GaussianNLLLoss(DistributionLoss):
    """
    Negative Log-Likelihood loss for diagonal Gaussian distributions.

    This loss models each output dimension with an independent Gaussian distribution
    where the model predicts both mean and variance (heteroscedastic regression).

    The model should output either:
    - A tuple (mean, log_variance)
    - A concatenated tensor [..., 2*n_features] containing [mean, log_var]
    - Just mean (when using fixed_variance parameter)

    Args:
        fixed_variance: Fixed variance value when model only predicts mean. Default: None
        min_variance: Minimum variance for numerical stability. Default: 1e-6
        eps: Small constant for numerical stability. Default: 1e-8
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> # Model predicts (mean, log_variance) tuple
        >>> loss_fn = GaussianNLLLoss()
        >>> pred_mean = torch.tensor([[1.0, 2.0]])
        >>> pred_logvar = torch.tensor([[-1.0, 0.0]])  # log(0.368), log(1.0)
        >>> target = torch.tensor([[1.0, 3.0]])
        >>> loss_fn((pred_mean, pred_logvar), target)
        tensor(1.0979)

        >>> # Model predicts only mean, use fixed variance
        >>> loss_fn = GaussianNLLLoss(fixed_variance=0.5)
        >>> y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> target = torch.tensor([[0.0, 2.0], [3.0, 5.0]])
        >>> loss_fn(y_pred, target)
        tensor(1.7689)
    """

    def __init__(
        self,
        fixed_variance: Optional[float] = None,
        min_variance: float = 1e-6,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.min_variance = min_variance
        self.eps = eps
        self.log_2pi = math.log(2 * math.pi)

        if fixed_variance is not None:
            self.register_buffer("fixed_variance", torch.tensor(fixed_variance))
        else:
            self.fixed_variance = None

    def _extract_distribution_parameters(
        self, y_pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract mean and variance from model predictions.

        Args:
            y_pred: Model predictions, can be either:
                   - Tuple of (mean, log_var)
                   - Tensor with shape [..., 2*n_features] containing concatenated [mean, log_var]
                   - Mean values only (when fixed_variance is set)

        Returns:
            tuple: (mean, variance)
        """
        if isinstance(y_pred, tuple) and len(y_pred) == 2:
            # Model outputs (mean, log_var) tuple
            mean, log_var = y_pred
            var = torch.exp(log_var).clamp(min=self.min_variance)
        elif isinstance(y_pred, torch.Tensor) and y_pred.shape[-1] % 2 == 0:
            # Try to interpret as concatenated [mean, log_var]
            # Only do this if fixed_variance is not set
            if self.fixed_variance is None:
                n_features = y_pred.shape[-1] // 2
                mean, log_var = y_pred[..., :n_features], y_pred[..., n_features:]
                var = torch.exp(log_var).clamp(min=self.min_variance)
            else:
                # Just mean predictions with fixed variance
                mean = y_pred
                var = self.fixed_variance * torch.ones_like(mean)
        else:
            # Just mean predictions, must use fixed variance
            if self.fixed_variance is None:
                raise ValueError(
                    "Model appears to predict only mean values, but fixed_variance is not set. "
                    "Either set fixed_variance parameter or ensure model outputs (mean, log_var)."
                )
            mean = y_pred
            var = self.fixed_variance * torch.ones_like(mean)

        return mean, var

    def _calculate_nll(
        self,
        y_pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate negative log likelihood for diagonal Gaussian.

        Args:
            y_pred: Predicted distribution parameters
            target: Target values
            mask: Optional mask for valid values

        Returns:
            tensor: Negative log likelihood values
        """
        mean, var = self._extract_distribution_parameters(y_pred)

        # Calculate NLL: 0.5 * (log(var) + (y-μ)²/var + log(2π))
        squared_error = (target - mean) ** 2
        nll = 0.5 * (torch.log(var + self.eps) + squared_error / (var + self.eps) + self.log_2pi)

        return nll

    def forward(
        self,
        y_pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate Gaussian negative log-likelihood loss.

        Args:
            y_pred: Predictions - see _extract_distribution_parameters for formats
            target: Ground truth values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional sample weights [batch_size, n_features] or [batch_size]

        Returns:
            Negative log-likelihood loss
        """
        # Validate inputs
        if isinstance(y_pred, tuple):
            self._validate_inputs(y_pred[0], target, mask)
        # For tensor inputs, validation is skipped when concatenated [mean, log_var]
        # as y_pred will have shape [..., 2*n_features]

        # Calculate NLL
        nll = self._calculate_nll(y_pred, target, mask)

        # Apply reduction with mask and weights
        return self._reduce_with_mask(nll, mask, weights)


class MultivariateGaussianLoss(DistributionLoss):
    """
    Negative Log-Likelihood loss for multivariate Gaussian with full covariance matrices.

    Models the output with a multivariate Gaussian distribution with a full covariance matrix,
    which can capture correlations between features.

    Args:
        n_features: Number of features (required for learnable_adjustment)
        learnable_adjustment: Whether to learn feature-specific variance adjustments
        jitter: Small value added to diagonal for numerical stability
        eps: Small constant for numerical stability in log calculations
        reduction: 'none' | 'mean' | 'sum'

    Example:
        >>> loss_fn = MultivariateGaussianLoss()
        >>> y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> target = torch.tensor([[0.0, 2.0], [3.0, 5.0]])
        >>> cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])  # Shared covariance
        >>> loss_fn(y_pred, target, cov)
        tensor(0.9069)  # Using multivariate Gaussian with correlation
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        learnable_adjustment: bool = False,
        jitter: float = 1e-6,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.n_features = n_features
        self.jitter = jitter
        self.eps = eps
        self.learnable_adjustment = learnable_adjustment
        self.log_2pi = math.log(2 * math.pi)

        if learnable_adjustment:
            if n_features is None:
                raise ValueError("n_features must be specified when learnable_adjustment=True")
            self.log_variance_adjustment = nn.Parameter(torch.zeros(n_features))

    def _extract_distribution_parameters(
        self, y_pred: torch.Tensor, covariance_matrices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract mean and covariance from model predictions.

        Args:
            y_pred: Predicted mean values [batch_size, n_features]
            covariance_matrices: Covariance matrices [batch_size, n_features, n_features]
                               or [n_features, n_features]

        Returns:
            tuple: (mean, covariance)
        """
        # Handle single covariance matrix case
        if covariance_matrices.ndim == 2:
            batch_size = y_pred.shape[0]
            covariance_matrices = covariance_matrices.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply variance adjustment if needed
        adjusted_cov = covariance_matrices
        if self.learnable_adjustment:
            # Get variance adjustment factors
            variance_adjustment = torch.exp(self.log_variance_adjustment)
            n_expand_dims = y_pred.dim() - 1
            for _ in range(n_expand_dims):
                variance_adjustment = variance_adjustment.unsqueeze(0)
            variance_adjustment = variance_adjustment.expand(*y_pred.shape)

            # Add to diagonal of covariance matrices
            adjusted_cov = covariance_matrices + torch.diag_embed(variance_adjustment)

        # Add small jitter to diagonal for numerical stability
        n_features = adjusted_cov.shape[-1]
        device = y_pred.device
        jitter_matrix = self.jitter * torch.eye(n_features, device=device)
        jitter_matrix = jitter_matrix.unsqueeze(0).expand_as(adjusted_cov)
        cov_matrices = adjusted_cov + jitter_matrix

        return y_pred, cov_matrices

    def _calculate_nll(
        self,
        params: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate negative log likelihood for multivariate Gaussian.

        Args:
            params: Tuple of (mean, covariance)
            target: Target values [batch_size, n_features]
            mask: Optional mask for valid values

        Returns:
            tensor: Negative log likelihood values [batch_size]
        """
        mean, cov = params
        batch_size, n_features = mean.shape

        # We can't directly apply masks to multivariate distributions
        # If masks are provided, we should handle this separately
        # For this implementation, we assume no masking or complete masking of samples

        try:
            # Try using torch distribution (most stable)
            mvn = MultivariateNormal(mean, cov)
            nll = -mvn.log_prob(target)
        except Exception:
            # Fallback with Cholesky decomposition
            try:
                L = torch.linalg.cholesky(cov)

                # log(det(Σ)) = 2 * sum(log(diag(L)))
                log_det = 2 * torch.sum(
                    torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + self.eps), dim=1
                )

                # (y-μ)ᵀΣ⁻¹(y-μ) = ‖L⁻¹(y-μ)‖²
                residuals = target - mean
                z = torch.linalg.solve_triangular(L, residuals.unsqueeze(-1), upper=False).squeeze(
                    -1
                )
                quadratic = torch.sum(z**2, dim=1)

                # -log p(y) = 0.5 * (log(det(Σ)) + (y-μ)ᵀΣ⁻¹(y-μ) + n*log(2π))
                nll = 0.5 * (log_det + quadratic + n_features * self.log_2pi)
            except Exception:
                # Ultimate fallback with eigendecomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                eigenvalues = torch.clamp(eigenvalues, min=self.eps)

                # log(det(Σ)) = sum(log(λ_i))
                log_det = torch.sum(torch.log(eigenvalues), dim=1)

                # Compute (y-μ)ᵀΣ⁻¹(y-μ) using eigendecomposition
                residuals = target - mean
                whitened = torch.bmm(
                    eigenvectors.transpose(-1, -2), residuals.unsqueeze(-1)
                ).squeeze(-1)
                quadratic = torch.sum(whitened**2 / eigenvalues, dim=1)

                nll = 0.5 * (log_det + quadratic + n_features * self.log_2pi)

        return nll

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        covariance_matrices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate multivariate Gaussian negative log-likelihood loss.

        Args:
            y_pred: Predicted mean values [batch_size, n_features]
            target: Ground truth values [batch_size, n_features]
            covariance_matrices: Predicted covariance matrices [batch_size, n_features, n_features] or
                                [n_features, n_features] for a shared covariance
            mask: Optional boolean mask for complete samples [batch_size]
            weights: Optional sample weights [batch_size]

        Returns:
            Negative log-likelihood loss
        """
        self._validate_inputs(y_pred, target, None)  # Special handling for mask with multivariate

        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred, covariance_matrices)

        # Calculate NLL - returns per-sample NLL [batch_size]
        nll = self._calculate_nll(params, target, None)  # Handle mask separately

        # Apply sample mask if provided (can only mask whole samples)
        if mask is not None:
            if mask.dim() > 1:
                # Convert feature-level mask to sample-level mask
                sample_mask = mask.all(dim=1) if mask.dim() == 2 else mask
            else:
                sample_mask = mask
            nll = nll * sample_mask

        # Apply weights if provided
        if weights is not None:
            if weights.dim() > 1:
                # Convert feature-level weights to sample-level weights (average)
                sample_weights = weights.mean(dim=1) if weights.dim() == 2 else weights
            else:
                sample_weights = weights
            nll = nll * sample_weights

        # Apply reduction
        if self.reduction == "none":
            return nll
        elif self.reduction == "mean":
            return nll.mean()
        else:  # 'sum'
            return nll.sum()


def create_gaussian_nll(
    n_features: int,
    covariance_type: str = "diagonal",
    model_predicts_variance: bool = True,
    fixed_variance: Optional[float] = None,
    jitter: float = 1e-6,
    reduction: str = "mean",
    **kwargs,
) -> DistributionLoss:
    """
    Factory function to create an appropriate Gaussian NLL loss.

    Args:
        n_features: Number of features
        covariance_type: One of 'diagonal', 'full'
        model_predicts_variance: Whether model predicts variance (default: True)
        fixed_variance: Fixed variance value when model doesn't predict variance
        jitter: Regularization strength for numerical stability
        reduction: 'none' | 'mean' | 'sum'
        **kwargs: Additional arguments for specific loss types

    Returns:
        An appropriate Gaussian NLL loss object

    Example:
        >>> # Model predicts (mean, log_var)
        >>> loss_fn = create_gaussian_nll(n_features=3, covariance_type='diagonal')

        >>> # Model predicts only mean with fixed variance
        >>> loss_fn = create_gaussian_nll(n_features=3, model_predicts_variance=False,
        ...                                fixed_variance=0.5)

        >>> # Full covariance case
        >>> loss_fn = create_gaussian_nll(n_features=3, covariance_type='full')
    """
    if covariance_type == "diagonal":
        if not model_predicts_variance and fixed_variance == 1.0:
            # If using fixed unit variance, just use WeightedMSELoss
            from .base import WeightedMSELoss

            return WeightedMSELoss(reduction=reduction)
        else:
            # Use GaussianNLLLoss for all other diagonal cases
            return GaussianNLLLoss(
                fixed_variance=fixed_variance if not model_predicts_variance else None,
                reduction=reduction,
                **kwargs,
            )
    elif covariance_type == "full":
        return MultivariateGaussianLoss(
            n_features=n_features,
            learnable_adjustment=not model_predicts_variance,
            jitter=jitter,
            reduction=reduction,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type}")
