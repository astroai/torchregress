"""
Evidential Regression for uncertainty quantification.

This module implements Evidential Deep Learning for regression, which places
a higher-order prior distribution over the parameters of a Gaussian likelihood.
This enables principled separation of aleatoric and epistemic uncertainty.

The method uses a Normal-Inverse-Gamma (NIG) distribution as a conjugate prior
over the mean and variance of a Gaussian, allowing for closed-form uncertainty
quantification.

References:
    - Amini et al. "Deep Evidential Regression" (NeurIPS 2020)
    - Sensoy et al. "Evidential Deep Learning to Quantify Classification Uncertainty" (NeurIPS 2018)
"""

import math
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from .base import DistributionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("evidential")
class EvidentialRegressionLoss(DistributionLoss):
    """
    Evidential Regression loss using Normal-Inverse-Gamma (NIG) prior.

    This loss function models uncertainty by placing a NIG prior over the
    Gaussian parameters (mean and variance). The model outputs four parameters
    (γ, ν, α, β) that define the NIG distribution:

    - γ (gamma): predicted mean
    - ν (nu): virtual number of observations for the mean
    - α (alpha): shape parameter of inverse gamma (for variance)
    - β (beta): scale parameter of inverse gamma (for variance)

    The NIG distribution provides:
    - Aleatoric uncertainty: from β/(α-1) (expected variance of data)
    - Epistemic uncertainty: from β/(ν(α-1)) (uncertainty about mean)

    Args:
        coeff_nig: Coefficient for NIG regularization term. Default: 0.01
            Higher values encourage less overconfident predictions
        reduction: Loss reduction method ('mean', 'sum', 'none'). Default: 'mean'

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from torchregress.losses import EvidentialRegressionLoss
        >>>
        >>> # Model must output 4 parameters per target dimension
        >>> class EvidentialModel(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.net = nn.Sequential(
        >>>             nn.Linear(10, 64),
        >>>             nn.ReLU(),
        >>>             nn.Linear(64, 4)  # Output: gamma, nu, alpha, beta
        >>>         )
        >>>
        >>>     def forward(self, x):
        >>>         out = self.net(x)
        >>>         gamma = out[:, 0:1]  # mean
        >>>         nu = F.softplus(out[:, 1:2]) + 0.01  # > 0
        >>>         alpha = F.softplus(out[:, 2:3]) + 1.01  # > 1
        >>>         beta = F.softplus(out[:, 3:4]) + 0.01  # > 0
        >>>         return torch.cat([gamma, nu, alpha, beta], dim=1)
        >>>
        >>> model = EvidentialModel()
        >>> loss_fn = EvidentialRegressionLoss(coeff_nig=0.01)
        >>>
        >>> # Training
        >>> x = torch.randn(32, 10)
        >>> y = torch.randn(32, 1)
        >>> y_pred = model(x)
        >>> loss = loss_fn(y_pred, y)
        >>>
        >>> # Inference with uncertainty
        >>> model.eval()
        >>> with torch.no_grad():
        >>>     params = model(x_test)
        >>>     mean, ale_unc, epi_unc = loss_fn.predict_with_uncertainty(params)

    Notes:
        - Model must output 4 values per target dimension: [gamma, nu, alpha, beta]
        - Constraints: nu > 0, alpha > 1, beta > 0
        - Use softplus activation for nu, alpha, beta to ensure constraints
        - Epistemic uncertainty decreases with more evidence (higher nu, alpha)
        - Aleatoric uncertainty is data-dependent and irreducible

    Mathematical Details:
        NIG(μ, σ² | γ, ν, α, β) is defined as:
        - μ | σ² ~ N(γ, σ²/ν)  (normal distribution for mean)
        - σ² ~ InvGamma(α, β)  (inverse gamma for variance)

        Expected values:
        - E[μ] = γ
        - E[σ²] = β/(α-1) for α > 1  (aleatoric uncertainty)
        - Var[μ] = β/(ν(α-1)) for α > 1  (epistemic uncertainty)
    """

    def __init__(
        self,
        coeff_nig: float = 0.01,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.coeff_nig = coeff_nig

        if coeff_nig < 0:
            raise ValueError(f"coeff_nig must be >= 0, got {coeff_nig}")

    def _extract_nig_parameters(self, y_pred: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Extract NIG parameters from model output.

        Args:
            y_pred: Model output [batch_size, 4*n_features]
                Format: [gamma, nu, alpha, beta] for each feature

        Returns:
            Tuple of (gamma, nu, alpha, beta)
            Each with shape [batch_size, n_features]

        Raises:
            ValueError: If output doesn't have correct shape
        """
        if y_pred.dim() < 2 or y_pred.shape[-1] % 4 != 0:
            raise ValueError(
                f"y_pred must have shape [..., 4*n_features], got {y_pred.shape}. "
                f"Evidential regression requires 4 outputs per target dimension."
            )

        n_features = y_pred.shape[-1] // 4

        # Split into 4 parameters
        gamma = y_pred[..., :n_features]  # mean
        nu = y_pred[..., n_features : 2 * n_features]  # evidence for mean
        alpha = y_pred[..., 2 * n_features : 3 * n_features]  # shape for variance
        beta = y_pred[..., 3 * n_features :]  # scale for variance

        return gamma, nu, alpha, beta

    def _nig_nll(
        self,
        target: Tensor,
        gamma: Tensor,
        nu: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> Tensor:
        """
        Negative log-likelihood for Normal-Inverse-Gamma distribution.

        Computes the NLL of observing target under the NIG prior defined by
        (gamma, nu, alpha, beta).

        Args:
            target: Ground truth values [batch_size, n_features]
            gamma: Mean parameter [batch_size, n_features]
            nu: Evidence for mean [batch_size, n_features]
            alpha: Shape of inverse gamma [batch_size, n_features]
            beta: Scale of inverse gamma [batch_size, n_features]

        Returns:
            NLL per sample [batch_size, n_features]
        """
        # Student-t distribution (predictive distribution of NIG)
        # NLL = 0.5 * log(pi/nu) - alpha*log(2*beta) + (alpha+0.5)*log(nu*(target-gamma)^2 + 2*beta)
        #       + loggamma(alpha) - loggamma(alpha+0.5)

        # For numerical stability, we compute in parts
        residual = target - gamma
        residual_sq = residual**2

        # Term 1: log normalizing constant
        nll = 0.5 * torch.log(math.pi / nu)

        # Term 2: -alpha * log(2*beta)
        nll -= alpha * torch.log(2.0 * beta + 1e-6)

        # Term 3: (alpha + 0.5) * log(nu * residual^2 + 2*beta)
        nll += (alpha + 0.5) * torch.log(nu * residual_sq + 2.0 * beta + 1e-6)

        # Term 4: log Gamma functions (approximate for large alpha)
        # loggamma(alpha) - loggamma(alpha + 0.5)
        nll += torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

        return nll

    def _nig_regularizer(
        self,
        target: Tensor,
        gamma: Tensor,
        nu: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> Tensor:
        """
        NIG regularization term to penalize overconfident predictions.

        This term encourages the model to output higher uncertainty (lower nu, alpha)
        when predictions are wrong, preventing overconfidence on incorrect predictions.

        Args:
            target: Ground truth values
            gamma: Predicted mean
            nu: Evidence for mean
            alpha: Shape parameter
            beta: Scale parameter

        Returns:
            Regularization term per sample
        """
        # Error-based regularization
        error = torch.abs(target - gamma)

        # Penalize small nu and alpha (high confidence) when error is large
        # This encourages: high error → high uncertainty
        reg = error * (2.0 * nu + alpha)

        return reg

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Compute evidential regression loss.

        Args:
            y_pred: Model predictions [batch_size, 4*n_features]
                Must contain [gamma, nu, alpha, beta] for each feature
            target: Ground truth values [batch_size, n_features]
            mask: Optional boolean mask for missing values
            weights: Optional sample weights
            **kwargs: Additional arguments

        Returns:
            Evidential loss combining NLL and regularization

        Raises:
            ValueError: If shapes are incompatible
        """
        # Extract NIG parameters
        gamma, nu, alpha, beta = self._extract_nig_parameters(y_pred)

        # Validate target shape
        if target.shape[-1] != gamma.shape[-1]:
            raise ValueError(
                f"Target dimension {target.shape[-1]} doesn't match "
                f"predicted dimension {gamma.shape[-1]}"
            )

        # Compute negative log-likelihood
        nll = self._nig_nll(target, gamma, nu, alpha, beta)

        # Compute regularization term
        reg = self._nig_regularizer(target, gamma, nu, alpha, beta)

        # Total loss: NLL + coefficient * regularization
        loss = nll + self.coeff_nig * reg

        # Apply weights if provided
        if weights is not None:
            if weights.dim() == 1:
                # Expand to match loss shape
                weights = weights.view(-1, 1).expand_as(loss)
            loss = loss * weights

        return self._reduce_with_mask(loss, mask, None)

    def predict_with_uncertainty(self, y_pred: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get predictions with decomposed uncertainty.

        Args:
            y_pred: Model output [batch_size, 4*n_features]

        Returns:
            Tuple of (mean, aleatoric_uncertainty, epistemic_uncertainty):
                - mean: Predicted mean E[μ] = γ
                - aleatoric_uncertainty: E[σ²] = β/(α-1) (data noise)
                - epistemic_uncertainty: Var[μ] = β/(ν(α-1)) (model uncertainty)

        Example:
            >>> model.eval()
            >>> with torch.no_grad():
            >>>     params = model(x_test)
            >>>     mean, ale, epi = loss_fn.predict_with_uncertainty(params)
            >>>
            >>> # Total uncertainty
            >>> total_unc = ale + epi
            >>>
            >>> # Prediction intervals (assuming Gaussian)
            >>> std_total = torch.sqrt(total_unc)
            >>> lower = mean - 1.96 * std_total  # 95% CI
            >>> upper = mean + 1.96 * std_total
        """
        gamma, nu, alpha, beta = self._extract_nig_parameters(y_pred)

        # Mean prediction
        mean = gamma

        # Aleatoric uncertainty: expected variance of the data
        # E[σ²] = β / (α - 1) for α > 1
        aleatoric = beta / (alpha - 1.0 + 1e-6)

        # Epistemic uncertainty: uncertainty about the mean
        # Var[μ] = β / (ν(α - 1)) for α > 1
        epistemic = beta / (nu * (alpha - 1.0) + 1e-6)

        return mean, aleatoric, epistemic

    def sample_predictions(self, y_pred: Tensor, n_samples: int = 100) -> Tensor:
        """
        Sample predictions from the evidential distribution.

        Samples from the NIG prior over (μ, σ²) and then from N(μ, σ²).

        Args:
            y_pred: Model output [batch_size, 4*n_features]
            n_samples: Number of samples to generate. Default: 100

        Returns:
            Samples [n_samples, batch_size, n_features]

        Example:
            >>> with torch.no_grad():
            >>>     params = model(x_test)
            >>>     samples = loss_fn.sample_predictions(params, n_samples=1000)
            >>>
            >>> # Compute statistics
            >>> mean = samples.mean(0)
            >>> std = samples.std(0)
            >>> percentiles = torch.quantile(samples, torch.tensor([0.05, 0.95]), dim=0)
        """
        gamma, nu, alpha, beta = self._extract_nig_parameters(y_pred)

        # Sample variance from Inverse Gamma
        # InvGamma(α, β) = 1 / Gamma(α, 1/β)
        # Using torch.distributions would be cleaner but more heavyweight
        # We use the property: if X ~ Gamma(α, β), then 1/X ~ InvGamma(α, β)

        # Sample from Gamma(alpha, beta)
        # Note: torch.distributions.Gamma uses rate parameterization (1/scale)
        from torch.distributions import Gamma, Normal

        # Shape for sampling: [n_samples, batch_size, n_features]
        alpha_expanded = alpha.unsqueeze(0).expand(n_samples, -1, -1)
        beta_expanded = beta.unsqueeze(0).expand(n_samples, -1, -1)

        # Sample variance: σ² ~ InvGamma(α, β)
        gamma_dist = Gamma(alpha_expanded, beta_expanded)
        variance_samples = 1.0 / gamma_dist.sample()  # InvGamma

        # Sample mean: μ | σ² ~ N(γ, σ²/ν)
        gamma_expanded = gamma.unsqueeze(0).expand(n_samples, -1, -1)
        nu_expanded = nu.unsqueeze(0).expand(n_samples, -1, -1)

        mean_variance = variance_samples / nu_expanded
        mean_dist = Normal(gamma_expanded, torch.sqrt(mean_variance))
        mean_samples = mean_dist.sample()

        # Sample final predictions: y | μ, σ² ~ N(μ, σ²)
        pred_dist = Normal(mean_samples, torch.sqrt(variance_samples))
        prediction_samples = pred_dist.sample()

        return prediction_samples

    def predict_interval(
        self,
        y_pred: Tensor,
        confidence: float = 0.95,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute prediction intervals using the Student-t predictive distribution.

        The predictive distribution for a Normal-Inverse-Gamma prior is a
        Student-t distribution with:
        - Location: γ (predicted mean)
        - Scale: sqrt(β * (1 + 1/ν) / (α - 1))
        - Degrees of freedom: 2α

        This method correctly accounts for the heavier tails of the Student-t
        compared to a Gaussian, especially when α is small (low evidence).

        Args:
            y_pred: Model output [batch_size, 4*n_features]
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower, upper) bounds, each [batch_size, n_features]

        Example:
            >>> model.eval()
            >>> with torch.no_grad():
            >>>     params = model(x_test)
            >>>     lower, upper = loss_fn.predict_interval(params, confidence=0.95)
            >>>
            >>> # Check coverage
            >>> in_interval = (y_test >= lower) & (y_test <= upper)
            >>> coverage = in_interval.float().mean()  # Should be ~0.95

        Notes:
            - For α close to 1, the Student-t has very heavy tails
            - With α > 30, the Student-t approaches Gaussian (1.96*std works)
            - The scale parameter differs from sqrt(aleatoric + epistemic)
        """
        gamma, nu, alpha, beta = self._extract_nig_parameters(y_pred)

        # Degrees of freedom for Student-t
        df = 2 * alpha

        # Scale parameter for Student-t predictive
        # This is sqrt(β * (1 + 1/ν) / (α - 1))
        # which differs from sqrt(aleatoric + epistemic)
        scale = torch.sqrt(beta * (1 + 1 / nu) / (alpha - 1.0 + 1e-6))

        # Student-t quantile using scipy
        # For large batches, this is computed element-wise
        from scipy import stats as scipy_stats

        # Compute t-quantile for each sample (may have different df)
        df_np = df.detach().cpu().numpy()
        t_quantile = scipy_stats.t.ppf((1 + confidence) / 2, df_np)
        t_quantile = torch.tensor(t_quantile, device=y_pred.device, dtype=y_pred.dtype)

        # Compute intervals
        lower = gamma - t_quantile * scale
        upper = gamma + t_quantile * scale

        return lower, upper

    def predict_interval_gaussian(
        self,
        y_pred: Tensor,
        confidence: float = 0.95,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute prediction intervals using Gaussian approximation.

        This uses mean ± z * sqrt(aleatoric + epistemic), which is only
        accurate when α is large (df > 30). For proper intervals, use
        `predict_interval()` instead.

        Args:
            y_pred: Model output [batch_size, 4*n_features]
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower, upper) bounds, each [batch_size, n_features]

        Warning:
            This approximation underestimates interval width when α is small,
            leading to under-coverage. Use predict_interval() for accurate
            coverage, especially with low-evidence models.
        """
        from scipy import stats as scipy_stats

        mean, aleatoric, epistemic = self.predict_with_uncertainty(y_pred)
        total_std = torch.sqrt(aleatoric + epistemic)

        z = scipy_stats.norm.ppf((1 + confidence) / 2)
        lower = mean - z * total_std
        upper = mean + z * total_std

        return lower, upper
