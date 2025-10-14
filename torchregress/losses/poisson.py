"""
Poisson regression loss functions.

This module provides specialized loss functions for count data regression based on
the Poisson distribution and its variants, which are appropriate for modeling:
- Count data (non-negative integers)
- Rate data (events per unit time/space)
- Rare event occurrences

For standard Poisson Negative Log-Likelihood, use WeightedPoissonNLLLoss
from the base module instead.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from .base import RegressionLoss


class PoissonDevianceLoss(RegressionLoss):
    """
    Poisson Deviance loss function, also known as G-statistic.

    The deviance is a measure of goodness-of-fit for Poisson regression models,
    defined as the difference between the log-likelihood of a saturated model
    and the current model.

    Args:
        log_input: If True, input is log(λ) rather than λ. Default: True
        eps: Small constant for numerical stability. Default: 1e-8
        learn_variance: Whether to use a learnable variance parameter. Default: False
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> loss_fn = PoissonDevianceLoss()
        >>> y_pred = torch.log(torch.tensor([1.0, 2.0, 3.0]))  # log(lambda)
        >>> target = torch.tensor([0.0, 1.0, 4.0])  # counts
        >>> loss_fn(y_pred, target)
    """

    def __init__(
        self,
        log_input: bool = True,
        eps: float = 1e-8,
        learn_variance: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.log_input = log_input
        self.eps = eps
        self.learn_variance = learn_variance

        if learn_variance:
            # Initialize learnable variance parameter as log(variance)
            self.log_variance = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate Poisson deviance loss.

        Args:
            y_pred: Predicted rate parameters λ or log(λ) [batch_size, n_features]
            target: Ground truth count values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]

        Returns:
            Poisson deviance loss value
        """
        self._validate_inputs(y_pred, target, mask)

        # Ensure non-negative targets
        if torch.any(target < 0):
            raise ValueError("Target values must be non-negative for Poisson regression")

        # Convert log_input to predicted rate
        if self.log_input:
            rate = torch.exp(y_pred)
        else:
            rate = y_pred

        # Initialize loss tensor
        loss = torch.zeros_like(target)

        # Calculate Poisson deviance: λ - y + y * log(y/λ)
        # First part: λ - y for all values
        loss = rate - target

        # Second part: y * log(y/λ) only for y > 0
        non_zero = target > 0
        if torch.any(non_zero):
            target_nz = target[non_zero]
            rate_nz = rate[non_zero] + self.eps
            loss[non_zero] = loss[non_zero] + target_nz * torch.log(target_nz / rate_nz)

        # Apply variance adjustment if using learnable variance
        if self.learn_variance:
            variance = torch.exp(self.log_variance)
            loss = loss / (variance + self.eps) + 0.5 * torch.log(variance + self.eps)

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


class PoissonLikelihoodRatioLoss(RegressionLoss):
    """
    Poisson Likelihood Ratio Loss for binned data, also known as Baker-Cousins Loss.

    This implements the likelihood ratio test statistic for Poisson binned data
    as defined in the PDG Review of Statistics:

    -2ln(λ) = 2∑[λᵢ - nᵢ + nᵢln(nᵢ/λᵢ)]

    Where:
    - nᵢ are the observed counts (target)
    - λᵢ are the expected counts (predictions)
    - For nᵢ = 0, the term nᵢln(nᵢ/λᵢ) = 0, so the contribution is just 2λᵢ

    This loss is useful for:
    - Histogram fitting in physics
    - Maximum likelihood estimation with binned data
    - Goodness-of-fit testing for count data

    Args:
        log_input: If True, input is log(λ) rather than λ. Default: True
        eps: Small constant for numerical stability. Default: 1e-8
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> # For binned histogram data
        >>> loss_fn = PoissonLikelihoodRatioLoss(log_input=False)
        >>> y_pred = torch.tensor([10.0, 20.0, 15.0])  # expected counts
        >>> target = torch.tensor([12.0, 18.0, 14.0])  # observed counts
        >>> loss_fn(y_pred, target)
        tensor(0.4749)
    """

    def __init__(self, log_input: bool = True, eps: float = 1e-8, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.log_input = log_input
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate Poisson likelihood ratio loss.

        Args:
            y_pred: Expected counts λᵢ (or log(λᵢ) if log_input=True) [batch_size, n_features]
            target: Observed counts nᵢ [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]

        Returns:
            Poisson likelihood ratio statistic value
        """
        self._validate_inputs(y_pred, target, mask)

        # Ensure non-negative targets
        if torch.any(target < 0):
            raise ValueError("Observed counts must be non-negative for Poisson statistics")

        # Convert log_input to expected counts
        if self.log_input:
            expected = torch.exp(y_pred)
        else:
            expected = y_pred

        # Ensure expected counts are positive
        expected = expected + self.eps

        # Create the loss tensor with the same shape as target
        loss = torch.zeros_like(target)

        # Calculate Poisson likelihood ratio statistic: 2∑[λᵢ - nᵢ + nᵢln(nᵢ/λᵢ)]
        # Calculate first part: 2(λᵢ - nᵢ)
        loss = 2.0 * (expected - target)

        # Add the 2nᵢln(nᵢ/λᵢ) term for bins where nᵢ > 0
        non_zero_mask = target > 0
        if torch.any(non_zero_mask):
            target_nz = target[non_zero_mask]
            expected_nz = expected[non_zero_mask]
            loss[non_zero_mask] += 2.0 * target_nz * torch.log(target_nz / expected_nz)

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


class ZeroInflatedPoissonNLLLoss(RegressionLoss):
    """
    Zero-Inflated Poisson Negative Log-Likelihood.

    This loss is suitable for count data with excess zeros beyond what would
    be expected under a standard Poisson distribution.

    Args:
        log_input: If True, input is log(λ) rather than λ. Default: True
        eps: Small constant for numerical stability. Default: 1e-8
        learn_variance: Whether to learn a global variance parameter. Default: False
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> loss_fn = ZeroInflatedPoissonNLLLoss()
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])  # lambda values
        >>> pi_logits = torch.tensor([-1.0, 0.0, 1.0])  # zero-inflation logits
        >>> target = torch.tensor([0.0, 0.0, 3.0])  # counts
        >>> loss_fn(y_pred, target, pi_logits)
    """

    def __init__(
        self,
        log_input: bool = True,
        eps: float = 1e-8,
        learn_variance: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.log_input = log_input
        self.eps = eps
        self.learn_variance = learn_variance

        if self.learn_variance:
            self.log_variance = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate Zero-Inflated Poisson NLL loss.

        Args:
            y_pred: Predicted rate parameters λ or log(λ) [batch_size, n_features]
            target: Ground truth count values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]
            **kwargs: Additional arguments. Can include 'pi_logits' for zero-inflation logits.

        Returns:
            Zero-inflated Poisson NLL loss value
        """
        # Extract pi_logits from kwargs
        pi_logits = kwargs.get("pi_logits", None)

        # If still None, try to extract from y_pred
        if pi_logits is None:
            if y_pred.shape[-1] % 2 == 0:
                # Assume y_pred contains both lambda and pi_logits
                half_dim = y_pred.shape[-1] // 2
                pi_logits = y_pred[..., half_dim:]
                y_pred = y_pred[..., :half_dim]
            else:
                raise ValueError(
                    "pi_logits must be provided either as argument or extracted from y_pred"
                )

        self._validate_inputs(y_pred, target, mask)

        # Ensure non-negative targets
        if torch.any(target < 0):
            raise ValueError("Target values must be non-negative for Poisson regression")

        # Convert log_input to predicted rate
        if self.log_input:
            rate = torch.exp(y_pred)
        else:
            rate = y_pred

        # Ensure rate and pi_logits have the same shape as target
        if rate.shape != target.shape:
            raise ValueError(f"Rate shape {rate.shape} must match target shape {target.shape}")
        if pi_logits.shape != target.shape:
            raise ValueError(
                f"Pi logits shape {pi_logits.shape} must match target shape {target.shape}"
            )

        # Calculate zero-inflation probability from logits
        pi = torch.sigmoid(pi_logits)

        # Initialize loss tensor
        loss = torch.zeros_like(target)

        # For zero targets: -log(pi + (1-pi) * exp(-lambda))
        zero_mask = target == 0
        if torch.any(zero_mask):
            exp_neg_rate = torch.exp(-rate)
            loss[zero_mask] = -torch.log(
                pi[zero_mask] + (1 - pi[zero_mask]) * exp_neg_rate[zero_mask] + self.eps
            )

        # For non-zero targets: -log(1-pi) + lambda - y*log(lambda) + log(y!)
        non_zero_mask = target > 0
        if torch.any(non_zero_mask):
            # Calculate log factorial of target values
            log_factorial = torch.lgamma(target[non_zero_mask] + 1)

            loss[non_zero_mask] = (
                -torch.log(1 - pi[non_zero_mask] + self.eps)
                + rate[non_zero_mask]
                - target[non_zero_mask] * torch.log(rate[non_zero_mask] + self.eps)
                + log_factorial
            )

        # Apply variance adjustment if using learnable variance
        if self.learn_variance:
            variance = torch.exp(self.log_variance)
            loss = loss / (variance + self.eps) + 0.5 * torch.log(variance + self.eps)

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


class NegativeBinomialNLLLoss(RegressionLoss):
    """
    Negative Binomial Negative Log-Likelihood loss.

    This loss is suitable for overdispersed count data where variance > mean.
    The negative binomial distribution has parameters μ (mean) and θ (dispersion).

    Args:
        learn_theta: Whether to learn the dispersion parameter θ. Default: False
        eps: Small constant for numerical stability. Default: 1e-8
        min_theta: Minimum value for θ parameter. Default: 1e-6
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> loss_fn = NegativeBinomialNLLLoss(learn_theta=True)
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])  # mean values
        >>> target = torch.tensor([0.0, 3.0, 5.0])  # counts
        >>> loss_fn(y_pred, target)
    """

    def __init__(
        self,
        learn_theta: bool = False,
        eps: float = 1e-8,
        min_theta: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.eps = eps
        self.min_theta = min_theta
        self.learn_theta = learn_theta

        if learn_theta:
            # Initialize θ as a learnable parameter (using log parameterization for positivity)
            self.log_theta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate Negative Binomial NLL loss.

        Args:
            y_pred: Predicted mean values μ [batch_size, n_features]
            target: Ground truth count values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]
            **kwargs: Additional arguments. Can include 'theta' for dispersion parameter.

        Returns:
            Negative Binomial NLL loss value
        """
        # Extract theta from kwargs
        theta = kwargs.get("theta", None)

        self._validate_inputs(y_pred, target, mask)

        # Ensure non-negative targets
        if torch.any(target < 0):
            raise ValueError("Target values must be non-negative for count regression")

        # Get dispersion parameter θ
        if self.learn_theta:
            theta_value = torch.exp(self.log_theta).clamp(min=self.min_theta)
        elif theta is not None:
            if isinstance(theta, (float, int)):
                theta_value = torch.tensor(float(theta), device=y_pred.device)
            else:
                theta_value = theta.clamp(min=self.min_theta)
        else:
            # Default θ value if not provided or learned
            theta_value = torch.tensor(1.0, device=y_pred.device)

        # Ensure positive mean predictions
        mu = torch.clamp(y_pred, min=self.eps)

        # Calculate Negative Binomial NLL
        # p = θ/(θ+μ), r = θ
        # NLL = -log(Γ(y+r)/Γ(y+1)Γ(r)) - y*log(p/(1-p)) - r*log(p)

        # Using log-probability formulation:
        log_theta = torch.log(theta_value + self.eps)
        logit_p = log_theta - torch.log(mu + theta_value + self.eps)  # logit(p) = log(p/(1-p))
        p = torch.sigmoid(logit_p)  # p = θ/(θ+μ)

        # Compute log factorial using Stirling's approximation for large values
        log_gamma_ypr = torch.lgamma(target + theta_value + self.eps)
        log_gamma_y1 = torch.lgamma(target + 1.0 + self.eps)
        log_gamma_r = torch.lgamma(theta_value + self.eps)

        # Negative log likelihood
        loss = -(
            log_gamma_ypr
            - log_gamma_y1
            - log_gamma_r
            + theta_value * (torch.log(p + self.eps))
            + target * torch.log(1 - p + self.eps)
        )

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)
