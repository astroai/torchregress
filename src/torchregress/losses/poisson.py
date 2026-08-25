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
import torch.nn.functional as F

from .base import RegressionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("poisson_deviance")
class PoissonDevianceLoss(RegressionLoss):
    """
    Poisson Deviance loss function, also known as G-statistic.

    The deviance is a measure of goodness-of-fit for Poisson regression models,
    defined as the difference between the log-likelihood of a saturated model
    and the current model.

    Args:
        log_input: If True, input is log(λ) rather than λ. Default: True
        eps: Small constant for numerical stability. Default: 1e-8
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
        reduction: str = "mean",
    ) -> None:
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
        # A3: clamp in log space before exponentiation to avoid overflow
        if self.log_input:
            rate = torch.exp(y_pred.clamp(max=30.0))
        else:
            rate = y_pred

        # Calculate Poisson deviance: λ - y + y * log(y/λ)
        # y * log(y/λ) = 0 for y = 0. Use torch.where to avoid log(0) and NaNs.
        target_safe = torch.where(target > 0, target, torch.ones_like(target))
        term = target * torch.log(target_safe / (rate + self.eps))
        loss = rate - target + torch.where(target > 0, term, 0.0)

        # Apply reduction with mask and weights
        return self._reduce(loss, mask, weights)


@register_regression_loss("poisson_likelihood_ratio")
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

    References
    ----------
    .. [1] Baker, S., & Cousins, R. D. (1984). Clarification of the use of chi-square
       and likelihood functions in fits to histograms.
       In *Nuclear Instruments and Methods in Physics Research*, 221(2), 437-442.
       https://doi.org/10.1016/0167-5087(84)90016-4
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

        # Calculate Poisson likelihood ratio statistic: 2∑[λᵢ - nᵢ + nᵢln(nᵢ/λᵢ)]
        # Use torch.where to avoid log(0) and NaNs for bins where target n_i = 0.
        target_safe = torch.where(target > 0, target, torch.ones_like(target))
        term = 2.0 * target * torch.log(target_safe / expected)
        loss = 2.0 * (expected - target) + torch.where(target > 0, term, 0.0)

        # Apply reduction with mask and weights
        return self._reduce(loss, mask, weights)


@register_regression_loss("zip")
class ZeroInflatedPoissonNLLLoss(RegressionLoss):
    """
    Zero-Inflated Poisson Negative Log-Likelihood.

    This loss is suitable for count data with excess zeros beyond what would
    be expected under a standard Poisson distribution.

    Args:
        log_input: If True, input is log(λ) rather than λ. Default: True
        eps: Small constant for numerical stability. Default: 1e-8
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Forward REQUIRES ``pi_logits`` (zero-inflation logits, same shape as the
    target); it is no longer guessed from an even/odd split of ``y_pred`` (A7).

    Example:
        >>> loss_fn = ZeroInflatedPoissonNLLLoss()
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])  # lambda values
        >>> pi_logits = torch.tensor([-1.0, 0.0, 1.0])  # zero-inflation logits
        >>> target = torch.tensor([0.0, 0.0, 3.0])  # counts
        >>> loss_fn(y_pred, target, pi_logits=pi_logits)

    References
    ----------
    .. [1] Lambert, D. (1992). Zero-Inflated Poisson Regression, with an
       Application to Defects in Manufacturing. Technometrics, 34(1), 1-14.
    """

    def __init__(
        self,
        log_input: bool = True,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.log_input = log_input
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        pi_logits: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate Zero-Inflated Poisson NLL loss.

        Args:
            y_pred: Predicted rate parameters λ or log(λ) [batch_size, n_features]
            target: Ground truth count values [batch_size, n_features]
            pi_logits: REQUIRED zero-inflation logits [batch_size, n_features] (A7).
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]

        Returns:
            Zero-inflated Poisson NLL loss value
        """
        if pi_logits is None and kwargs.get("pi_logits") is not None:
            pi_logits = kwargs["pi_logits"]
        if pi_logits is None:
            raise ValueError("pi_logits must be provided")

        self._validate_inputs(y_pred, target, mask)

        # Ensure non-negative targets
        if torch.any(target < 0):
            raise ValueError("Target values must be non-negative for Poisson regression")

        # Convert log_input to predicted rate
        # A3: clamp in log space before exponentiation to avoid overflow
        if self.log_input:
            rate = torch.exp(y_pred.clamp(max=30.0))
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

        # For zero targets: -log(pi + (1-pi) * exp(-lambda))
        exp_neg_rate = torch.exp(-rate)
        loss_zero = -torch.log(pi + (1.0 - pi) * exp_neg_rate + self.eps)

        # For non-zero targets: -log(1-pi) + lambda - y*log(lambda) + log(y!)
        # Use target_safe to avoid negative values or zero in lgamma and log
        target_safe = torch.where(target > 0, target, torch.ones_like(target))
        log_factorial = torch.lgamma(target_safe + 1.0)
        loss_nonzero = (
            -torch.log(1.0 - pi + self.eps)
            + rate
            - target * torch.log(rate + self.eps)
            + log_factorial
        )

        loss = torch.where(target == 0, loss_zero, loss_nonzero)

        # Apply reduction with mask and weights
        return self._reduce(loss, mask, weights)


@register_regression_loss("nbinom")
class NegativeBinomialNLLLoss(RegressionLoss):
    """
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
        logit_p = log_theta - torch.log(mu + self.eps)  # logit(p) = log(θ/μ) → p = θ/(θ+μ)

        # Compute log factorial using Stirling's approximation for large values
        log_gamma_ypr = torch.lgamma(target + theta_value + self.eps)
        log_gamma_y1 = torch.lgamma(target + 1.0 + self.eps)
        log_gamma_r = torch.lgamma(theta_value + self.eps)

        # Negative log likelihood.
        # A7: use -logsigmoid for log(p)/log(1-p) — numerically stable where
        # p is near 0 or 1, unlike the previous log(p + eps) form.
        loss = -(
            log_gamma_ypr
            - log_gamma_y1
            - log_gamma_r
            - theta_value * F.logsigmoid(-logit_p)
            - target * F.logsigmoid(logit_p)
        )

        # Apply reduction with mask and weights
        return self._reduce(loss, mask, weights)
