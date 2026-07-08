"""
Poisson-Gaussian mixture loss functions for signal processing applications.

This module provides loss functions that model signals as a mixture of
Poisson process (for count/shot noise) and Gaussian readout noise.

These models are especially useful for:
- Scientific imaging (e.g., microscopy)
- Low-light photography
- Medical imaging (e.g., CT scans, PET)
- Sensor data with both counting noise and electronic noise
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Union, cast

import torch
import torch.nn as nn
from torch.nn import PoissonNLLLoss

from .base import RegressionLoss
from .loss_registry import register_regression_loss


@dataclass(frozen=True)
class PoissonGaussianMixtureConfig:
    """Configuration for PoissonGaussianMixtureLoss."""

    eps: float = 1e-8
    learn_variance: bool = False
    initial_variance: float = 1.0
    min_variance: float = 1e-6
    log_input: bool = False
    mixture_weights: Optional[Union[float, str]] = None
    extra_variance_model: bool = False
    reduction: str = "mean"


@dataclass(frozen=True)
class EnhancedPoissonGaussianConfig:
    """Configuration for EnhancedPoissonGaussianMixtureLoss."""

    gain: Union[float, str] = 1.0
    offset: Union[float, str] = 0.0
    read_noise: Union[float, str] = 1.0
    shot_noise: Union[float, str] = 0.0
    log_input: bool = False
    calibration: bool = False
    reduction: str = "mean"


@dataclass(frozen=True)
class PoissonGaussianLikelihoodRatioConfig:
    """Configuration for PoissonGaussianLikelihoodRatioLoss."""

    log_input: bool = True
    eps: float = 1e-8
    learn_variance: bool = False
    initial_variance: float = 1.0
    reduction: str = "mean"


@register_regression_loss("poisson_gaussian_mixture")
class PoissonGaussianMixtureLoss(RegressionLoss):
    """
    Negative log-likelihood loss for a mixture of Gaussian (readout noise)
    and Poisson (count) noise, common in imaging and signal processing.

    This models the signal as a combination of:
    y ~ Poisson(λ) + N(0, σ²)

    Args:
        eps: Small constant for numerical stability. Default: 1e-8
        learn_variance: Whether to learn the Gaussian variance parameter. Default: False
        initial_variance: Initial value for Gaussian variance. Default: 1.0
        min_variance: Minimum variance value for numerical stability. Default: 1e-6
        log_input: Whether y_pred is provided as log(lambda). Default: False
        mixture_weights: How to weight the mixture components:
            - If None: Equal weighting (0.5, 0.5)
            - If float: Fixed weighting (mixture_weights, 1-mixture_weights)
            - If 'learn': Learn the mixture weights
        extra_variance_model: Whether to include a separate learned variance term. Default: False
        reduction: Method for loss reduction. Default: 'mean'

    Example:
        >>> # For imaging data with fixed noise variance
        >>> loss_fn = PoissonGaussianMixtureLoss(initial_variance=0.1, log_input=True)
        >>> y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))  # log(lambda)
        >>> target = torch.tensor([[11.2, 19.5], [4.8, 16.3]])  # noisy measurements
        >>> loss_fn(y_pred, target)
        tensor(1.8241)

    References
    ----------
    .. [1] Seifert, J., et al. (2023). Maximum-likelihood estimation in ptychography
       in the presence of Poisson-Gaussian noise statistics.
       In *Optics Letters*, 48(19), 5011-5014.
       https://arxiv.org/abs/2308.02436
    """

    def __init__(
        self,
        config: Optional[PoissonGaussianMixtureConfig] = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        if "reduction" not in kwargs and config is None:
            kwargs["reduction"] = reduction
        if config is None:
            config = PoissonGaussianMixtureConfig(**kwargs)
        super().__init__(reduction=config.reduction)
        self.config = config
        self.eps = config.eps
        self.learn_variance = config.learn_variance
        self.initial_variance = config.initial_variance
        self.min_variance = config.min_variance
        self.log_input = config.log_input
        self.mixture_weights = config.mixture_weights
        self.extra_variance_model = config.extra_variance_model

        # Initialize learnable parameters if needed
        if config.learn_variance:
            self.log_variance = nn.Parameter(
                torch.ones(1) * torch.log(torch.tensor(config.initial_variance))
            )

        if config.mixture_weights == "learn":
            self.weight_logit = nn.Parameter(torch.zeros(1))

        # Initialize Poisson loss using torch.nn implementation
        # Note: torch.nn.PoissonNLLLoss has an additional 'full' parameter not present
        # in the local version
        self.poisson_loss = PoissonNLLLoss(
            log_input=config.log_input,
            full=True,  # Set full=True to match behavior of original implementation
            eps=config.eps,
            reduction="none",
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        extra_var: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate the Poisson-Gaussian mixture loss.

        Args:
            y_pred: Predicted values (lambda for Poisson component)
            target: Target values
            mask: Optional mask for invalid values
            weights: Optional tensor of weights for each sample
            extra_var: Optional per-sample variance term

        Returns:
            Loss value
        """
        self._validate_inputs(y_pred, target, mask)

        # Get lambda (rate) parameter
        if self.log_input:
            lam = torch.exp(y_pred)
        else:
            lam = y_pred

        # Calculate Poisson component
        poisson_nll = self.poisson_loss(y_pred, target)

        # Calculate Gaussian component (using squared error)
        residuals = target - lam

        # Get variance parameter
        if self.learn_variance:
            variance = torch.exp(self.log_variance).clamp(min=self.min_variance)
        else:
            variance = torch.tensor(self.initial_variance, device=y_pred.device)

        # Add extra variance if provided
        if extra_var is not None and self.extra_variance_model:
            variance = variance + extra_var

        # Calculate Gaussian NLL: 0.5 * (log(2π) + log(σ²) + (y-μ)²/σ²)
        gaussian_nll = 0.5 * (
            torch.log(2 * math.pi * variance + self.eps) + residuals**2 / (variance + self.eps)
        )

        # Calculate mixture weights
        poisson_weight: float | torch.Tensor
        gaussian_weight: float | torch.Tensor
        if self.mixture_weights is None:
            # Equal weights (0.5, 0.5)
            poisson_weight = 0.5
            gaussian_weight = 0.5
        elif self.mixture_weights == "learn":
            # Learned weights using sigmoid to ensure they sum to 1
            poisson_weight = torch.sigmoid(self.weight_logit)
            gaussian_weight = 1 - poisson_weight
        else:
            # Fixed weights
            fixed_weight = cast(float, self.mixture_weights)
            poisson_weight = fixed_weight
            gaussian_weight = 1 - fixed_weight

        # Calculate weighted mixture
        mixture_nll = poisson_weight * poisson_nll + gaussian_weight * gaussian_nll

        # Apply reduction with mask and weights
        return self._reduce(mixture_nll, mask, weights)


@register_regression_loss("enhanced_poisson_gaussian_mixture")
class EnhancedPoissonGaussianMixtureLoss(RegressionLoss):
    """
    Advanced Poisson-Gaussian mixture loss with additional features for scientific applications.

    This extends the basic Poisson-Gaussian mixture with:
    1. Support for gain/scaling factor in the model
    2. Handling of offset/bias term
    3. Optional calibration parameters
    4. Multiple variance components

    Model: y ~ Poisson(g * λ + b) + N(0, σ₁² + σ₂² * λ)

    Args:
        gain: Fixed gain/scaling factor or 'learn' to make it learnable. Default: 1.0
        offset: Fixed offset/bias or 'learn' to make it learnable. Default: 0.0
        read_noise: Constant variance component (σ₁²). Default: 1.0
        shot_noise: Signal-dependent variance component (σ₂²). Default: 0.0
        log_input: Whether inputs are in log space. Default: False
        calibration: Whether to include calibration parameters. Default: False
        reduction: Method for loss reduction. Default: 'mean'

    Example:
        >>> # For scientific imaging with signal-dependent noise
        >>> loss_fn = EnhancedPoissonGaussianMixtureLoss(
        ...     gain='learn',
        ...     read_noise=0.2,
        ...     shot_noise=0.1
        ... )
        >>> y_pred = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        >>> target = torch.tensor([[9.8, 21.2], [28.5, 41.3]])
        >>> loss_fn(y_pred, target)
        tensor(1.4208)

    References
    ----------
    .. [1] Seifert, J., et al. (2023). Maximum-likelihood estimation in ptychography
       in the presence of Poisson-Gaussian noise statistics.
       In *Optics Letters*, 48(19), 5011-5014.
       https://arxiv.org/abs/2308.02436
    """

    def __init__(
        self,
        config: Optional[EnhancedPoissonGaussianConfig] = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        if "reduction" not in kwargs and config is None:
            kwargs["reduction"] = reduction
        if config is None:
            config = EnhancedPoissonGaussianConfig(**kwargs)
        super().__init__(reduction=config.reduction)
        self.config = config
        self.log_input = config.log_input
        self.calibration = config.calibration
        self.eps = 1e-8

        # Configure gain parameter
        self.learn_gain = config.gain == "learn"
        if self.learn_gain:
            self.log_gain = nn.Parameter(torch.tensor(0.0))  # Initialize with gain=1.0
        else:
            self.register_buffer("fixed_gain", torch.tensor(float(config.gain)))

        # Configure offset parameter
        self.learn_offset = config.offset == "learn"
        if self.learn_offset:
            self.offset = nn.Parameter(torch.tensor(0.0))  # Initialize with offset=0.0
        else:
            self.register_buffer("fixed_offset", torch.tensor(float(config.offset)))

        # Configure read noise (constant variance component)
        self.learn_read_noise = config.read_noise == "learn"
        if self.learn_read_noise:
            # Initialize with log(1.0) = 0.0
            self.log_read_noise = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("fixed_read_noise", torch.tensor(float(config.read_noise)))

        # Configure shot noise (signal-dependent variance component)
        self.learn_shot_noise = config.shot_noise == "learn"
        if self.learn_shot_noise:
            # Initialize with log(small value) for stability
            self.log_shot_noise = nn.Parameter(torch.tensor(-4.0))  # exp(-4) ≈ 0.018
        else:
            self.register_buffer("fixed_shot_noise", torch.tensor(float(config.shot_noise)))

        # Calibration parameters (optional)
        if self.calibration:
            # Initializing intercept (add) and slope (mult) parameters
            self.calib_add = nn.Parameter(torch.tensor(0.0))
            self.calib_mult = nn.Parameter(torch.tensor(1.0))

        # Initialize Poisson NLL using torch.nn implementation
        self.poisson_nll = PoissonNLLLoss(
            log_input=False,  # We handle log_input ourselves
            full=True,  # Set full=True to match behavior of original implementation
            eps=self.eps,
            reduction="none",
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate enhanced Poisson-Gaussian mixture loss.

        Args:
            y_pred: Predicted rate parameters or log(rate)
            target: Ground truth values
            mask: Optional mask for masked loss
            weights: Optional sample weights

        Returns:
            Negative log-likelihood loss
        """
        self._validate_inputs(y_pred, target, mask)

        # Convert from log space if needed
        if self.log_input:
            rate = torch.exp(y_pred)
        else:
            rate = y_pred

        # Apply calibration if enabled
        if self.calibration:
            rate = self.calib_mult * rate + self.calib_add

        # Get gain parameter
        gain: torch.Tensor
        if self.learn_gain:
            gain = torch.exp(self.log_gain)
        else:
            gain = cast(torch.Tensor, self.fixed_gain).to(y_pred.device)

        # Get offset parameter
        offset: torch.Tensor
        if self.learn_offset:
            offset = self.offset
        else:
            offset = cast(torch.Tensor, self.fixed_offset).to(y_pred.device)

        # Apply gain and offset to rate
        scaled_rate = gain * rate + offset
        scaled_rate = torch.clamp(scaled_rate, min=self.eps)  # Ensure positive rate

        # Calculate Poisson component
        poisson_loss = self.poisson_nll(scaled_rate, target)

        # Calculate variance components for Gaussian
        read_var: torch.Tensor
        if self.learn_read_noise:
            read_var = torch.exp(self.log_read_noise)
        else:
            read_var = cast(torch.Tensor, self.fixed_read_noise).to(y_pred.device)

        shot_coef: torch.Tensor
        if self.learn_shot_noise:
            shot_coef = torch.exp(self.log_shot_noise)
        else:
            shot_coef = cast(torch.Tensor, self.fixed_shot_noise).to(y_pred.device)

        # Total variance: read noise + shot noise * signal
        total_var = read_var + shot_coef * scaled_rate
        total_var = torch.clamp(total_var, min=self.eps)

        # Calculate Gaussian component
        squared_error = (target - scaled_rate) ** 2
        gaussian_loss = 0.5 * (
            squared_error / total_var + torch.log(total_var) + math.log(2.0 * math.pi)
        )

        # Combine losses - use equal weight for simplicity
        # Could be extended to learn weights if needed
        loss = 0.5 * (poisson_loss + gaussian_loss)

        # Apply reduction with mask and weights
        return self._reduce(loss, mask, weights)


@register_regression_loss("poisson_gaussian_likelihood_ratio")
class PoissonGaussianLikelihoodRatioLoss(RegressionLoss):
    """
    Likelihood ratio loss combining Poisson count processes and Gaussian noise.

    This implements a weighted combination of Poisson likelihood ratio and Gaussian components,
    suitable for binned data with mixed noise characteristics.

    Args:
        log_input: If True, input is log(λ) rather than λ. Default: True
        eps: Small constant for numerical stability. Default: 1e-8
        learn_variance: Whether to learn the Gaussian variance. Default: False
        initial_variance: Initial value for Gaussian variance. Default: 1.0
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> loss_fn = PoissonGaussianLikelihoodRatioLoss(log_input=True)
        >>> y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        >>> target = torch.tensor([[11.2, 19.5], [4.8, 16.3]])
        >>> loss_fn(y_pred, target)
    """

    def __init__(
        self,
        config: Optional[PoissonGaussianLikelihoodRatioConfig] = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        if "reduction" not in kwargs and config is None:
            kwargs["reduction"] = reduction
        if config is None:
            if "log_input" not in kwargs:
                # Class default
                kwargs["log_input"] = True
            config = PoissonGaussianLikelihoodRatioConfig(**kwargs)
        super().__init__(reduction=config.reduction)
        self.config = config
        self.log_input = config.log_input
        self.eps = config.eps
        self.learn_variance = config.learn_variance
        self.initial_variance = config.initial_variance

        if config.learn_variance:
            self.log_variance = nn.Parameter(
                torch.ones(1) * torch.log(torch.tensor(config.initial_variance))
            )

        # Still import PoissonLikelihoodRatioLoss from local .poisson module
        # This class doesn't exist in torch.nn
        from .poisson import PoissonLikelihoodRatioLoss

        self.poisson_lr_loss = PoissonLikelihoodRatioLoss(
            log_input=config.log_input, eps=config.eps, reduction="none"
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate combined Poisson-Gaussian likelihood ratio loss.

        Args:
            y_pred: Predicted values (lambda for Poisson or log(lambda) if log_input=True)
            target: Target values
            mask: Optional mask for invalid values
            weights: Optional tensor of weights for each sample

        Returns:
            Loss value
        """
        self._validate_inputs(y_pred, target, mask)

        # Calculate Poisson likelihood ratio component
        poisson_lr = self.poisson_lr_loss(y_pred, target)

        # Get lambda parameter
        if self.log_input:
            lam = torch.exp(y_pred)
        else:
            lam = y_pred

        # Calculate Gaussian likelihood ratio component
        residuals = target - lam

        # Get variance parameter
        if self.learn_variance:
            variance = torch.exp(self.log_variance).clamp(min=self.eps)
        else:
            variance = torch.tensor(self.initial_variance, device=y_pred.device)

        # Calculate Gaussian component
        gaussian_lr = residuals**2 / (variance + self.eps)

        # Combine components (equal weighting for simplicity)
        combined_lr = 0.5 * (poisson_lr + gaussian_lr)

        # Apply reduction with mask and weights
        return self._reduce(combined_lr, mask, weights)



