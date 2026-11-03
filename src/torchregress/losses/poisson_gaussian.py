"""
Poisson-Gaussian losses.

Two different likelihoods share the "Poisson-Gaussian" name:

- :class:`PoissonGaussianLoss` is the detector (convolution) model: a Poisson
  count ``K ~ Poisson(rate)`` read out as ``y = gain * K + offset + eps`` with
  Gaussian read noise ``eps ~ N(0, read_noise_variance)``. Use it for shot noise
  plus read noise (CCD/CMOS, photon counting, microscopy, low-dose imaging).
- :class:`PoissonGaussianContaminationMixtureLoss` and
  :class:`EnhancedPoissonGaussianContaminationMixtureLoss` are two-component mixtures
  ``w * Pois(y; .) + (1 - w) * N(y; .)``: each sample is drawn from *either* a
  Poisson *or* a Gaussian law. That is a contamination model, not the
  shot-plus-read-noise model.
  The former names ``PoissonGaussianMixtureLoss`` and
  ``EnhancedPoissonGaussianMixtureLoss`` remain as deprecated aliases and will be
  removed in the next minor release.
- :class:`PoissonGaussianLikelihoodRatioLoss` averages a Poisson deviance and a
  scaled squared error; it is not a likelihood of either model.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union, cast

import torch
import torch.nn as nn
from torch.nn import PoissonNLLLoss

from ._poisson_gaussian_kernel import poisson_gaussian_nll
from .base import RegressionLoss
from .loss_registry import register_deprecated_loss_key, register_regression_loss


def _safe_log_weight(weight: "float | torch.Tensor", like: torch.Tensor) -> torch.Tensor:
    """log(w) with a floor of 1e-12, on ``like``'s device/dtype (A1)."""
    if not torch.is_tensor(weight):
        weight = torch.tensor(float(weight), dtype=like.dtype, device=like.device)
    return torch.log(weight.clamp_min(1e-12))


@dataclass(frozen=True)
class PoissonGaussianConfig:
    """Configuration for PoissonGaussianLoss."""

    gain: float = 1.0
    offset: float = 0.0
    read_noise_variance: float = 1.0
    learn_gain: bool = False
    learn_offset: bool = False
    learn_read_noise: bool = False
    log_input: bool = False
    tail_sigma: float = 12.0
    max_terms: int = 4096
    reduction: str = "mean"


@dataclass(frozen=True)
class PoissonGaussianContaminationMixtureConfig:
    """Configuration for PoissonGaussianContaminationMixtureLoss."""

    eps: float = 1e-8
    learn_variance: bool = False
    initial_variance: float = 1.0
    min_variance: float = 1e-6
    log_input: bool = False
    mixture_weights: Optional[Union[float, str]] = None
    extra_variance_model: bool = False
    reduction: str = "mean"


@dataclass(frozen=True)
class EnhancedPoissonGaussianContaminationMixtureConfig:
    """Configuration for EnhancedPoissonGaussianContaminationMixtureLoss."""

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


@register_regression_loss("poisson_gaussian")
class PoissonGaussianLoss(RegressionLoss):
    r"""
    Poisson-Gaussian detector NLL (shot noise convolved with read noise).

    The latent count is Poisson and the measured value given the count is
    Gaussian:

        K ~ Poisson(λ),   y = g·K + b + ε,   ε ~ N(0, σ²)

        p(y) = Σ_k Pois(k; λ) · N(y; g·k + b, σ²)

    ``λ`` is the expected count (electrons/photons), ``g`` the gain in output
    units per count, ``b`` a signed offset (bias, background in output units)
    and ``σ²`` the read-noise variance in output units squared. ``target`` is
    in output units and may be non-integer or negative.

    The sum runs over a per-element count window that keeps at least
    ``tail_sigma`` standard deviations of the Poisson prior and of the
    observation-conditioned posterior over ``k`` (so outliers far from
    ``g·λ + b`` are scored on their own window). In-distribution windows are
    renormalised, so the returned law is a normalised density. Elements are
    processed in chunks of about ``2**22`` count terms; first derivatives are
    analytic posterior moments and need O(batch) memory. The window is chosen
    without gradient; changing it changes the finite mixture only by the
    discarded tail mass beyond ``tail_sigma`` standard deviations.

    Args:
        gain: Gain ``g`` (> 0), or its initial value when ``learn_gain``. Default: 1.0
        offset: Offset ``b``, or its initial value when ``learn_offset``. Default: 0.0
        read_noise_variance: Read-noise variance ``σ²`` (> 0), or its initial
            value when ``learn_read_noise``. Default: 1.0
        learn_gain: Learn ``log g`` as a parameter. Default: False
        learn_offset: Learn ``b`` as a parameter. Default: False
        learn_read_noise: Learn ``log σ²`` as a parameter. Default: False
        log_input: ``y_pred`` is ``log λ`` instead of ``λ``. Default: False
        tail_sigma: Support radius in standard deviations. Default: 12.0
        max_terms: Maximum count-window width; wider windows raise
            ``ValueError`` (use the Gaussian limit there). Default: 4096
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    ``forward`` also accepts per-element ``gain``, ``offset`` and
    ``read_noise_variance`` tensors (broadcastable to ``target``) that replace
    the module values for that call, e.g. per-pixel calibration maps.

    At high counts the law tends to ``N(g·λ + b, g²·λ + σ²)``; as ``σ → 0``
    with ``g = 1, b = 0`` it tends to ``Poisson(λ)`` on integer targets.

    Example:
        >>> loss_fn = PoissonGaussianLoss(gain=2.0, offset=100.0, read_noise_variance=9.0)
        >>> rate = torch.tensor([[3.0, 40.0]], requires_grad=True)
        >>> target = torch.tensor([[104.5, 181.0]])
        >>> loss_fn(rate, target).backward()

    References
    ----------
    .. [1] D. L. Snyder, A. M. Hammoud, R. L. White (1993). Image recovery from
       data acquired with a charge-coupled-device camera. JOSA A 10(5), 1014-1023.
       https://doi.org/10.1364/JOSAA.10.001014
    .. [2] J. Seifert et al. (2023). Maximum-likelihood estimation in
       ptychography in the presence of Poisson-Gaussian noise statistics.
       Optics Letters 48(22), 6027-6030. https://arxiv.org/abs/2308.02436
    """

    def __init__(
        self,
        config: Optional[PoissonGaussianConfig] = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        if "reduction" not in kwargs and config is None:
            kwargs["reduction"] = reduction
        if config is None:
            config = PoissonGaussianConfig(**kwargs)
        super().__init__(reduction=config.reduction)
        if not (math.isfinite(config.gain) and config.gain > 0):
            raise ValueError("gain must be finite and positive")
        if not (math.isfinite(config.read_noise_variance) and config.read_noise_variance > 0):
            raise ValueError("read_noise_variance must be finite and positive")
        if not math.isfinite(config.offset):
            raise ValueError("offset must be finite")
        if not (math.isfinite(config.tail_sigma) and config.tail_sigma > 0):
            raise ValueError("tail_sigma must be finite and positive")
        if int(config.max_terms) < 2:
            raise ValueError("max_terms must be at least 2")
        self.config = config
        self.log_input = config.log_input
        self.tail_sigma = float(config.tail_sigma)
        self.max_terms = int(config.max_terms)
        # Fixed values stay Python floats (exact in float64, movable to any device);
        # learnable ones are unconstrained parameters initialised at those values.
        if config.learn_gain:
            self.log_gain = nn.Parameter(torch.tensor(math.log(config.gain)))
        if config.learn_offset:
            self.offset = nn.Parameter(torch.tensor(float(config.offset)))
        if config.learn_read_noise:
            self.log_read_noise_variance = nn.Parameter(
                torch.tensor(math.log(config.read_noise_variance))
            )

    def detector_parameters(
        self, like: torch.Tensor
    ) -> "tuple[torch.Tensor | float, torch.Tensor | float, torch.Tensor | float]":
        """Current ``(gain, offset, read_noise_variance)`` in ``like``'s dtype."""
        cfg = self.config
        gain = self.log_gain.to(like.dtype).exp() if cfg.learn_gain else cfg.gain
        offset = self.offset.to(like.dtype) if cfg.learn_offset else cfg.offset
        var = (
            self.log_read_noise_variance.to(like.dtype).exp()
            if cfg.learn_read_noise
            else cfg.read_noise_variance
        )
        return gain, offset, var

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        gain: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        read_noise_variance: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate the Poisson-Gaussian detector NLL.

        Args:
            y_pred: Expected count ``λ >= 0`` (or ``log λ`` if ``log_input``)
            target: Measured values in output units
            gain, offset, read_noise_variance: Optional per-call overrides,
                broadcastable to ``target`` (pass by keyword)
            mask: Optional boolean mask (False = missing; may hold NaN there)
            weights: Optional sample weights

        Returns:
            Loss value according to ``reduction``
        """
        self._validate_inputs(y_pred, target, mask)
        if mask is not None:
            # Masked entries may hold NaN/Inf; replace them before the kernel's
            # fail-closed checks (and before ``exp``, so no NaN reaches the
            # backward pass) and drop them again in ``_reduce``.
            keep = mask.to(dtype=torch.bool, device=target.device)
            fill = 0.0 if self.log_input else 1.0
            y_pred = torch.where(keep, y_pred, torch.full_like(y_pred, fill))
            target = torch.where(keep, target, torch.zeros_like(target))
        rate = torch.exp(y_pred) if self.log_input else y_pred
        g, b, var = self.detector_parameters(target)
        g = gain if gain is not None else g
        b = offset if offset is not None else b
        var = read_noise_variance if read_noise_variance is not None else var
        nll = poisson_gaussian_nll(
            target,
            rate,
            var,
            g,
            b,
            tail_sigma=self.tail_sigma,
            max_terms=self.max_terms,
        )
        return self._reduce(nll, mask, weights)


@register_regression_loss("poisson_gaussian_contamination_mixture")
class PoissonGaussianContaminationMixtureLoss(RegressionLoss):
    """
    Two-component Poisson/Gaussian mixture NLL (a contamination model).

    Each sample is modelled as coming from *one* of two laws: with probability
    ``w`` a Poisson law with rate ``λ`` and with probability ``1 - w`` a
    Gaussian ``N(λ, σ²)``. The loss is the exact mixture NLL

        -log( w·Pois(y; λ) + (1 - w)·N(y; λ, σ²) )

    computed with ``torch.logsumexp``.

    This is **not** the shot-noise-plus-read-noise detector model
    ``y = K + ε`` with ``K ~ Poisson(λ)`` and ``ε ~ N(0, σ²)``, whose density
    is the convolution ``Σ_k Pois(k; λ) N(y; k, σ²)``. For detector data use
    :class:`PoissonGaussianLoss`. The mixture can be read as an
    ε-contamination likelihood (Huber, 1964), where ``1 - w`` is the fraction
    of samples that follow the Gaussian law rather than the Poisson law.

    The Poisson component is ``torch.nn.PoissonNLLLoss(full=True)``, which is
    evaluated on continuous and possibly negative targets: there it is not a
    normalised density, so the two components are not on a common measure and
    ``w`` is not a calibrated contamination fraction.

    Args:
        eps: Small constant for numerical stability. Default: 1e-8
        learn_variance: Whether to learn the Gaussian variance parameter. Default: False
        initial_variance: Initial value for Gaussian variance. Default: 1.0
        min_variance: Minimum variance value for numerical stability. Default: 1e-6
        log_input: Whether y_pred is provided as log(lambda). Default: False
        mixture_weights: Weight ``w`` of the Poisson component:
            - If None: Equal weighting (0.5, 0.5)
            - If float: Fixed weighting (mixture_weights, 1-mixture_weights)
            - If 'learn': Learn the mixture weights
        extra_variance_model: Whether to include a separate learned variance term. Default: False
        reduction: Method for loss reduction. Default: 'mean'

    Example:
        >>> loss_fn = PoissonGaussianContaminationMixtureLoss(initial_variance=0.1, log_input=True)
        >>> y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))  # log(lambda)
        >>> target = torch.tensor([[11.2, 19.5], [4.8, 16.3]])
        >>> loss_fn(y_pred, target)
        tensor(1.8241)

    References
    ----------
    .. [1] P. J. Huber (1964). Robust estimation of a location parameter.
       Annals of Mathematical Statistics 35(1), 73-101.
       https://doi.org/10.1214/aoms/1177703732
    .. [2] G. J. McLachlan, D. Peel (2000). Finite Mixture Models. Wiley.
       https://doi.org/10.1002/0471721182
    """

    def __init__(
        self,
        config: Optional[PoissonGaussianContaminationMixtureConfig] = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        if "reduction" not in kwargs and config is None:
            kwargs["reduction"] = reduction
        if config is None:
            config = PoissonGaussianContaminationMixtureConfig(**kwargs)
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
            y_pred: Rate ``λ`` (or ``log λ`` if ``log_input``), shared by both components
            target: Target values
            mask: Optional mask for invalid values
            weights: Optional tensor of weights for each sample
            extra_var: Optional per-sample variance term

        Returns:
            Loss value
        """
        self._validate_inputs(y_pred, target, mask)

        # Get lambda (rate) parameter
        # A3: clamp in log space before exponentiation to avoid overflow
        if self.log_input:
            lam = torch.exp(y_pred.clamp(max=30.0))
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

        # True mixture NLL (A1): -log(w_p * exp(-poisson_nll) + w_g * exp(-gaussian_nll))
        # computed via logsumexp for numerical stability. The previous weighted
        # arithmetic mean was the likelihood of neither component.
        log_wp = _safe_log_weight(poisson_weight, poisson_nll)
        log_wg = _safe_log_weight(gaussian_weight, gaussian_nll)
        mixture_nll = -torch.logsumexp(
            torch.stack([log_wp - poisson_nll, log_wg - gaussian_nll], dim=-1), dim=-1
        )

        # Apply reduction with mask and weights
        return self._reduce(mixture_nll, mask, weights)


@register_regression_loss("enhanced_poisson_gaussian_contamination_mixture")
class EnhancedPoissonGaussianContaminationMixtureLoss(RegressionLoss):
    """
    Two-component Poisson/Gaussian mixture NLL with gain, offset and a
    signal-dependent Gaussian variance (a contamination model).

    With ``μ = g·λ + b`` (``λ`` optionally passed through an affine
    ``calibration``), the loss is the equal-weight mixture NLL

        -log( 0.5·Pois(y; μ) + 0.5·N(y; μ, σ₁² + σ₂²·μ) )

    computed with ``torch.logsumexp``. Each sample is treated as coming from
    one of the two laws.

    This is **not** the detector model ``y = g·K + b + ε`` with
    ``K ~ Poisson(λ)`` and ``ε ~ N(0, σ²)``: there the Poisson rate is ``λ``
    (not ``g·λ + b``) and the density is the convolution
    ``Σ_k Pois(k; λ) N(y; g·k + b, σ²)``. For detector data use
    :class:`PoissonGaussianLoss`. With ``offset = 0`` and ``shot_noise = g`` the
    Gaussian component alone is the moment-matched approximation
    ``N(g·λ, g²·λ + σ₁²)`` of the detector model; the Poisson component is
    ``torch.nn.PoissonNLLLoss(full=True)`` on ``y`` in output units, which is a
    count law only when ``g = 1`` and ``y`` is a non-negative integer.

    Args:
        gain: Fixed gain/scaling factor or 'learn' to make it learnable. Default: 1.0
        offset: Fixed offset/bias or 'learn' to make it learnable. Default: 0.0
        read_noise: Constant variance component (σ₁²). Default: 1.0
        shot_noise: Signal-dependent variance coefficient (σ₂²). Default: 0.0
        log_input: Whether inputs are in log space. Default: False
        calibration: Whether to include calibration parameters. Default: False
        reduction: Method for loss reduction. Default: 'mean'

    Example:
        >>> loss_fn = EnhancedPoissonGaussianContaminationMixtureLoss(
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
    .. [1] P. J. Huber (1964). Robust estimation of a location parameter.
       Annals of Mathematical Statistics 35(1), 73-101.
       https://doi.org/10.1214/aoms/1177703732
    .. [2] G. J. McLachlan, D. Peel (2000). Finite Mixture Models. Wiley.
       https://doi.org/10.1002/0471721182
    """

    def __init__(
        self,
        config: Optional[EnhancedPoissonGaussianContaminationMixtureConfig] = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        if "reduction" not in kwargs and config is None:
            kwargs["reduction"] = reduction
        if config is None:
            config = EnhancedPoissonGaussianContaminationMixtureConfig(**kwargs)
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
        # A3: clamp in log space before exponentiation to avoid overflow
        if self.log_input:
            rate = torch.exp(y_pred.clamp(max=30.0))
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

        # True mixture NLL with equal weights (A1) — same logsumexp form as
        # PoissonGaussianContaminationMixtureLoss; the previous arithmetic mean was not the
        # NLL of the Poisson-Gaussian mixture.
        loss = -torch.logsumexp(
            torch.stack(
                [
                    _safe_log_weight(0.5, poisson_loss) - poisson_loss,
                    _safe_log_weight(0.5, gaussian_loss) - gaussian_loss,
                ],
                dim=-1,
            ),
            dim=-1,
        )

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
        # A3: clamp in log space before exponentiation to avoid overflow
        if self.log_input:
            lam = torch.exp(y_pred.clamp(max=30.0))
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


# ---------------------------------------------------------------------------
# Deprecated aliases (to be removed in the next minor release)
# ---------------------------------------------------------------------------

_ALIAS_NOTE = (
    "It computes a two-component mixture w*Pois + (1-w)*N, not the Poisson-Gaussian "
    "detector convolution; use {new} for the mixture, or PoissonGaussianLoss for "
    "shot-plus-read-noise detector data."
)


def _warn_alias(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in the next minor release. "
        + _ALIAS_NOTE.format(new=new),
        DeprecationWarning,
        stacklevel=3,
    )


@register_regression_loss("poisson_gaussian_mixture")
class PoissonGaussianMixtureLoss(PoissonGaussianContaminationMixtureLoss):
    """Deprecated alias of :class:`PoissonGaussianContaminationMixtureLoss`."""

    def __init__(
        self,
        config: Optional[PoissonGaussianContaminationMixtureConfig] = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        _warn_alias("PoissonGaussianMixtureLoss", "PoissonGaussianContaminationMixtureLoss")
        super().__init__(config, reduction, **kwargs)


@register_regression_loss("enhanced_poisson_gaussian_mixture")
class EnhancedPoissonGaussianMixtureLoss(EnhancedPoissonGaussianContaminationMixtureLoss):
    """Deprecated alias of :class:`EnhancedPoissonGaussianContaminationMixtureLoss`."""

    def __init__(
        self,
        config: Optional[EnhancedPoissonGaussianContaminationMixtureConfig] = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        _warn_alias(
            "EnhancedPoissonGaussianMixtureLoss",
            "EnhancedPoissonGaussianContaminationMixtureLoss",
        )
        super().__init__(config, reduction, **kwargs)


register_deprecated_loss_key("poisson_gaussian_mixture", "poisson_gaussian_contamination_mixture")
register_deprecated_loss_key(
    "enhanced_poisson_gaussian_mixture", "enhanced_poisson_gaussian_contamination_mixture"
)

_DEPRECATED_CONFIGS = {
    "PoissonGaussianMixtureConfig": "PoissonGaussianContaminationMixtureConfig",
    "EnhancedPoissonGaussianConfig": "EnhancedPoissonGaussianContaminationMixtureConfig",
}


def __getattr__(name: str) -> Any:
    new = _DEPRECATED_CONFIGS.get(name)
    if new is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"{name} is deprecated and will be removed in the next minor release; use {new}.",
        DeprecationWarning,
        stacklevel=2,
    )
    return globals()[new]
