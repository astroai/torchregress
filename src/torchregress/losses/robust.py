"""
Robust loss functions for regression.

This module provides implementations of loss functions that are
robust against outliers, such as:
- Pseudo-Huber loss: Differentiable approximation to Huber loss
- Log-Cosh loss: Smooth approximation of Huber loss
- Various other robust alternatives to standard regression losses

Note: For standard Huber loss with masking and weights, use WeightedHuberLoss
from the base module instead.
"""

import math
from typing import Any, Optional, cast

import torch
import torch.nn as nn

from ..utils.validation import validate_positive, validate_range, validate_weights
from .base import RegressionLoss
from .loss_registry import register_regression_loss
from .utils_robust import huber_elementwise, log_cosh, tukey_biweight

_BARRON_ALPHA_EPS = 1e-6
_BARRON_SCALE_EPS = 1e-8
# Width of the |alpha| window around 0 where the analytic Taylor series of the
# Cauchy-like limit is used instead of the generic expression (which is 0/0 there).
_BARRON_TAYLOR_TOL = 1e-2
# Floor on |alpha - 2| used by the generic expression.  d(rho)/d(alpha) diverges
# logarithmically as alpha -> 2 (both one-sided derivatives go to +infinity);
# inflating |alpha - 2| by this floor turns the divergence into a large-but-finite
# gradient while leaving loss values unchanged to <1e-4 outside |alpha - 2| < 1e-3.
# ponytail: exact rho is not C^1 at alpha = 2, so any finite gradient requires
# regularizing the curvature at some scale; 1e-4 keeps the distortion negligible.
_BARRON_CURVATURE_FLOOR = 1e-4
# Anchors of log Z(alpha) = log integral(exp(-rho(u, alpha))) du, the partition
# function of the density associated with the Barron loss.  Closed forms exist at
# alpha in {0, 1, 2}: pi*sqrt(2), 2*e*K_1(1) (modified Bessel function of the
# second kind, order 1, at 1), sqrt(2*pi).
_BARRON_LOG_PARTITION_ANCHORS = (
    math.log(math.pi * math.sqrt(2.0)),
    math.log(2.0 * math.e * 0.6019072301972346),
    0.5 * math.log(2.0 * math.pi),
)


def _barron_elementwise(
    residuals: torch.Tensor,
    alpha: torch.Tensor | float,
    scale: torch.Tensor | float,
    eps: float = _BARRON_SCALE_EPS,
) -> torch.Tensor:
    """
    Barron's general robust loss ``rho(r / c, alpha)``.

    The expression is differentiable with respect to ``alpha`` everywhere,
    including the singular points ``alpha = 0`` and ``alpha = 2``:

    - Around ``alpha = 0`` the generic expression ``(beta/alpha)*expm1(u)`` is a
      0/0 limit; the analytic Taylor series of the Cauchy-like limit
      ``(beta/2) * lam * (1 + u/2 + u^2/6)`` (with ``lam = log1p(z^2/beta)``,
      ``u = alpha * lam / 2``, error O(u^3)) is used in a small window.  Both
      sides of the window carry gradient with respect to ``alpha``.
    - At ``alpha = 2`` the true loss is not C^1 in ``alpha`` (logarithmically
      divergent slope), so ``|alpha - 2|`` is inflated by
      ``_BARRON_CURVATURE_FLOOR``, which yields finite, nonzero gradients while
      preserving values to machine precision at the point itself.

    Reference:
        Barron, J. T. "A General and Adaptive Robust Loss Function." CVPR, 2019.
    """
    alpha_tensor = torch.as_tensor(alpha, dtype=residuals.dtype, device=residuals.device)
    scale_tensor = torch.as_tensor(scale, dtype=residuals.dtype, device=residuals.device).clamp(
        min=eps
    )

    squared_scaled = (residuals / scale_tensor) ** 2

    # Smooth replacement for |alpha - 2|: identical outside ~1e-3 of the point,
    # bounded away from zero so the exponent/log below never see beta = 0.
    beta = torch.sqrt((alpha_tensor - 2.0) ** 2 + _BARRON_CURVATURE_FLOOR**2)
    lam = torch.log1p(squared_scaled / beta)
    u = alpha_tensor * lam / 2.0

    # Exact analytic series around the regular point alpha = 0; reduces exactly to
    # log1p(0.5 * z^2) there.
    cauchy_limit = (beta / 2.0) * lam * (1.0 + u / 2.0 + u * u / 6.0)

    # Generic branch, rewritten with expm1 for numerical stability; equivalent to
    # (beta / alpha) * ((z^2 / beta + 1) ** (alpha / 2) - 1).
    alpha_safe = torch.where(
        alpha_tensor >= 0,
        alpha_tensor.clamp(min=_BARRON_ALPHA_EPS),
        alpha_tensor.clamp(max=-_BARRON_ALPHA_EPS),
    )
    generic = (beta / alpha_safe) * torch.expm1(u)

    return torch.where(alpha_tensor.abs() <= _BARRON_TAYLOR_TOL, cauchy_limit, generic)


def _log_barron_partition(alpha: torch.Tensor | float) -> torch.Tensor:
    """
    ``log Z(alpha)`` where ``Z(alpha) = integral exp(-rho(u, alpha)) du`` is the
    partition function of the density induced by the Barron loss (Barron, 2019,
    Eq. 17 normalization).  Adding ``log Z`` makes losses comparable across
    shapes; together with ``log(scale)`` it makes the objective scale-consistent.

    Closed forms exist at ``alpha in {0, 1, 2}``; elsewhere this returns the
    quadratic interpolation of ``log Z`` through those exact anchors.
    ponytail: for alpha < 0 the integral diverges, so the polynomial extension
    below 0 is an approximation ceiling, not a normalizer.
    """
    log_z0, log_z1, log_z2 = _BARRON_LOG_PARTITION_ANCHORS
    alpha_tensor = torch.as_tensor(alpha)
    return (
        log_z0 * (alpha_tensor - 1.0) * (alpha_tensor - 2.0) / 2.0
        - log_z1 * alpha_tensor * (alpha_tensor - 2.0)
        + log_z2 * alpha_tensor * (alpha_tensor - 1.0) / 2.0
    )


@register_regression_loss("pseudo_huber")
class PseudoHuberLoss(RegressionLoss):
    """
    Pseudo-Huber Loss: a smooth approximation to the Huber loss.

    L(y, f(x)) = delta^2 * (sqrt(1 + ((y - f(x))/delta)^2) - 1)

    Args:
        delta: Controls the smoothness. Default: 1.0
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> loss_fn = PseudoHuberLoss(delta=1.0)
        >>> y_pred = torch.tensor([0.0, 1.0, 2.0])
        >>> target = torch.tensor([0.0, 2.0, 1.0])
        >>> loss_fn(y_pred, target)
        tensor(0.6213)  # Smoother than standard Huber loss
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.delta = validate_positive(delta, "delta")

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate Pseudo-Huber loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Pseudo-Huber loss value
        """
        self._validate_inputs(y_pred, target, mask)

        # Calculate scaled difference
        scaled_diff = (target - y_pred) / self.delta

        # Calculate pseudo-huber
        loss = self.delta**2 * (torch.sqrt(1.0 + scaled_diff**2) - 1.0)

        # Apply reduction with mask and weights
        return self._reduce(loss, mask, weights)


@register_regression_loss("barron")
class BarronLoss(RegressionLoss):
    """
    Barron's general robust loss.

    This is a continuous family that interpolates between quadratic,
    Cauchy-like, and redescending robust penalties through the shape
    parameter ``alpha``.

    Args:
        alpha: Robustness shape parameter. ``alpha=2`` recovers quadratic loss,
            ``alpha=0`` gives a Cauchy-like penalty, and smaller values become
            increasingly outlier-robust.
        scale: Positive scale parameter controlling the transition point.
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    References
    ----------
    .. [1] Barron, J. T. (2019). A General and Adaptive Robust Loss Function.
       In *CVPR 2019*. https://arxiv.org/abs/1701.03077
    """

    def __init__(self, alpha: float = 1.0, scale: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        if not math.isfinite(alpha):
            raise ValueError(f"alpha must be finite, got {alpha}")
        self.alpha = alpha
        self.scale = validate_positive(scale, "scale")

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Calculate Barron's robust loss."""
        self._validate_inputs(y_pred, target, mask)
        residuals = target - y_pred
        loss = _barron_elementwise(residuals, self.alpha, self.scale)
        return self._reduce(loss, mask, weights)


@register_regression_loss("adaptive_robust")
class AdaptiveRobustLoss(RegressionLoss):
    """
    Trainable Barron-style robust loss.

    ``alpha`` and ``scale`` are stored as constrained parameters so they can be
    optimized jointly with the model by adding ``loss_fn.parameters()`` to the
    optimizer parameter list.

    The objective is the normalized Barron loss (Barron, 2019, Eq. 17):
    ``rho(r/c, alpha) + log(c) + log Z(alpha)``.  Without the normalization
    terms the unnormalized ``rho(r/c, alpha) -> 0`` as ``c -> inf`` and joint
    optimization lets the scale diverge; with them the objective is
    scale-consistent (e.g. at ``alpha = 2`` the optimal scale is the residual
    RMS).

    Args:
        alpha_init: Initial shape parameter.
        scale_init: Initial positive scale parameter.
        alpha_min: Lower bound for the learned alpha.
        alpha_max: Upper bound for the learned alpha.
        learn_alpha: Whether to optimize alpha.
        learn_scale: Whether to optimize scale.
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    References
    ----------
    .. [1] Barron, J. T. (2019). A General and Adaptive Robust Loss Function.
       In *CVPR 2019*. https://arxiv.org/abs/1701.03077
    """

    def __init__(
        self,
        alpha_init: float = 1.0,
        scale_init: float = 1.0,
        alpha_min: float = -8.0,
        alpha_max: float = 2.0,
        learn_alpha: bool = True,
        learn_scale: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        if not math.isfinite(alpha_init):
            raise ValueError(f"alpha_init must be finite, got {alpha_init}")
        if not math.isfinite(alpha_min) or not math.isfinite(alpha_max):
            raise ValueError("alpha_min and alpha_max must be finite")
        if alpha_min >= alpha_max:
            raise ValueError("alpha_min must be strictly less than alpha_max")
        validate_range(alpha_init, alpha_min, alpha_max, "alpha_init")
        validate_positive(scale_init, "scale_init")

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        alpha_prob = (alpha_init - alpha_min) / (alpha_max - alpha_min)
        alpha_prob = min(max(alpha_prob, _BARRON_ALPHA_EPS), 1.0 - _BARRON_ALPHA_EPS)
        raw_scale = scale_init - _BARRON_SCALE_EPS
        scale_raw = raw_scale + math.log(-math.expm1(-raw_scale))

        self._alpha_logits = nn.Parameter(
            torch.tensor(math.log(alpha_prob) - math.log1p(-alpha_prob), dtype=torch.float32),
            requires_grad=learn_alpha,
        )
        self._scale_raw = nn.Parameter(
            torch.tensor(scale_raw, dtype=torch.float32),
            requires_grad=learn_scale,
        )

    @property
    def alpha(self) -> torch.Tensor:
        alpha_prob = torch.sigmoid(self._alpha_logits)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_prob

    @property
    def scale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._scale_raw) + _BARRON_SCALE_EPS

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Calculate the normalized adaptive robust loss with the current learned parameters."""
        self._validate_inputs(y_pred, target, mask)
        residuals = target - y_pred
        alpha = self.alpha.to(dtype=residuals.dtype, device=residuals.device)
        scale = self.scale.to(dtype=residuals.dtype, device=residuals.device)
        loss = _barron_elementwise(residuals, alpha, scale)
        # Barron (2019) Eq. 17: log(scale) + log Z(alpha) makes the objective
        # scale-consistent so joint optimization cannot drive c -> inf.
        loss = loss + torch.log(scale) + _log_barron_partition(alpha)
        return self._reduce(loss, mask, weights)

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha.detach().item():.4f}, "
            f"scale={self.scale.detach().item():.4f}, "
            f"alpha_range=({self.alpha_min}, {self.alpha_max})"
        )


@register_regression_loss("log_cosh")
class LogCoshLoss(RegressionLoss):
    """
    Log-Cosh Loss: A smooth approximation of Huber loss.

    L(y, f(x)) = log(cosh(y - f(x)))

    Args:
        scale: Scaling factor to control smoothness. Default: 1.0
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> loss_fn = LogCoshLoss()
        >>> y_pred = torch.tensor([0.0, 1.0, 2.0])
        >>> target = torch.tensor([0.0, 2.0, 1.0])
        >>> loss_fn(y_pred, target)
        tensor(0.3266)  # Smooth but approaches |x| for large values
    """

    def __init__(self, scale: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.scale = validate_positive(scale, "scale")

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate Log-Cosh loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Log-Cosh loss value
        """
        self._validate_inputs(y_pred, target, mask)

        # Calculate scaled difference and apply log-cosh
        diff = self.scale * (target - y_pred)

        # Use a stable formula for log-cosh
        abs_diff = torch.abs(diff)
        loss = abs_diff + torch.log1p(torch.exp(-2.0 * abs_diff)) - math.log(2.0)

        # Apply reduction with mask and weights
        return self._reduce(loss, mask, weights)


@register_regression_loss("charbonnier")
class CharbonnierLoss(RegressionLoss):
    """
    Charbonnier Loss: A smooth alternative to L1 loss.

    L(y, f(x)) = sqrt((y - f(x))^2 + eps^2)

    Args:
        eps: Small constant for numerical stability. Default: 1e-3
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> loss_fn = CharbonnierLoss(eps=1e-3)
        >>> y_pred = torch.tensor([0.0, 1.0, 2.0])
        >>> target = torch.tensor([0.0, 2.0, 1.0])
        >>> loss_fn(y_pred, target)
        tensor(0.6667)  # Similar to L1 but differentiable at zero
    """

    def __init__(self, eps: float = 1e-3, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.eps = validate_positive(eps, "eps")

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate Charbonnier loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Charbonnier loss value
        """
        self._validate_inputs(y_pred, target, mask)

        # Calculate squared difference and apply charbonnier formula
        squared_diff = (target - y_pred) ** 2
        loss = torch.sqrt(squared_diff + self.eps**2)

        # Apply reduction with mask and weights
        return self._reduce(loss, mask, weights)


@register_regression_loss("tukey_biweight")
class TukeyBiweightLoss(RegressionLoss):
    """
    Tukey's biweight (bisquare) loss for robust regression.

    This loss completely ignores errors beyond a certain threshold, making it
    highly robust against outliers, but non-convex.

    L(y, f(x)) = c^2/6 * (1 - (1 - (r/c)^2)^3)  if |r| <= c
                 c^2/6                          otherwise

    where r = y - f(x) and c is the tuning constant.

    Args:
        c: Tuning constant (typical value 4.685)
        reduction: 'none' | 'mean' | 'sum'

    Example:
        >>> loss_fn = TukeyBiweightLoss(c=4.685)
        >>> y_pred = torch.tensor([0.0, 2.0, 10.0])
        >>> target = torch.tensor([0.0, 0.0, 0.0])
        >>> loss_fn(y_pred, target)
        tensor(1.8287)  # The large error is effectively capped

    References
    ----------
    .. [1] Beaton, A. E., & Tukey, J. W. (1974). The Fitting of Power Series,
       Meaning Polynomials, Illustrated on Band-Spectroscopic Data.
       In *Technometrics*, 16(2), 147-185.
       https://doi.org/10.1080/00401706.1974.10489171
    """

    def __init__(self, c: float = 4.685, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.c = validate_positive(c, "c")
        self.c_squared_over_6 = c**2 / 6.0

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate Tukey's biweight loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Tukey's biweight loss
        """
        self._validate_inputs(y_pred, target, mask)

        # Tukey biweight:
        #   L(r) = c²/6 * (1 - (1 - (r/c)²)³)              for |r| <= c
        #        = c²/6                                    for |r| >  c
        # Use ``torch.where`` (single fused kernel) rather than masked
        # assignment: the latter allocates a full ``ones_like`` tensor and
        # then writes to it from a CPU/GPU sync, which is unnecessary work
        # and breaks some autograd paths.
        residuals = target - y_pred
        abs_residuals = torch.abs(residuals)
        within = abs_residuals <= self.c
        squared_scaled = (residuals / self.c) ** 2
        inlier = self.c_squared_over_6 * (1.0 - (1.0 - squared_scaled) ** 3)
        loss = torch.where(within, inlier, torch.full_like(residuals, self.c_squared_over_6))

        return self._reduce(loss, mask, weights)


@register_regression_loss("cauchy")
class CauchyLoss(RegressionLoss):
    """
    Cauchy loss for robust regression.

    This loss uses the negative log of the Cauchy distribution density.
    It is very robust to outliers but non-convex.

    L(y, f(x)) = log(1 + (r/c)^2)

    where r = y - f(x) and c is the scale parameter.

    Args:
        c: Scale parameter
        reduction: 'none' | 'mean' | 'sum'

    Example:
        >>> loss_fn = CauchyLoss(c=1.0)
        >>> y_pred = torch.tensor([0.0, 1.0, 10.0])
        >>> target = torch.tensor([0.0, 0.0, 0.0])
        >>> loss_fn(y_pred, target)
        tensor(1.7009)  # Large outliers have limited impact
    """

    def __init__(
        self,
        c: Optional[float] = None,
        scale: Optional[float] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        if c is not None and scale is not None:
            raise ValueError("Provide either c or scale for CauchyLoss, not both.")
        if scale is not None:
            c = scale
        if c is None:
            c = 1.0
        self.c = validate_positive(c, "c")

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate Cauchy loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Cauchy loss
        """
        self._validate_inputs(y_pred, target, mask)

        # Calculate residuals
        residuals = target - y_pred

        # Calculate Cauchy loss
        scaled_residuals = residuals / self.c
        loss = torch.log(1.0 + scaled_residuals**2)

        # Apply reduction with mask and weights
        return self._reduce(loss, mask, weights)


@register_regression_loss("cvar")
class CVaRLoss(RegressionLoss):
    """
    Conditional Value at Risk (CVaR) loss for tail-focused regression.

    CVaR focuses on the worst-performing samples by averaging the top
    alpha fraction of per-sample losses. It can be combined with robust
    base losses (e.g., Huber) to stabilize training under noisy labels.

    Args:
        alpha: Fraction of worst losses to average (0 < alpha <= 1).
        base_loss: Base loss to apply before CVaR ("mse", "mae", "huber",
            "log_cosh", "cauchy", "tukey").
        delta: Huber threshold for base_loss="huber".
        c: Scale for base_loss in {"cauchy", "tukey"}.
        scale: Scale for base_loss="log_cosh".
        reduction: 'none' | 'mean' | 'sum'. For 'mean'/'sum', applies CVaR.

    References
    ----------
    .. [1] Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk.
       In *Journal of Risk*, 2(3), 21-41.
       https://doi.org/10.1023/A:1008126422718
    """

    def __init__(
        self,
        alpha: float = 0.1,
        base_loss: str = "mse",
        delta: float = 1.0,
        c: float = 1.0,
        scale: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.alpha = validate_range(alpha, 0.0, 1.0, "alpha")
        if self.alpha <= 0:
            raise ValueError("alpha must be in (0, 1].")
        self.base_loss = base_loss.lower()
        # ``float(...)``: validate_positive passes tensors through, but these
        # constants feed scalar-only helpers (huber/log_cosh/tukey).
        self.delta = float(validate_positive(delta, "delta"))
        self.c = float(validate_positive(c, "c"))
        self.scale = float(validate_positive(scale, "scale"))

        supported = {"mse", "mae", "huber", "log_cosh", "cauchy", "tukey"}
        if self.base_loss not in supported:
            raise ValueError(f"base_loss must be one of {sorted(supported)}, got {self.base_loss}")

    def _compute_elementwise_loss(
        self, residuals: torch.Tensor, abs_residuals: torch.Tensor
    ) -> torch.Tensor:
        if self.base_loss == "mse":
            return residuals**2
        if self.base_loss == "mae":
            return abs_residuals
        if self.base_loss == "huber":
            # A11: shared robust helpers (were local copies)
            return huber_elementwise(residuals, self.delta)
        if self.base_loss == "log_cosh":
            return log_cosh(residuals, self.scale)
        if self.base_loss == "cauchy":
            scaled = residuals / self.c
            return torch.log1p(scaled**2)

        return tukey_biweight(residuals, self.c)

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate CVaR loss.
        """
        self._validate_inputs(y_pred, target, mask)

        residuals = target - y_pred
        abs_residuals = torch.abs(residuals)
        elem_loss = self._compute_elementwise_loss(residuals, abs_residuals)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask > 0
            mask_float = mask.to(elem_loss.dtype)
            dims = tuple(range(1, elem_loss.dim()))
            masked = elem_loss * mask_float
            valid = mask_float.sum(dim=dims).clamp(min=1)
            per_sample = masked.sum(dim=dims) / valid
        else:
            per_sample = elem_loss.mean(dim=tuple(range(1, elem_loss.dim())))

        if weights is not None:
            weights = validate_weights(weights, per_sample.shape[0])
            weights = cast(torch.Tensor, weights)
            per_sample = per_sample * weights

        if self.reduction == "none":
            return per_sample

        # CVaR selects the worst α fraction of *samples*; per_sample must
        # already be 1‑D (one scalar loss per batch element).  Use the
        # explicit batch dimension rather than .numel() so that a future
        # refactoring that accidentally preserves multi-dimensional shape
        # is caught by the guard below instead of silently producing
        # wrong top‑k indices.
        if per_sample.dim() != 1:
            raise RuntimeError(
                "Internal error: CVaRLoss per-sample loss tensor has shape "
                f"{tuple(per_sample.shape)}; expected 1‑D [batch_size]. "
                "This indicates a bug in the per-sample aggregation above."
            )
        batch_size = per_sample.shape[0]
        k = max(1, int(math.ceil(self.alpha * batch_size)))
        topk = torch.topk(per_sample, k=k, largest=True).values
        if self.reduction == "sum":
            return torch.sum(topk)
        return torch.mean(topk)
