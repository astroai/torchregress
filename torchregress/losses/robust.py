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
from typing import Any, Optional

import torch

from ..utils.validation import validate_positive, validate_range, validate_weights
from .base import RegressionLoss
from .loss_registry import register_regression_loss

# Remove HuberLoss as it's redundant with WeightedHuberLoss in base


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
        return self._reduce_with_mask(loss, mask, weights)


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
        return self._reduce_with_mask(loss, mask, weights)


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
        return self._reduce_with_mask(loss, mask, weights)





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

        # Calculate residuals
        residuals = target - y_pred
        abs_residuals = torch.abs(residuals)

        # Calculate loss based on whether residuals exceed threshold
        scaled_residuals = abs_residuals / self.c
        squared_scaled_residuals = scaled_residuals**2

        # For |r| <= c: c^2/6 * (1 - (1 - (r/c)^2)^3)
        # For |r| >  c: c^2/6
        mask_within = abs_residuals <= self.c
        loss = torch.ones_like(residuals) * self.c_squared_over_6
        if torch.any(mask_within):
            loss[mask_within] = self.c_squared_over_6 * (
                1.0 - (1.0 - squared_scaled_residuals[mask_within]) ** 3
            )

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)

















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

    def __init__(self, c: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
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
        return self._reduce_with_mask(loss, mask, weights)


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
        self.delta = validate_positive(delta, "delta")
        self.c = validate_positive(c, "c")
        self.scale = validate_positive(scale, "scale")
        self.c_squared_over_6 = self.c**2 / 6.0

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
            return torch.where(
                abs_residuals <= self.delta,
                0.5 * residuals**2,
                self.delta * (abs_residuals - 0.5 * self.delta),
            )
        if self.base_loss == "log_cosh":
            diff = self.scale * residuals
            abs_diff = torch.abs(diff)
            return abs_diff + torch.log1p(torch.exp(-2.0 * abs_diff)) - math.log(2.0)
        if self.base_loss == "cauchy":
            scaled = residuals / self.c
            return torch.log1p(scaled**2)

        scaled = abs_residuals / self.c
        squared = scaled**2
        loss = torch.ones_like(residuals) * self.c_squared_over_6
        mask_within = abs_residuals <= self.c
        if torch.any(mask_within):
            loss[mask_within] = self.c_squared_over_6 * (1.0 - (1.0 - squared[mask_within]) ** 3)
        return loss

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
            per_sample = per_sample * weights

        if self.reduction == "none":
            return per_sample

        k = max(1, int(math.ceil(self.alpha * per_sample.numel())))
        topk = torch.topk(per_sample, k=k, largest=True).values
        if self.reduction == "sum":
            return torch.sum(topk)
        return torch.mean(topk)
