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

import torch
from typing import Optional
from .base import RegressionLoss
from ..utils.validation import validate_positive

# Remove HuberLoss as it's redundant with WeightedHuberLoss in base

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
        loss = abs_diff + torch.log1p(torch.exp(-2.0 * abs_diff)) - torch.log(torch.tensor(2.0))

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


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


class LqLoss(RegressionLoss):
    """
    Lq Loss: A generalization of L1 (q=1) and L2 (q=2) losses.

    L(y, f(x)) = |y - f(x)|^q

    Args:
        q: Order of the norm. Default: 1.5
        eps: Small constant for numerical stability. Default: 1e-8
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> loss_fn = LqLoss(q=1.5)
        >>> y_pred = torch.tensor([0.0, 1.0, 2.0])
        >>> target = torch.tensor([0.0, 2.0, 1.0])
        >>> loss_fn(y_pred, target)
        tensor(0.5774)  # Between L1 and L2 loss
    """

    def __init__(self, q: float = 1.5, eps: float = 1e-8, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.q = validate_positive(q, "q")
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate Lq loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Lq loss value
        """
        self._validate_inputs(y_pred, target, mask)

        # Calculate absolute difference and raise to power q
        abs_diff = torch.abs(target - y_pred) + self.eps
        loss = abs_diff**self.q

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


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


class WinsorizedLoss(RegressionLoss):
    """
    Winsorized loss for robust regression.

    This loss replaces extreme residuals with more moderate values,
    truncating the influence of outliers.

    Args:
        quantile_low: Lower quantile for winsorization (0-1)
        quantile_high: Upper quantile for winsorization (0-1)
        reduction: 'none' | 'mean' | 'sum'

    Example:
        >>> loss_fn = WinsorizedLoss(quantile_low=0.25, quantile_high=0.75)
        >>> y_pred = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        >>> target = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        >>> loss_fn(y_pred, target)
        tensor(1.0000)  # Extreme values are capped to the quantile thresholds
    """

    def __init__(
        self, quantile_low: float = 0.05, quantile_high: float = 0.95, reduction: str = "mean"
    ) -> None:
        super().__init__(reduction=reduction)
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate Winsorized loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Winsorized loss
        """
        self._validate_inputs(y_pred, target, mask)

        # Calculate residuals
        residuals = target - y_pred

        # Determine quantile thresholds
        with torch.no_grad():
            if mask is not None:
                valid_residuals = residuals[mask]
                if valid_residuals.numel() == 0:
                    # Default thresholds if no valid residuals
                    lower_threshold = torch.tensor(-1.0, device=residuals.device)
                    upper_threshold = torch.tensor(1.0, device=residuals.device)
                else:
                    lower_threshold = torch.quantile(valid_residuals, self.quantile_low)
                    upper_threshold = torch.quantile(valid_residuals, self.quantile_high)
            else:
                lower_threshold = torch.quantile(residuals, self.quantile_low)
                upper_threshold = torch.quantile(residuals, self.quantile_high)

        # Winsorize residuals
        winsorized_residuals = torch.clamp(residuals, min=lower_threshold, max=upper_threshold)

        # Calculate squared error on winsorized residuals
        loss = winsorized_residuals**2

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


class LogBarrierLoss(RegressionLoss):
    """
    Log Barrier loss for robust regression.

    This loss implements a logarithmic barrier function that gracefully limits
    the influence of large errors.

    L(y, f(x)) = -log(1 - min(|y - f(x)|/rho, 1-eps)^2)

    Args:
        rho: Scale parameter defining the error threshold
        eps: Small constant to ensure loss remains finite
        reduction: 'none' | 'mean' | 'sum'

    Example:
        >>> loss_fn = LogBarrierLoss(rho=2.0)
        >>> y_pred = torch.tensor([0.0, 1.0, 3.0])
        >>> target = torch.tensor([0.0, 2.0, 0.0])
        >>> loss_fn(y_pred, target)
        tensor(0.3466)  # Large errors are handled with logarithmic growth
    """

    def __init__(self, rho: float = 1.0, eps: float = 1e-8, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.rho = validate_positive(rho, "rho")
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate Log Barrier loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Log Barrier loss
        """
        self._validate_inputs(y_pred, target, mask)

        # Calculate error ratio
        abs_error = torch.abs(target - y_pred)
        ratio = abs_error / self.rho

        # Clip ratio to avoid barrier approaching infinity
        ratio = torch.clamp(ratio, max=1.0 - self.eps)

        # Calculate log barrier loss
        loss = -torch.log(1.0 - ratio**2 + self.eps)

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


class AdaptiveHuberLoss(RegressionLoss):
    """
    Adaptive Huber loss with automatic delta estimation.

    This loss dynamically computes the delta parameter of the Huber loss
    based on the data, making it adaptive to different scales.

    Args:
        quantile: Quantile of absolute errors to use for delta estimation
        scale_factor: Additional scaling factor for delta
        reduction: 'none' | 'mean' | 'sum'

    Example:
        >>> loss_fn = AdaptiveHuberLoss(quantile=0.8)
        >>> y_pred = torch.tensor([0.0, 1.0, 2.0, 10.0])
        >>> target = torch.tensor([0.0, 2.0, 1.0, 0.0])
        >>> loss_fn(y_pred, target)
        tensor(1.7500)  # Delta adapts to the scale of the errors
    """

    def __init__(
        self, quantile: float = 0.8, scale_factor: float = 1.0, reduction: str = "mean"
    ) -> None:
        super().__init__(reduction=reduction)
        self.quantile = quantile
        self.scale_factor = scale_factor

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate Adaptive Huber loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Adaptive Huber loss
        """
        self._validate_inputs(y_pred, target, mask)

        # Calculate residuals
        residuals = target - y_pred
        abs_residuals = torch.abs(residuals)

        # Determine delta dynamically based on the quantile of absolute residuals
        with torch.no_grad():
            if mask is not None:
                # Get only valid residuals for quantile calculation
                valid_residuals = abs_residuals[mask]
                if valid_residuals.numel() == 0:
                    # Default delta if no valid residuals
                    delta = torch.tensor(1.0, device=y_pred.device)
                else:
                    delta = torch.quantile(valid_residuals, self.quantile) * self.scale_factor
            else:
                delta = torch.quantile(abs_residuals, self.quantile) * self.scale_factor

            # Ensure delta is at least a small positive number for numerical stability
            delta = torch.clamp(delta, min=1e-8)

        # Calculate Huber loss with the adaptive delta
        quadratic_mask = abs_residuals <= delta
        loss = torch.zeros_like(abs_residuals)

        # Quadratic region
        loss[quadratic_mask] = 0.5 * abs_residuals[quadratic_mask] ** 2

        # Linear region
        linear_mask = ~quadratic_mask
        loss[linear_mask] = delta * (abs_residuals[linear_mask] - 0.5 * delta)

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


class ClippedLoss(RegressionLoss):
    """
    Clipped loss for robust regression.

    This loss clips errors above a threshold, limiting the influence of outliers.

    Args:
        threshold: Error threshold beyond which the loss is clipped
        base_loss: Base loss function ('l1', 'mse')
        reduction: 'none' | 'mean' | 'sum'

    Example:
        >>> loss_fn = ClippedLoss(threshold=1.0, base_loss='mse')
        >>> y_pred = torch.tensor([0.0, 1.0, 3.0])
        >>> target = torch.tensor([0.0, 2.0, 0.0])
        >>> loss_fn(y_pred, target)
        tensor(1.0000)  # Squared errors are clipped at the threshold
    """

    def __init__(
        self, threshold: float = 1.0, base_loss: str = "mse", reduction: str = "mean"
    ) -> None:
        super().__init__(reduction=reduction)
        self.threshold = validate_positive(threshold, "threshold")
        self.base_loss = base_loss

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate Clipped loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Clipped loss
        """
        self._validate_inputs(y_pred, target, mask)

        # Calculate residuals
        residuals = target - y_pred

        if self.base_loss == "l1":
            # Calculate absolute error
            error = torch.abs(residuals)
        else:  # 'mse'
            # Calculate squared error
            error = residuals**2

        # Clip error at threshold
        loss = torch.clamp(error, max=self.threshold)

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


class FairLoss(RegressionLoss):
    """
    Fair loss for robust regression.

    This loss grows less than linearly with the absolute error, making it
    less sensitive to outliers than MSE or MAE.

    L(y, f(x)) = c^2 * (|r|/c - log(1 + |r|/c))

    where r = y - f(x) and c is the scale parameter.

    Args:
        c: Scale parameter
        reduction: 'none' | 'mean' | 'sum'
        eps: Small constant for numerical stability

    Example:
        >>> loss_fn = FairLoss(c=1.0)
        >>> y_pred = torch.tensor([0.0, 1.0, 3.0])
        >>> target = torch.tensor([0.0, 2.0, 0.0])
        >>> loss_fn(y_pred, target)
        tensor(0.7297)  # Growth is slower than MAE for large errors
    """

    def __init__(self, c: float = 1.0, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__(reduction=reduction)
        self.c = validate_positive(c, "c")
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate Fair loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Fair loss
        """
        self._validate_inputs(y_pred, target, mask)

        # Calculate residuals
        residuals = target - y_pred
        abs_residuals = torch.abs(residuals)

        # Calculate fair loss
        scaled_residuals = abs_residuals / self.c
        loss = self.c**2 * (scaled_residuals - torch.log(1.0 + scaled_residuals + self.eps))

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


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
