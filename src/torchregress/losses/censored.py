"""Censored and interval-censored regression losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..utils.validation import validate_range, validate_weights
from .base import BaseLoss
from .loss_registry import register_regression_loss

_LOG_SQRT_2PI = 0.5 * torch.log(torch.tensor(2.0 * torch.pi))


def _log1mexp(x: Tensor) -> Tensor:
    """``log(1 - exp(x))`` for ``x < 0`` (Maechler 2012, "Accurately computing
    log(1 - exp(-|a|))")."""
    return torch.where(
        x > -0.6931471805599453, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x))
    )


def _log_normal_interval_prob(z_low: Tensor, z_up: Tensor) -> Tensor:
    """``log(Phi(z_up) - Phi(z_low))`` for ``z_low < z_up``, accurate in both tails.

    The interval is reflected (``Phi(b) - Phi(a) = Phi(-a) - Phi(-b)``) so that
    it never lies in the upper tail, where ``Phi`` rounds to one, and the
    difference is evaluated as ``log Phi(b) + log1mexp(log Phi(a) - log Phi(b))``.
    The log-difference is capped at ``-finfo.eps`` so a degenerate interval
    that collapses under rounding yields a large finite NLL instead of ``inf``.
    """
    flip = (z_low + z_up) > 0
    a = torch.where(flip, -z_up, z_low)
    b = torch.where(flip, -z_low, z_up)
    log_a = torch.special.log_ndtr(a)
    log_b = torch.special.log_ndtr(b)
    diff = (log_a - log_b).clamp(max=-torch.finfo(log_a.dtype).eps)
    return log_b + _log1mexp(diff)


def _extract_mean_and_var(
    y_pred: Tensor | tuple[Tensor, Tensor],
    *,
    log_variance: bool,
    kwargs: dict[str, Any],
    eps: float,
) -> tuple[Tensor, Tensor]:
    if isinstance(y_pred, (tuple, list)):
        if len(y_pred) != 2:
            raise ValueError("y_pred tuple must have (mean, variance/log_variance)")
        mean, var_or_log = y_pred
    else:
        if "var" in kwargs:
            mean = y_pred
            var_or_log = kwargs["var"]
        elif "log_var" in kwargs:
            mean = y_pred
            var_or_log = kwargs["log_var"]
            log_variance = True
        else:
            raise ValueError("Provide y_pred as (mean, var/log_var) or pass var/log_var in kwargs")

    if log_variance:
        var = torch.exp(var_or_log.clamp(min=-20.0, max=20.0))
    else:
        var = var_or_log
    var = var.clamp(min=eps, max=1e6)
    return mean, var


def _validate_censoring_inputs(
    target: Tensor,
    censoring: Tensor | None,
    lower_bound: Tensor | None,
    upper_bound: Tensor | None,
    mask: Tensor | None,
) -> None:
    if censoring is not None and censoring.shape != target.shape:
        raise ValueError("censoring must have same shape as target")
    if lower_bound is not None and lower_bound.shape != target.shape:
        raise ValueError("lower_bound must have same shape as target")
    if upper_bound is not None and upper_bound.shape != target.shape:
        raise ValueError("upper_bound must have same shape as target")
    if mask is not None and mask.shape != target.shape:
        raise ValueError("mask must have same shape as target")


@register_regression_loss("censored_gaussian_nll")
class CensoredGaussianNLLLoss(BaseLoss):
    """Gaussian NLL supporting right/left and interval censoring.

    Censor encoding:
    - `0`: observed target
    - `1`: right-censored (true value >= target)
    - `-1`: left-censored (true value <= target)
    Interval-censoring can be supplied with explicit `lower_bound`/`upper_bound`.
    """

    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-8,
        log_variance: bool = True,
    ) -> None:
        super().__init__(reduction=reduction)
        self.eps = float(eps)
        self.log_variance = log_variance

    def forward(
        self,
        y_pred: Tensor | tuple[Tensor, Tensor],
        target: Tensor,
        censoring: Tensor | None = None,
        lower_bound: Tensor | None = None,
        upper_bound: Tensor | None = None,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        mean, var = _extract_mean_and_var(
            y_pred,
            log_variance=self.log_variance,
            kwargs=kwargs,
            eps=self.eps,
        )
        self._validate_inputs(mean, target, mask)
        _validate_censoring_inputs(target, censoring, lower_bound, upper_bound, mask)

        if weights is not None:
            weights = validate_weights(weights, target.shape[0])

        std = torch.sqrt(var).clamp_min(self.eps)
        z_target = (target - mean) / std
        # Censored terms via log_ndtr: the previous -log(clamp(Phi, eps)) form
        # saturated at -log(eps) ~ 18.4 nats with an exactly-zero gradient once
        # the censoring point was > ~5.6 sigma into the tail.
        log_cdf_target = torch.special.log_ndtr(z_target)
        log_surv_target = torch.special.log_ndtr(-z_target)
        logpdf = -0.5 * z_target.pow(2) - torch.log(std) - _LOG_SQRT_2PI.to(std.device, std.dtype)

        if censoring is None:
            censoring = torch.zeros_like(target, dtype=torch.int64)
        censoring_i = censoring.long()
        if torch.any(~((censoring_i == -1) | (censoring_i == 0) | (censoring_i == 1))):
            raise ValueError("censoring values must be in {-1, 0, 1}")

        observed_mask = censoring_i == 0
        right_mask = censoring_i == 1
        left_mask = censoring_i == -1

        nll = torch.zeros_like(target, dtype=mean.dtype)

        interval_mask = torch.zeros_like(observed_mask)
        if lower_bound is not None and upper_bound is not None:
            interval_mask = (
                (upper_bound > lower_bound)
                & torch.isfinite(lower_bound)
                & torch.isfinite(upper_bound)
            )
            # Non-interval entries get a dummy finite interval so their
            # (discarded) branch never produces inf/NaN gradients.
            low = torch.where(interval_mask, lower_bound, mean.detach() - std.detach())
            up = torch.where(interval_mask, upper_bound, mean.detach() + std.detach())
            log_p_int = _log_normal_interval_prob((low - mean) / std, (up - mean) / std)
            nll[interval_mask] = -log_p_int[interval_mask]

            observed_mask = observed_mask & (~interval_mask)
            right_mask = right_mask & (~interval_mask)
            left_mask = left_mask & (~interval_mask)

        nll[observed_mask] = -logpdf[observed_mask]
        nll[right_mask] = -log_surv_target[right_mask]
        nll[left_mask] = -log_cdf_target[left_mask]

        return self._reduce(nll, mask=mask, weights=weights)


@register_regression_loss("censored_quantile")
class CensoredQuantileLoss(BaseLoss):
    """Quantile loss variant for censored / interval-censored targets."""

    def __init__(self, quantile: float = 0.5, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.quantile = float(validate_range(quantile, 0.0, 1.0, "quantile"))

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        censoring: Tensor | None = None,
        lower_bound: Tensor | None = None,
        upper_bound: Tensor | None = None,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        self._validate_inputs(y_pred, target, mask)
        _validate_censoring_inputs(target, censoring, lower_bound, upper_bound, mask)

        if weights is not None:
            weights = validate_weights(weights, target.shape[0])

        if censoring is None:
            censoring = torch.zeros_like(target, dtype=torch.int64)
        censoring_i = censoring.long()

        q = self.quantile
        error = target - y_pred

        observed_loss = torch.maximum(q * error, (q - 1.0) * error)
        right_loss = q * torch.relu(target - y_pred)
        left_loss = (1.0 - q) * torch.relu(y_pred - target)

        loss = torch.zeros_like(target, dtype=y_pred.dtype)
        loss[censoring_i == 0] = observed_loss[censoring_i == 0]
        loss[censoring_i == 1] = right_loss[censoring_i == 1]
        loss[censoring_i == -1] = left_loss[censoring_i == -1]

        if lower_bound is not None and upper_bound is not None:
            interval_mask = (
                (upper_bound > lower_bound)
                & torch.isfinite(lower_bound)
                & torch.isfinite(upper_bound)
            )
            interval_loss = q * torch.relu(lower_bound - y_pred) + (1.0 - q) * torch.relu(
                y_pred - upper_bound
            )
            loss[interval_mask] = interval_loss[interval_mask]

        return self._reduce(loss, mask=mask, weights=weights)


@register_regression_loss("aft")
class AFTLoss(BaseLoss):
    """Log-normal accelerated failure time (AFT) loss with censoring support."""

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__(reduction=reduction)
        self.eps = float(eps)

    def forward(
        self,
        y_pred: Tensor | tuple[Tensor, Tensor],
        target: Tensor,
        censoring: Tensor | None = None,
        lower_bound: Tensor | None = None,
        upper_bound: Tensor | None = None,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        if isinstance(y_pred, (tuple, list)):
            if len(y_pred) != 2:
                raise ValueError("AFTLoss expects (loc, log_scale) tuple")
            loc, log_scale = y_pred
        else:
            if "log_scale" not in kwargs:
                raise ValueError("Provide y_pred as (loc, log_scale) or pass log_scale in kwargs")
            loc = y_pred
            log_scale = kwargs["log_scale"]

        self._validate_inputs(loc, target, mask)
        _validate_censoring_inputs(target, censoring, lower_bound, upper_bound, mask)

        if weights is not None:
            weights = validate_weights(weights, target.shape[0])

        if censoring is None:
            censoring = torch.zeros_like(target, dtype=torch.int64)
        censoring_i = censoring.long()
        if torch.any(~((censoring_i == -1) | (censoring_i == 0) | (censoring_i == 1))):
            raise ValueError("censoring values must be in {-1, 0, 1}")

        scale = torch.exp(log_scale.clamp(min=-7.0, max=7.0)).clamp(min=self.eps, max=1e3)
        safe_target = target.clamp_min(self.eps)
        log_t = torch.log(safe_target)

        z = (log_t - loc) / scale
        # log_ndtr instead of -log(clamp(Phi, eps)) (no saturation in the tails)
        log_cdf = torch.special.log_ndtr(z)
        log_surv = torch.special.log_ndtr(-z)
        logpdf = (
            -torch.log(safe_target)
            - torch.log(scale)
            - 0.5 * z.pow(2)
            - _LOG_SQRT_2PI.to(scale.device, scale.dtype)
        )

        observed_mask = censoring_i == 0
        right_mask = censoring_i == 1
        left_mask = censoring_i == -1

        nll = torch.zeros_like(target, dtype=loc.dtype)

        interval_mask = torch.zeros_like(observed_mask)
        if lower_bound is not None and upper_bound is not None:
            interval_mask = (upper_bound > lower_bound) & (upper_bound > 0) & (lower_bound > 0)
            z_low = (torch.log(lower_bound.clamp_min(self.eps)) - loc) / scale
            z_up = (torch.log(upper_bound.clamp_min(self.eps)) - loc) / scale
            # Dummy finite interval where the branch is not taken (see above).
            z_low = torch.where(interval_mask, z_low, torch.full_like(z_low, -1.0))
            z_up = torch.where(interval_mask, z_up, torch.full_like(z_up, 1.0))
            log_p_int = _log_normal_interval_prob(z_low, z_up)
            nll[interval_mask] = -log_p_int[interval_mask]

            observed_mask = observed_mask & (~interval_mask)
            right_mask = right_mask & (~interval_mask)
            left_mask = left_mask & (~interval_mask)

        nll[observed_mask] = -logpdf[observed_mask]
        nll[right_mask] = -log_surv[right_mask]
        nll[left_mask] = -log_cdf[left_mask]

        return self._reduce(nll, mask=mask, weights=weights)


__all__ = [
    "CensoredGaussianNLLLoss",
    "CensoredQuantileLoss",
    "AFTLoss",
]
