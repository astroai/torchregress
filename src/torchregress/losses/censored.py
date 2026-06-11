"""Censored and interval-censored regression losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..utils.distributions import normal_cdf
from ..utils.validation import validate_range, validate_weights
from .base import BaseLoss
from .loss_registry import register_regression_loss

_LOG_SQRT_2PI = 0.5 * torch.log(torch.tensor(2.0 * torch.pi))


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
        cdf_target = normal_cdf(z_target).clamp(self.eps, 1.0 - self.eps)
        surv_target = (1.0 - cdf_target).clamp_min(self.eps)
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
            z_low = (lower_bound - mean) / std
            z_up = (upper_bound - mean) / std
            cdf_low = normal_cdf(z_low)
            cdf_up = normal_cdf(z_up)
            interval_prob = (cdf_up - cdf_low).clamp_min(self.eps)
            nll[interval_mask] = -torch.log(interval_prob[interval_mask])

            observed_mask = observed_mask & (~interval_mask)
            right_mask = right_mask & (~interval_mask)
            left_mask = left_mask & (~interval_mask)

        nll[observed_mask] = -logpdf[observed_mask]
        nll[right_mask] = -torch.log(surv_target[right_mask])
        nll[left_mask] = -torch.log(cdf_target[left_mask])

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
        cdf = normal_cdf(z).clamp(self.eps, 1.0 - self.eps)
        surv = (1.0 - cdf).clamp_min(self.eps)
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
            p_int = (normal_cdf(z_up) - normal_cdf(z_low)).clamp_min(self.eps)
            nll[interval_mask] = -torch.log(p_int[interval_mask])

            observed_mask = observed_mask & (~interval_mask)
            right_mask = right_mask & (~interval_mask)
            left_mask = left_mask & (~interval_mask)

        nll[observed_mask] = -logpdf[observed_mask]
        nll[right_mask] = -torch.log(surv[right_mask])
        nll[left_mask] = -torch.log(cdf[left_mask])

        return self._reduce(nll, mask=mask, weights=weights)


__all__ = [
    "CensoredGaussianNLLLoss",
    "CensoredQuantileLoss",
    "AFTLoss",
]
