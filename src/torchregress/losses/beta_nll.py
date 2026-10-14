"""
Beta-NLL loss for heteroscedastic Gaussian regression.

Down-weights variance collapse by scaling each per-element Gaussian NLL term
with ``var^{beta}`` computed from a detached variance, following Seitzer,
Tavakoli, Antic, Martius (2022), "On the Pitfalls of Heteroscedastic
Uncertainty Estimation with Probabilistic Neural Networks", ICLR 2022.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union, cast

import torch

from .gaussian import GaussianNLLLoss
from .loss_registry import register_regression_loss


@register_regression_loss("beta_nll")
class BetaNLLLoss(GaussianNLLLoss):
    """
    Heteroscedastic Gaussian NLL with detached variance rescaling (β-NLL).

    Uses the same prediction formats as :class:`GaussianNLLLoss`: tuple
    ``(mean, log_variance)``, concatenated ``[mean, log_variance]`` along
    ``split_dim``, and the same ``min_variance`` / ``eps`` stabilisation.

    The per-element loss is ``var.detach().pow(beta) * nll_per_dim`` where
    ``nll_per_dim`` is the standard diagonal Gaussian negative log-likelihood
    including the ``log(2π)`` term, summed over the feature dimension (the
    last dim of ``>= 2``-D inputs) AFTER the per-element weighting
    (paper-exact form); 1-D ``[B]`` inputs are treated as one scalar target
    per sample. For ``beta == 0`` this matches :class:`GaussianNLLLoss`.

    Learned variance is required; ``fixed_variance`` is not supported.

    Args:
        beta: Non-negative exponent on the detached ``var`` in the rescaling
            ``var.detach() ** beta`` (``0`` recovers the plain Gaussian NLL terms).
        min_variance: Floor applied after ``exp(log_var)``.
        eps: Small constant inside ``log`` and divisions for numerical stability.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
        split_dim: Dimension along which concatenated predictions are split in half.

    References
    ----------
    .. [1] Seitzer, M., Tavakoli, A., Antic, D., & Martius, G. (2022).
       On the Pitfalls of Heteroscedastic Uncertainty Estimation with
       Probabilistic Neural Networks. In *ICLR 2022*.
       https://arxiv.org/abs/2203.09168
    """

    def __init__(
        self,
        beta: float = 0.5,
        *,
        min_variance: float = 1e-6,
        eps: float = 1e-8,
        reduction: str = "mean",
        split_dim: int = -1,
    ) -> None:
        super().__init__(
            covariance_type="diagonal",
            fixed_variance=None,
            min_variance=min_variance,
            eps=eps,
            reduction=reduction,
            split_dim=split_dim,
        )
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        self.beta = beta

    def forward(
        self,
        y_pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        mean, var = self._extract_distribution_parameters(y_pred)
        self._validate_inputs(mean, target, mask)
        nll_per_dim = 0.5 * (
            math.log(2 * math.pi)
            + torch.log(var + self.eps)
            + (target - mean) ** 2 / (var + self.eps)
        )
        coef = var.detach().clamp_min(self.eps).pow(self.beta)
        weighted = nll_per_dim * coef  # [B, D] or [B]
        # Sum over the feature dim only when there is one: for 1-D [B] inputs
        # every element is its own sample, and summing over dim -1 would sum
        # over the batch (A-LOSS-002).
        has_feature_dim = weighted.dim() >= 2
        if mask is not None:
            # Preserve partial rows: zero-fill per-element then sum per-sample.
            # Previously summed before _reduce forced mask.all(row) discard.
            # Bool-convert first (BaseLoss._reduce policy): float masks from
            # dense/spatial loaders would fail torch.where's condition check.
            mask_bool = mask.to(dtype=torch.bool)
            masked = torch.where(mask_bool, weighted, torch.zeros_like(weighted))
            if not has_feature_dim:
                return self._reduce(masked, mask=mask_bool, weights=weights)
            summed = masked.sum(dim=-1)  # [B]
            # Per-sample valid mask for _reduce (exclude fully-masked rows)
            sample_mask = mask_bool.any(dim=-1)
            if weights is not None and weights.shape == mask_bool.shape:
                # Per-feature weights -> per-sample weight averaged over the
                # unmasked features of each row (BaseLoss._reduce policy).
                valid = mask_bool.to(weights.device)
                w_valid = torch.where(valid, weights, torch.zeros_like(weights))
                weights = w_valid.sum(dim=-1) / valid.sum(dim=-1).clamp_min(1).to(weights.dtype)
            return self._reduce(summed, mask=sample_mask, weights=weights)
        if not has_feature_dim:
            return self._reduce(weighted, mask=None, weights=weights)
        summed = weighted.sum(dim=-1)  # [B] sum over features per paper
        return self._reduce(summed, mask=None, weights=weights)


def beta_nll_loss(
    y_pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    target: torch.Tensor,
    beta: float,
    *,
    min_variance: float = 1e-6,
    eps: float = 1e-8,
    reduction: str = "mean",
    split_dim: int = -1,
    mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Functional β-NLL for diagonal Gaussian predictions.

    See :class:`BetaNLLLoss` for argument semantics.

    References
    ----------
    .. [1] Seitzer, M., Tavakoli, A., Antic, D., & Martius, G. (2022).
       On the Pitfalls of Heteroscedastic Uncertainty Estimation with
       Probabilistic Neural Networks. In *ICLR 2022*.
       https://arxiv.org/abs/2203.09168
    """
    fn = BetaNLLLoss(
        beta=beta,
        min_variance=min_variance,
        eps=eps,
        reduction=reduction,
        split_dim=split_dim,
    )
    return cast(torch.Tensor, fn(y_pred, target, mask=mask, weights=weights))
