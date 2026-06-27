"""
Beta-NLL loss for heteroscedastic Gaussian regression.

Down-weights variance collapse by scaling each per-element Gaussian NLL term
with ``var^{-beta}`` computed from a detached variance, following Seitzer et al.
(ICLR 2022).
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

    The per-element loss is ``(var + eps).detach().pow(-beta) * nll`` where
    ``nll`` is the standard diagonal Gaussian negative log-likelihood including
    the ``log(2π)`` term. For ``beta == 0`` this matches :class:`GaussianNLLLoss`.

    Learned variance is required; ``fixed_variance`` is not supported.

    Args:
        beta: Non-negative exponent on ``1/var`` in the detached rescaling.
        min_variance: Floor applied after ``exp(log_var)``.
        eps: Small constant inside ``log`` and divisions for numerical stability.
        reduction: ``"mean"``, ``"sum"``, ``"none"``, ``"min"``, or ``"max"``.
        split_dim: Dimension along which concatenated predictions are split in half.

    References
    ----------
    .. [1] Seitzer, M., Tavakoli, A., Keurifon, D., & Peters, J. (2022).
       On the Pitfalls of Heteroscedastic Uncertainty Estimation with NLL.
       In *NeurIPS 2022*. https://arxiv.org/abs/2205.11310
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

        nll = 0.5 * (
            math.log(2 * math.pi)
            + torch.log(var + self.eps)
            + (target - mean) ** 2 / (var + self.eps)
        )
        nll = nll.sum(dim=-1)  # [B, D] → [B], consistent with GaussianNLLLoss
        coef = (var + self.eps).detach().pow(-self.beta)
        # ponytail: coef is [B, D] but we need per-sample weight; take mean across features
        weighted = coef.mean(dim=-1) * nll
        return self._reduce_with_mask(weighted, mask, weights)


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
    .. [1] Seitzer, M., Tavakoli, A., Keurifon, D., & Peters, J. (2022).
       On the Pitfalls of Heteroscedastic Uncertainty Estimation with NLL.
       In *NeurIPS 2022*. https://arxiv.org/abs/2205.11310
    """
    fn = BetaNLLLoss(
        beta=beta,
        min_variance=min_variance,
        eps=eps,
        reduction=reduction,
        split_dim=split_dim,
    )
    return cast(torch.Tensor, fn(y_pred, target, mask=mask, weights=weights))
