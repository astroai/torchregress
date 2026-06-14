"""
Faithful heteroscedastic Gaussian objective (mean / variance decoupling).

The variance branch uses squared residuals with a **stopped-gradient** mean so
that aleatoric calibration does not distort the mean estimate through the
heteroscedastic likelihood, a common failure mode of joint Gaussian NLL training.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch

from .gaussian import GaussianNLLLoss
from .loss_registry import register_regression_loss


@register_regression_loss("faithful_gaussian")
class FaithfulGaussianLoss(GaussianNLLLoss):
    """
    Combined mean squared error + variance NLL with a detached mean in the NLL residual.

    Let :math:`\\mu_\\theta(x)` and :math:`\\sigma^2_\\theta(x)` be the predicted mean
    and variance. The per-element objective is:

    .. math::

        \\lambda_{\\mu} (\\mu - y)^2 + \\lambda_{v} \\cdot \\frac{1}{2}
        \\left( \\log(2\\pi\\sigma^2) + \\frac{(y - \\mathrm{sg}(\\mu))^2}{\\sigma^2} \\right)

    where :math:`\\mathrm{sg}` is stop-gradient. The mean receives gradients only from
    the MSE term; the variance head receives gradients from the NLL term. This
    mirrors the intent of *faithful* heteroscedastic training: preserve point
    prediction quality while learning a noise model.

    Accepts the same ``y_pred`` formats as :class:`GaussianNLLLoss` (tuple
    ``(mean, log_variance)`` or concatenated tensor).

    Parameters
    ----------
    mean_weight:
        Multiplier on :math:`(\\mu - y)^2`. Set to ``0`` to train variance only
        (mean still forwarded for the detached residual).
    variance_weight:
        Multiplier on the Gaussian NLL terms (including :math:`\\log 2\\pi`).
    min_variance, eps, reduction, split_dim:
        Same meaning as :class:`GaussianNLLLoss`.

    See Also
    --------
    GaussianNLLLoss : Joint NLL without decoupling.
    BetaNLLLoss : Variance-detached *reweighting* of the joint NLL.
    """

    def __init__(
        self,
        *,
        mean_weight: float = 1.0,
        variance_weight: float = 1.0,
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
        if mean_weight < 0 or variance_weight < 0:
            raise ValueError("mean_weight and variance_weight must be non-negative.")
        self.mean_weight = float(mean_weight)
        self.variance_weight = float(variance_weight)

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

        # Avoid `0.0 * term` when a weight is zero — that can still attach `term` to the graph.
        if self.mean_weight > 0.0:
            mse_part = self.mean_weight * (mean - target) ** 2
        else:
            mse_part = torch.zeros_like(mean)

        if self.variance_weight > 0.0:
            mean_detached = mean.detach()
            nll_var = 0.5 * (
                math.log(2 * math.pi)
                + torch.log(var + self.eps)
                + (target - mean_detached) ** 2 / (var + self.eps)
            )
            var_part = self.variance_weight * nll_var
        else:
            var_part = torch.zeros_like(mean)

        per_elem = mse_part + var_part
        return self._reduce_with_mask(per_elem, mask, weights)
