"""
Faithful heteroscedastic Gaussian objective (mean / variance decoupling).

The variance branch uses squared residuals with a **stopped-gradient** mean so
that aleatoric calibration does not distort the mean estimate through the
heteroscedastic likelihood, a common failure mode of joint Gaussian NLL training.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F

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
        Multiplier on the mean term. A scalar applies to every output; a 1D
        sequence or tensor of length ``D`` gives one weight per output (the last
        dimension of the mean). All weights must be finite and non-negative.
        Set to ``0`` to train variance only (mean still forwarded for the
        detached residual).
    mean_loss:
        ``"mse"`` (default) uses :math:`(\\mu - y)^2`. ``"huber"`` uses twice the
        Huber loss with threshold ``huber_delta``: identical to ``"mse"`` for
        errors below the threshold, linear beyond it, which limits the pull of
        outlying targets on the mean.
    huber_delta:
        Huber threshold, in target units. Used only when ``mean_loss="huber"``.
    variance_weight:
        Multiplier on the Gaussian NLL terms (including :math:`\\log 2\\pi`).
    min_variance, eps, reduction, split_dim:
        Same meaning as :class:`GaussianNLLLoss`.

    See Also
    --------
    GaussianNLLLoss : Joint NLL without decoupling.
    BetaNLLLoss : Variance-detached *reweighting* of the joint NLL.
    """

    mean_weight: torch.Tensor

    def __init__(
        self,
        *,
        mean_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,
        mean_loss: Literal["mse", "huber"] = "mse",
        huber_delta: float = 1.0,
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
        mean_weight_tensor = torch.as_tensor(mean_weight, dtype=torch.float32)
        if mean_weight_tensor.dim() > 1:
            raise ValueError("mean_weight must be a scalar or a 1D sequence.")
        if not torch.isfinite(mean_weight_tensor).all() or (mean_weight_tensor < 0).any():
            raise ValueError("mean_weight must be finite and non-negative.")
        if variance_weight < 0:
            raise ValueError("variance_weight must be non-negative.")
        if mean_loss not in ("mse", "huber"):
            raise ValueError(f"mean_loss must be 'mse' or 'huber', got {mean_loss!r}")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive.")
        self.register_buffer("mean_weight", mean_weight_tensor)
        self.mean_loss = mean_loss
        self.huber_delta = float(huber_delta)
        self.variance_weight = float(variance_weight)

    @property
    def _has_mean_term(self) -> bool:
        """Whether any ``mean_weight`` entry is positive, read from the live buffer."""
        return bool((self.mean_weight > 0).any())

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
        # Read from the buffer on every call (not cached at __init__): the
        # buffer is restored by load_state_dict and may be ramped by curricula.
        if self._has_mean_term:
            weight = self.mean_weight.to(device=mean.device, dtype=mean.dtype)
            if weight.dim() == 1 and weight.shape[0] != mean.shape[-1]:
                raise ValueError(
                    f"mean_weight has {weight.shape[0]} entries but the mean has "
                    f"{mean.shape[-1]} outputs."
                )
            if self.mean_loss == "huber":
                # x2 so the quadratic zone equals (mu - y)^2 and mean_weight keeps
                # the same meaning for both options.
                mean_term = 2.0 * F.huber_loss(
                    mean, target, reduction="none", delta=self.huber_delta
                )
            else:
                mean_term = (mean - target) ** 2
            mse_part = weight * mean_term
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
        return self._reduce(per_elem, mask, weights)
