"""Log-variance bounds shared by ensemble prediction and Gaussian NLL training.

Heteroscedastic ensemble members are usually trained with
:class:`torchregress.losses.GaussianNLLLoss`, which converts a predicted
log-variance to a variance via ``exp(log_var.clamp(log(min_variance), 30))`` with
``min_variance=1e-6`` by default. Prediction must apply the *same* bounds;
otherwise a member that legitimately learned a log-variance outside a narrower
range (e.g. unstandardised targets with sd ~ 50, or very precise targets with
sd ~ 1e-3) reports a silently wrong aleatoric variance.
"""

from __future__ import annotations

import math

from torch import Tensor

from torchregress.utils.gaussian_output import variance_from_logvar

#: ``log(min_variance)`` for the default ``GaussianNLLLoss(min_variance=1e-6)``.
GAUSSIAN_NLL_MIN_LOGVAR: float = math.log(1.0e-6)
#: Upper log-variance clamp applied by ``GaussianNLLLoss``.
GAUSSIAN_NLL_MAX_LOGVAR: float = 30.0


def member_variance_from_logvar(log_var: Tensor) -> Tensor:
    """Member log-variance to variance with the ``GaussianNLLLoss`` training bounds."""
    return variance_from_logvar(
        log_var,
        min_logvar=GAUSSIAN_NLL_MIN_LOGVAR,
        max_logvar=GAUSSIAN_NLL_MAX_LOGVAR,
    )


__all__ = [
    "GAUSSIAN_NLL_MAX_LOGVAR",
    "GAUSSIAN_NLL_MIN_LOGVAR",
    "member_variance_from_logvar",
]
