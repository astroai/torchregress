"""
Student-t Negative Log-Likelihood loss for robust regression.

This implements the negative log-likelihood of a Student-t distribution
with configurable degrees of freedom (ν) and scale (σ), generalizing both
the Cauchy loss (ν=1) and the Gaussian NLL (ν→∞).

This is the direct probabilistic loss that RHMF (Hilder et al. 2026)
optimizes via IRLS — unlike the existing CauchyLoss, this supports
arbitrary ν rather than being fixed at ν=1.

References:
    RHMF: "Robust Heteroskedastic Matrix Factorization" (arXiv:2607.08081)
    Lange, Little, & Taylor (1989): "Robust statistical modeling using the
    t distribution." JASA 84(408): 881-896.
"""

import math
from typing import Any, Optional

import torch
from torch import Tensor

from .base import RegressionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("student_t")
class StudentTLoss(RegressionLoss):
    """Negative log-likelihood of a zero-mean scaled Student-t distribution.

    L(y, f(x)) = -log p(y | f(x)) where

        p(y | μ, σ, ν) = StudentT_ν(y | μ, σ²)

    The NLL is:

        NLL = log Γ(ν/2) - log Γ((ν+1)/2) + ½ log(ν π)
              + ½ (ν + 1) log(1 + (r/σ)² / ν) + log σ

    where r = y - μ, σ is the scale, and ν is the degrees of freedom.

    Special cases:
    - ν = 1  → Cauchy NLL (log(1 + (r/σ)²) + log(πσ))
    - ν → ∞ → Gaussian NLL (½ (r/σ)² + log(σ) + ½ log(2π))

    The scale σ controls the soft threshold between quadratic behavior
    (small residuals) and logarithmic behavior (large residuals),
    analogous to Q in RHMF.

    Args:
        nu: Degrees of freedom (> 0). Default: 1.0 (Cauchy).
            ν = 1 gives maximal robustness; larger ν approaches Gaussian.
        scale: Scale parameter σ (> 0). Default: 1.0.
            Controls where the quadratic→logarithmic transition occurs.
            Scales with the data noise level (analogous to RHMF's Q).
        reduction: Loss reduction ('mean', 'sum', 'none'). Default: 'mean'.

    Example:
        >>> loss = StudentTLoss(nu=1.0, scale=5.0)
        >>> y_pred = torch.tensor([0.0, 1.0, 10.0])
        >>> y_true = torch.tensor([0.0, 0.0, 0.0])
        >>> loss(y_pred, y_true)
        tensor(...)  # 10.0 residual is downweighted vs MSE

    Example (matching RHMF toy experiment ν=1, Q=5):
        >>> import torch
        >>> loss = StudentTLoss(nu=1.0, scale=5.0)
        >>> r = torch.tensor([0.5, 2.0, 10.0])
        >>> # Compare to RHMF's w^robust = Q²/(Q² + r²)
        >>> w_robust = 25.0 / (25.0 + r**2)
        >>> w_robust
        tensor([0.9901, 0.8621, 0.2000])

    References
    ----------
    .. [1] Hilder et al. (2026). Robust Heteroskedastic Matrix Factorization.
       arXiv:2607.08081.
    .. [2] Lange, K. L., Little, R. J. A., & Taylor, J. M. G. (1989).
       Robust Statistical Modeling Using the t Distribution.
       Journal of the American Statistical Association, 84(408), 881-896.
    """

    def __init__(
        self,
        nu: float = 1.0,
        scale: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        if nu <= 0:
            raise ValueError(f"nu must be > 0, got {nu}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        self.nu = nu
        self.scale = scale

        # Pre-compute constant term (depends only on ν, not data)
        # From Student-t density: log Γ(ν/2) - log Γ((ν+1)/2) + ½ log(νπ)
        self._log_norm = float(
            math.lgamma(nu / 2.0) - math.lgamma((nu + 1.0) / 2.0) + 0.5 * math.log(nu * math.pi)
        )

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute Student-t NLL.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask for valid values
            weights: Optional per-sample weights

        Returns:
            Reduced Student-t NLL.
        """
        self._validate_inputs(y_pred, target, mask)

        residual = target - y_pred
        scaled_sq = (residual / self.scale) ** 2

        # NLL = log_norm + ½(ν+1) log(1 + r²/(ν σ²)) + log(σ)
        nll = (
            self._log_norm
            + 0.5 * (self.nu + 1.0) * torch.log1p(scaled_sq / self.nu)
            + math.log(self.scale)
        )

        return self._reduce(nll, mask, weights)

    def extra_repr(self) -> str:
        return f"nu={self.nu}, scale={self.scale}"
