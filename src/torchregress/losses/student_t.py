"""
Student-t likelihood for robust regression with known per-observation precision.

This module holds the Student-t family used by :class:`StudentTLoss`, by the
``"student_t"`` weights of :func:`torchregress.algorithms.robust_weights`, and by
:func:`torchregress.algorithms.batched_iteratively_reweighted_fit`:

* :func:`student_t_nll` -- exact negative log-likelihood of ``t_nu(0, sigma^2)``
  with ``sigma^2 = 1 / precision``, normalising constants included, masked and
  summed per row (``nu = inf`` is the Gaussian);
* :func:`effective_precision` -- the variance model ``1/precision + eta^2 m^2``
  (known noise plus a model-proportional error);
* :func:`student_t_efficiency` -- ``E[psi']^2 / E[psi^2]`` of the ``t_nu``
  location score under Gaussian or ``t_nu`` noise (the latter is the Fisher
  factor ``(nu + 1)/(nu + 3)``).

References:
    Lange, Little, & Taylor (1989): "Robust statistical modeling using the
    t distribution." JASA 84(408): 881-896.
    Huber (1964): "Robust estimation of a location parameter." Ann. Math.
    Statist. 35(1): 73-101.
"""

import functools
import math
from typing import Any, Literal, Optional

import numpy as np
import torch
from torch import Tensor

from .base import RegressionLoss
from .loss_registry import register_regression_loss

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)

#: Gauss--Hermite nodes of :func:`student_t_efficiency` for ``nu >= 3``. The
#: integrands have poles at ``±i√nu``, so the rule converges slowly for small
#: ``nu`` (``1e-6`` relative error at 0.5, 3% at 0.1); from ``nu = 3`` it is at
#: float64 round-off (``≈ 1e-15`` against mpmath).
_HERMITE_NODES = 200

#: Below this ``nu`` :func:`student_t_efficiency` uses the closed form, which is at
#: round-off there (``≲ 3e-15``) but loses digits to cancellation as ``nu`` grows
#: (``3e-14`` at 10, ``2e-11`` at 100).
_EFFICIENCY_CLOSED_FORM_NU = 3.0


def _validated_nu(nu: float) -> float:
    value = float(nu)
    if math.isnan(value) or value <= 0.0:
        raise ValueError(f"nu must be > 0 (math.inf for the Gaussian), got {nu}")
    return value


#: Above this ``ν`` the log normaliser uses its asymptotic series: the difference
#: of two ``lgamma`` values cancels catastrophically (``3e-14`` absolute at
#: ``ν = 100``, ``1e-6`` at ``1e9``, ``O(1)`` at ``1e15``), while the series is
#: within ``2e-15`` of mpmath from ``ν = 100`` up.
_LOG_NORM_SERIES_NU = 100.0


def _student_t_log_norm(nu: float) -> float:
    """``log Γ(ν/2) − log Γ((ν+1)/2) + ½ log(νπ)`` for finite ``ν``.

    For large ``ν`` this is ``½ log 2π + 1/(4ν) − 1/(24ν³) + 1/(20ν⁵) + O(ν⁻⁷)``,
    which tends to the Gaussian constant of the ``ν = ∞`` branch.
    """
    if nu >= _LOG_NORM_SERIES_NU:
        inv = 1.0 / nu  # Horner form: no overflow of ν⁵ for huge ν
        return _HALF_LOG_2PI + inv * (0.25 + inv * inv * (-1.0 / 24.0 + inv * inv / 20.0))
    return math.lgamma(0.5 * nu) - math.lgamma(0.5 * (nu + 1.0)) + 0.5 * math.log(nu * math.pi)


def _student_t_terms(residual: Tensor, precision: Tensor, nu: float) -> Tensor:
    """Elementwise ``t_ν(0, 1/precision)`` NLL (unmasked; ``ν = ∞`` is Gaussian)."""
    z2 = residual.square() * precision
    if math.isinf(nu):
        return 0.5 * z2 - 0.5 * torch.log(precision) + _HALF_LOG_2PI
    return _student_t_log_norm(nu) - 0.5 * torch.log(precision) + _student_t_rho(z2, nu)


def _student_t_weight(z2: Tensor, nu: float) -> Tensor:
    """IRLS multiplier ``ψ(z)/z = (ν+1)/(ν+z²)`` of the ``t_ν`` NLL; 1 at ``ν = ∞``."""
    if math.isinf(nu):
        return torch.ones_like(z2)
    return (nu + 1.0) / (nu + z2)


def _student_t_rho(z2: Tensor, nu: float) -> Tensor:
    """Data part ``ρ(z) = ½(ν+1) log(1 + z²/ν)`` of the ``t_ν`` NLL; ``½z²`` at ``ν = ∞``."""
    if math.isinf(nu):
        return 0.5 * z2
    return 0.5 * (nu + 1.0) * torch.log1p(z2 / nu)


def student_t_nll(
    residual: Tensor,
    precision: Tensor,
    nu: float,
    *,
    mask: Optional[Tensor] = None,
    dim: int | tuple[int, ...] | None = -1,
) -> Tensor:
    r"""Exact Student-t negative log-likelihood with per-element precision.

    Each element is scored under ``t_ν(0, σ²)`` with scale ``σ = precision^{-1/2}``:

    .. math::

        -\log p(r \mid \nu, \lambda) = \log\Gamma(\tfrac{\nu}{2})
        - \log\Gamma(\tfrac{\nu+1}{2}) + \tfrac12\log(\nu\pi)
        - \tfrac12\log\lambda + \tfrac{\nu+1}{2}\log\!\left(1 + \frac{\lambda r^2}{\nu}\right),

    and ``nu = math.inf`` gives the Gaussian ``½λr² − ½log λ + ½log 2π``. The
    scale is ``σ`` itself (not ``σ√((ν−2)/ν)``), so the IRLS weight of this NLL is
    ``λ(ν+1)/(ν + λr²)`` and any ``ν > 0`` is valid.

    Args:
        residual: Residuals ``r`` (any shape).
        precision: Precisions ``λ = 1/σ²``, broadcastable to ``residual``.
            ``λ = 0`` on an unmasked element gives ``+inf``; non-finite inputs
            propagate (fail closed). Use ``mask`` to drop observations.
        nu: Degrees of freedom ``ν > 0``, or ``math.inf``.
        mask: Optional boolean mask (``True`` = observed), broadcastable to
            ``residual``. Masked elements contribute exactly zero, whatever
            their residual or precision (``NaN`` included).
        dim: Dimension(s) summed over (default ``-1``: one NLL per row of a
            ``[B, N]`` batch). ``None`` returns the elementwise NLL.

    Returns:
        The NLL summed over ``dim`` (or elementwise for ``dim=None``).
    """
    nu = _validated_nu(nu)
    residual, precision = torch.broadcast_tensors(residual, precision)
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype torch.bool")
        mask = torch.broadcast_to(mask.to(residual.device), residual.shape)
        residual = torch.where(mask, residual, torch.zeros_like(residual))
        precision = torch.where(mask, precision, torch.ones_like(precision))
    terms = _student_t_terms(residual, precision, nu)
    if mask is not None:
        terms = torch.where(mask, terms, torch.zeros_like(terms))
    return terms if dim is None else terms.sum(dim=dim)


def effective_precision(precision: Tensor, model: Tensor, eta: float | Tensor) -> Tensor:
    r"""Precision of the variance model ``1/λ + η² m²``.

    .. math::

        \lambda_{\mathrm{eff}} = \frac{1}{1/\lambda + \eta^2 m^2}
        = \frac{\lambda}{1 + \eta^2 m^2 \lambda},

    i.e. known noise plus an error proportional to the model ``m`` (a
    fractional template or calibration error ``η``). Zero precision (no
    information) stays zero for a finite model; non-finite model values
    propagate (fail closed). For ``eta == 0`` (a Python number) ``precision``
    itself is returned, not a copy.

    A float ``eta`` is checked on the host at no cost. A tensor ``eta`` is
    checked with one reduction read back to the host: a device
    synchronisation on every call when it lives on an accelerator, so a loop
    that passes a per-row tensor pays it at each iteration. A float ``eta``
    (one value for every row) avoids it.

    Args:
        precision: Known precisions ``λ ≥ 0``.
        model: Model values ``m`` (broadcastable to ``precision``), usually frozen
            at the previous iterate of an IRLS fit.
        eta: Fractional model error ``η ≥ 0``: a float, or a tensor broadcastable
            to ``precision`` (e.g. one value per row, ``[B, 1]``) whose entries
            are all finite and non-negative.

    Raises:
        ValueError: ``eta`` is negative or non-finite, or a tensor ``eta`` does
            not broadcast to the shape of ``precision``.
    """
    if isinstance(eta, Tensor):
        eta = eta.to(dtype=precision.dtype, device=precision.device)
        try:
            fits = torch.broadcast_shapes(eta.shape, precision.shape) == precision.shape
        except RuntimeError:
            fits = False
        if not fits:
            raise ValueError(
                f"eta shape {tuple(eta.shape)} must broadcast to precision shape "
                f"{tuple(precision.shape)}"
            )
        if not bool((torch.isfinite(eta) & (eta >= 0.0)).all()):
            raise ValueError("eta must be finite and >= 0")
        eta2 = eta.square()
    else:
        value = float(eta)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"eta must be finite and >= 0, got {eta!r}")
        if value == 0.0:
            return precision
        eta2 = value * value
    m2 = model.to(precision.dtype).square()
    return precision / (1.0 + eta2 * m2 * precision)


@functools.lru_cache(maxsize=64)
def student_t_efficiency(nu: float, noise: Literal["gaussian", "student_t"] = "gaussian") -> float:
    r"""Location efficiency ``E[ψ']² / E[ψ²]`` of the ``t_ν`` estimator.

    ``ψ(z) = (ν+1) z / (ν + z²)`` is the ``t_ν`` location score at unit scale. Its
    asymptotic variance is ``σ² E[ψ²]/E[ψ']²``, so the ratio is the information the
    ``t_ν`` M-estimator extracts per unit Gaussian information ``1/σ²``:

    * ``noise="gaussian"`` -- the noise is ``N(0, σ²)``; a closed form in
      ``erfc`` for ``ν < 3`` and 200-node Gauss--Hermite quadrature above, both
      within a few ``1e-15`` (relative) of high-precision quadrature
      (``≈ 0.942`` at ``ν = 5``, ``→ 0`` like ``√ν`` as ``ν → 0``);
    * ``noise="student_t"`` -- the noise is ``t_ν(0, σ²)``; the estimator is the
      MLE and the ratio is the Fisher factor ``(ν + 1)/(ν + 3)`` (``0.75`` at ``ν = 5``).

    Both are 1 for ``ν = ∞``. Multiply a Gaussian (inverse-variance) Fisher
    information by this factor to get the asymptotic ``t_ν`` information.
    """
    nu = _validated_nu(nu)
    if noise not in ("gaussian", "student_t"):
        raise ValueError(f"noise must be 'gaussian' or 'student_t', got {noise!r}")
    if math.isinf(nu):
        return 1.0
    if noise == "student_t":
        return (nu + 1.0) / (nu + 3.0)
    if nu < _EFFICIENCY_CLOSED_FORM_NU:
        # With J₁ = E[1/(ν+Z²)] = √(π/(2ν)) e^{ν/2} erfc(√(ν/2)) and, by parts,
        # J₂ = E[1/(ν+Z²)²] = (1 − (ν−1)J₁)/(2ν):
        # E[ψ²] = (ν+1)²(J₁ − νJ₂) = (ν+1)²((ν+1)J₁ − 1)/2 and
        # E[ψ'] = (ν+1)(2νJ₂ − J₁) = (ν+1)(1 − νJ₁).
        s = math.sqrt(0.5 * nu)
        tail = math.exp(s * s) * math.erfc(s)
        j1 = math.sqrt(0.5 * math.pi) / math.sqrt(nu) * tail
        nu_j1 = math.sqrt(0.5 * math.pi * nu) * tail
        return 2.0 * (1.0 - nu_j1) ** 2 / ((nu + 1.0) * j1 - 1.0)
    x, w = np.polynomial.hermite.hermgauss(_HERMITE_NODES)
    r = math.sqrt(2.0) * x
    psi = (nu + 1.0) * r / (nu + r * r)
    dpsi = (nu + 1.0) * (nu - r * r) / (nu + r * r) ** 2
    e_psi2 = float((w * psi * psi).sum()) / math.sqrt(math.pi)
    e_dpsi = float((w * dpsi).sum()) / math.sqrt(math.pi)
    return e_dpsi * e_dpsi / e_psi2


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
    - ν = ∞ (``math.inf``) → Gaussian NLL (½ (r/σ)² + log(σ) + ½ log(2π))

    The scale σ controls the soft threshold between quadratic behavior
    (small residuals) and logarithmic behavior (large residuals),
    analogous to Q in RHMF. For a per-element scale (known inverse variances)
    use the functional :func:`student_t_nll`.

    Args:
        nu: Degrees of freedom (> 0, or ``math.inf``). Default: 1.0 (Cauchy).
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
        if not nu > 0:
            raise ValueError(f"nu must be > 0, got {nu}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        self.nu = nu
        self.scale = scale

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
        # NLL = log_norm + ½(ν+1) log(1 + r²/(ν σ²)) + log(σ); log_norm = ½ log 2π at ν = ∞
        log_norm = _HALF_LOG_2PI if math.isinf(self.nu) else _student_t_log_norm(self.nu)
        nll = log_norm + _student_t_rho(scaled_sq, self.nu) + math.log(self.scale)

        return self._reduce(nll, mask, weights)

    def extra_repr(self) -> str:
        return f"nu={self.nu}, scale={self.scale}"
