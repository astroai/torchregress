"""
Gaussian 2-Wasserstein **bound surrogate**: squared L2 on means plus squared Frobenius
distance between principal matrix square roots of covariances.

This implements the common upper-bound objective

.. math::

    \\|\\hat\\mu - \\mu\\|_2^2 + \\|\\hat\\Sigma^{1/2} - \\Sigma^{1/2}\\|_F^2

where :math:`\\Sigma^{1/2}` is the **symmetric positive semi-definite** principal matrix
square root (computed here via eigen-decomposition). It is a useful training signal for
joint mean–covariance supervision; it is **not** the exact Gaussian 2-Wasserstein
distance in full generality.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, cast

import torch

from .base import BaseLoss
from .loss_registry import register_regression_loss

CovarianceParameterization = Literal["diagonal", "covariance", "cholesky", "sqrt"]


class _SymmetricSqrt(torch.autograd.Function):
    """Principal square root of symmetric PSD matrices with a Sylvester backward.

    ``torch.linalg.eigh``'s backward contains ``1 / (lambda_i - lambda_j)`` and
    is NaN whenever two eigenvalues coincide (e.g. an isotropic ``sigma^2 I``),
    although the derivative of :math:`\\Sigma^{1/2}` is finite there. For
    :math:`X = \\Sigma^{1/2}` the differential solves the Sylvester equation
    :math:`X\\,dX + dX\\,X = d\\Sigma`, so in the eigenbasis
    :math:`\\Sigma = Q\\Lambda Q^\\top` the vector-Jacobian product is

    .. math::

        \\bar\\Sigma = Q\\left[\\frac{(Q^\\top G Q)_{ij}}{s_i + s_j}\\right]Q^\\top,
        \\qquad s = \\Lambda^{1/2},

    with :math:`G` the symmetrised output gradient, i.e. the divided difference
    of :math:`\\sqrt{\\cdot}`, which is finite for
    repeated eigenvalues. Entries involving eigenvalues clamped at ``eps``
    use the divided difference of the clamped map (zero derivative below
    ``eps``).
    """

    @staticmethod
    def forward(ctx: Any, sigma: torch.Tensor, eps: float) -> torch.Tensor:
        evals, vecs = torch.linalg.eigh(sigma)
        s = torch.clamp(evals, min=eps).sqrt()
        ctx.save_for_backward(evals, vecs, s)
        ctx.eps = eps
        return (vecs * s.unsqueeze(-2)) @ vecs.transpose(-1, -2)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, None]:
        evals, vecs, s = ctx.saved_tensors
        # sigma is symmetric by contract: return the symmetric gradient.
        g = 0.5 * (grad_out + grad_out.transpose(-1, -2))
        g_eig = vecs.transpose(-1, -2) @ g @ vecs
        s_i, s_j = s.unsqueeze(-1), s.unsqueeze(-2)
        kernel = 1.0 / (s_i + s_j)
        clamped = evals < ctx.eps
        any_clamped = clamped.unsqueeze(-1) | clamped.unsqueeze(-2)
        if bool(any_clamped.any()):
            # Divided difference (f(l_i) - f(l_j)) / (l_i - l_j) of the clamped
            # map f(l) = sqrt(max(l, eps)); zero when both are clamped.
            dl = evals.unsqueeze(-1) - evals.unsqueeze(-2)
            nonzero = dl != 0
            safe_dl = torch.where(nonzero, dl, torch.ones_like(dl))
            divided = torch.where(nonzero, (s_i - s_j) / safe_dl, torch.zeros_like(dl))
            kernel = torch.where(any_clamped, divided, kernel)
        grad_sigma = vecs @ (g_eig * kernel) @ vecs.transpose(-1, -2)
        return grad_sigma, None


def symmetric_spd_matrix_sqrt(sigma: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """
    Principal matrix square root of batched symmetric positive semi-definite matrices.

    Uses ``torch.linalg.eigh``. For SPD :math:`\\Sigma = Q \\Lambda Q^\\top`, returns
    :math:`Q \\Lambda^{1/2} Q^\\top` with eigenvalues clamped below by ``eps`` before
    the square root.

    The backward pass solves the Sylvester equation
    :math:`X\\,dX + dX\\,X = d\\Sigma` in the eigenbasis instead of
    differentiating through ``eigh``, so gradients stay finite for repeated
    eigenvalues (e.g. isotropic covariances :math:`\\sigma^2 I`). The returned
    gradient w.r.t. ``sigma`` is symmetric. Double backward is not supported.

    Parameters
    ----------
    sigma : torch.Tensor
        Symmetric PSD matrices of shape ``[..., D, D]``.
    eps : float, default=1e-8
        Floor applied to the eigenvalues before the square root.

    Returns
    -------
    torch.Tensor
        Symmetric PSD square roots, same shape, dtype and device as ``sigma``.
    """
    return cast(torch.Tensor, _SymmetricSqrt.apply(sigma, eps))


def _batch_frobenius_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a - b
    return (diff * diff).sum(dim=(-2, -1))


def _ensure_batch_matrix(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"{name} must be 2D [D, D] or 3D [B, D, D], got shape {tuple(x.shape)}")


def _align_batch_matrix(x: torch.Tensor, batch: int, *, name: str) -> torch.Tensor:
    xb = _ensure_batch_matrix(x, name=name)
    if xb.shape[0] == 1 and batch > 1:
        return xb.expand(batch, -1, -1)
    if xb.shape[0] != batch:
        raise ValueError(f"{name} batch {xb.shape[0]} does not match pred_mean batch {batch}")
    return xb


@register_regression_loss("gaussian_wasserstein_bound")
class GaussianWassersteinBoundLoss(BaseLoss):
    """
    Surrogate loss for supervised Gaussian mean and covariance.

    Per sample (before reduction):

    .. math::

        \\lambda_{\\mu} \\|\\hat\\mu - \\mu\\|_2^2
        + \\lambda_{\\Sigma} \\|\\hat S - S\\|_F^2

    where in ``"covariance"`` and ``"cholesky"`` modes :math:`\\hat S` and :math:`S`
    are principal matrix square roots of the corresponding SPD covariances (with
    ``jitter`` added on the diagonal for numerical stability). In ``"sqrt"`` mode the
    tensors **are** those roots. In ``"diagonal"`` mode the covariance term is
    :math:`\\sum_i (\\sqrt{\\hat v_i} - \\sqrt{v_i})^2` for positive variances
    :math:`\\hat v_i, v_i`.

    Args:
        covariance_parameterization: How ``pred_covariance`` and ``target_covariance``
            are interpreted: ``"diagonal"`` (positive variances), ``"covariance"`` (SPD
            matrices), ``"cholesky"`` (lower-triangular :math:`L` with
            :math:`\\Sigma = L L^\\top`), or ``"sqrt"`` (symmetric PSD roots :math:`S`
            with :math:`\\Sigma \\approx S S` for symmetric :math:`S`).
        mean_weight: Multiplier for the mean squared error term.
        covariance_weight: Multiplier for the covariance / root term.
        eps: Floor inside diagonal square roots.
        jitter: Added to the diagonal of full matrices before ``matrix_sqrt``.
        reduction: ``"mean"``, ``"sum"``, ``"none"``, ``"min"``, or ``"max"``.
    """

    def __init__(
        self,
        *,
        covariance_parameterization: CovarianceParameterization = "covariance",
        mean_weight: float = 1.0,
        covariance_weight: float = 1.0,
        eps: float = 1e-8,
        jitter: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        allowed: tuple[str, ...] = ("diagonal", "covariance", "cholesky", "sqrt")
        if covariance_parameterization not in allowed:
            raise ValueError(
                f"covariance_parameterization must be one of {allowed}, "
                f"got {covariance_parameterization!r}"
            )
        self.covariance_parameterization = covariance_parameterization
        self.mean_weight = float(mean_weight)
        self.covariance_weight = float(covariance_weight)
        self.eps = float(eps)
        self.jitter = float(jitter)

    def _spd_with_jitter(self, sigma: torch.Tensor) -> torch.Tensor:
        d = sigma.shape[-1]
        eye = torch.eye(d, device=sigma.device, dtype=sigma.dtype)
        return sigma + self.jitter * eye

    def _matrix_sqrt(self, sigma: torch.Tensor) -> torch.Tensor:
        return symmetric_spd_matrix_sqrt(self._spd_with_jitter(sigma), eps=self.eps)

    def _covariance_from_cholesky(self, scale_tril: torch.Tensor) -> torch.Tensor:
        return scale_tril @ scale_tril.transpose(-1, -2)

    def forward(
        self,
        pred_mean: torch.Tensor,
        target_mean: torch.Tensor,
        pred_covariance: torch.Tensor,
        target_covariance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        if pred_mean.shape != target_mean.shape:
            raise ValueError(
                f"pred_mean shape {tuple(pred_mean.shape)} must match "
                f"target_mean shape {tuple(target_mean.shape)}"
            )
        if pred_mean.dim() < 2:
            raise ValueError("pred_mean and target_mean must be at least 2D [batch, dim]")

        batch, dim = pred_mean.shape[0], pred_mean.shape[-1]
        mean_term = (pred_mean - target_mean).pow(2).sum(dim=-1)

        mode = self.covariance_parameterization
        if mode == "diagonal":
            if (
                pred_covariance.shape != pred_mean.shape
                or target_covariance.shape != pred_mean.shape
            ):
                raise ValueError(
                    "In diagonal mode pred_covariance and target_covariance must match "
                    f"pred_mean shape {tuple(pred_mean.shape)}"
                )
            vp = pred_covariance.clamp(min=self.eps)
            vt = target_covariance.clamp(min=self.eps)
            cov_term = (vp.sqrt() - vt.sqrt()).pow(2).sum(dim=-1)
        else:
            pc = _align_batch_matrix(pred_covariance, batch, name="pred_covariance")
            tc = _align_batch_matrix(target_covariance, batch, name="target_covariance")
            if pc.shape[-2:] != (dim, dim) or tc.shape[-2:] != (dim, dim):
                raise ValueError(
                    f"In {mode} mode covariance tensors must end with [D, D] = "
                    f"[{dim}, {dim}], got pred {tuple(pc.shape)} and target {tuple(tc.shape)}"
                )
            if mode == "sqrt":
                cov_term = _batch_frobenius_squared(pc, tc)
            elif mode == "covariance":
                cov_term = _batch_frobenius_squared(self._matrix_sqrt(pc), self._matrix_sqrt(tc))
            elif mode == "cholesky":
                sig_p = self._covariance_from_cholesky(pc)
                sig_t = self._covariance_from_cholesky(tc)
                cov_term = _batch_frobenius_squared(
                    self._matrix_sqrt(sig_p),
                    self._matrix_sqrt(sig_t),
                )
            else:
                raise RuntimeError(f"unhandled mode {mode!r}")

        loss_per_sample = self.mean_weight * mean_term + self.covariance_weight * cov_term
        return self._reduce(loss_per_sample, mask, weights)


def gaussian_wasserstein_bound_loss(
    pred_mean: torch.Tensor,
    target_mean: torch.Tensor,
    pred_covariance: torch.Tensor,
    target_covariance: torch.Tensor,
    *,
    covariance_parameterization: CovarianceParameterization = "covariance",
    mean_weight: float = 1.0,
    covariance_weight: float = 1.0,
    eps: float = 1e-8,
    jitter: float = 1e-6,
    reduction: str = "mean",
    mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional form of :class:`GaussianWassersteinBoundLoss`."""
    fn = GaussianWassersteinBoundLoss(
        covariance_parameterization=covariance_parameterization,
        mean_weight=mean_weight,
        covariance_weight=covariance_weight,
        eps=eps,
        jitter=jitter,
        reduction=reduction,
    )
    return cast(
        torch.Tensor,
        fn(pred_mean, target_mean, pred_covariance, target_covariance, mask=mask, weights=weights),
    )
