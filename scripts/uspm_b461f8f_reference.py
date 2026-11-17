"""Frozen reference: uspm's own Student-t IRLS at commit b461f8f (pre-torchregress).

uspm (a downstream spectral-fitting package, MIT, same author) replaced these
functions with torchregress primitives after b461f8f, so a current uspm
checkout no longer carries an independent implementation to compare against.
This module keeps that implementation (``src/uspm/v2/operator.py``,
``src/uspm/v2/refine.py`` and ``src/uspm/likelihood/spectral.py`` at b461f8f;
only the imports and the line wrapping differ) as the reference of
``scripts/check_uspm_irls_parity.py``. It is not part of the package or the
test suite. Do not edit it except to re-freeze it from a named uspm commit.
"""

# ruff: noqa: RUF001, RUF002, RUF003
# mypy: ignore-errors

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

USPM_COMMIT = "b461f8f"


def diagonal_student_t_spectral_nll(
    residual: Tensor,
    inverse_variance: Tensor,
    valid: Tensor,
    *,
    degrees_of_freedom: float,
) -> Tensor:
    """Independent per-pixel Student-t NLL, summed per spectrum (``[batch]``).

    The formulation's diagonal law: ``scale_j = sigma_j sqrt((nu-2)/nu)`` so the
    delivered variance is preserved. Unlike the multivariate t, whose single
    shared scale only rescales the whole spectrum by its mean misfit, each pixel
    here has its own tail, so a cosmic ray or a sky residual is down-weighted
    against the other pixels rather than inflating them all (REVIEW F5).
    """
    if residual.ndim != 2:
        raise ValueError("residual must have shape [batch, pixel]")
    if inverse_variance.shape != residual.shape or valid.shape != residual.shape:
        raise ValueError("residual, inverse_variance, and valid must align")
    nu = float(degrees_of_freedom)
    if not math.isfinite(nu) or nu <= 2.0:
        raise ValueError("degrees_of_freedom must be finite and greater than 2")
    valid = valid.to(device=residual.device, dtype=torch.bool)
    inverse_variance = inverse_variance.to(device=residual.device, dtype=residual.dtype)
    usable = valid & torch.isfinite(residual) & torch.isfinite(inverse_variance)
    usable &= inverse_variance > 0.0
    if bool((valid & ~usable).any()):
        raise ValueError("valid spectral coordinates require finite residual and ivar > 0")
    safe_ivar = torch.where(usable, inverse_variance, torch.ones_like(residual))
    safe_residual = torch.where(usable, residual, torch.zeros_like(residual))
    scaled_ivar = safe_ivar * (nu / (nu - 2.0))
    standardized_sq = torch.square(safe_residual) * scaled_ivar
    constant = math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu) - 0.5 * math.log(math.pi * nu)
    log_density = (
        constant
        + 0.5 * torch.log(scaled_ivar)
        - 0.5 * (nu + 1.0) * torch.log1p(standardized_sq / nu)
    )
    return -torch.where(usable, log_density, torch.zeros_like(log_density)).sum(dim=-1)


def student_t_fisher_factor(nu: float) -> float:
    """Location Fisher information of ``t_ν(0, σ²)`` relative to the Gaussian."""
    return 1.0 if math.isinf(nu) else (nu + 1.0) / (nu + 3.0)


@functools.lru_cache(maxsize=16)
def student_t_gaussian_efficiency(nu: float) -> float:
    """``E[ψ']² / E[ψ²]`` of the ``t_ν`` location score ``ψ(r) = (ν+1) r/(ν+r²)``
    under ``r ~ N(0, 1)``: the information of the ``t_ν`` estimator (scale
    ``σ``) about a location when the noise is Gaussian with standard deviation
    ``σ``, relative to the Gaussian estimator's (1 at ``ν = ∞``; 0.942 at
    ``ν = 5``, where :func:`student_t_fisher_factor` assumes ``t_ν`` noise and
    gives 0.75).

    The Laplace σ of :func:`refine_mode` uses it: the photon σ and the
    template-error ``η m̄`` are Gaussian standard deviations
    (:class:`uspm.v2.nmf.EtaProfile`), so ``1/ivar_eff`` is the noise
    variance and ``σ_θ`` the t-estimator's asymptotic σ under it. 200-node
    Gauss–Hermite quadrature (the integrands are smooth; exact to round-off).
    """
    if math.isinf(nu):
        return 1.0
    x, w = np.polynomial.hermite.hermgauss(200)
    r = math.sqrt(2.0) * x
    psi = (nu + 1.0) * r / (nu + r * r)
    dpsi = (nu + 1.0) * (nu - r * r) / (nu + r * r) ** 2
    e_psi2 = float((w * psi * psi).sum()) / math.sqrt(math.pi)
    e_dpsi = float((w * dpsi).sum()) / math.sqrt(math.pi)
    return e_dpsi * e_dpsi / e_psi2


@dataclass(frozen=True)
class IRLSConfig:
    """IRLS settings.

    Attributes:
        nu: Student-t degrees of freedom, ``> 2`` (the scale ``σ = ivar^−½``
            needs a finite variance, :func:`student_t_nll`) or ``math.inf``
            for the Gaussian (one weighted least-squares solve).
        max_iter: iteration cap; non-converged fits are reported, not hidden.
        tol: converged when every coefficient moves by less than ``tol`` of its
            standard error (shared calibration: the projected Gauss–Newton
            step at the current weights).
        ridge: added to the unit diagonal of the Jacobi-scaled normal matrix
            only when its Cholesky factorisation fails (rank deficiency); fits
            that needed it are flagged. 1e−8 caps the scaled condition number
            near 1e8, well inside float64.
        eta: template-error scale ``η`` of FORMULATION_V2 §5.1: the variance
            is ``σ_p² + η² m̄_p²`` (:func:`effective_ivar`), ``m̄`` frozen at
            the previous iterate. A model parameter of the survey, estimated
            on held-out data (:func:`uspm.v2.nmf.train_nmf`); 0 is the
            photon-noise-only likelihood.
        calibration_sigma: widths of the Gaussian prior ``c_m ~ N(0, σ_m²)``
            on the shared calibration coefficients ``c_1..c_J``
            (FORMULATION_V2 §8.4; :func:`irls_fit_shared`), in the
            likelihood: ``σ_m`` is entry ``min(m, len) − 1`` (the last width
            repeats for higher degrees). Without it the gauge ``c_0 ≡ 1``
            fixes ``P`` only on average over the pixel range: where the
            source or the likelihood covers part of it, ``a → 0`` with
            ``|c| → ∞`` fits as well and the profile has no finite optimum.
            The default (0.2 for the tilt ``c_1``, the top of §8.4's 5–20 %
            spectrophotometric width, as no photometric centre is used yet;
            0.1 for ``j ≥ 2``, which are shrunk) is negligible next to the
            data wherever they constrain ``P`` (a BOSS spectrum at S/N 10
            carries ~10⁴ nats per unit ``c_1``) and binds in the directions
            they do not. Empty: no prior. The per-template ablation ignores it.
    """

    nu: float = 5.0
    max_iter: int = 50
    tol: float = 1.0e-3
    ridge: float = 1.0e-8
    eta: float = 0.0
    calibration_sigma: tuple[float, ...] = (0.2, 0.1)

    def __post_init__(self) -> None:
        sigma = tuple(float(v) for v in self.calibration_sigma)
        if not all(math.isfinite(v) and v > 0.0 for v in sigma):
            raise ValueError(
                f"calibration_sigma must be finite and > 0, got {self.calibration_sigma!r}"
            )
        object.__setattr__(self, "calibration_sigma", sigma)
        if not (self.nu > 2.0):
            raise ValueError(f"nu must be > 2 or inf, got {self.nu!r}")
        if not (math.isfinite(self.eta) and self.eta >= 0.0):
            raise ValueError(f"eta must be finite and >= 0, got {self.eta!r}")
        if self.max_iter < 1 or not (self.tol > 0.0) or not (self.ridge > 0.0):
            raise ValueError("max_iter >= 1, tol > 0 and ridge > 0 are required")


def calibration_precision(
    config: IRLSConfig, degree: int, *, dtype=torch.float64, device=None
) -> Tensor:
    """Prior precisions ``1/σ_m²`` of ``c_1..c_J`` (``[J]``; zeros without a prior)."""
    sigma = config.calibration_sigma
    if not sigma:
        return torch.zeros(degree, dtype=dtype, device=device)
    widths = [sigma[min(m, len(sigma)) - 1] for m in range(1, degree + 1)]
    return torch.tensor(widths, dtype=dtype, device=device).pow(-2)


def effective_ivar(ivar: Tensor, model: Tensor, eta: float) -> Tensor:
    """``1/(1/ivar + η² m̄²)`` = ``ivar/(1 + η² m̄² ivar)``; ``ivar`` itself at η = 0.

    The template-error variance of FORMULATION_V2 §5.1 at the frozen model
    ``m̄``. Zero ``ivar`` (no information) stays zero.
    """
    if eta == 0.0:
        return ivar
    m2 = model.to(ivar.dtype).square()
    return ivar / (1.0 + (eta * eta) * m2 * ivar)


def _irls_weights(iv: Tensor, residual: Tensor, model: Tensor, config: IRLSConfig):
    """IRLS weights ``ivar_eff (ν+1)/(ν + r² ivar_eff)`` (``ivar_eff`` for ν = ∞)."""
    ive = effective_ivar(iv, model, config.eta)
    if math.isinf(config.nu):
        return ive.clone() if ive is iv else ive
    r2 = residual.square() * ive
    return ive * (config.nu + 1.0) / (config.nu + r2)


def student_t_nll(residual: Tensor, ivar: Tensor, mask: Tensor, nu: float) -> Tensor:
    """Per-spectrum NLL of ``residual`` under ``t_ν(0, σ²)``, ``σ = ivar^−½``.

    Uses :func:`uspm.likelihood.spectral.diagonal_student_t_spectral_nll`,
    whose scale is ``σ√((ν−2)/ν)``, with ``ivar·(ν−2)/ν`` so the scale is ``σ``
    (the IRLS weight ``(ν+1)/(ν+r²)`` of FORMULATION_V2 §5.1). ``ν = ∞`` is
    the Gaussian. Requires ``ν > 2`` otherwise.
    """
    if math.isinf(nu):
        safe = torch.where(mask, ivar, torch.ones_like(ivar))
        terms = (
            0.5 * residual.square() * safe - 0.5 * torch.log(safe) + 0.5 * math.log(2.0 * math.pi)
        )
        return torch.where(mask, terms, torch.zeros_like(terms)).sum(-1)
    scaled = ivar * ((nu - 2.0) / nu)
    return diagonal_student_t_spectral_nll(
        torch.where(mask, residual, torch.zeros_like(residual)),
        torch.where(mask, scaled, torch.zeros_like(scaled)),
        mask,
        degrees_of_freedom=nu,
    )


@dataclass
class IRLSResult:
    """Profiled per-observation fit.

    ``chi2`` is the plain ``Σ ivar r²`` over likelihood pixels and ``dof`` is
    ``N_lik − n_params``; ``chi2_eff`` is ``Σ ivar_eff r²`` with the
    template-error variance (``effective_ivar`` ``[B, P]``, zero off the
    likelihood; equal to ``chi2`` at ``η = 0``); ``nll`` is the Student-t (or
    Gaussian) NLL there at the effective scale, normalisation included, so
    deviances on the same pixels compare across models and ``η``.
    ``failed`` rows (no more likelihood pixels than parameters, non-finite
    flux or ivar on a likelihood pixel, or a normal matrix that is singular
    even with the ridge) are not fitted: their coefficients, model, weights,
    ``chi2``, ``dof`` and ``nll`` are zero and ``converged`` is False, so sums
    over a batch skip them. Those zeros are not a fit: only a caller that
    reads ``failed`` may ask for them (``strict=False`` in :func:`irls_fit`
    and :func:`profile_batch`); by default an unfittable row raises.
    ``n_iter`` counts the iterations of this solve only (a warm-started solve
    counts from its warm start).

    ``coefficients`` is ``[B, K·(J+1)]`` per-template (:func:`irls_fit`) and
    ``[B, K + J]`` = ``(a_1..a_K, c_1..c_J)`` for the shared calibration
    (:func:`irls_fit_shared`); :meth:`PreparedBatch.amplitudes` reads either.
    ``free`` ``[B, n]`` marks the parameters free at the solution (not on a
    bound, not an empty column) and ``n_params`` counts them: ``dof =
    N_lik − n_params``. ``dead_columns`` counts empty template columns.

    ``prior_nll`` is the calibration prior ``½ Σ_m c_m²/σ_m²`` of the shared
    calibration (:attr:`IRLSConfig.calibration_sigma`; zero per-template), and
    ``nll`` includes it: the profiled objective.
    """

    coefficients: Tensor
    model: Tensor
    weights: Tensor
    chi2: Tensor
    dof: Tensor
    nll: Tensor
    n_iter: Tensor
    converged: Tensor
    ridge_used: Tensor
    dead_columns: Tensor
    failed: Tensor
    n_params: Tensor
    free: Tensor
    chi2_eff: Tensor
    effective_ivar: Tensor
    prior_nll: Tensor


def _solve_normal(
    design: Tensor, weights: Tensor, target: Tensor, ridge: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Weighted LS per row, Jacobi-scaled Cholesky; ridge only on failure.

    Returns coefficients, coefficient standard errors (from the weighted normal
    matrix), whether a ridge was needed, how many columns were empty, and
    which rows are singular even with the ridge (their coefficients and
    standard errors are zero; the caller decides what to do with them).
    """
    xw = design * weights.unsqueeze(-1)
    normal = design.transpose(1, 2) @ xw
    rhs = (xw * target.unsqueeze(-1)).sum(1)
    diag = torch.diagonal(normal, dim1=-2, dim2=-1)
    dead = diag <= 0.0
    scale = torch.where(dead, torch.ones_like(diag), diag.sqrt())
    scaled = normal / (scale.unsqueeze(-1) * scale.unsqueeze(-2))
    # Empty columns: unit diagonal and zero rhs pin the coefficient to 0.
    scaled = scaled + torch.diag_embed(dead.to(normal.dtype))
    chol, info = torch.linalg.cholesky_ex(scaled)
    failed = info != 0
    singular = torch.zeros_like(failed)
    if bool(failed.any()):
        eye = torch.eye(normal.shape[-1], dtype=normal.dtype, device=normal.device)
        chol2, info2 = torch.linalg.cholesky_ex(scaled + ridge * eye)
        singular = failed & (info2 != 0)
        # A singular row gets an identity factor, so the solve stays finite.
        chol2 = torch.where(singular[:, None, None], eye, chol2)
        chol = torch.where(failed[:, None, None], chol2, chol)
    rhs_s = torch.where(dead | singular[:, None], torch.zeros_like(rhs), rhs / scale)
    coef_s = torch.cholesky_solve(rhs_s.unsqueeze(-1), chol).squeeze(-1)
    inv_diag = torch.diagonal(torch.cholesky_inverse(chol), dim1=-2, dim2=-1)
    se = inv_diag.clamp_min(0.0).sqrt() / scale
    se = torch.where(singular[:, None], torch.zeros_like(se), se)
    coef = coef_s / scale
    return coef, se, failed, dead, singular


def _irls_start(
    flux: Tensor,
    ivar: Tensor,
    likelihood: Tensor,
    n_par: int,
    config: IRLSConfig,
    initial_weights: Tensor | None,
    strict: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``(f, iv, n_lik, failed, weights)``: the data on the likelihood and the start."""
    # Off the likelihood, flux and ivar do not enter (a NaN there must not
    # reach the weights through ``0 · NaN``).
    f = flux.to(torch.float64)
    f = torch.where(likelihood, f, torch.zeros_like(f))
    iv = torch.where(likelihood, ivar.to(torch.float64), torch.zeros_like(f))
    n_lik = likelihood.sum(-1)
    failed = n_lik <= n_par
    if strict and bool(failed.any()):
        raise ValueError("an observation has no more likelihood pixels than parameters")
    bad = likelihood & ~(torch.isfinite(f) & torch.isfinite(iv) & (iv >= 0.0))
    nonfinite = bad.any(-1)
    if strict and bool(nonfinite.any()):
        raise ValueError("an observation has non-finite flux or invalid ivar on a likelihood pixel")
    failed |= nonfinite
    if initial_weights is None or (math.isinf(config.nu) and config.eta == 0.0):
        weights = iv.clone()
    else:
        if initial_weights.shape != iv.shape:
            raise ValueError("initial_weights must be [B, P]")
        warm = torch.where(likelihood, initial_weights.to(torch.float64), torch.zeros_like(iv))
        valid = (torch.isfinite(warm) & (warm >= 0.0) & ((warm > 0.0) == (iv > 0.0))).all(-1)
        weights = torch.where(valid[:, None], warm, iv)
    return f, iv, n_lik, failed, weights


def _irls_result(
    coef: Tensor,
    model: Tensor,
    f: Tensor,
    iv: Tensor,
    likelihood: Tensor,
    n_lik: Tensor,
    failed: Tensor,
    n_params: Tensor,
    free: Tensor,
    config: IRLSConfig,
    prior_nll: Tensor | None = None,
    **state: Tensor,
) -> IRLSResult:
    """The result at the final coefficients: weights, χ², NLL, dof (module rules)."""
    residual = f - model
    r2 = residual.square() * iv
    weights = _irls_weights(iv, residual, model, config)
    use = likelihood & ~failed[:, None]
    weights = torch.where(use, weights, torch.zeros_like(weights))
    chi2 = torch.where(use, r2, torch.zeros_like(r2)).sum(-1)
    ive = effective_ivar(iv, model, config.eta)
    ive = torch.where(use, ive, torch.zeros_like(ive))
    if config.eta == 0.0:
        chi2_eff = chi2
    else:
        r2e = residual.square() * ive
        chi2_eff = torch.where(use, r2e, torch.zeros_like(r2e)).sum(-1)
    nll = student_t_nll(residual, ive, use, config.nu)
    if prior_nll is None:
        prior_nll = torch.zeros_like(nll)
    prior_nll = torch.where(failed, torch.zeros_like(prior_nll), prior_nll)
    nll = nll + prior_nll
    n_params = torch.where(failed, torch.zeros_like(n_params), n_params)
    dof = torch.where(failed, torch.zeros_like(n_lik), n_lik - n_params)
    return IRLSResult(
        coefficients=coef,
        model=torch.where(failed[:, None], torch.zeros_like(model), model),
        weights=weights,
        chi2=chi2,
        dof=dof,
        nll=nll,
        converged=state["converged"] & ~failed,
        n_iter=state["n_iter"],
        ridge_used=state["ridge_used"],
        dead_columns=state["dead_columns"],
        failed=failed,
        n_params=n_params,
        free=free & ~failed[:, None],
        chi2_eff=chi2_eff,
        effective_ivar=ive,
        prior_nll=prior_nll,
    )


def irls_fit(
    design: Tensor,
    flux: Tensor,
    ivar: Tensor,
    likelihood: Tensor,
    config: IRLSConfig,
    *,
    initial_weights: Tensor | None = None,
    strict: bool = True,
) -> IRLSResult:
    """Profile unconstrained linear coefficients per row of a batch by IRLS.

    The per-template calibration (and any linear design).

    Args:
        design: float64 ``[B, P, n]``.
        flux, ivar: ``[B, P]``.
        likelihood: bool ``[B, P]``; only these pixels enter the fit.
        initial_weights: optional ``[B, P]`` IRLS weights to start from (a
            warm start, e.g. the previous fit's ``weights`` of a nearby
            problem); ``ivar`` otherwise. The fixed point and the stopping
            rule are the same; the result moves within the tolerance. A row
            whose warm weights are not IRLS weights of this problem (non-finite,
            negative, or zero where ``ivar`` is positive, e.g. a failed fit's
            zeros) starts from ``ivar``.
        strict: raise on an unfittable row (``ValueError`` for too few
            likelihood pixels or non-finite data, ``RuntimeError`` for a
            normal matrix singular even with the ridge). With ``False`` such
            rows are flagged in ``failed`` instead of failing the batch
            (:class:`IRLSResult`); only callers that read ``failed`` may pass it.

    Each iteration solves only the rows that have not converged (the others
    keep their state), so a batch costs ``Σ n_iter`` row solves rather than
    ``B · max n_iter``.
    """
    if design.dtype != torch.float64:
        raise TypeError("design must be float64")
    n_par = design.shape[-1]
    f, iv, n_lik, failed, weights = _irls_start(
        flux, ivar, likelihood, n_par, config, initial_weights, strict
    )
    # One weighted solve suffices only without reweighting (Gaussian, η = 0).
    one_solve = math.isinf(config.nu) and config.eta == 0.0
    b = design.shape[0]
    converged = torch.zeros(b, dtype=torch.bool, device=design.device)
    n_iter = torch.zeros(b, dtype=torch.int64, device=design.device)
    ridge_used = torch.zeros(b, dtype=torch.bool, device=design.device)
    coef = design.new_zeros(b, n_par)
    dead = torch.zeros(b, n_par, dtype=torch.bool, device=design.device)
    tiny = torch.finfo(torch.float64).tiny
    for iteration in range(1, config.max_iter + 1):
        act = (~(converged | failed)).nonzero().squeeze(-1)
        if act.numel() == 0:
            break
        whole = act.numel() == b
        x = design if whole else design[act]
        f_a = f if whole else f[act]
        new, se, ridged, dead_a, singular = _solve_normal(
            x, weights if whole else weights[act], f_a, config.ridge
        )
        if strict and bool(singular.any()):
            raise RuntimeError("normal matrix is not positive definite even with the ridge")
        ridge_used[act] |= ridged
        dead[act] = dead_a
        failed[act] |= singular
        ok = ~singular
        step = ((new - coef[act]).abs() / se.clamp_min(tiny)).amax(-1)
        coef[act] = torch.where(ok[:, None], new, coef[act])
        n_iter[act] = iteration
        if one_solve:
            converged[act] = ok
            break
        done = ok & (step < config.tol) if iteration > 1 else torch.zeros_like(ok)
        converged[act] = done
        update = ok & ~done
        if bool(update.any()):
            model_a = (x @ coef[act].unsqueeze(-1)).squeeze(-1)
            iv_a = iv if whole else iv[act]
            fresh = _irls_weights(iv_a, f_a - model_a, model_a, config)
            weights[act] = torch.where(update[:, None], fresh, weights[act])
    coef = torch.where(failed[:, None], torch.zeros_like(coef), coef)
    model = (design @ coef.unsqueeze(-1)).squeeze(-1)
    return _irls_result(
        coef,
        model,
        f,
        iv,
        likelihood,
        n_lik,
        failed,
        n_par - dead.sum(-1),
        ~dead,
        config,
        converged=converged,
        n_iter=n_iter,
        ridge_used=ridge_used,
        dead_columns=dead.sum(-1),
    )


def _passive_cholesky(
    scaled: Tensor, passive: Tensor, ridge: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Cholesky of ``scaled`` on the passive block (identity elsewhere).

    Returns the factor, whether the ridge was needed and which rows are
    singular even with it (their factor is the identity).
    """
    n = scaled.shape[-1]
    eye = torch.eye(n, dtype=scaled.dtype, device=scaled.device)
    pair = passive.unsqueeze(-1) & passive.unsqueeze(-2)
    system = torch.where(pair, scaled, torch.zeros_like(scaled)) + torch.diag_embed(
        (~passive).to(scaled.dtype)
    )
    chol, info = torch.linalg.cholesky_ex(system)
    ridged = info != 0
    singular = torch.zeros_like(ridged)
    if bool(ridged.any()):
        chol2, info2 = torch.linalg.cholesky_ex(
            system + ridge * torch.diag_embed(passive.to(scaled.dtype))
        )
        singular = ridged & (info2 != 0)
        chol2 = torch.where(singular[:, None, None], eye, chol2)
        chol = torch.where(ridged[:, None, None], chol2, chol)
    return chol, ridged, singular


#: KKT tolerance of :func:`nnls_normal` on the Jacobi-scaled gradient, relative
#: to its largest entry: freeing a variable whose scaled gradient is below it
#: lowers the objective by less than ``½ (tol |g|_∞)²``, ~1e−13 χ² units at
#: S/N 10⁴ per observation.
NNLS_KKT_TOL = 1.0e-10


def nnls_normal(
    normal: Tensor,
    rhs: Tensor,
    n_constrained: int,
    start: Tensor | None = None,
    *,
    ridge: float = 1.0e-8,
    kkt_tol: float = NNLS_KKT_TOL,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``min ½ xᵀNx − gᵀx`` s.t. ``x_i ≥ 0`` for ``i < n_constrained``, per row.

    Batched Lawson–Hanson active-set method on the Jacobi-scaled normal
    equations (``N [B, n, n]`` positive semi-definite, ``g [B, n]``, float64);
    the remaining variables are free. Finite: each outer step frees the
    constrained variable with the largest scaled gradient (KKT multiplier)
    above ``kkt_tol · (1 + |g_s|_∞)``; each inner step solves the passive
    block and, when a constrained variable would go non-positive, moves to the
    first bound on the segment and pins it. ``start`` (feasible after
    clipping) warm-starts the passive set. Empty columns (zero diagonal) are
    pinned at 0. A passive block that is singular takes the ridge (flagged);
    a row singular even with it keeps its feasible iterate and is flagged.

    Returns ``(x, se, ridged, dead, singular, passive)``: the solution, the
    standard errors ``√(N_PP⁻¹)_ii`` on the final passive set ``P`` (``1/√N_ii``
    off it), and per row/variable flags.
    """
    b, n = rhs.shape
    dev, dt = rhs.device, rhs.dtype
    diag = torch.diagonal(normal, dim1=-2, dim2=-1)
    dead = diag <= 0.0
    scale = torch.where(dead, torch.ones_like(diag), diag.sqrt())
    scaled = normal / (scale.unsqueeze(-1) * scale.unsqueeze(-2))
    scaled = torch.where(
        (dead.unsqueeze(-1) | dead.unsqueeze(-2)), torch.zeros_like(scaled), scaled
    ) + torch.diag_embed(dead.to(dt))
    gs = torch.where(dead, torch.zeros_like(rhs), rhs / scale)
    constrained = torch.arange(n, device=dev) < n_constrained
    if start is None:
        x = torch.zeros_like(rhs)
    else:
        x = torch.where(dead, torch.zeros_like(rhs), start.to(dt) * scale)
        x = torch.where(constrained & ~(x > 0.0), torch.zeros_like(x), x)
    passive = ~dead & (~constrained | (x > 0.0))
    tol = kkt_tol * (1.0 + gs.abs().amax(-1, keepdim=True))
    ridged = torch.zeros(b, dtype=torch.bool, device=dev)
    singular = torch.zeros_like(ridged)
    going = torch.ones_like(ridged)
    inf = torch.full_like(x, math.inf)
    # The passive set of the last factorisation: the standard errors reuse it
    # when the set has not changed since (bit-identical, one Cholesky saved).
    fact_passive = None
    chol = None
    for _ in range(3 * n + 3):
        pending = going & ~singular
        for _ in range(n + 1):
            if not bool(pending.any()):
                break
            chol, r_used, sing = _passive_cholesky(scaled, passive, ridge)
            fact_passive = passive
            ridged |= r_used & pending
            singular |= sing & pending
            pending &= ~sing
            z = torch.cholesky_solve(
                torch.where(passive, gs, torch.zeros_like(gs)).unsqueeze(-1), chol
            ).squeeze(-1)
            z = torch.where(passive, z, torch.zeros_like(z))
            infeasible = pending.unsqueeze(-1) & passive & constrained & ~(z > 0.0)
            stuck = infeasible.any(-1)
            accept = pending & ~stuck
            x = torch.where(accept.unsqueeze(-1), z, x)
            # Move from x (feasible) towards z up to the first bound on the way.
            ratio = torch.where(infeasible, x / (x - z), inf)
            alpha = ratio.amin(-1, keepdim=True).clamp(0.0, 1.0)
            moved = x + alpha * (z - x)
            hit = stuck.unsqueeze(-1) & (
                (infeasible & (ratio <= alpha)) | (passive & constrained & ~(moved > 0.0))
            )
            x = torch.where(stuck.unsqueeze(-1), moved, x)
            x = torch.where(hit, torch.zeros_like(x), x)
            passive = passive & ~hit
            pending = stuck
        lam = gs - (scaled @ x.unsqueeze(-1)).squeeze(-1)
        cand = going.unsqueeze(-1) & constrained & ~passive & ~dead & (lam > tol)
        going = cand.any(-1) & ~singular
        if not bool(going.any()):
            break
        best = torch.where(cand, lam, -inf).argmax(-1)
        pick = torch.nn.functional.one_hot(best, n).to(torch.bool)
        passive = passive | (pick & going.unsqueeze(-1))
    if fact_passive is None or not bool(torch.equal(fact_passive, passive)):
        chol, _, _ = _passive_cholesky(scaled, passive, ridge)
    inv_diag = torch.diagonal(torch.cholesky_inverse(chol), dim1=-2, dim2=-1)
    se_s = torch.where(passive, inv_diag.clamp_min(0.0).sqrt(), torch.ones_like(x))
    return x / scale, se_s / scale, ridged, dead, singular, passive


#: Levenberg–Marquardt damping of :func:`irls_fit_shared`, relative to the
#: diagonal of the Gauss–Newton normal matrix: the first λ tried after an
#: undamped step overshoots, and the floor λ decays to after accepted steps
#: (below it the damped step equals the undamped one to ~1e−12 relative).
LM_START = 1.0e-3


LM_MIN = 1.0e-12


def irls_fit_shared(
    basis: Tensor,
    legendre: Tensor,
    flux: Tensor,
    ivar: Tensor,
    likelihood: Tensor,
    config: IRLSConfig,
    *,
    initial_weights: Tensor | None = None,
    initial_coefficients: Tensor | None = None,
    strict: bool = True,
    max_damping: int = 16,
) -> IRLSResult:
    """Profile ``m = P(x)·Σ_k a_k b_k``, ``a ≥ 0``, ``P = 1 + Σ_m c_m L_m``, by IRLS.

    Args:
        basis: float64 ``[B, P, K]``; legendre: float64 ``[B, P, J+1]``.
        flux, ivar, likelihood, initial_weights, strict: as :func:`irls_fit`.
        initial_coefficients: optional ``[B, K + J]`` start (e.g. the previous
            fit of a nearby problem; negative ``a`` are clipped to 0, a
            non-finite row starts from zero). Together with ``initial_weights``
            it moves the result within the tolerance.
        max_damping: Levenberg–Marquardt increases (×10 from the row's
            current λ, :data:`LM_START` at first) before a step is dropped.

    At fixed weights the problem is bilinear in ``(a, c)``. Each iteration
    takes one projected Gauss–Newton step: linearised about ``(a⁰, c⁰)``,
    ``m ≈ P⁰ B a + (Σ_{m≥1} c_m L_m) s⁰ − (P⁰ − 1) s⁰`` (``s⁰ = B a⁰``) is
    linear in the new ``(a, c)``, and its weighted least squares with
    ``a ≥ 0`` is solved exactly (:func:`nnls_normal`, warm-started on the
    current active set). When that step increases the weighted objective at
    the current weights, Levenberg–Marquardt damping (``λ diag N``, ×10 until
    it does not increase; :data:`LM_START`) replaces it. With Student-t weights that
    objective majorises the NLL at the frozen variance ``1/ivar + η² m̄²``
    (``log(1 + r²/ν)`` is concave in ``r²``), so at ``η = 0`` the reported
    NLL never increases. For ``η > 0``, ``m̄`` is the previous iterate's model
    (FORMULATION_V2 §5.1): each iteration lowers the NLL at the variance it
    froze, but the reported NLL (at the current model's variance) may rise
    between iterations; the iteration is a fixed-point iteration in ``m̄``,
    not a descent on that NLL. The weights are then refreshed. The first step
    from ``(0, 0)`` is the non-negative least squares at ``P ≡ 1``.

    Converged when the full (undamped) projected Gauss–Newton step at the
    current weights moves every parameter by less than ``tol`` of its
    standard error (from the Gauss–Newton normal matrix on the free set),
    after at least two iterations with Student-t weights; a damped step never
    counts, and a row that finds no decrease at any damping stops and is
    reported not converged (unless the predicted decrease is below the
    objective's float64 resolution). At the fixed point the KKT conditions of the
    profiled NLL hold: zero gradient in ``c`` and in every positive ``a_k``,
    a non-positive one in every ``a_k = 0``.

    The calibration prior ``½ Σ_m c_m²/σ_m²`` (:attr:`IRLSConfig.calibration_sigma`,
    FORMULATION_V2 §8.4) is part of the objective: it adds ``1/σ_m²`` to the
    ``c`` block of every Gauss–Newton normal matrix (hence to the standard
    errors) and to the weighted objective of the step acceptance, and
    ``nll`` includes it (``prior_nll``). It bounds ``c`` when the data fix
    ``P`` on part of the pixel range only, so ``a`` stays finite and
    identified instead of ``a → 0``, ``|c| → ∞``.

    A row whose ``a`` are all zero has no calibration (``c = 0``, the prior
    mode, no free parameter). ``coefficients`` are ``(a_1..a_K, c_1..c_J)``.
    """
    if basis.dtype != torch.float64 or legendre.dtype != torch.float64:
        raise TypeError("basis and legendre must be float64")
    b, _, k = basis.shape
    j = legendre.shape[-1] - 1
    n_par = k + j
    f, iv, n_lik, failed, weights = _irls_start(
        flux, ivar, likelihood, n_par, config, initial_weights, strict
    )
    # Without reweighting (Gaussian, η = 0) only the Gauss–Newton steps iterate.
    fixed_weights = math.isinf(config.nu) and config.eta == 0.0
    dev = basis.device
    cal = legendre[..., 1:]
    converged = torch.zeros(b, dtype=torch.bool, device=dev)
    n_iter = torch.zeros(b, dtype=torch.int64, device=dev)
    ridge_used = torch.zeros(b, dtype=torch.bool, device=dev)
    dead = torch.zeros(b, k, dtype=torch.bool, device=dev)
    theta = basis.new_zeros(b, n_par)
    if initial_coefficients is not None:
        if initial_coefficients.shape != theta.shape:
            raise ValueError(f"initial_coefficients must be [B, {n_par}]")
        warm = initial_coefficients.to(theta)
        usable = torch.isfinite(warm).all(-1, keepdim=True)
        warm = torch.where(usable, warm, theta)
        warm[:, :k] = warm[:, :k].clamp_min(0.0)
        no_source = ~(warm[:, :k] > 0.0).any(-1, keepdim=True)
        warm[:, k:] = torch.where(no_source, torch.zeros_like(warm[:, k:]), warm[:, k:])
        theta = warm
    tiny = torch.finfo(torch.float64).tiny
    # The calibration prior ½ Σ c_m²/σ_m² (in the NLL; ×2 in the weighted
    # objective Σ w r², whose half majorises the NLL). The linearised unknowns
    # are the new c themselves, so it adds to the normal matrix only.
    precision = calibration_precision(config, j, device=dev)
    prior_block = torch.diag_embed(torch.cat([precision.new_zeros(k), precision])).unsqueeze(0)

    def objective(x, leg, f_a, w, par):
        poly = 1.0 + (leg @ par[:, k:].unsqueeze(-1)).squeeze(-1)
        source = (x @ par[:, :k].unsqueeze(-1)).squeeze(-1)
        q = (w * (f_a - poly * source).square()).sum(-1)
        return q + (precision * par[:, k:].square()).sum(-1), poly * source

    stalled = torch.zeros(b, dtype=torch.bool, device=dev)
    damping = torch.full((b,), LM_START, dtype=torch.float64, device=dev)
    for iteration in range(1, config.max_iter + 1):
        act = (~(converged | failed | stalled)).nonzero().squeeze(-1)
        if act.numel() == 0:
            break
        whole = act.numel() == b
        x = basis if whole else basis[act]
        leg = cal if whole else cal[act]
        f_a = f if whole else f[act]
        w = weights if whole else weights[act]
        par0 = theta[act]
        poly = 1.0 + (leg @ par0[:, k:].unsqueeze(-1)).squeeze(-1)
        source = (x @ par0[:, :k].unsqueeze(-1)).squeeze(-1)
        z = torch.cat([poly.unsqueeze(-1) * x, leg * source.unsqueeze(-1)], -1)
        target = f_a + (poly - 1.0) * source
        zw = z * w.unsqueeze(-1)
        normal = z.transpose(1, 2) @ zw + prior_block
        rhs = (zw * target.unsqueeze(-1)).sum(1)
        new, se, ridged, dead_a, singular, _ = nnls_normal(normal, rhs, k, par0, ridge=config.ridge)
        if strict and bool(singular.any()):
            raise RuntimeError("normal matrix is not positive definite even with the ridge")
        ridge_used[act] |= ridged
        dead[act] = dead_a[:, :k]
        failed[act] |= singular
        ok = ~singular
        step = new - par0
        q0, _ = objective(x, leg, f_a, w, par0)
        # The full (undamped) projected Gauss–Newton step decides convergence:
        # at a KKT point of the weighted problem it is zero. A step below the
        # tolerance is taken as it is: it changes the objective at round-off
        # level, where the comparison means nothing.
        sub_tol = ok & ((step.abs() / se.clamp_min(tiny)).amax(-1) < config.tol)
        par = par0.clone()
        accepted = sub_tol.clone()
        par[accepted] = new[accepted]
        pending = ok & ~accepted
        q1 = q0
        if bool(pending.any()):
            q1, _ = objective(x, leg, f_a, w, new)
            good = pending & (q1 < q0)
            par[good] = new[good]
            accepted |= good
            pending &= ~good
            lam_a = damping[act]
            damping[act] = torch.where(good, (lam_a / 10.0).clamp_min(LM_MIN), lam_a)
        # Levenberg–Marquardt: the undamped step overshoots (a curved
        # bilinear valley), so solve the damped problem
        # (N + λ diag N) x = g + λ diag(N) θ⁰ with the bounds, λ ×10 until
        # the objective does not increase. Damping turns the step towards the
        # scaled gradient instead of shortening it along the overshooting
        # direction: halving crawled along such valleys for hundreds of
        # iterations (tests/v2/test_shared_calibration.py).
        lam = damping[act].clone()
        diag_n = torch.diagonal(normal, dim1=-2, dim2=-1)
        for _ in range(max_damping):
            rows = pending.nonzero().squeeze(-1)
            if rows.numel() == 0:
                break
            d = diag_n[rows] * lam[rows].unsqueeze(-1)
            trial, _, _, _, sing_d, _ = nnls_normal(
                normal[rows] + torch.diag_embed(d),
                rhs[rows] + d * par0[rows],
                k,
                par0[rows],
                ridge=config.ridge,
            )
            q, _ = objective(x[rows], leg[rows], f_a[rows], w[rows], trial)
            good = ~sing_d & (q < q0[rows])
            hit = rows[good]
            par[hit] = trial[good]
            accepted[hit] = True
            pending[hit] = False
            damping[act[hit]] = (lam[hit] / 10.0).clamp_min(LM_MIN)
            lam[rows[~good]] *= 10.0
        # Convex combinations of feasible points; clip the round-off.
        par[:, :k] = par[:, :k].clamp_min(0.0)
        no_source = ~(par[:, :k] > 0.0).any(-1, keepdim=True)
        par[:, k:] = torch.where(no_source, torch.zeros_like(par[:, k:]), par[:, k:])
        theta[act] = torch.where(ok.unsqueeze(-1), par, par0)
        n_iter[act] = iteration
        # Never converged on a damped or dropped step: a damped step moves
        # the parameters by less than tol while the full step is still ~σ.
        # A row with no decrease even at the largest damping makes no
        # progress at these weights: it stops, and it is converged only when
        # no decrease could have been seen. Along the full step d the
        # objective is φ(t) = q(θ⁰ + t d), φ'(0) = −2 Δ with Δ the decrease
        # the linearised problem predicts; the quadratic through φ(0), φ'(0)
        # and φ(1) allows at most Δ² / (φ(1) − φ(0) + 2Δ), and when that is
        # below the float64 resolution of φ (a sum of n_lik terms:
        # n_lik·ε·q0) the row is at its optimum to machine precision (the
        # GN model misses the bilinear curvature, so Δ alone overstates it).
        stuck = ok & ~accepted
        resolved = stuck & ~sub_tol
        if bool(resolved.any()):
            lin = target - (z @ new.unsqueeze(-1)).squeeze(-1)
            q_lin = (w * lin.square()).sum(-1)
            predicted = q0 - q_lin - (precision * new[:, k:].square()).sum(-1)
            curvature = q1 - q0 + 2.0 * predicted
            reachable = torch.where(
                curvature > 0.0,
                predicted.square() / curvature.clamp_min(tiny),
                predicted,
            )
            n_a = n_lik if whole else n_lik[act]
            floor = n_a.to(q0.dtype) * torch.finfo(q0.dtype).eps * q0.abs()
            sub_tol = sub_tol | (resolved & (reachable <= floor))
        stalled[act] |= stuck
        if fixed_weights:
            converged[act] = sub_tol
            continue
        done = sub_tol if iteration > 1 else torch.zeros_like(ok)
        converged[act] = done
        update = ok & ~done & ~stuck
        if bool(update.any()):
            _, model_a = objective(x, leg, f_a, w, theta[act])
            iv_a = iv if whole else iv[act]
            fresh = _irls_weights(iv_a, f_a - model_a, model_a, config)
            weights[act] = torch.where(update.unsqueeze(-1), fresh, weights[act])
    theta = torch.where(failed.unsqueeze(-1), torch.zeros_like(theta), theta)
    positive = theta[:, :k] > 0.0
    any_positive = positive.any(-1, keepdim=True)
    free = torch.cat([positive, any_positive.expand(-1, j)], -1)
    poly = 1.0 + (cal @ theta[:, k:].unsqueeze(-1)).squeeze(-1)
    model = poly * (basis @ theta[:, :k].unsqueeze(-1)).squeeze(-1)
    return _irls_result(
        theta,
        model,
        f,
        iv,
        likelihood,
        n_lik,
        failed,
        free.sum(-1),
        free,
        config,
        0.5 * (precision * theta[:, k:].square()).sum(-1),
        converged=converged,
        n_iter=n_iter,
        ridge_used=ridge_used,
        dead_columns=dead.sum(-1),
    )
