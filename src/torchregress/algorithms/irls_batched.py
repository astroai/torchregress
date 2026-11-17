"""Batched IRLS for many independent small regression problems."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType

import torch
from torch import Tensor

from ..losses.student_t import effective_precision, student_t_nll
from .irls_fit import (
    RobustWeightName,
    WeightFunction,
    _robust_objective,
    _student_t_nu,
    robust_weights,
)

_RUNNING = -1


class IRLSRowStatus(IntEnum):
    """Per-row outcome of :func:`batched_iteratively_reweighted_fit` (``int8`` codes).

    Correspondence with :data:`IRLSTermination` of :func:`iteratively_reweighted_fit`:
    ``CONVERGED`` is ``"converged"``, ``MAX_ITERATIONS`` is ``"max_iterations"``,
    ``EMPTY`` is ``"empty"``, ``FAILED`` covers ``"breakdown"`` (a singular or
    failed solve) and ``"non_finite"``; ``STALLED`` (no damped step accepted) has
    no single-problem counterpart.
    """

    CONVERGED = 0
    """The full step at the current weights moved every parameter by < ``step_tol`` σ."""
    MAX_ITERATIONS = 1
    """``max_iter`` reached before convergence."""
    STALLED = 2
    """No decrease of the weighted objective at any damping: no progress possible."""
    FAILED = 3
    """Non-finite data, step, weights or objective, a singular solve, or too few
    observations. Row outputs are ``NaN``."""
    EMPTY = 4
    """No observation with positive precision. Row outputs are ``NaN``."""


@dataclass(frozen=True)
class BatchedIRLSStep:
    """What a caller's weighted solve returns for the rows it was given.

    Attributes:
        theta: ``[b, n]`` parameters after the full (undamped) step at the
            supplied weights.
        standard_errors: ``[b, n]`` parameter standard errors at those weights
            (e.g. from the weighted normal matrix); the convergence test divides
            the step by them. ``inf`` marks a parameter with no information.
            These are the conditional errors of the weighted least-squares solve,
            not asymptotic ``t_ν`` errors (see :class:`BatchedIRLSFitResult`).
        failed: optional ``[b]`` boolean mask of rows the solver could not solve
            (e.g. a singular normal matrix); they are marked ``FAILED``.
        predicted_objective: optional ``[b]`` value of the solver's local model
            (e.g. Gauss--Newton) of the weighted objective ``q`` at ``theta``. With
            a damped solve it lets a row whose step no damping accepts be declared
            converged when the predicted decrease is non-negative (up to
            round-off) and the decrease still reachable is below the float
            resolution of ``q``.
        ridge_used: optional ``[b]`` boolean mask of rows whose solve was
            regularised (e.g. :attr:`NormalEquationsSolution.ridge_used`: a
            rank-deficient normal matrix). Accumulated per row into
            :attr:`BatchedIRLSFitResult.ridge_used` so callers can fail closed.
    """

    theta: Tensor
    standard_errors: Tensor
    failed: Tensor | None = None
    predicted_objective: Tensor | None = None
    ridge_used: Tensor | None = None


@dataclass(frozen=True)
class BatchedIRLSProblem:
    """Caller-owned model and weighted solve of a batch of independent rows.

    Every callback receives ``rows``: ``None`` when it is called on the whole
    batch in order, otherwise the ``int64`` global indices of the rows it gets
    (finished rows are dropped from later iterations). Tensors passed in hold
    only those rows; ``weights`` and ``target`` are the controller's working
    buffers, valid for the duration of the call only (they are compacted and
    overwritten in place later): do not keep or modify them. Any object with
    these attributes is accepted in place of this dataclass.

    Contract on ``rows``: every callback of one iteration (``solve``, the
    weight refresh's ``predict`` and ``penalty``) receives the identical object,
    and the object is reused across iterations until rows finish; the callbacks
    of one decrease test or one damping attempt (on a subset of the rows) also
    share one object. Per-row gathers of the caller's operator (basis, design,
    calibration polynomials) can thus be cached on ``rows is last_rows``.

    Attributes:
        predict: ``predict(theta, rows) -> model`` ``[b, N]``.
        solve: ``solve(theta, weights, target, rows) -> BatchedIRLSStep``. The full
            step of ``min_θ Σ_i w_i (y_i − m_i(θ))² + 2·penalty(θ)`` at the given
            weights ``[b, N]`` (zero on masked or zero-precision entries, where
            ``target`` is zero too). Bounds, priors and linearisations live here
            (see :func:`solve_normal_equations`).
        damped_solve: optional ``damped_solve(theta, weights, target, rows,
            damping) -> theta`` ``[b, n]``: a damped step (e.g. Levenberg--Marquardt
            ``N + λ diag N``) for the per-row ``damping`` ``λ`` ``[b]``. When given,
            a full step is accepted only if it lowers the weighted objective;
            otherwise ``λ`` grows ×10 until a damped step does, and a damped step
            never counts towards convergence.
        penalty: optional ``penalty(theta, rows) -> [b]`` (e.g. a Gaussian prior
            ``½ Σ c²/σ²``) in negative-log-likelihood units, added to the reported
            objective and, doubled, to the weighted objective. Its curvature
            belongs in ``solve``.
        exact_solve: ``solve`` returns the exact minimiser at the given weights
            (linear least squares, NNLS). With weights that do not depend on the
            residuals one solve then converges: its result is taken as it is,
            without the decrease test of ``damped_solve`` (the start may be
            infeasible and fit better than any feasible point).
    """

    predict: Callable[[Tensor, Tensor | None], Tensor]
    solve: Callable[[Tensor, Tensor, Tensor, Tensor | None], BatchedIRLSStep]
    damped_solve: Callable[[Tensor, Tensor, Tensor, Tensor | None, Tensor], Tensor] | None = None
    penalty: Callable[[Tensor, Tensor | None], Tensor] | None = None
    exact_solve: bool = False


@dataclass(frozen=True)
class BatchedIRLSFitConfig:
    """Settings of :func:`batched_iteratively_reweighted_fit`.

    Attributes:
        weight: robust family (``"student_t"`` by default; any
            :func:`robust_weights` name or a callable of the standardized residual).
        weight_params: family parameters (empty by default; ``"student_t"`` uses
            ``nu=5`` unless given, and ``nu=math.inf`` is the Gaussian).
        eta: fractional model error ``η ≥ 0`` of the variance ``1/λ + η² m̄²``
            (:func:`effective_precision`), ``m̄`` the previous iterate's model.
        max_iter: iteration cap per row.
        step_tol: convergence threshold on ``max_k |Δθ_k| / se_k``, the largest
            full-step change in standard-error units. Unlike the relative-norm
            :attr:`IRLSFitConfig.parameter_tol`, it is scale-free per parameter.
        min_iter: iterations required before a row may converge when the weights
            depend on the residuals (the first solve uses the starting weights).
        min_observations: observations with positive precision a row needs to be
            fitted; ``None`` means one more than the number of parameters.
        max_damping: damped solves tried per iteration before a row stalls.
        damping_start: initial damping ``λ``; divided by 10 after every accepted
            step (down to ``damping_min``).
        damping_min: floor of ``λ``.
    """

    weight: RobustWeightName | WeightFunction = "student_t"
    weight_params: Mapping[str, float] = field(default_factory=dict)
    eta: float = 0.0
    max_iter: int = 50
    step_tol: float = 1.0e-3
    min_iter: int = 2
    min_observations: int | None = None
    max_damping: int = 16
    damping_start: float = 1.0e-3
    damping_min: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.max_iter < 1 or self.min_iter < 1 or self.max_damping < 0:
            raise ValueError("max_iter >= 1, min_iter >= 1 and max_damping >= 0 are required")
        if not (math.isfinite(self.step_tol) and self.step_tol > 0.0):
            raise ValueError("step_tol must be finite and positive")
        if not (math.isfinite(self.eta) and self.eta >= 0.0):
            raise ValueError("eta must be finite and >= 0")
        if not (0.0 < self.damping_min <= self.damping_start < math.inf):
            raise ValueError("0 < damping_min <= damping_start < inf is required")
        if self.min_observations is not None and self.min_observations < 1:
            raise ValueError("min_observations must be >= 1 or None")
        object.__setattr__(self, "weight_params", MappingProxyType(dict(self.weight_params)))
        # Validate the family and its parameters once, eagerly.
        robust_weights(torch.zeros(1, dtype=torch.float64), self.weight, self.weight_params)


@dataclass(frozen=True)
class BatchedIRLSFitResult:
    """Per-row result of :func:`batched_iteratively_reweighted_fit`.

    Rows that are ``FAILED`` or ``EMPTY`` carry ``NaN`` in every floating output.

    Attributes:
        theta: ``[B, n]`` parameters.
        standard_errors: ``[B, n]`` from the last weighted solve of the row: the
            conditional errors of that weighted least-squares problem, which set
            the convergence scale. They are *not* asymptotic ``t_ν`` errors: the
            Student-t IRLS weights average 1 under ``t_ν`` noise, so their normal
            matrix estimates the Gaussian information and these errors are too
            small by about ``√((ν+3)/(ν+1))`` under ``t_ν`` noise. For asymptotic
            errors invert ``student_t_efficiency(ν, noise) · Jᵀ diag(λ_eff) J``
            (the normal matrix of ``effective_precision`` without the IRLS
            weights), or use a sandwich estimate.
        model: ``[B, N]`` model at ``theta`` (values off the likelihood are
            returned as ``predict`` gives them and not checked).
        weights: ``[B, N]`` IRLS weights ``λ_eff ψ(u)/u`` at ``theta`` (zero off the
            likelihood: masked or zero-precision entries).
        effective_precision: ``[B, N]`` ``λ_eff = 1/(1/λ + η² m²)`` at ``theta``
            (zero off the likelihood).
        objective: ``[B]`` exact Student-t (Gaussian at ``ν = ∞``) negative
            log-likelihood at ``λ_eff``, constants included, plus ``penalty``; for
            other families ``Σ ρ(u) + penalty``.
        penalty: ``[B]`` the caller's penalty at ``theta`` (zeros without one).
        status: ``[B]`` ``int8`` :class:`IRLSRowStatus` codes.
        iterations: ``[B]`` weighted solves of the row.
        n_observations: ``[B]`` observations with positive precision.
        ridge_used: ``[B]`` rows for which any ``solve`` reported
            :attr:`BatchedIRLSStep.ridge_used` (a regularised, possibly
            non-identified fit: its ``theta`` and errors are ridge-limited).
    """

    theta: Tensor
    standard_errors: Tensor
    model: Tensor
    weights: Tensor
    effective_precision: Tensor
    objective: Tensor
    penalty: Tensor
    status: Tensor
    iterations: Tensor
    n_observations: Tensor
    ridge_used: Tensor

    @property
    def converged(self) -> Tensor:
        return self.status == IRLSRowStatus.CONVERGED

    @property
    def failed(self) -> Tensor:
        return self.status == IRLSRowStatus.FAILED

    @property
    def stalled(self) -> Tensor:
        return self.status == IRLSRowStatus.STALLED


@dataclass(frozen=True)
class _Family:
    weight: RobustWeightName | WeightFunction
    params: Mapping[str, float]
    eta: float
    nu: float | None  # Student-t degrees of freedom, None for other families

    def weights_at(
        self, residual: Tensor, prec_eff: Tensor, unused: Tensor, out: Tensor | None = None
    ) -> Tensor:
        """``λ_eff ψ(u)/u`` given ``λ_eff``, zero where ``unused``.

        Overwrites ``residual`` (a temporary) with ``z² = r² λ_eff``; written into
        ``out`` when given (it must not alias ``residual`` or ``prec_eff``).
        """
        z2 = residual.square_().mul_(prec_eff)
        if self.nu is None:
            weights = robust_weights(z2.sqrt_(), self.weight, self.params).mul_(prec_eff)
            if out is not None:
                weights = out.copy_(weights)
        elif math.isinf(self.nu):
            weights = prec_eff.clone() if out is None else out.copy_(prec_eff)
        else:
            # (λ_eff (ν+1)) / (ν + z²): the operation order of the reference implementation.
            weights = torch.mul(prec_eff, self.nu + 1.0, out=out).div_(z2.add_(self.nu))
        return weights.masked_fill_(unused, 0.0)

    def objective(self, residual: Tensor, prec_eff: Tensor, use: Tensor) -> Tensor:
        if self.nu is not None:
            return student_t_nll(residual, prec_eff, self.nu, mask=use)
        z = (residual.square() * prec_eff).sqrt()
        rho = _robust_objective(z, self.weight, self.params)
        return torch.where(use, rho, torch.zeros_like(rho)).sum(-1)


def batched_iteratively_reweighted_fit(
    problem: BatchedIRLSProblem,
    initial_theta: Tensor,
    target: Tensor,
    precision: Tensor,
    mask: Tensor | None = None,
    config: BatchedIRLSFitConfig | None = None,
    *,
    initial_weights: Tensor | None = None,
) -> BatchedIRLSFitResult:
    r"""Fit many independent rows by IRLS under a Student-t (or other robust) likelihood.

    Row ``b`` has observations ``y_{bi}`` with known precisions ``λ_{bi}``, a model
    ``m_b(θ_b)`` and parameters ``θ_b``. With ``λ_eff = 1/(1/λ + η² m̄²)`` and
    ``u = (y − m)√λ_eff`` each iteration solves, for every unfinished row,

    .. math::

        \theta^{(t+1)} = \arg\min_\theta \sum_i w_i^{(t)} (y_i - m_i(\theta))^2
        + 2\,\mathrm{penalty}(\theta),
        \qquad w_i^{(t)} = \lambda_{\mathrm{eff},i}\,\frac{\psi(u_i^{(t)})}{u_i^{(t)}},

    through the caller's ``problem.solve`` (and ``damped_solve``), then refreshes
    the weights. For ``weight="student_t"`` the weights are
    ``λ_eff (ν+1)/(ν + u²)`` and, at ``η = 0``, every accepted step lowers the
    exact ``t_ν`` NLL (the weighted objective majorises it). For ``η > 0``, ``m̄``
    is frozen at the previous iterate: a fixed-point iteration in ``m̄``.

    **Convergence** is certified per row on the *full* (undamped) step at the
    current weights: ``max_k |Δθ_k| / se_k < step_tol``, after at least ``min_iter``
    iterations when the weights depend on the residuals. A damped or rejected
    step never counts. A row whose step no damping accepts stops as ``STALLED``,
    unless the solver's ``predicted_objective`` shows that the decrease still
    reachable is below the float resolution of the objective (then it is
    converged, if ``min_iter`` allows). Finished rows leave the active set, so
    a batch costs ``Σ_b iterations_b`` row solves, and there is no per-row
    Python loop: one host synchronisation per controller iteration (with a
    ``damped_solve``, plus one for the decrease test and one per damping
    attempt), excluding the caller's solve (the non-negative path of
    :func:`solve_normal_equations` adds about one per active-set step).

    **Fail closed.** Rows with non-finite target or precision on an observed
    entry, a non-finite start, fewer than ``min_observations`` usable
    observations, a failed or non-finite solve, or non-finite weights or
    objective are ``FAILED``; rows without usable observations are ``EMPTY``.
    Their outputs are ``NaN``, never zeros that look like a fit.

    Args:
        problem: the caller's model and weighted solve (:class:`BatchedIRLSProblem`).
        initial_theta: ``[B, n]`` starting parameters.
        target: ``[B, N]`` observations ``y``.
        precision: ``[B, N]`` known base precisions ``λ ≥ 0`` (inverse variances,
            as :attr:`IRLSFitConfig.base_precision`; not the effective precision
            :func:`iteratively_reweighted_fit` passes to its weighted fit);
            zero-precision entries leave the likelihood.
        mask: optional ``[B, N]`` boolean (``True`` = observed). Masked entries
            may hold any value, ``NaN`` included.
        config: :class:`BatchedIRLSFitConfig`.
        initial_weights: optional ``[B, N]`` warm-start IRLS weights (e.g. a
            previous fit of a nearby problem). A row whose weights are not
            finite, non-negative and positive exactly where ``λ`` is positive
            starts from ``λ``; ignored when the weights are fixed.

    Returns:
        :class:`BatchedIRLSFitResult`.
    """
    cfg = config or BatchedIRLSFitConfig()
    _validate(initial_theta, target, precision, mask, initial_weights)
    dtype, device = target.dtype, target.device
    batch, n_par = initial_theta.shape
    family = _family(cfg)
    fixed_weights = family.nu is not None and math.isinf(family.nu) and cfg.eta == 0.0
    one_solve = fixed_weights and problem.exact_solve

    # --- data on the likelihood ------------------------------------------------
    # The controller keeps working copies of the rows still running only (target,
    # precision, unused entries and weights): about 3.1 [b, N] buffers, plus up
    # to 2 temporaries in the weight refresh. The result is re-derived from the
    # caller's tensors in row chunks.
    theta = initial_theta.detach().clone()
    use, bad = _likelihood(target, precision, mask)
    invalid = bad.any(-1) | ~torch.isfinite(theta).all(-1)
    del bad
    y_w = torch.where(use, target, 0.0).detach()
    prec_w = torch.where(use, precision, 0.0).detach()
    n_obs = use.sum(-1)
    min_obs = n_par + 1 if cfg.min_observations is None else cfg.min_observations
    status = torch.full((batch,), _RUNNING, dtype=torch.int8, device=device)
    status = torch.where(n_obs == 0, int(IRLSRowStatus.EMPTY), status)
    status = torch.where(
        invalid | ((n_obs > 0) & (n_obs < min_obs)), int(IRLSRowStatus.FAILED), status
    ).to(torch.int8)
    theta = torch.where(torch.isfinite(theta), theta, torch.zeros_like(theta))

    if fixed_weights:
        w_w = prec_w  # never refreshed
    elif initial_weights is not None:
        warm = torch.where(use, initial_weights.to(dtype).detach(), 0.0)
        valid = (torch.isfinite(warm) & (warm >= 0.0) & ((warm > 0.0) == (prec_w > 0.0))).all(-1)
        w_w = torch.where(valid[:, None], warm, prec_w)
        del warm
    else:
        w_w = prec_w.clone()
    unused_w = ~use
    del use

    standard_errors = torch.full((batch, n_par), math.nan, dtype=dtype, device=device)
    iterations = torch.zeros(batch, dtype=torch.int64, device=device)
    ridge_used = torch.zeros(batch, dtype=torch.bool, device=device)
    damping = torch.full((batch,), cfg.damping_start, dtype=dtype, device=device)
    # With a damped solve the decrease test needs q(θ⁰) at the current weights.
    # The weight refresh evaluates the model at the same θ, so it stores q there
    # for the next iteration instead of predicting it again.
    cache_q = problem.damped_solve is not None and not one_solve and not fixed_weights
    q_w: Tensor | None = None
    tiny = torch.finfo(dtype).tiny
    eps = torch.finfo(dtype).eps

    # ``act``: global indices of the working rows (``None``: every row, in order).
    # It changes only when rows finish, so callbacks see the same object across
    # iterations until then.
    act: Tensor | None = None
    for iteration in range(1, cfg.max_iter + 1):
        running = _take(status, act) == _RUNNING
        keep = running.nonzero().squeeze(-1)  # the one host sync
        if keep.numel() == 0:
            break
        if keep.numel() != y_w.shape[0]:
            # Rows finished: shrink the working buffers once (each old buffer is
            # released as soon as its compacted copy exists).
            act = keep if act is None else act[keep]
            y_w = y_w.index_select(0, keep)
            prec_w = prec_w.index_select(0, keep)
            unused_w = unused_w.index_select(0, keep)
            w_w = prec_w if fixed_weights else w_w.index_select(0, keep)
            if q_w is not None:
                q_w = q_w.index_select(0, keep)
        rows = act
        th0 = _take(theta, rows)
        step = problem.solve(th0, w_w, y_w, rows)
        full, se, ok = _checked_step(step, th0)
        if step.ridge_used is not None:
            _put(ridge_used, rows, _take(ridge_used, rows) | step.ridge_used)
        full = torch.where(ok[:, None], full, th0)
        ratio = ((full - th0).abs() / se.clamp_min(tiny)).amax(-1)
        sub_tol = ok & (ratio < cfg.step_tol)
        new = full
        stuck = torch.zeros_like(ok)
        # One exact solve at fixed weights is the minimiser itself: it is taken
        # unconditionally (the start may be infeasible, or lower than any
        # feasible point), never damped.
        if problem.damped_solve is not None and not one_solve:
            new, stuck, sub_tol, lam = _accept_or_damp(
                problem,
                cfg,
                th0,
                full,
                w_w,
                y_w,
                rows,
                ok,
                sub_tol,
                step,
                _take(damping, rows),
                _take(n_obs, rows),
                tiny,
                eps,
                q_w if cache_q and iteration > 1 else None,
            )
            _put(damping, rows, lam)
        if one_solve:
            done = ok & ~stuck
        elif fixed_weights or iteration >= cfg.min_iter:
            done = sub_tol
        else:
            done = torch.zeros_like(ok)
        row_status = torch.full_like(ok, _RUNNING, dtype=torch.int8)
        row_status = torch.where(stuck, int(IRLSRowStatus.STALLED), row_status)
        row_status = torch.where(done, int(IRLSRowStatus.CONVERGED), row_status)
        row_status = torch.where(~ok, int(IRLSRowStatus.FAILED), row_status).to(torch.int8)
        update = ok & ~done & ~stuck
        if not fixed_weights:
            # Rows that finish here never use their weights again (the result
            # recomputes them at the final parameters), so every working row is
            # refreshed in place: no select pass over [b, N].
            model = problem.predict(new, rows)
            _check_model(model, y_w)
            residual = y_w - model
            prec_eff = effective_precision(prec_w, model, family.eta)
            del model
            r2 = residual.square() if cache_q else None
            family.weights_at(residual, prec_eff, unused_w, out=w_w)
            del residual, prec_eff
            # One reduction instead of an isfinite pass: a non-finite weight makes the sum so.
            fresh_ok = torch.isfinite(w_w.sum(-1))
            row_status = torch.where(update & ~fresh_ok, int(IRLSRowStatus.FAILED), row_status).to(
                torch.int8
            )
            if r2 is not None:
                # Bit-identical to _weighted_objective(new) at the fresh weights.
                misfit = r2.mul_(w_w).masked_fill_(~(w_w > 0.0), 0.0)
                q_w = misfit.sum(-1) + 2.0 * _penalty(problem, new, rows)
                del r2, misfit
        _put(theta, rows, new)
        _put(standard_errors, rows, torch.where(ok[:, None], se, _take(standard_errors, rows)))
        _put(iterations, rows, torch.full_like(_take(iterations, rows), iteration))
        _put(status, rows, row_status)

    del y_w, prec_w, unused_w, w_w, q_w
    status = torch.where(status == _RUNNING, int(IRLSRowStatus.MAX_ITERATIONS), status)
    return _result(
        problem,
        family,
        theta,
        standard_errors,
        status.to(torch.int8),
        iterations,
        target,
        precision,
        mask,
        n_obs,
        ridge_used,
    )


def _likelihood(target: Tensor, precision: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    """Entries on the likelihood, and observed entries with invalid data.

    On the likelihood: observed, finite target and precision, precision > 0.
    Invalid: observed with a non-finite target or precision, or a negative one.
    """
    usable = torch.isfinite(target) & torch.isfinite(precision) & (precision >= 0.0)
    bad = ~usable
    use = usable.logical_and_(precision > 0.0)
    if mask is not None:
        bad.logical_and_(mask)
        use.logical_and_(mask)
    return use, bad


def _family(cfg: BatchedIRLSFitConfig) -> _Family:
    nu = _student_t_nu(cfg.weight_params) if cfg.weight == "student_t" else None
    return _Family(cfg.weight, cfg.weight_params, cfg.eta, nu)


def _take(tensor: Tensor, rows: Tensor | None) -> Tensor:
    return tensor if rows is None else tensor[rows]


def _put(tensor: Tensor, rows: Tensor | None, values: Tensor) -> None:
    if rows is None:
        tensor.copy_(values)
    else:
        tensor[rows] = values


def _validate(
    theta: Tensor,
    target: Tensor,
    precision: Tensor,
    mask: Tensor | None,
    initial_weights: Tensor | None,
) -> None:
    if not target.is_floating_point():
        raise TypeError("target must be floating point")
    if target.ndim != 2:
        raise ValueError("target must have shape [B, N]")
    if theta.ndim != 2 or theta.shape[0] != target.shape[0] or theta.shape[1] < 1:
        raise ValueError("initial_theta must have shape [B, n] with n >= 1")
    for name, tensor in (("precision", precision), ("initial_weights", initial_weights)):
        if tensor is not None and tensor.shape != target.shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} must be {tuple(target.shape)}")
    for name, tensor in (("initial_theta", theta), ("precision", precision)):
        if tensor.dtype != target.dtype or tensor.device != target.device:
            raise ValueError(f"{name} must share dtype and device with target")
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype torch.bool")
        if mask.shape != target.shape or mask.device != target.device:
            raise ValueError("mask must have the shape and device of target")


def _checked_step(step: BatchedIRLSStep, theta0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(step, BatchedIRLSStep):
        raise TypeError("problem.solve must return a BatchedIRLSStep")
    if step.theta.shape != theta0.shape or step.standard_errors.shape != theta0.shape:
        raise ValueError(
            f"solve returned theta {tuple(step.theta.shape)} and standard errors "
            f"{tuple(step.standard_errors.shape)}; expected {tuple(theta0.shape)}"
        )
    full = step.theta.detach().to(theta0)
    se = step.standard_errors.detach().to(theta0)
    ok = torch.isfinite(full).all(-1) & (se >= 0.0).all(-1)
    if step.failed is not None:
        if step.failed.shape != theta0.shape[:1] or step.failed.dtype != torch.bool:
            raise ValueError("BatchedIRLSStep.failed must be a boolean [b] tensor")
        ok = ok & ~step.failed
    if step.predicted_objective is not None and step.predicted_objective.shape != theta0.shape[:1]:
        raise ValueError("BatchedIRLSStep.predicted_objective must have shape [b]")
    if step.ridge_used is not None and (
        step.ridge_used.shape != theta0.shape[:1] or step.ridge_used.dtype != torch.bool
    ):
        raise ValueError("BatchedIRLSStep.ridge_used must be a boolean [b] tensor")
    return full, se, ok


def _check_model(model: Tensor, target: Tensor) -> None:
    if model.shape != target.shape:
        raise ValueError(f"predict returned {tuple(model.shape)}; expected {tuple(target.shape)}")


def _penalty(problem: BatchedIRLSProblem, theta: Tensor, rows: Tensor | None) -> Tensor:
    if problem.penalty is None:
        return torch.zeros(theta.shape[0], dtype=theta.dtype, device=theta.device)
    value = problem.penalty(theta, rows)
    if value.shape != theta.shape[:1]:
        raise ValueError(f"penalty returned {tuple(value.shape)}; expected {theta.shape[0]}")
    return value.detach().to(theta)


def _weighted_objective(
    problem: BatchedIRLSProblem, theta: Tensor, weights: Tensor, target: Tensor, rows: Tensor | None
) -> Tensor:
    """``q(θ) = Σ w (y − m(θ))² + 2·penalty(θ)`` at frozen weights."""
    model = problem.predict(theta, rows)
    _check_model(model, target)
    misfit = weights * (target - model).square()
    misfit = torch.where(weights > 0.0, misfit, torch.zeros_like(misfit))
    return misfit.sum(-1) + 2.0 * _penalty(problem, theta, rows)


def _accept_or_damp(
    problem: BatchedIRLSProblem,
    cfg: BatchedIRLSFitConfig,
    th0: Tensor,
    full: Tensor,
    w: Tensor,
    y_a: Tensor,
    rows: Tensor | None,
    ok: Tensor,
    sub_tol: Tensor,
    step: BatchedIRLSStep,
    lam: Tensor,
    n_obs: Tensor,
    tiny: float,
    eps: float,
    q0: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Accept the full step on decrease, else Levenberg--Marquardt damping.

    ``q0`` is the weighted objective at ``th0`` when the caller already has it.
    Returns the new parameters, the stuck rows (no decrease at any damping),
    the convergence flags (full step below ``step_tol``, or stuck at round-off) and
    the updated damping.
    """
    assert problem.damped_solve is not None
    # A full step below the tolerance is taken as it is: it changes the
    # objective at round-off level, where the comparison means nothing.
    accepted = sub_tol.clone()
    pending = ok & ~sub_tol
    if q0 is None:
        q0 = _weighted_objective(problem, th0, w, y_a, rows)
    # q at the full step, on the rows whose step is still to be tested only
    # (q1 is read on stuck rows, a subset of them).
    local = pending.nonzero().squeeze(-1)
    if local.numel() == th0.shape[0]:
        q1 = _weighted_objective(problem, full, w, y_a, rows)
    else:
        q1 = q0.clone()
        if local.numel() > 0:
            sub_rows = local if rows is None else rows[local]
            q1[local] = _weighted_objective(problem, full[local], w[local], y_a[local], sub_rows)
    good = pending & (q1 < q0)
    accepted |= good
    pending &= ~good
    lam = torch.where(good, (lam / 10.0).clamp_min(cfg.damping_min), lam)
    trial_lam = lam.clone()
    new = torch.where(accepted[:, None], full, th0)
    for _ in range(cfg.max_damping):
        local = pending.nonzero().squeeze(-1)
        if local.numel() == 0:
            break
        sub_rows = local if rows is None else rows[local]
        if rows is None and local.numel() == th0.shape[0]:
            sub_rows_arg: Tensor | None = None
        else:
            sub_rows_arg = sub_rows
        trial = problem.damped_solve(
            th0[local], w[local], y_a[local], sub_rows_arg, trial_lam[local]
        )
        if trial.shape != (local.numel(), th0.shape[1]):
            raise ValueError("damped_solve must return [b, n] parameters")
        trial = trial.detach().to(th0)
        finite = torch.isfinite(trial).all(-1)
        safe = torch.where(finite[:, None], trial, th0[local])
        q = _weighted_objective(problem, safe, w[local], y_a[local], sub_rows_arg)
        hit = finite & (q < q0[local])
        new[local] = torch.where(hit[:, None], trial, new[local])
        accepted[local] = hit
        pending[local] = ~hit
        lam[local] = torch.where(
            hit, (trial_lam[local] / 10.0).clamp_min(cfg.damping_min), lam[local]
        )
        trial_lam[local] = torch.where(hit, trial_lam[local], trial_lam[local] * 10.0)
    stuck = ok & ~accepted
    if step.predicted_objective is not None:
        # Along the full step d, φ(t) = q(θ⁰ + t d) has φ'(0) = −2Δ with Δ the
        # decrease the local model predicts; the quadratic through φ(0), φ'(0)
        # and φ(1) allows at most Δ²/(φ(1) − φ(0) + 2Δ). Below the float
        # resolution of q (a sum of n terms: n·ε·|q⁰|) the row is at its optimum
        # to machine precision. A model that predicts an increase (Δ below
        # −floor) or no curvature certifies nothing; at a true optimum Δ is
        # ±round-off, hence the tolerance on its sign. NaN fails both tests.
        predicted = q0 - step.predicted_objective.detach().to(q0)
        curvature = q1 - q0 + 2.0 * predicted
        reachable = torch.where(
            torch.isfinite(curvature) & (curvature > 0.0),
            predicted.square() / curvature.clamp_min(tiny),
            predicted.abs(),
        )
        floor = n_obs.to(q0.dtype) * eps * q0.abs()
        sub_tol = sub_tol | (stuck & (predicted >= -floor) & (reachable <= floor))
    return new, stuck, sub_tol, lam


#: Rows evaluated at a time by the final result: bounds its temporaries to
#: ``O(_RESULT_CHUNK · N)`` whatever the batch size.
_RESULT_CHUNK = 2048


def _result(
    problem: BatchedIRLSProblem,
    family: _Family,
    theta: Tensor,
    standard_errors: Tensor,
    status: Tensor,
    iterations: Tensor,
    target: Tensor,
    precision: Tensor,
    mask: Tensor | None,
    n_obs: Tensor,
    ridge_used: Tensor,
) -> BatchedIRLSFitResult:
    """Evaluate every row at its final parameters, in row chunks of the inputs."""
    batch = theta.shape[0]
    dtype, device = target.dtype, target.device
    fitted = (status != IRLSRowStatus.FAILED) & (status != IRLSRowStatus.EMPTY)
    theta_safe = torch.where(fitted[:, None], theta, torch.zeros_like(theta))
    model = torch.empty(target.shape, dtype=dtype, device=device)
    weights = torch.empty_like(model)
    prec_out = torch.empty_like(model)
    objective = torch.empty(batch, dtype=dtype, device=device)
    penalty = torch.empty_like(objective)
    model_ok = torch.empty(batch, dtype=torch.bool, device=device)
    for lo in range(0, batch, _RESULT_CHUNK):
        hi = min(lo + _RESULT_CHUNK, batch)
        rows = None if (lo == 0 and hi == batch) else torch.arange(lo, hi, device=device)
        target_c, precision_c = target[lo:hi], precision[lo:hi]
        use_c, _ = _likelihood(target_c, precision_c, None if mask is None else mask[lo:hi])
        unused_c = ~use_c
        y_c = torch.where(use_c, target_c, 0.0).detach()
        prec_c = torch.where(use_c, precision_c, 0.0).detach()
        model_c = problem.predict(theta_safe[lo:hi], rows).detach()
        _check_model(model_c, y_c)
        model[lo:hi] = model_c
        # Only the model on the likelihood is checked: values off it (masked or
        # zero-precision entries) are never used, so they may be non-finite.
        model_ok[lo:hi] = torch.isfinite(model_c.masked_fill(unused_c, 0.0).sum(-1))
        residual = y_c - model_c
        prec_eff = effective_precision(prec_c, model_c, family.eta)
        if prec_eff is not prec_c:  # η > 0
            prec_eff.masked_fill_(unused_c, 0.0)
        prec_out[lo:hi] = prec_eff
        pen = _penalty(problem, theta_safe[lo:hi], rows)
        penalty[lo:hi] = pen
        objective[lo:hi] = family.objective(residual, prec_eff, use_c) + pen
        family.weights_at(residual, prec_eff, unused_c, out=weights[lo:hi])
        del model_c, residual, prec_eff, prec_c, y_c, use_c, unused_c
    finite = (
        torch.isfinite(objective)
        & model_ok
        & torch.isfinite(weights.sum(-1))
        & torch.isfinite(theta).all(-1)
    )
    status = torch.where(fitted & ~finite, int(IRLSRowStatus.FAILED), status).to(torch.int8)
    drop = ~(fitted & finite)
    nan = math.nan
    return BatchedIRLSFitResult(
        theta=theta.masked_fill_(drop[:, None], nan),
        standard_errors=standard_errors.masked_fill_(drop[:, None], nan),
        model=model.masked_fill_(drop[:, None], nan),
        weights=weights.masked_fill_(drop[:, None], nan),
        effective_precision=prec_out.masked_fill_(drop[:, None], nan),
        objective=objective.masked_fill_(drop, nan),
        penalty=penalty.masked_fill_(drop, nan),
        status=status,
        iterations=iterations,
        n_observations=n_obs,
        ridge_used=ridge_used,
    )
