"""Batched normal-equation solves for many small (non-negative) least-squares problems."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

#: Default KKT tolerance of the non-negative solve. A constrained variable at its
#: bound is freed only while its multiplier ``μ_j`` on the Jacobi-scaled equations
#: exceeds ``kkt_tol · (1 + |g_s|_∞)``: a mixed absolute/relative threshold, in
#: units of σ when the weights are true precisions. Freeing a variable with
#: multiplier ``μ_j`` lowers the objective by ``μ_j² / (2 S_j) ≥ μ_j² / 2``, where
#: ``S_j ≤ 1`` is the Schur complement of its scaled diagonal on the free block,
#: so collinear columns (small ``S_j``) amplify the forgone decrease.
NNLS_KKT_TOL = 1.0e-10


@dataclass(frozen=True)
class NormalEquationsSolution:
    """Per-row solution of ``min ½ xᵀNx − gᵀx`` (optionally with ``x_i ≥ 0``).

    Attributes:
        solution: ``[..., n]`` minimiser; ``NaN`` on ``singular`` rows.
        standard_errors: ``[..., n]`` ``√((N_FF)⁻¹)_ii`` on the free set ``F``
            (the inverse of the normal matrix restricted to it; of its ridged
            scaled form on ``ridge_used`` rows), ``1/√N_ii`` for a
            variable held at its bound, ``inf`` for an empty column (zero
            diagonal: no information) and ``NaN`` on ``singular`` rows.
        free: ``[..., n]`` variables neither at a bound nor empty.
        dead: ``[..., n]`` empty columns (``N_ii ≤ 0``), pinned at 0 (``False`` on
            rows with non-finite input).
        ridge_used: ``[...]`` rows that took the ridge (a factorisation failed,
            or, with ``rank_tol > 0``, the final system was numerically
            rank-deficient): their solution and standard errors are regularised.
        singular: ``[...]`` rows singular even with the ridge, or with non-finite
            input (any ``NaN``/``±inf`` in their ``N`` or ``g``); not solved, and
            ``free`` is ``False`` on them.
    """

    solution: Tensor
    standard_errors: Tensor
    free: Tensor
    dead: Tensor
    ridge_used: Tensor
    singular: Tensor


def solve_normal_equations(
    normal: Tensor,
    rhs: Tensor,
    *,
    nonnegative: Tensor | None = None,
    start: Tensor | None = None,
    ridge: float = 1.0e-8,
    kkt_tol: float = NNLS_KKT_TOL,
    rank_tol: float = 0.0,
) -> NormalEquationsSolution:
    r"""Solve many small least-squares problems given their normal equations.

    Each row solves

    .. math::

        \min_x \tfrac12 x^\mathsf{T} N x - g^\mathsf{T} x
        \quad\text{s.t.}\quad x_i \ge 0 \ (i \in \mathcal C),

    for a positive semi-definite ``N = XᵀWX (+ prior precision)`` and
    ``g = XᵀWy (+ prior term)``; ``𝒞`` is empty unless ``nonnegative`` is given.
    The system is Jacobi-scaled (unit diagonal) before factorisation, so the
    ``ridge`` is relative. By default it is added to the scaled diagonal (and
    the row flagged in ``ridge_used``) only where the Cholesky factorisation
    fails. On ridged rows the solution and standard errors are those of
    ``N_s + ridge·I``: regularised, so a non-identified direction gets a
    finite (ridge-limited) error rather than ``inf``. A factorisation that
    succeeds is not regularised, however badly conditioned: rows whose
    minimiser is identified in floating point are never biased.

    With ``rank_tol > 0`` (opt-in) a row is also ridged where its final system
    is numerically rank-deficient: the smallest eigenvalue of the scaled
    matrix (of its block on the final free set, with ``nonnegative``) is at
    most ``rank_tol`` times the largest. The test is rank-revealing and
    monotone in conditioning (one batched ``eigvalsh``), and in the non-negative solve it
    is applied to the final free set only: the active-set steps ridge failed
    factorisations alone, so the free set reached does not depend on the test,
    and a flagged row is then re-solved as the strictly convex problem
    ``N_s + ridge·I`` (warm-started from that solution), whose minimiser does
    not depend on ``start``. The test sees the free set reached only: where the
    minimisers form a set and the active set stops at a full-rank vertex of
    it (a variable at its bound), the solution is a minimiser and the row is
    not flagged.
    Empty columns (``N_ii ≤ 0``) are pinned at 0.

    Without ``nonnegative`` the solve is one batched Cholesky (and a ridged one
    of the rows where it fails) with no host synchronisation on an accelerator;
    on the CPU, where reading a flag costs nothing, the ridged factorisation and
    the swap of non-finite rows run only where they are needed. With it, a
    batched Lawson--Hanson active-set method runs on the scaled normal
    equations: each outer step frees the constrained variable with the largest
    scaled gradient (KKT multiplier) above ``kkt_tol · (1 + |g_s|_∞)``; each
    inner step solves the free block and, when a constrained variable would
    turn non-positive, moves to the first bound on the segment and pins it.
    ``start`` warm-starts the free set (e.g. the previous solution of a nearby
    problem). Rows terminate independently, and every step works on the rows
    still in play only: about one host synchronisation per active-set step, at
    ``O(b·n³)`` for the ``b`` rows still pending (one factorisation each; a
    ridged one only for rows where it fails). The standard errors reuse each
    row's last factorisation, which is of its final free set unless the step
    budget ran out right after a bound was hit: those rows alone are factorised
    again, after one more host synchronisation (and, with ``rank_tol > 0``, one
    batched ``eigvalsh`` and one more host synchronisation).

    Args:
        normal: ``[..., n, n]`` symmetric positive semi-definite matrices.
        rhs: ``[..., n]`` right-hand sides ``g``.
        nonnegative: optional boolean ``[n]`` (or broadcastable to ``rhs``) mask of
            the variables constrained to ``x_i ≥ 0``.
        start: optional ``[..., n]`` warm start of the non-negative solve (clipped
            to the bounds; non-finite entries start at 0). Ignored without
            ``nonnegative``.
        ridge: positive relative ridge added to the scaled diagonal where the
            factorisation fails (or, with ``rank_tol > 0``, the final system
            is numerically rank-deficient).
        rank_tol: smallest-to-largest eigenvalue ratio of the scaled final
            system at or below which a row counts as rank-deficient and takes
            the ridge. ``0`` (default) takes the ridge only where the
            factorisation fails. ``n·eps`` (about ``2e-16·n`` in float64) flags
            the rows whose system is singular to working precision.
        kkt_tol: positive KKT tolerance of the non-negative solve (threshold
            ``kkt_tol · (1 + |g_s|_∞)`` on the scaled multipliers).

    Returns:
        :class:`NormalEquationsSolution`.
    """
    _validate(normal, rhs, ridge, kkt_tol)
    if not (math.isfinite(rank_tol) and rank_tol >= 0.0):
        raise ValueError("rank_tol must be finite and >= 0")
    batch_shape = rhs.shape[:-1]
    n = rhs.shape[-1]
    flat = rhs.ndim == 2
    normal_2d = normal if flat else normal.reshape(-1, n, n)
    rhs_2d = rhs if flat else rhs.reshape(-1, n)
    # Rows with non-finite input are not solved. They are swapped for a trivial
    # system first so that they cannot poison batch-wide quantities (the KKT
    # tolerance, the multipliers, the active-set loop) of the other rows.
    bad = ~(torch.isfinite(normal_2d).all(-1).all(-1) & torch.isfinite(rhs_2d).all(-1))
    if _host_any(bad):
        eye = torch.eye(n, dtype=rhs.dtype, device=rhs.device)
        normal_2d = torch.where(bad[:, None, None], eye, normal_2d)
        rhs_2d = torch.where(bad[:, None], 0.0, rhs_2d)
    scaled, gs, scale, dead = _jacobi_scale(normal_2d, rhs_2d)
    if nonnegative is None:
        x_s, se_s, free, ridged, singular = _solve_unconstrained(scaled, gs, dead, ridge, rank_tol)
    else:
        if nonnegative.dtype != torch.bool:
            raise TypeError("nonnegative must be a boolean mask")
        constrained = torch.broadcast_to(nonnegative.to(rhs.device), rhs.shape)
        constrained = constrained if flat else constrained.reshape(-1, n)
        x0 = torch.zeros_like(rhs_2d)
        if start is not None:
            if start.shape != rhs.shape:
                raise ValueError(f"start shape {tuple(start.shape)} must be {tuple(rhs.shape)}")
            warm = start.to(rhs_2d) if flat else start.to(rhs_2d).reshape(-1, n)
            x0 = torch.where(dead | ~torch.isfinite(warm), x0, warm * scale)
            x0 = torch.where(constrained & ~(x0 > 0.0), 0.0, x0)
        x_s, se_s, free, ridged, singular = _solve_nonnegative(
            scaled, gs, dead, constrained, x0, ridge, rank_tol, kkt_tol
        )
    solution = x_s / scale
    standard_errors = (se_s / scale).masked_fill(dead, math.inf)
    singular = singular | bad | ~torch.isfinite(solution).all(-1)
    if _host_any(singular):
        solution = solution.masked_fill(singular[:, None], math.nan)
        standard_errors = standard_errors.masked_fill(singular[:, None], math.nan)
        free = free & ~singular[:, None]
    if not flat:
        solution, standard_errors = solution.reshape(rhs.shape), standard_errors.reshape(rhs.shape)
        free, dead = free.reshape(rhs.shape), dead.reshape(rhs.shape)
        ridged, singular = ridged.reshape(batch_shape), singular.reshape(batch_shape)
    return NormalEquationsSolution(
        solution=solution,
        standard_errors=standard_errors,
        free=free,
        dead=dead,
        ridge_used=ridged,
        singular=singular,
    )


def _reads_are_free(tensor: Tensor) -> bool:
    """Whether reading a flag of ``tensor`` on the host is free (on the CPU).

    Where it is, work needed by a few rows only (the ridged factorisation, the
    swap of non-finite rows) runs on those rows alone; on an accelerator it
    runs unconditionally instead of synchronising with the host. The results
    are the same: rows are processed independently.
    """
    return tensor.device.type == "cpu"


def _host_any(mask: Tensor) -> bool:
    """``mask.any()`` where reading it is free (:func:`_reads_are_free`), else ``True``."""
    return not _reads_are_free(mask) or bool(mask.any())


def _validate(normal: Tensor, rhs: Tensor, ridge: float, kkt_tol: float) -> None:
    if not (normal.is_floating_point() and rhs.is_floating_point()):
        raise TypeError("normal and rhs must be floating point")
    if normal.dtype != rhs.dtype or normal.device != rhs.device:
        raise ValueError("normal and rhs must share dtype and device")
    if rhs.ndim < 1 or rhs.shape[-1] == 0:
        raise ValueError("rhs must have shape [..., n] with n >= 1")
    if normal.shape != rhs.shape + rhs.shape[-1:]:
        raise ValueError(
            f"normal shape {tuple(normal.shape)} must be rhs shape + (n,): "
            f"{tuple(rhs.shape + rhs.shape[-1:])}"
        )
    if not (math.isfinite(ridge) and ridge > 0.0):
        raise ValueError("ridge must be finite and positive")
    if not (math.isfinite(kkt_tol) and kkt_tol > 0.0):
        raise ValueError("kkt_tol must be finite and positive")


def _jacobi_scale(normal: Tensor, rhs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Unit-diagonal scaling; empty columns decoupled with a unit diagonal."""
    diag = torch.diagonal(normal, dim1=-2, dim2=-1)
    dead = diag <= 0.0
    if not _host_any(dead):
        # The decoupling below would add zeros only (``+ 0.0``: the same signed zeros).
        scale = diag.sqrt()
        scaled = normal / (scale.unsqueeze(-1) * scale.unsqueeze(-2)) + 0.0
        return scaled, rhs / scale, scale, dead
    scale = torch.where(dead, 1.0, diag.sqrt())
    scaled = normal / (scale.unsqueeze(-1) * scale.unsqueeze(-2))
    pair = dead.unsqueeze(-1) | dead.unsqueeze(-2)
    scaled = torch.where(pair, 0.0, scaled) + torch.diag_embed(dead.to(normal.dtype)).to(
        device=normal.device, dtype=normal.dtype
    )
    gs = torch.where(dead, 0.0, rhs / scale)
    return scaled, gs, scale, dead


def _rank_deficient(system: Tensor, rank_tol: float) -> Tensor:
    """Rows whose (Jacobi-scaled) ``system`` has ``λ_min ≤ rank_tol · λ_max``.

    Rank-revealing and monotone in conditioning, unlike the Cholesky pivot ratio, which
    can exceed ``λ_min`` by many orders of magnitude. Variables outside a free
    block carry a unit diagonal, an eigenvalue 1 that changes neither extreme of
    a unit-diagonal block (its ``λ_max ≥ 1 ≥ λ_min``).
    """
    eigenvalues = torch.linalg.eigvalsh(system)
    return eigenvalues[..., 0] <= rank_tol * eigenvalues[..., -1]


def _ridge_failed_rows(
    system: Tensor,
    chol: Tensor,
    failed: Tensor,
    ridge: float,
    ridged_variables: Tensor | None,
    gather: bool,
) -> tuple[Tensor, Tensor]:
    """Swap the factor of the ``failed`` rows for that of the ridged ``system``.

    The ridge is added to the diagonal entries ``ridged_variables`` (``[b, n]``;
    all of them when ``None``). With ``gather`` only the failed rows are
    refactorised (a host read of their index, and no work when no row
    failed); otherwise every row is, unconditionally (no host
    synchronisation). Rows are factorised independently, so the factors agree
    either way. Rows singular even with the ridge get an identity factor.
    Returns the factor and the singular rows.
    """
    rows = failed.nonzero().squeeze(-1) if gather else None  # host read
    if rows is not None and rows.numel() == 0:
        return chol, torch.zeros_like(failed)
    eye = torch.eye(system.shape[-1], dtype=system.dtype, device=system.device)
    sub = system if rows is None else system[rows]
    if ridged_variables is None:
        term = ridge * eye
    else:
        mask = ridged_variables if rows is None else ridged_variables[rows]
        term = ridge * torch.diag_embed(mask.to(system.dtype)).to(
            device=system.device, dtype=system.dtype
        )
    chol_r, info_r = torch.linalg.cholesky_ex(sub + term)
    singular_r = info_r != 0
    if rows is None:
        singular = failed & singular_r
        chol_r = torch.where(singular[:, None, None], eye, chol_r)
        return torch.where(failed[:, None, None], chol_r, chol), singular
    chol_r = torch.where(singular_r[:, None, None], eye, chol_r)
    singular = torch.zeros_like(failed)
    singular[rows] = singular_r
    return chol.index_copy(0, rows, chol_r), singular


def _solve_unconstrained(
    scaled: Tensor, gs: Tensor, dead: Tensor, ridge: float, rank_tol: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    chol, info = torch.linalg.cholesky_ex(scaled)
    ridged = info != 0
    if rank_tol > 0.0:
        ridged = ridged | _rank_deficient(scaled, rank_tol)
    chol, singular = _ridge_failed_rows(scaled, chol, ridged, ridge, None, _reads_are_free(scaled))
    x = torch.cholesky_solve(gs.unsqueeze(-1), chol).squeeze(-1)
    inv_diag = torch.diagonal(torch.cholesky_inverse(chol), dim1=-2, dim2=-1)
    return x, inv_diag.clamp_min(0.0).sqrt(), ~dead, ridged, singular


def _free_block(scaled: Tensor, free: Tensor) -> Tensor:
    """``scaled`` on the free block, identity elsewhere."""
    pair = free.unsqueeze(-1) & free.unsqueeze(-2)
    return torch.where(pair, scaled, 0.0) + torch.diag_embed((~free).to(scaled.dtype)).to(
        device=scaled.device, dtype=scaled.dtype
    )


def _free_block_cholesky(
    scaled: Tensor, free: Tensor, ridge: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Cholesky factor of ``scaled`` on the free block (identity elsewhere).

    The ridge is taken where the factorisation fails, and only there (on the
    CPU the ridged system is built and factorised for those rows alone).
    """
    system = _free_block(scaled, free)
    chol, info = torch.linalg.cholesky_ex(system)
    ridged = info != 0
    chol, singular = _ridge_failed_rows(system, chol, ridged, ridge, free, _reads_are_free(scaled))
    return chol, ridged, singular


def _solve_nonnegative(
    scaled: Tensor,
    gs: Tensor,
    dead: Tensor,
    constrained: Tensor,
    x: Tensor,
    ridge: float,
    rank_tol: float,
    kkt_tol: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Non-negative solve; rank-deficient final free blocks re-solved ridged."""
    x, free, ridged, singular, chol, fresh = _active_set(
        scaled, gs, dead, constrained, x, ridge, kkt_tol
    )
    if rank_tol > 0.0:
        deficient = ~singular & _rank_deficient(_free_block(scaled, free), rank_tol)
        rows = deficient.nonzero().squeeze(-1)  # host synchronisation (opt-in)
        if rows.numel() > 0:
            # The ridged problem is strictly convex: its minimiser is unique, so it
            # does not depend on the warm start (the unridged solution).
            live = (~dead[rows]).to(scaled.dtype)
            scaled = scaled.clone()
            scaled[rows] = scaled[rows] + ridge * torch.diag_embed(live).to(
                device=live.device, dtype=live.dtype
            )
            x_r, free_r, _, singular_r, chol_r, fresh_r = _active_set(
                scaled[rows], gs[rows], dead[rows], constrained[rows], x[rows], ridge, kkt_tol
            )
            x = x.index_copy(0, rows, x_r)
            chol = chol.index_copy(0, rows, chol_r)
            free[rows], fresh[rows] = free_r, fresh_r
            ridged[rows] = True
            singular[rows] = singular_r
    # The standard errors need the factor of the final free set. A row's last
    # active-set factor is that factor unless its free set changed after it (an
    # exhausted step budget): only those rows are factorised again.
    stale = (~fresh).nonzero().squeeze(-1)  # host synchronisation
    if stale.numel() > 0:
        chol_s, _, _ = _free_block_cholesky(scaled[stale], free[stale], ridge)
        chol = chol.index_copy(0, stale, chol_s)
    inv_diag = torch.diagonal(torch.cholesky_inverse(chol), dim1=-2, dim2=-1)
    se = torch.where(free, inv_diag.clamp_min(0.0).sqrt(), 1.0)
    return x, se, free, ridged, singular


def _active_set(
    scaled: Tensor,
    gs: Tensor,
    dead: Tensor,
    constrained: Tensor,
    x: Tensor,
    ridge: float,
    kkt_tol: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Batched Lawson--Hanson on Jacobi-scaled normal equations (feasible ``x``).

    Returns the solution, the final free sets, the rows whose factorisations
    took the ridge or were singular even with it, each row's last free-block
    Cholesky factor and whether that factor is of the final free set
    (``fresh``). Every step works on the rows still in play only (``going`` for
    the outer KKT step, ``pending`` for the inner feasibility steps), gathered
    by index unless they are all the rows: each step costs ``O(b·n³)`` on those
    rows and one host synchronisation (the index of the rows that carry on),
    whatever the batch size.
    """
    b, n = gs.shape
    free = ~dead & (~constrained | (x > 0.0))
    tol = kkt_tol * (1.0 + gs.abs().amax(-1))
    ridged = torch.zeros(b, dtype=torch.bool, device=gs.device)
    singular = torch.zeros_like(ridged)
    fresh = torch.zeros_like(ridged)
    # ``chol`` holds each row's last factor. It starts as a placeholder (every row
    # is factorised in the first step) and may be a step's own factor (after a
    # step on every row): ``shared`` copies it before the first write into it,
    # which would otherwise alter a tensor that autograd may have saved.
    chol, shared = scaled, True
    going = torch.arange(b, device=gs.device)
    columns = torch.arange(n, device=gs.device)
    for _ in range(3 * n + 3):
        pending = going
        for _ in range(n + 1):
            if pending.numel() == 0:
                break
            every = pending.numel() == b
            if every:
                free_p, gs_p, x_p, scaled_p, constrained_p = free, gs, x, scaled, constrained
            else:
                free_p, gs_p, x_p = free[pending], gs[pending], x[pending]
                scaled_p, constrained_p = scaled[pending], constrained[pending]
            bounded = free_p & constrained_p
            chol_p, used, sing = _free_block_cholesky(scaled_p, free_p, ridge)
            rhs_p = torch.where(free_p, gs_p, 0.0).unsqueeze(-1)
            z = torch.where(free_p, torch.cholesky_solve(rhs_p, chol_p).squeeze(-1), 0.0)
            infeasible = ~sing.unsqueeze(-1) & bounded & ~(z > 0.0)
            stuck = infeasible.any(-1)
            accept = ~sing & ~stuck
            x_p = torch.where(accept.unsqueeze(-1), z, x_p)
            # Move from x (feasible) towards z up to the first bound on the way.
            ratio = torch.where(infeasible, x_p / (x_p - z), math.inf)
            alpha = ratio.amin(-1, keepdim=True).clamp(0.0, 1.0)
            moved = x_p + alpha * (z - x_p)
            hit = stuck.unsqueeze(-1) & (
                (infeasible & (ratio <= alpha)) | (bounded & ~(moved > 0.0))
            )
            x_p = torch.where(stuck.unsqueeze(-1), moved, x_p)
            x_p = torch.where(hit, 0.0, x_p)
            # The factor is of the final free set unless a bound was hit.
            fresh_p = ~hit.any(-1)
            if every:
                ridged, singular = ridged | used, singular | sing
                x, free, chol, fresh, shared = x_p, free_p & ~hit, chol_p, fresh_p, True
            else:
                ridged[pending] = ridged[pending] | used
                singular[pending] = singular[pending] | sing
                x = x.index_copy(0, pending, x_p)
                free[pending] = free_p & ~hit
                if shared:
                    chol, shared = chol.clone(), False
                chol[pending] = chol_p
                fresh[pending] = fresh_p
            pending = pending[stuck]  # the one host synchronisation of the step
        every = going.numel() == b
        if every:
            free_g, x_g, scaled_g, fresh_g = free, x, scaled, fresh
            gs_g, singular_g, constrained_g, dead_g, tol_g = gs, singular, constrained, dead, tol
        else:
            free_g, x_g, scaled_g, fresh_g = free[going], x[going], scaled[going], fresh[going]
            gs_g, singular_g = gs[going], singular[going]
            constrained_g, dead_g, tol_g = constrained[going], dead[going], tol[going]
        multiplier = gs_g - (scaled_g @ x_g.unsqueeze(-1)).squeeze(-1)
        candidate = (
            ~singular_g.unsqueeze(-1)
            & constrained_g
            & ~free_g
            & ~dead_g
            & (multiplier > tol_g.unsqueeze(-1))
        )
        best = torch.where(candidate, multiplier, -math.inf).argmax(-1)
        more = candidate.any(-1)
        pick = (columns == best.unsqueeze(-1)) & more.unsqueeze(-1)
        if every:
            free, fresh = free_g | pick, fresh_g & ~more
        else:
            free[going] = free_g | pick
            fresh[going] = fresh_g & ~more
        going = going[more]  # the one host synchronisation of the step
        if going.numel() == 0:
            break
    return x, free, ridged, singular, chol, fresh
