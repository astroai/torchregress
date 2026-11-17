"""Parity of torchregress's Student-t / batched IRLS primitives with uspm's own IRLS.

uspm (a downstream spectral-fitting package) carried its own Student-t IRLS until
commit b461f8f and now imports the torchregress replacements, so the reference is
that implementation, frozen in ``scripts/uspm_b461f8f_reference.py``:

* ``student_t_nll`` vs ``student_t_nll``;
* ``effective_precision`` x ``robust_weights(..., "student_t")`` vs ``_irls_weights``;
* ``student_t_efficiency`` vs ``student_t_fisher_factor`` /
  ``student_t_gaussian_efficiency``;
* ``solve_normal_equations(nonnegative=...)`` vs ``nnls_normal`` (cold and warm
  starts, partly constrained, a single row, and many rows of one call that
  take the ridge);
* ``batched_iteratively_reweighted_fit`` with a linear solve vs ``irls_fit`` and with
  uspm's bilinear Gauss--Newton/LM solve as callbacks vs ``irls_fit_shared``.

It is not part of the test suite (it times the fits and takes about a minute)::

    PYTHONPATH=src python scripts/check_uspm_irls_parity.py

Exit status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import torch
from torch import Tensor

from torchregress.algorithms import (
    BatchedIRLSFitConfig,
    BatchedIRLSProblem,
    BatchedIRLSStep,
    batched_iteratively_reweighted_fit,
    robust_weights,
    solve_normal_equations,
)
from torchregress.losses import effective_precision, student_t_efficiency, student_t_nll

DTYPE = torch.float64
FAILURES: list[str] = []


def report(name: str, value: float, limit: float) -> None:
    ok = value <= limit
    print(f"{'ok  ' if ok else 'FAIL'} {name:<62s} {value:.3e} (<= {limit:.0e})")
    if not ok:
        FAILURES.append(name)


def rel(a: Tensor, b: Tensor) -> float:
    return float(((a - b).abs() / b.abs().clamp_min(1e-300)).max())


def take(t: Tensor, rows: Tensor | None) -> Tensor:
    return t if rows is None else t[rows]


def check_elementwise(uspm_op, uspm_refine) -> None:  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(0)
    residual = 3.0 * torch.randn(64, 500, generator=g, dtype=DTYPE)
    ivar = 0.1 + torch.rand(64, 500, generator=g, dtype=DTYPE)
    model = 5.0 * torch.randn(64, 500, generator=g, dtype=DTYPE)
    mask = torch.rand(64, 500, generator=g) > 0.2
    for nu in (2.5, 3.0, 5.0, 10.0, math.inf):
        ours = student_t_nll(residual, ivar, nu, mask=mask)
        theirs = uspm_op.student_t_nll(residual, ivar, mask, nu)
        report(f"student_t_nll nu={nu}", rel(ours, theirs), 1e-12)
        for eta in (0.0, 0.05):
            config = uspm_op.IRLSConfig(nu=nu, eta=eta)
            prec_eff = effective_precision(ivar, model, eta)
            u = residual * prec_eff.sqrt()
            ours_w = prec_eff * robust_weights(u, "student_t", {"nu": nu})
            theirs_w = uspm_op._irls_weights(ivar, residual, model, config)
            report(f"IRLS weights nu={nu} eta={eta}", rel(ours_w, theirs_w), 1e-12)
            report(
                f"effective_precision eta={eta}",
                rel(prec_eff, uspm_op.effective_ivar(ivar, model, eta)),
                1e-15,
            )
    for nu in (2.1, 3.0, 4.0, 5.0, 8.0, 30.0, 1e3, math.inf):
        fisher = abs(
            student_t_efficiency(nu, "student_t") - uspm_refine.student_t_fisher_factor(nu)
        )
        gauss = abs(student_t_efficiency(nu) - uspm_refine.student_t_gaussian_efficiency(nu))
        report(f"fisher factor nu={nu}", fisher, 1e-12)
        report(f"gaussian efficiency nu={nu}", gauss, 1e-12)


def check_nnls(uspm_op) -> None:  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(1)
    b, p, n = 256, 80, 9
    x = torch.randn(b, p, n, generator=g, dtype=DTYPE)
    y = torch.randn(b, p, generator=g, dtype=DTYPE)
    collinear = x.clone()
    collinear[: b // 4, :, 2] = collinear[: b // 4, :, 0] + collinear[: b // 4, :, 1]
    # An exact duplicate of an unconstrained (always free) column: the plain
    # factorisation fails on nearly all of those rows, so many rows of one call
    # take the ridge (a collinear constrained column is mostly pinned instead).
    duplicated = x.clone()
    duplicated[: b // 4, :, 8] = duplicated[: b // 4, :, 7]
    warm = torch.randn(b, n, generator=g, dtype=DTYPE)
    for label, design, k, start, rows in (
        ("k=6", x, 6, None, slice(None)),
        ("k=9", x, 9, None, slice(None)),
        ("k=6 warm", x, 6, warm, slice(None)),
        ("k=6 one row", x, 6, None, slice(0, 1)),
        ("k=6 collinear constrained column", collinear, 6, warm, slice(None)),
        ("k=6 ridged rows", duplicated, 6, None, slice(None)),
        ("k=6 ridged rows warm", duplicated, 6, warm, slice(None)),
    ):
        xd, yd = design[rows], y[rows]
        normal = xd.transpose(1, 2) @ xd
        rhs = (xd * yd.unsqueeze(-1)).sum(1)
        x0 = None if start is None else start[rows]
        ours = solve_normal_equations(normal, rhs, nonnegative=torch.arange(n) < k, start=x0)
        sol, se, ridged, _, singular, passive = uspm_op.nnls_normal(normal, rhs, k, x0)
        report(f"nnls {label}: solution (abs)", float((ours.solution - sol).abs().max()), 1e-12)
        report(f"nnls {label}: standard errors", rel(ours.standard_errors, se), 1e-12)
        report(f"nnls {label}: free set mismatches", float((ours.free != passive).sum()), 0.0)
        flags = (ours.ridge_used != ridged).sum() + (ours.singular != singular).sum()
        report(f"nnls {label}: ridge/singular mismatches", float(flags), 0.0)
        if "ridged rows" in label:
            # Most of the b // 4 duplicated rows take the ridge in the same call.
            count = int(ours.ridge_used.sum())
            short = float(max(0, b // 8 - count))
            report(f"nnls {label}: ridged rows short of {b // 8} (got {count})", short, 0.0)


def linear_problem(design: Tensor) -> BatchedIRLSProblem:
    def predict(theta: Tensor, rows: Tensor | None) -> Tensor:
        return (take(design, rows) @ theta.unsqueeze(-1)).squeeze(-1)

    def solve(theta: Tensor, w: Tensor, y: Tensor, rows: Tensor | None) -> BatchedIRLSStep:
        x = take(design, rows)
        xw = x * w.unsqueeze(-1)
        out = solve_normal_equations(x.transpose(1, 2) @ xw, (xw * y.unsqueeze(-1)).sum(1))
        return BatchedIRLSStep(
            out.solution, out.standard_errors, failed=out.singular, ridge_used=out.ridge_used
        )

    return BatchedIRLSProblem(predict, solve, exact_solve=True)


def shared_calibration_problem(
    basis: Tensor, legendre: Tensor, prior: Tensor
) -> BatchedIRLSProblem:
    """uspm's ``m = P(x) Σ_k a_k b_k`` (a ≥ 0, ``P = 1 + Σ c_m L_m``) as callbacks."""
    k = basis.shape[-1]
    cal = legendre[..., 1:]
    n = k + cal.shape[-1]
    bounds = torch.arange(n) < k
    prior_block = torch.diag_embed(torch.cat([prior.new_zeros(k), prior]))

    def parts(theta: Tensor, rows: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x, leg = take(basis, rows), take(cal, rows)
        poly = 1.0 + (leg @ theta[:, k:].unsqueeze(-1)).squeeze(-1)
        source = (x @ theta[:, :k].unsqueeze(-1)).squeeze(-1)
        return x, leg, poly, source

    def predict(theta: Tensor, rows: Tensor | None) -> Tensor:
        _, _, poly, source = parts(theta, rows)
        return poly * source

    def linearised(theta, w, y, rows):  # type: ignore[no-untyped-def]
        x, leg, poly, source = parts(theta, rows)
        z = torch.cat([poly.unsqueeze(-1) * x, leg * source.unsqueeze(-1)], -1)
        target = y + (poly - 1.0) * source
        zw = z * w.unsqueeze(-1)
        return z, target, z.transpose(1, 2) @ zw + prior_block, (zw * target.unsqueeze(-1)).sum(1)

    def solve(theta: Tensor, w: Tensor, y: Tensor, rows: Tensor | None) -> BatchedIRLSStep:
        z, target, normal, rhs = linearised(theta, w, y, rows)
        out = solve_normal_equations(normal, rhs, nonnegative=bounds, start=theta)
        lin = target - (z @ out.solution.unsqueeze(-1)).squeeze(-1)
        model_q = (w * lin.square()).sum(-1) + (prior * out.solution[:, k:].square()).sum(-1)
        return BatchedIRLSStep(
            out.solution, out.standard_errors, out.singular, model_q, out.ridge_used
        )

    def damped(theta: Tensor, w: Tensor, y: Tensor, rows: Tensor | None, lam: Tensor) -> Tensor:
        _, _, normal, rhs = linearised(theta, w, y, rows)
        d = torch.diagonal(normal, dim1=-2, dim2=-1) * lam.unsqueeze(-1)
        out = solve_normal_equations(
            normal + torch.diag_embed(d),
            rhs + d * theta,
            nonnegative=bounds,
            start=theta,
        )
        return out.solution

    def penalty(theta: Tensor, rows: Tensor | None) -> Tensor:
        return 0.5 * (prior * theta[:, k:].square()).sum(-1)

    return BatchedIRLSProblem(predict, solve, damped, penalty)


def synthetic(b: int, p: int, k: int, j: int, seed: int):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    grid = torch.linspace(-1.0, 1.0, p, dtype=DTYPE)
    centres = torch.rand(b, 1, k, generator=g, dtype=DTYPE) * 2.0 - 1.0
    basis = 1.0 + torch.exp(-0.5 * ((grid[None, :, None] - centres) / 0.15) ** 2)
    leg = torch.stack([torch.special.legendre_polynomial_p(grid, m) for m in range(j + 1)], -1)
    legendre = leg.expand(b, p, j + 1).contiguous()
    a = torch.rand(b, k, generator=g, dtype=DTYPE) * (torch.rand(b, k, generator=g) > 0.3)
    c = 0.1 * torch.randn(b, j, generator=g, dtype=DTYPE)
    model = (1.0 + (legendre[..., 1:] @ c.unsqueeze(-1)).squeeze(-1)) * (
        basis @ a.unsqueeze(-1)
    ).squeeze(-1)
    ivar = 400.0 * (0.5 + torch.rand(b, p, generator=g, dtype=DTYPE))
    flux = model + torch.randn(b, p, generator=g, dtype=DTYPE) * ivar.rsqrt()
    flux = torch.where(torch.rand(b, p, generator=g) < 0.02, flux + 1.0, flux)
    likelihood = torch.rand(b, p, generator=g) > 0.05
    return basis, legendre, flux, ivar, likelihood


def check_fits(uspm_op, b: int, p: int) -> None:  # type: ignore[no-untyped-def]
    k, j = 5, 2
    basis, legendre, flux, ivar, lik = synthetic(b, p, k, j, seed=2)
    design = (basis.unsqueeze(-1) * legendre.unsqueeze(-2)).reshape(b, p, k * (j + 1))
    for nu, eta in ((5.0, 0.0), (5.0, 0.03), (math.inf, 0.0)):
        cfg = uspm_op.IRLSConfig(nu=nu, eta=eta)
        ours_cfg = BatchedIRLSFitConfig(
            weight_params={"nu": nu}, eta=eta, step_tol=cfg.tol, max_iter=cfg.max_iter
        )
        theta0 = torch.zeros(b, design.shape[-1], dtype=DTYPE)
        t0 = time.perf_counter()
        theirs = uspm_op.irls_fit(design, flux, ivar, lik, cfg)
        t1 = time.perf_counter()
        ours = batched_iteratively_reweighted_fit(
            linear_problem(design), theta0, flux, ivar, lik, ours_cfg
        )
        t2 = time.perf_counter()
        tag = f"irls_fit nu={nu} eta={eta}"
        report(
            f"{tag}: coefficients / se",
            float(((ours.theta - theirs.coefficients).abs() / ours.standard_errors).max()),
            1e-9,
        )
        report(f"{tag}: nll", rel(ours.objective, theirs.nll), 1e-12)
        report(
            f"{tag}: converged mismatches", float((ours.converged != theirs.converged).sum()), 0.0
        )
        report(f"{tag}: n_iter mismatches", float((ours.iterations != theirs.n_iter).sum()), 0.0)
        report(
            f"{tag}: ridge_used mismatches",
            float((ours.ridge_used != theirs.ridge_used).sum()),
            0.0,
        )
        print(f"     timing uspm {t1 - t0:.3f}s  torchregress {t2 - t1:.3f}s  (B={b}, N={p})")

        prior = uspm_op.calibration_precision(cfg, j)
        t0 = time.perf_counter()
        theirs = uspm_op.irls_fit_shared(basis, legendre, flux, ivar, lik, cfg)
        t1 = time.perf_counter()
        ours = batched_iteratively_reweighted_fit(
            shared_calibration_problem(basis, legendre, prior),
            torch.zeros(b, k + j, dtype=DTYPE),
            flux,
            ivar,
            lik,
            ours_cfg,
        )
        t2 = time.perf_counter()
        tag = f"irls_fit_shared nu={nu} eta={eta}"
        both = ours.converged & theirs.converged
        diff = (ours.theta - theirs.coefficients).abs() / ours.standard_errors.clamp_min(1e-300)
        report(f"{tag}: coefficients / se (both converged)", float(diff[both].max()), 1e-9)
        report(f"{tag}: nll (both converged)", rel(ours.objective[both], theirs.nll[both]), 1e-12)
        report(
            f"{tag}: converged mismatches", float((ours.converged != theirs.converged).sum()), 0.0
        )
        mean_iter = (ours.iterations.double().mean(), theirs.n_iter.double().mean())
        print(
            f"     mean iterations torchregress {mean_iter[0]:.2f} uspm {mean_iter[1]:.2f}; "
            f"timing uspm {t1 - t0:.3f}s  torchregress {t2 - t1:.3f}s"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--pixels", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    import uspm_b461f8f_reference as reference  # noqa: PLC0415 (next to this script)

    print(f"reference: uspm {reference.USPM_COMMIT} (scripts/uspm_b461f8f_reference.py)")
    check_elementwise(reference, reference)
    check_nnls(reference)
    check_fits(reference, args.rows, args.pixels)
    print(f"\n{len(FAILURES)} failure(s)" + (": " + ", ".join(FAILURES) if FAILURES else ""))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
