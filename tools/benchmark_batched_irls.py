"""Throughput of batched IRLS vs a per-row loop over ``iteratively_reweighted_fit``.

Many independent Student-t (nu = 5) linear fits, one per row: ``N`` observations
with known precisions, 2% outliers, ``n`` parameters. The batched controller
solves all rows at once (``batched_iteratively_reweighted_fit`` with
``solve_normal_equations``); the baseline loops over rows calling the scalar
controller. Each configuration runs in a fresh subprocess so its peak RSS is its
own. Rows/s for the loop are measured on at most ``--loop-rows`` rows (its cost
is linear in rows by construction).

``--max-outlier-fraction F`` draws each row's outlier fraction uniformly in
``[0, F]`` instead of the fixed 2%, so rows converge after different numbers of
iterations and the controller's compaction of finished rows shows up in the
solve-call sizes (``row solves`` = Σ rows per call = Σ_b iterations_b).

    pixi run python tools/benchmark_batched_irls.py --rows 100 1000 10000
    pixi run python tools/benchmark_batched_irls.py --rows 1000 --max-outlier-fraction 0.3
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time

import torch

from torchregress.algorithms import (
    BatchedIRLSFitConfig,
    BatchedIRLSProblem,
    BatchedIRLSStep,
    IRLSCallbacks,
    IRLSFitConfig,
    batched_iteratively_reweighted_fit,
    iteratively_reweighted_fit,
    solve_normal_equations,
)

NU = 5.0
CHUNK = 1024  # rows per normal-matrix block inside the caller's solve


def make_data(  # type: ignore[no-untyped-def]
    rows: int, pixels: int, n: int, seed: int = 0, max_outliers: float | None = None
):
    g = torch.Generator().manual_seed(seed)
    grid = torch.linspace(-1.0, 1.0, pixels, dtype=torch.float64)
    basis = torch.stack([torch.special.legendre_polynomial_p(grid, m) for m in range(n)], -1)
    design = basis * (1.0 + 0.2 * torch.rand(rows, 1, n, generator=g, dtype=torch.float64))
    theta = torch.randn(rows, n, generator=g, dtype=torch.float64)
    precision = 100.0 * (0.5 + torch.rand(rows, pixels, generator=g, dtype=torch.float64))
    target = (design @ theta.unsqueeze(-1)).squeeze(-1)
    target += torch.randn(rows, pixels, generator=g, dtype=torch.float64) * precision.rsqrt()
    if max_outliers is None:
        fraction = torch.full((rows, 1), 0.02)
    else:  # heterogeneous rows: different iteration counts
        fraction = max_outliers * torch.rand(rows, 1, generator=g)
    target += 2.0 * (torch.rand(rows, pixels, generator=g) < fraction)
    return design, target, precision


def batched_problem(design: torch.Tensor, calls: list[int]) -> BatchedIRLSProblem:
    def take(rows: torch.Tensor | None, lo: int, hi: int) -> torch.Tensor:
        return design[lo:hi] if rows is None else design[rows[lo:hi]]

    def predict(theta: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
        parts = [
            (take(rows, lo, lo + CHUNK) @ theta[lo : lo + CHUNK].unsqueeze(-1)).squeeze(-1)
            for lo in range(0, theta.shape[0], CHUNK)
        ]
        return torch.cat(parts)

    def solve(
        theta: torch.Tensor, w: torch.Tensor, y: torch.Tensor, rows: torch.Tensor | None
    ) -> BatchedIRLSStep:
        calls.append(theta.shape[0])
        normal, rhs = [], []
        for lo in range(0, theta.shape[0], CHUNK):
            x = take(rows, lo, lo + CHUNK)
            w_c = w[lo : lo + CHUNK]
            normal.append(x.transpose(1, 2) @ (x * w_c.unsqueeze(-1)))
            rhs.append((x.transpose(1, 2) @ (w_c * y[lo : lo + CHUNK]).unsqueeze(-1)).squeeze(-1))
        out = solve_normal_equations(torch.cat(normal), torch.cat(rhs))
        return BatchedIRLSStep(out.solution, out.standard_errors, failed=out.singular)

    return BatchedIRLSProblem(predict, solve, exact_solve=True)


def run_batched(
    rows: int, pixels: int, n: int, max_outliers: float | None = None
) -> dict[str, float]:
    design, target, precision = make_data(rows, pixels, n, max_outliers=max_outliers)
    calls: list[int] = []
    problem = batched_problem(design, calls)
    theta0 = torch.zeros(rows, n, dtype=torch.float64)
    start = time.perf_counter()
    result = batched_iteratively_reweighted_fit(
        problem, theta0, target, precision, config=BatchedIRLSFitConfig(weight_params={"nu": NU})
    )
    seconds = time.perf_counter() - start
    return {
        "seconds": seconds,
        "rows_per_s": rows / seconds,
        "mean_iterations": float(result.iterations.double().mean()),
        "max_iterations": float(result.iterations.max()),
        "converged_fraction": float(result.converged.double().mean()),
        "solve_calls": float(len(calls)),
        "row_solves": float(sum(calls)),
    }


def run_loop(rows: int, pixels: int, n: int, max_outliers: float | None = None) -> dict[str, float]:
    design, target, precision = make_data(rows, pixels, n, max_outliers=max_outliers)
    iterations = converged = 0
    start = time.perf_counter()
    for row in range(rows):
        x, y, lam = design[row], target[row], precision[row]
        root = lam.sqrt()

        def residuals(beta: torch.Tensor, x=x, y=y, root=root) -> torch.Tensor:  # type: ignore[no-untyped-def]
            return (y - x @ beta) * root

        def weighted_fit(beta: torch.Tensor, prec: torch.Tensor, x=x, y=y) -> torch.Tensor:  # type: ignore[no-untyped-def]
            del beta
            xw = x * prec.unsqueeze(-1)
            return torch.linalg.solve(x.T @ xw, xw.T @ y)

        result = iteratively_reweighted_fit(
            IRLSCallbacks(residuals, weighted_fit),
            torch.zeros(n, dtype=torch.float64),
            IRLSFitConfig(
                weight="student_t",
                weight_params={"nu": NU},
                scale="none",
                base_precision=lam,
                objective_tol=1e-10,
                weight_tol=None,
            ),
        )
        iterations += result.iterations
        converged += int(result.converged)
    seconds = time.perf_counter() - start
    return {
        "seconds": seconds,
        "rows_per_s": rows / seconds,
        "mean_iterations": iterations / rows,
        "converged_fraction": converged / rows,
    }


def child(
    mode: str, rows: int, pixels: int, n: int, threads: int, max_outliers: float | None
) -> None:
    torch.set_num_threads(threads)
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    run = run_batched if mode == "batched" else run_loop
    stats = run(rows, pixels, n, max_outliers)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats["peak_rss_mb"] = peak / 1024.0
    stats["peak_rss_over_import_mb"] = (peak - base) / 1024.0
    print(json.dumps(stats))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--pixels", type=int, default=4096)
    parser.add_argument("--params", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--loop-rows", type=int, default=200)
    parser.add_argument(
        "--max-outlier-fraction",
        type=float,
        default=None,
        help="draw each row's outlier fraction in [0, F] (heterogeneous convergence)",
    )
    parser.add_argument("--child", nargs=2, metavar=("MODE", "ROWS"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(
            args.child[0],
            int(args.child[1]),
            args.pixels,
            args.params,
            args.threads,
            args.max_outlier_fraction,
        )
        return
    outliers = (
        "2% outliers"
        if args.max_outlier_fraction is None
        else f"outlier fraction U[0, {args.max_outlier_fraction}] per row"
    )
    print(f"N={args.pixels} n={args.params} threads={args.threads} nu={NU} float64 cpu, {outliers}")
    header = f"{'mode':<8} {'B':>6} {'rows/s':>10} {'s':>8} {'iters':>6} {'conv':>5} {'peak MB':>8}"
    print(header)
    for rows in args.rows:
        for mode, count in (("batched", rows), ("loop", min(rows, args.loop_rows))):
            out = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--child",
                    mode,
                    str(count),
                    "--pixels",
                    str(args.pixels),
                    "--params",
                    str(args.params),
                    "--threads",
                    str(args.threads),
                ]
                + (
                    []
                    if args.max_outlier_fraction is None
                    else ["--max-outlier-fraction", str(args.max_outlier_fraction)]
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            stats = json.loads(out.stdout.strip().splitlines()[-1])
            print(
                f"{mode:<8} {rows:>6} {stats['rows_per_s']:>10.1f} {stats['seconds']:>8.2f} "
                f"{stats['mean_iterations']:>6.2f} {stats['converged_fraction']:>5.2f} "
                f"{stats['peak_rss_over_import_mb']:>8.0f}"
                + (
                    f"  solve calls {stats['solve_calls']:.0f}, row solves "
                    f"{stats['row_solves']:.0f} (max iters {stats['max_iterations']:.0f})"
                    if mode == "batched"
                    else ""
                )
            )


if __name__ == "__main__":
    main()
