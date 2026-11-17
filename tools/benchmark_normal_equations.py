"""Per-call time of ``solve_normal_equations`` on many small (non-negative) problems.

The regime of a caller that solves one or a few objects at a time inside its own
loop (a Gauss--Newton refinement, a damping search): ``B`` rows of ``n``
parameters, the first ``k`` constrained to be non-negative, normal equations from a
weighted design with a positive constrained block and a mixed-sign truth, so the
active set does work. Reports the minimum over ``--repeats`` of the mean time per
call, interleaving the solvers, and the non-negative solves' active-set work
(factorisations per row). ``--uspm-reference`` also times the frozen uspm
``nnls_normal`` (``scripts/uspm_b461f8f_reference.py``) on the same inputs and
checks that the solutions agree.

    pixi run python tools/benchmark_normal_equations.py
    pixi run python tools/benchmark_normal_equations.py --batch 1 8 --uspm-reference
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

from torchregress.algorithms import solve_normal_equations

DTYPE = torch.float64


def make_problem(b: int, n: int, k: int, pixels: int, seed: int) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, pixels, n, generator=g, dtype=DTYPE)
    x[..., :k] = x[..., :k].abs() + 0.1
    truth = torch.randn(b, n, generator=g, dtype=DTYPE)
    noise = 0.3 * torch.randn(b, pixels, generator=g, dtype=DTYPE)
    y = (x @ truth.unsqueeze(-1)).squeeze(-1) + noise
    w = 0.5 + torch.rand(b, pixels, generator=g, dtype=DTYPE)
    xw = x * w.unsqueeze(-1)
    return x.transpose(1, 2) @ xw, (xw * y.unsqueeze(-1)).sum(1)


def per_call_ms(solvers: dict[str, Callable[[], object]], calls: int, repeats: int) -> dict:
    best = {name: float("inf") for name in solvers}
    for solve in solvers.values():
        solve()  # warm-up
    for _ in range(repeats):
        for name, solve in solvers.items():
            start = time.perf_counter()
            for _ in range(calls):
                solve()
            best[name] = min(best[name], (time.perf_counter() - start) / calls * 1e3)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 512])
    parser.add_argument("--params", type=int, nargs="+", default=[6, 12, 20])
    parser.add_argument("--constrained", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--pixels", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--uspm-reference", action="store_true")
    parser.add_argument("--json", type=Path, help="write the rows here")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    reference = None
    if args.uspm_reference:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import uspm_b461f8f_reference as reference  # noqa: PLC0415

    import torchregress.algorithms.normal_equations as module  # noqa: PLC0415

    rows = []
    for b in args.batch:
        for n in args.params:
            for k in args.constrained:
                if k > n:
                    continue
                normal, rhs = make_problem(b, n, k, args.pixels, seed=1000 * n + 10 * k + b)
                mask = torch.arange(n) < k
                solvers: dict[str, Callable[[], object]] = {
                    "nonnegative": lambda: solve_normal_equations(normal, rhs, nonnegative=mask),
                    "unconstrained": lambda: solve_normal_equations(normal, rhs),
                }
                if reference is not None:
                    solvers["uspm_nnls_normal"] = lambda: reference.nnls_normal(normal, rhs, k)
                calls = max(1, round(200 / (1 + b / 8)))
                times = per_call_ms(solvers, calls, args.repeats)
                factor = module._free_block_cholesky
                sizes: list[int] = []

                def counting(scaled, free, ridge):  # type: ignore[no-untyped-def]
                    sizes.append(scaled.shape[0])
                    return factor(scaled, free, ridge)

                module._free_block_cholesky = counting  # type: ignore[assignment]
                try:
                    out = solve_normal_equations(normal, rhs, nonnegative=mask)
                finally:
                    module._free_block_cholesky = factor  # type: ignore[assignment]
                row = {
                    "batch": b,
                    "params": n,
                    "constrained": k,
                    **{f"{name}_ms": value for name, value in times.items()},
                    "factorisations_per_row": sum(sizes) / b,
                }
                if reference is not None:
                    sol = reference.nnls_normal(normal, rhs, k)[0]
                    row["max_abs_diff_vs_uspm"] = float((out.solution - sol).abs().max())
                rows.append(row)
                text = "  ".join(
                    f"{key} {value:.3g}" if isinstance(value, float) else f"{key} {value}"
                    for key, value in row.items()
                )
                print(text, flush=True)
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
