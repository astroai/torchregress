"""Scaling of ``marginalize_linear_nuisance`` with rows, samples and nuisance rank.

Forward + backward (w.r.t. ``y_pred``) of the marginal Gaussian NLL for ``B`` rows
of ``N`` samples with ``m`` nuisance columns (Legendre baseline, flat prior,
float64, 10% masked samples), with the basis shared ``[N, m]`` or per row
``[B, N, m]``. The comparator is ``torch.distributions.LowRankMultivariateNormal``
with factor ``U`` (per row, unit prior covariance; it has no masks or flat
priors). Each configuration runs in a fresh subprocess so its peak RSS is its own.

    pixi run python tools/benchmark_linear_nuisance.py --rows 16 256 --pixels 1000 10000 --rank 5 50
"""

from __future__ import annotations

import argparse
import itertools
import json
import resource
import subprocess
import sys
import time

import torch
from torch.distributions import LowRankMultivariateNormal

from torchregress.losses import marginalize_linear_nuisance


def _data(rows: int, pixels: int, rank: int, seed: int = 0):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    grid = torch.linspace(-1.0, 1.0, pixels, dtype=torch.float64)
    basis = torch.stack([torch.special.legendre_polynomial_p(grid, k) for k in range(rank)], -1)
    precision = 100.0 * (0.5 + torch.rand(rows, pixels, generator=g, dtype=torch.float64))
    target = torch.randn(rows, pixels, generator=g, dtype=torch.float64) * precision.rsqrt()
    mask = torch.rand(rows, pixels, generator=g) > 0.1
    return basis, target, precision, mask


def _child(mode: str, rows: int, pixels: int, rank: int, threads: int, repeats: int) -> None:
    torch.set_num_threads(threads)
    basis, target, precision, mask = _data(rows, pixels, rank)
    pred = torch.zeros_like(target, requires_grad=True)
    if mode == "per_row":
        basis = basis.expand(rows, pixels, rank).contiguous()

    def run() -> None:
        pred.grad = None
        if mode == "low_rank_mvn":
            dist = LowRankMultivariateNormal(
                pred, basis.expand(rows, pixels, rank), 1.0 / precision
            )
            loss = -dist.log_prob(target).sum()
        else:
            loss = marginalize_linear_nuisance(
                pred, target, precision, basis, prior_precision=None, mask=mask
            ).nll.sum()
        loss.backward()

    run()  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    best = min(times)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(
        json.dumps(
            {
                "mode": mode,
                "rows": rows,
                "pixels": pixels,
                "rank": rank,
                "seconds": best,
                "rows_per_s": rows / best,
                "msamples_per_s": rows * pixels / best / 1e6,
                "peak_rss_mb": peak_mb,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, nargs="+", default=[16, 256])
    parser.add_argument("--pixels", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--rank", type=int, nargs="+", default=[5, 50])
    parser.add_argument("--modes", nargs="+", default=["shared", "per_row", "low_rank_mvn"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--max-elements",
        type=float,
        default=2e8,
        help="skip configurations with rows * pixels * rank above this (memory bound)",
    )
    parser.add_argument(
        "--child", nargs=4, metavar=("MODE", "ROWS", "PIXELS", "RANK"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.child:
        mode, rows, pixels, rank = args.child
        _child(mode, int(rows), int(pixels), int(rank), args.threads, args.repeats)
        return

    print(
        f"{'mode':>13} {'B':>6} {'N':>6} {'m':>3} {'s/iter':>9} {'rows/s':>10} {'Msamp/s':>8} {'RSS MB':>7}"
    )
    for mode, rows, pixels, rank in itertools.product(
        args.modes, args.rows, args.pixels, args.rank
    ):
        if mode != "shared" and rows * pixels * rank > args.max_elements:
            print(f"{mode:>13} {rows:>6} {pixels:>6} {rank:>3}  skipped (> --max-elements)")
            continue
        cmd = [
            sys.executable,
            __file__,
            "--child", mode, str(rows), str(pixels), str(rank),
            "--threads", str(args.threads),
            "--repeats", str(args.repeats),
        ]  # fmt: skip
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        s = json.loads(out.stdout.strip().splitlines()[-1])
        print(
            f"{mode:>13} {rows:>6} {pixels:>6} {rank:>3} {s['seconds']:>9.4f} "
            f"{s['rows_per_s']:>10.1f} {s['msamples_per_s']:>8.2f} {s['peak_rss_mb']:>7.0f}"
        )


if __name__ == "__main__":
    main()
