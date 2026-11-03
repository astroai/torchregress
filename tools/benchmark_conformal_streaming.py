"""Benchmark density-conformal calibration and prediction at realistic sizes.

Times one case per invocation and reports wall time and peak memory as JSON:

- ``cd_calibrate``: ``ConformalizedDensity.calibrate_log_prob`` with a 6-D zuko NSF.
- ``cd_contains``: ``ConformalizedDensity.contains`` on the test split.
- ``cd_interval``: scalar ``ConformalizedDensity.predict_interval`` (1-D NSF).
- ``waldo_density_calibrate``: ``WaldoDensityConformal.calibrate`` with an MDN sampler.
- ``waldo_posterior_calibrate``: ``WaldoPosteriorCalibrator.calibrate`` on given samples.
- ``waldo_interval``: scalar ``WaldoConformal.predict_interval``.

Peak memory is ``torch.cuda.max_memory_allocated`` on CUDA and the growth of the
process ``ru_maxrss`` high-water mark on CPU (run one case per process for a clean
CPU number).

Example::

    pixi run python tools/benchmark_conformal_streaming.py --case cd_calibrate \
        --device cuda --n-cal 1000000
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from typing import Callable

import torch
import zuko
from torch import Tensor, nn

from torchregress.losses import (
    ConformalizedDensity,
    WaldoConformal,
    WaldoDensityConformal,
    WaldoPosteriorCalibrator,
)

CASES = (
    "cd_calibrate",
    "cd_contains",
    "cd_interval",
    "waldo_density_calibrate",
    "waldo_posterior_calibrate",
    "waldo_interval",
)


class _MDN(nn.Module):
    """Small diagonal Gaussian mixture density network."""

    def __init__(self, context: int, dim: int, components: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.k = components
        self.net = nn.Sequential(
            nn.Linear(context, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, components * (1 + 2 * dim)),
        )

    def dist(self, context: Tensor) -> torch.distributions.Distribution:
        out = self.net(context)
        logits = out[:, : self.k]
        loc, log_scale = out[:, self.k :].reshape(-1, self.k, 2, self.dim).unbind(dim=2)
        comp = torch.distributions.Independent(
            torch.distributions.Normal(loc, log_scale.clamp(-5, 2).exp()), 1
        )
        mix = torch.distributions.Categorical(logits=logits)
        return torch.distributions.MixtureSameFamily(mix, comp)

    def sample(self, context: Tensor, n: int) -> Tensor:
        return self.dist(context).sample((n,)).transpose(0, 1)  # [N, S, D]


def _scalar(value: Tensor | None) -> float | None:
    return None if value is None else float(value)


def _peak_reset(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        return float(torch.cuda.memory_allocated(device))
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _peak_read(device: torch.device, base: float) -> float:
    """Peak memory growth in MiB since ``_peak_reset``."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return (torch.cuda.max_memory_allocated(device) - base) / 2**20
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    scale = 1.0 if sys.platform == "darwin" else 1024.0  # bytes on macOS, KiB on Linux
    return (rss - base) * scale / 2**20


def _timed(device: torch.device, fn: Callable[[], object]) -> tuple[float, float]:
    base = _peak_reset(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - t0, _peak_read(device, base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-cal", type=int, default=200_000)
    parser.add_argument("--n-test", type=int, default=200_000)
    parser.add_argument("--n-interval", type=int, default=2_000)
    parser.add_argument("--dim", type=int, default=6)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    def data(n: int, d: int) -> tuple[Tensor, Tensor]:
        ctx = torch.randn(n, args.context, generator=gen).to(device)
        y = torch.randn(n, d, generator=gen).to(device)
        return ctx, y

    case = args.case
    detail: dict[str, object] = {}
    if case in ("cd_calibrate", "cd_contains", "cd_interval"):
        d = 1 if case == "cd_interval" else args.dim
        flow = zuko.flows.NSF(d, args.context, transforms=3, hidden_features=[64, 64]).to(device)
        flow.eval()

        def log_prob(c: Tensor, y: Tensor) -> Tensor:
            return flow(c).log_prob(y)

        def nld(c: Tensor, y: Tensor) -> Tensor:
            return -flow(c).log_prob(y)

        cp = ConformalizedDensity(alpha=0.1, grid_size=args.grid_size)
        ctx, y = data(args.n_cal, d)
        if case == "cd_calibrate":
            elapsed, peak = _timed(device, lambda: cp.calibrate_log_prob(log_prob, ctx, y))
            detail["threshold"] = _scalar(cp.threshold)
        else:
            with torch.no_grad():
                cp.calibrate_log_prob(log_prob, ctx[:5000], y[:5000])
            if case == "cd_contains":
                tctx, ty = data(args.n_test, d)
                elapsed, peak = _timed(device, lambda: cp.contains(nld, tctx, ty))
            else:
                tctx, _ = data(args.n_interval, d)
                out: list[tuple[Tensor, Tensor]] = []
                elapsed, peak = _timed(
                    device, lambda: out.append(cp.predict_interval(nld, tctx, -6.0, 6.0))
                )
                detail["mean_width"] = float((out[0][1] - out[0][0]).mean())
    elif case in ("waldo_density_calibrate", "waldo_posterior_calibrate"):
        mdn = _MDN(args.context, args.dim).to(device)
        ctx, _ = data(args.n_cal, args.dim)
        with torch.no_grad():
            y = mdn.sample(ctx[: min(args.n_cal, 50_000)], 1)[:, 0]
            y = y.repeat((args.n_cal + y.shape[0] - 1) // y.shape[0], 1)[: args.n_cal]
        if case == "waldo_density_calibrate":
            wd = WaldoDensityConformal(alpha=0.1, covariance_type="full", n_samples=args.n_samples)
            elapsed, peak = _timed(device, lambda: wd.calibrate(mdn.sample, ctx, y))
            detail["threshold"] = _scalar(wd.threshold)
        else:
            # Plain Gaussian draws: no sampler intermediates inflate the RSS baseline.
            samples = y.unsqueeze(1) + 0.5 * torch.randn(
                args.n_cal, args.n_samples, args.dim, generator=gen
            ).to(device)
            wp = WaldoPosteriorCalibrator(alpha=0.1, covariance_type="full")
            elapsed, peak = _timed(device, lambda: wp.calibrate(samples, y))
            detail["threshold"] = _scalar(wp.threshold)
            detail["samples_mib"] = samples.numel() * samples.element_size() / 2**20
    else:  # waldo_interval
        gen_mean = torch.randn(args.n_cal, 1, generator=gen).to(device)
        std = torch.full_like(gen_mean, 0.5)
        target = gen_mean + std * torch.randn(args.n_cal, 1, generator=gen).to(device)
        wc = WaldoConformal(alpha=0.1, critical_bins=10)
        wc.calibrate(gen_mean, target, std**2)
        tm = torch.randn(args.n_interval, 1, generator=gen).to(device)
        ts = torch.full_like(tm, 0.5)
        elapsed, peak = _timed(
            device,
            lambda: wc.predict_interval(tm, ts, y_min=-6, y_max=6, grid_size=args.grid_size),
        )

    print(
        json.dumps(
            {
                "case": case,
                "device": str(device),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "n_cal": args.n_cal,
                "n_test": args.n_test,
                "n_interval": args.n_interval,
                "dim": args.dim,
                "n_samples": args.n_samples,
                "grid_size": args.grid_size,
                "seconds": round(elapsed, 4),
                "peak_mib": round(peak, 1),
                **detail,
            }
        )
    )


if __name__ == "__main__":
    main()
