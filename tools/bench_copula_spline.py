"""Forward+backward timing of ``GaussianCopulaSplineHead`` with ragged missingness.

Examples::

    pixi run python tools/bench_copula_spline.py --device cpu
    pixi run -e cuda python tools/bench_copula_spline.py --device cuda --steps 50
    # compare with another implementation of the module, e.g. an older revision:
    git show <rev>:src/torchregress/models/copula_spline.py > /tmp/old.py
    pixi run python tools/bench_copula_spline.py --module /tmp/old.py
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from collections.abc import Callable

import torch


def _head_class(path: str | None) -> type:
    if path is None:
        from torchregress.models.copula_spline import GaussianCopulaSplineHead

        return GaussianCopulaSplineHead
    spec = importlib.util.spec_from_file_location("copula_under_test", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GaussianCopulaSplineHead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--module", help="path to an alternative copula_spline.py")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--dims", type=int, nargs="+", default=[6, 12])
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--p-obs", type=float, default=0.7, help="per-coordinate P(observed)")
    parser.add_argument("--nodes", type=int, default=16, help="noise nodes; 0 skips the timing")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    head_class = _head_class(args.module)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    for dim in args.dims:
        torch.manual_seed(0)
        head = head_class(16, dim, width=128).to(device=device, dtype=dtype)
        with torch.no_grad():
            head.net[-1].weight.normal_(0, 0.05)
        x = torch.randn(args.batch, 16, device=device, dtype=dtype)
        y = torch.randn(args.batch, dim, device=device, dtype=dtype)
        sigma = torch.full_like(y, 0.1)
        nodes = torch.randn(max(args.nodes, 2), dim, device=device, dtype=dtype)
        masks = {
            "complete": torch.ones(args.batch, dim, dtype=torch.bool, device=device),
            "missing": torch.rand(args.batch, dim, device=device) < args.p_obs,
        }
        for label, mask in masks.items():
            patterns = len(torch.unique(mask, dim=0))
            modes: list[tuple[str, Callable[[], torch.Tensor]]] = [
                ("intrinsic", lambda m=mask: head.intrinsic_log_prob(x, y, m))
            ]
            if args.nodes:
                modes.append(
                    (
                        f"observed K={args.nodes}",
                        lambda m=mask: head.observed_log_prob(x, y, sigma, m, nodes)[0],
                    )
                )
            for name, fn in modes:
                for _ in range(2):  # warm-up
                    head.zero_grad(set_to_none=True)
                    (-fn().mean()).backward()
                sync()
                start = time.perf_counter()
                for _ in range(args.steps):
                    head.zero_grad(set_to_none=True)
                    (-fn().mean()).backward()
                sync()
                seconds = (time.perf_counter() - start) / args.steps
                print(
                    f"{device.type} D={dim:2d} {label:8s} patterns={patterns:5d} "
                    f"{name:14s} {seconds:.4f} s/step"
                )


if __name__ == "__main__":
    main()
