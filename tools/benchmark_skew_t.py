"""Per-step training time of the skew-t NLL head: early-exit Lentz vs fixed-depth CF.

The Student-t CDF inside ``SkewTLoss`` evaluates a continued fraction. The
early-exit Lentz loop (``_betacf``, the CPU default) tests convergence every
term, a device-to-host sync on accelerators; the fixed-depth tree evaluation
(``_betacf_fixed``, the default off the CPU) is sync-free and CUDA-graph
capturable. This times one Adam step of a small MLP + ``SkewTLoss`` with each
evaluation forced, eagerly and (on CUDA) as a replayed CUDA graph, and counts
kernel launches and synchronising calls per step.

    pixi run python tools/benchmark_skew_t.py --device cpu
    pixi run -e cuda python tools/benchmark_skew_t.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import torch

import torchregress.losses.families as fam
from torchregress.losses import SkewTLoss

PATHS: dict[str, Callable[[torch.device], bool]] = {
    "lentz": lambda device: False,
    "fixed": lambda device: True,
}


def _setup(dev: torch.device, batch: int, capturable: bool):  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    x = torch.randn(batch, 16, device=dev)
    y = 2.0 * torch.randn(batch, device=dev)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4)
    ).to(device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, capturable=capturable)
    loss_fn = SkewTLoss()

    def step() -> None:
        opt.zero_grad(set_to_none=not capturable)
        loss_fn(model(x), y).backward()
        opt.step()

    return model, x, y, step


def _time_step(dev: torch.device, batch: int, steps: int, graph: bool) -> float:
    _, _, _, step = _setup(dev, batch, capturable=graph)
    cuda = dev.type == "cuda"
    if graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        run = g.replay
    else:
        for _ in range(3):
            step()
        run = step
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        run()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1e3


def _count_launches(dev: torch.device, batch: int) -> dict[str, int]:
    from torch.profiler import ProfilerActivity, profile

    _, _, _, step = _setup(dev, batch, capturable=False)
    step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events()]
    launches = sum(n.startswith(("cudaLaunchKernel", "cuLaunchKernel")) for n in names)
    # Stream syncs inside the step (the explicit device sync after it is not
    # counted); device-to-device cudaMemcpyAsync copies do not block the host.
    syncs = sum(n == "cudaStreamSynchronize" for n in names)
    return {"kernel_launches": launches, "host_syncs": syncs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batches", type=int, nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    batches = args.batches or ([1024, 4096, 16384] if cuda else [256, 1024, 4096])
    steps = args.steps or (50 if cuda else 20)
    original = fam._cf_fixed_depth
    results: dict[str, dict[str, object]] = {}
    try:
        for batch in batches:
            row: dict[str, object] = {}
            for name, path in PATHS.items():
                fam._cf_fixed_depth = path
                row[f"{name}_eager_ms"] = round(_time_step(dev, batch, steps, graph=False), 3)
                if cuda:
                    if name == "fixed":
                        ms = _time_step(dev, batch, steps, graph=True)
                        row[f"{name}_graph_ms"] = round(ms, 3)
                    row[f"{name}_per_step"] = _count_launches(dev, batch)
            results[str(batch)] = row
            print(batch, json.dumps(row), flush=True)
        if cuda:
            # Last, since a failed capture can leave the context unusable.
            fam._cf_fixed_depth = PATHS["lentz"]
            try:
                _time_step(dev, batches[0], 1, graph=True)
                results["lentz_graph_capture"] = {"ok": True}
            except RuntimeError as exc:
                results["lentz_graph_capture"] = {"ok": False, "error": str(exc)[:120]}
    finally:
        fam._cf_fixed_depth = original
    print(json.dumps({"device": str(dev), "torch": torch.__version__, "results": results}))


if __name__ == "__main__":
    main()
