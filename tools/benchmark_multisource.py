"""Time the multi-source marginal losses (forward + backward) on synthetic data.

Example::

    pixi run python -m tools.benchmark_multisource --objects 50000 --device cpu
    pixi run python -m tools.benchmark_multisource --objects 50000 --device cuda --robust

The problem mimics the harness multi-source suite: ``K = 12`` packed measurement
slots from several surveys (cross-correlated within a block), ragged
missingness, a ``D``-dimensional shared latent and a linearized observation map.
"""

from __future__ import annotations

import argparse
import time

import torch

from torchregress.losses import (
    MarginalizedMultiSourceGaussianNLLLoss,
    PackedObservations,
    RobustMultiSourceGaussianNLLLoss,
)

# Slot -> block (two surveys share one block, as in the harness): 12 slots, 4 blocks.
_BLOCKS = (0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 3)


def make_problem(
    objects: int, dims: int, device: torch.device, dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PackedObservations]:
    """Return ``(mean, scale_tril, offset, observations)`` with ragged missingness."""
    gen = torch.Generator().manual_seed(seed)
    blocks = torch.tensor(_BLOCKS)
    k = len(blocks)
    mean = torch.randn(objects, dims, generator=gen, dtype=dtype)
    raw = 0.3 * torch.randn(objects, dims, dims, generator=gen, dtype=dtype)
    lower = torch.tril(raw, -1) + torch.diag_embed(0.5 + raw.diagonal(dim1=-2, dim2=-1).abs())
    jac = 0.2 * torch.randn(objects, k, dims, generator=gen, dtype=dtype)
    jac[:, torch.arange(k), torch.arange(k) % dims] += 1.0
    offset = 0.05 * torch.randn(objects, k, generator=gen, dtype=dtype)
    sigma = 0.1 + 0.2 * torch.rand(objects, k, generator=gen, dtype=dtype)
    same = (blocks[:, None] == blocks[None, :]).to(dtype)
    corr = torch.eye(k, dtype=dtype) + 0.3 * (same - torch.eye(k, dtype=dtype))
    cov = sigma[:, :, None] * corr * sigma[:, None, :]
    mask = torch.rand(objects, k, generator=gen) < 0.6
    values = torch.randn(objects, k, generator=gen, dtype=dtype)
    values[~mask] = float("nan")
    obs = PackedObservations(
        values.to(device),
        torch.zeros(objects, k, dtype=dtype, device=device),
        jac.to(device),
        cov.to(device),
        mask.to(device),
        blocks.expand(objects, k).clone().to(device),
    )
    return mean.to(device), lower.to(device), offset.to(device), obs


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_loss(
    loss: torch.nn.Module,
    objects: int,
    dims: int,
    device: torch.device,
    dtype: torch.dtype,
    repeats: int,
) -> float:
    """Return the best forward+backward wall time in seconds."""
    mean0, lower0, offset, obs = make_problem(objects, dims, device, dtype)
    best = float("inf")
    for _ in range(repeats + 1):  # the first pass warms up kernels and the allocator
        mean = mean0.clone().requires_grad_(True)
        lower = lower0.clone().requires_grad_(True)
        mapped = torch.einsum("bkd,bd->bk", obs.jacobian, mean) + offset
        packed = PackedObservations(
            obs.values, mapped, obs.jacobian, obs.measurement_covariance, obs.mask, obs.block_ids
        )
        _sync(device)
        start = time.perf_counter()
        loss(mean, lower, packed).backward()
        _sync(device)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=int, default=50_000)
    parser.add_argument("--dims", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--robust", action="store_true", help="also time the robust mixture")
    parser.add_argument("--robust-objects", type=int, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cases: list[tuple[str, torch.nn.Module, int]] = [
        (f"marginal[{m}]", MarginalizedMultiSourceGaussianNLLLoss(method=m), args.objects)
        for m in ("auto", "direct", "information")
    ]
    if args.robust:
        robust_objects = args.robust_objects or args.objects
        cases.append(("robust[auto]", RobustMultiSourceGaussianNLLLoss(), robust_objects))
    print(f"device={device} dtype={args.dtype} K={len(_BLOCKS)} D={args.dims}")
    for name, loss, objects in cases:
        seconds = time_loss(loss, objects, args.dims, device, dtype, args.repeats)
        print(
            f"{name:16s} objects={objects:7d} {seconds * 1e3:10.1f} ms "
            f"{objects / seconds:12.0f} objects/s"
        )


if __name__ == "__main__":
    main()
