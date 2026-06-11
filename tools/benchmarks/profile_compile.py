"""Compare eager vs torch.compile-wrapped MultivariateGaussianLoss.forward.

torch.compile on CPU uses the Inductor backend: graph capture + kernel fusion
are available, CUDA-graph capture is not.  Compile cost is paid on the
first call, so warmup includes a fresh ``torch._dynamo.reset()`` and a
few trigger compilations before timing.
"""

import argparse
import math
import statistics
import time

import torch


def time_callable(fn, iters: int) -> float:
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters


def bench(label, fn, *, warmup=20, iters=100, repeats=5):
    for _ in range(warmup):
        fn()
    times = [time_callable(fn, iters) for _ in range(repeats)]
    med = statistics.median(times)
    print(
        f"  {label:46s}  median {med * 1e6:8.2f} us  "
        f"(min {min(times) * 1e6:.2f}, max {max(times) * 1e6:.2f})"
    )
    return med


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1024)
    parser.add_argument("--D", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(0)
    B, D = args.B, args.D
    device = torch.device("cpu")
    dtype = torch.float32
    jitter = 1e-6
    eps = 1e-8

    mean = torch.randn(B, D, device=device, dtype=dtype)
    target = torch.randn(B, D, device=device, dtype=dtype)
    A = torch.randn(B, D, D, device=device, dtype=dtype)
    cov = A @ A.transpose(-1, -2) + torch.eye(D, device=device) * 1e-3
    eye = torch.eye(D, device=device, dtype=dtype) * jitter

    def fwd_default():
        diff = (target - mean).unsqueeze(-1)
        cov_jit = cov + eye
        L = torch.linalg.cholesky(cov_jit)
        sol = torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1)
        quad = torch.sum(sol**2, dim=-1)
        log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + eps), dim=-1)
        nll = 0.5 * (log_det + quad + D * math.log(2 * math.pi))
        return nll.mean()

    fwd_compiled = torch.compile(fwd_default, dynamic=False, fullgraph=False)

    print(f"=== torch.compile comparison | B={B} D={D} CPU float32 ===")

    print("--- Eager (no compile) ---")
    t_eager = bench("default fwd (eager)", fwd_default)

    print("--- torch.compile (inductor CPU) ---")
    # Pay the compile cost outside the timed window.
    print("  [warmup includes compile, may print dynamo messages]")

    def compiled_runner():
        return fwd_compiled()

    t_compiled = bench("default fwd (compiled)", compiled_runner, warmup=30)

    speedup = t_eager / t_compiled if t_compiled > 0 else float("inf")
    print(
        f"\n  Eager  : {t_eager * 1e6:8.2f} us/iter\n"
        f"  Compile: {t_compiled * 1e6:8.2f} us/iter\n"
        f"  Speedup: {speedup:.2f}x"
    )


if __name__ == "__main__":
    main()
