"""Section-level profile of MultivariateGaussianLoss.forward.

Times each step of the forward pass (cov prep, Cholesky, solve_triangular,
log_det, quadratic form, reduction) and compares the default
``cholesky + solve_triangular`` path against two alternatives:

  - ``cholesky_inverse + matmul``:  log|Sigma| = -2 sum log L_diag then
    Sigma^{-1} via ``torch.cholesky_inverse`` and the quadratic form as
    ``(y-mu)^T Sigma^{-1} (y-mu)``.
  - ``cholesky_solve + tri_to_dense``: pull the solve through the same
    factorization without the ``triangular_solve`` overhead.

Outputs a section table (median us/iter) and a final verdict.
"""

import argparse
import math
import statistics
import time

import torch

from torchregress.losses.gaussian import reduce_per_sample


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
    parser.add_argument("--label", default="AFTER (current refactor)")
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

    print(f"=== {args.label} | B={B} D={D} CPU float32 | iters=100 repeats=5 ===")

    # Section breakdown of the current forward.
    cov_with_jitter = cov + torch.eye(D, device=device) * jitter
    diff = (target - mean).unsqueeze(-1)  # [B, D, 1]

    print("--- Section timings ---")
    bench(
        "cov + jitter (in-place clone + diag embed)",
        lambda: cov + torch.eye(D, device=device) * jitter,
    )
    bench("cholesky (B, D, D)", lambda: torch.linalg.cholesky(cov_with_jitter))
    bench("diff = (target - mean).unsqueeze(-1)", lambda: (target - mean).unsqueeze(-1))

    L = torch.linalg.cholesky(cov_with_jitter)
    bench("solve_triangular(L, diff)", lambda: torch.linalg.solve_triangular(L, diff, upper=False))
    bench(
        "diagonal(L) + log + sum (log_det)",
        lambda: 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + eps), dim=-1),
    )
    bench(
        "quad = sum(sol**2, dim=-1)",
        lambda: torch.sum(
            torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1) ** 2, dim=-1
        ),
    )

    bench(
        "reduce_per_sample(nll, None, None, 'mean')",
        lambda: reduce_per_sample(torch.zeros(B), None, None, "mean"),
    )

    # Full forward: original (cholesky + solve_triangular) vs alternatives.
    print("--- Full forward (cholesky + solve_triangular, current) ---")

    def fwd_default():
        diff_local = (target - mean).unsqueeze(-1)
        L = torch.linalg.cholesky(cov_with_jitter)
        sol = torch.linalg.solve_triangular(L, diff_local, upper=False).squeeze(-1)
        quad = torch.sum(sol**2, dim=-1)
        log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + eps), dim=-1)
        nll = 0.5 * (log_det + quad + D * math.log(2 * math.pi))
        return reduce_per_sample(nll, None, None, "mean")

    bench("default fwd", fwd_default)

    print("--- Alternative: cholesky_inverse + matmul (Sigma^-1 y) ---")

    def fwd_chol_inverse():
        L = torch.linalg.cholesky(cov_with_jitter)
        Sigma_inv = torch.cholesky_inverse(L)
        quad = torch.einsum("bij,bj->bi", Sigma_inv, target - mean)
        quad = torch.sum(quad * (target - mean), dim=-1)
        log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + eps), dim=-1)
        nll = 0.5 * (log_det + quad + D * math.log(2 * math.pi))
        return reduce_per_sample(nll, None, None, "mean")

    bench("cholesky_inverse fwd", fwd_chol_inverse)

    print("--- Alternative: cholesky_solve with identity (gets Sigma^-1) ---")

    def fwd_chol_solve():
        L = torch.linalg.cholesky(cov_with_jitter)
        eye_batch = torch.eye(D, device=device, dtype=dtype).expand(B, D, D)
        Sigma_inv = torch.cholesky_solve(eye_batch, L, upper=False)
        quad = torch.einsum("bij,bj->bi", Sigma_inv, target - mean)
        quad = torch.sum(quad * (target - mean), dim=-1)
        log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + eps), dim=-1)
        nll = 0.5 * (log_det + quad + D * math.log(2 * math.pi))
        return reduce_per_sample(nll, None, None, "mean")

    bench("cholesky_solve(I) fwd", fwd_chol_solve)

    print("--- torch.profiler: default fwd ---")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        for _ in range(50):
            fwd_default()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))


if __name__ == "__main__":
    main()
