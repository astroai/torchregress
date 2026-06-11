"""Before/after benchmark for GaussianCRPSLoss and MultivariateGaussianLoss.

CPU, B=1024, D=5, 100 timed iterations per repeat, 5 repeats.
Median reported to reduce noise; also prints min/max.
"""

import argparse
import statistics
import time

import torch

from torchregress.losses.gaussian import GaussianCRPSLoss, MultivariateGaussianLoss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="label printed in the header")
    args = parser.parse_args()

    torch.manual_seed(0)
    B, D = 1024, 5
    ITERS = 100
    WARMUP = 20
    REPEATS = 5

    device = torch.device("cpu")
    dtype = torch.float32

    mean_base = torch.randn(B, D, device=device, dtype=dtype)
    target = torch.randn(B, D, device=device, dtype=dtype)
    log_var = torch.zeros(B, D, device=device, dtype=dtype)

    A = torch.randn(B, D, D, device=device, dtype=dtype)
    cov = A @ A.transpose(-1, -2) + torch.eye(D, device=device) * 1e-3

    def fresh_mean() -> torch.Tensor:
        return torch.randn_like(mean_base).requires_grad_(True)

    def time_callable(fn, iters: int) -> float:
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - start) / iters

    def bench(label, fn, *, warmup=WARMUP, iters=ITERS, repeats=REPEATS):
        for _ in range(warmup):
            fn()
        times = [time_callable(fn, iters) for _ in range(repeats)]
        med = statistics.median(times)
        print(
            f"  {label:42s}  median {med * 1e6:8.2f} us/iter  "
            f"(min {min(times) * 1e6:.2f}, max {max(times) * 1e6:.2f})"
        )
        return med

    print(f"=== {args.label} | B={B} D={D} CPU float32 iters={ITERS} repeats={REPEATS} ===")

    print("--- GaussianCRPSLoss ---")
    crps_loss = GaussianCRPSLoss(reduction="mean")
    crps_fwd = bench("forward only", lambda: crps_loss((mean_base, log_var), target))

    def crps_fb():
        m = fresh_mean()
        out = crps_loss((m, log_var), target)
        out.backward()

    crps_fb_t = bench("forward + backward", crps_fb)
    print(f"  CRPS fwd-only={crps_fwd * 1e6:.2f}us  fwd+bwd={crps_fb_t * 1e6:.2f}us")

    print("--- MultivariateGaussianLoss ---")
    mvn_loss = MultivariateGaussianLoss()
    mvn_fwd = bench("forward only", lambda: mvn_loss(mean_base, target, cov))

    def mvn_fb():
        m = fresh_mean()
        out = mvn_loss(m, target, cov)
        out.backward()

    mvn_fb_t = bench("forward + backward", mvn_fb)
    print(f"  MVN  fwd-only={mvn_fwd * 1e6:.2f}us  fwd+bwd={mvn_fb_t * 1e6:.2f}us")


if __name__ == "__main__":
    main()
