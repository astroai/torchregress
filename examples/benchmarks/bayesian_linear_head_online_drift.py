"""
Online label drift on fixed Gaussian features: ``RecursiveBayesianHead`` vs batch refit.

Phase A fits weights ``w1``; phase B switches to ``w2`` while feature distribution stays
the same. Compares streaming ``partial_fit`` under different ``forgetting_factor`` values to
a **batch oracle** refit on phase-B data only (same total sample count as phase B).

Note: ``partial_fit`` scales precision ``Lambda`` by ``forgetting_factor`` when ``< 1`` but
does not scale ``h``; this is the intended v1 behaviour (see module docstring). The script
still reports empirical adaptation quality under that update.
"""

from __future__ import annotations

import argparse

import torch

from torchregress.test_time import BayesianLinearHead, RecursiveBayesianHead


def _rmse(head: BayesianLinearHead, phi: torch.Tensor, y: torch.Tensor) -> float:
    pred = head.predict(phi, return_std=False)
    return float(torch.sqrt(torch.mean((pred["mean"] - y) ** 2)).item())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--n-phase-a", type=int, default=400)
    p.add_argument("--n-phase-b", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--noise", type=float, default=0.35)
    p.add_argument("--prior-precision", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float64
    device = torch.device("cpu")
    sigma2 = args.noise**2

    w1 = torch.randn(args.dim, 1, device=device, dtype=dtype)
    w2 = torch.randn(args.dim, 1, device=device, dtype=dtype)

    def stream_xy(w: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        phi = torch.randn(n, args.dim, device=device, dtype=dtype)
        y = phi @ w + args.noise * torch.randn(n, 1, device=device, dtype=dtype)
        return phi, y

    phi_te, y_te = stream_xy(w2, args.n_test)

    phi_b, y_b = stream_xy(w2, args.n_phase_b)
    oracle = BayesianLinearHead(
        in_features=args.dim,
        fit_intercept=False,
        prior_precision=args.prior_precision,
        noise_variance=float(sigma2),
    ).fit(phi_b, y_b)
    rmse_oracle = _rmse(oracle, phi_te, y_te)

    forgetting_grid = [1.0, 0.995, 0.99, 0.98]
    print(f"phase B oracle RMSE (fit on phase B only): {rmse_oracle:.5f}")
    print(f"{'forgetting':>12}  {'rmse_after_stream':>18}")

    for ff in forgetting_grid:
        head = RecursiveBayesianHead(
            in_features=args.dim,
            fit_intercept=False,
            prior_precision=args.prior_precision,
            noise_variance=float(sigma2),
            forgetting_factor=float(ff),
        )
        phi_a, y_a = stream_xy(w1, args.n_phase_a)
        for start in range(0, phi_a.shape[0], args.batch_size):
            sl = slice(start, start + args.batch_size)
            head.partial_fit(phi_a[sl], y_a[sl])
        for start in range(0, phi_b.shape[0], args.batch_size):
            sl = slice(start, start + args.batch_size)
            head.partial_fit(phi_b[sl], y_b[sl])
        rmse = _rmse(head, phi_te, y_te)
        print(f"{ff:12.4f}  {rmse:18.5f}")


if __name__ == "__main__":
    main()
