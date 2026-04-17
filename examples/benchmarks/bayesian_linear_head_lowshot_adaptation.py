"""
Low-shot adaptation on fixed Gaussian features: conjugate BLR vs ridge MAP.

Ridge with L2 penalty ``alpha`` matches the **posterior mean** of ``BayesianLinearHead`` when
``prior_precision = alpha / noise_variance`` (Gaussian prior on weights, known homoscedastic
noise). The benchmark highlights **predictive Gaussian NLL** on held-out points, where BLR
uses full predictive variance while an oracle ridge baseline uses only the known noise
variance (no epistemic inflation).
"""

from __future__ import annotations

import argparse

import torch

from torchregress.test_time import BayesianLinearHead


def _ridge_weights(phi: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """Ridge MAP with intercept column omitted (caller augments if needed)."""
    d = phi.shape[1]
    a = phi.T @ phi + alpha * torch.eye(d, device=phi.device, dtype=phi.dtype)
    b = phi.T @ y
    return torch.linalg.solve(a, b)


def _rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())


def _gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> float:
    v = var.clamp(min=eps)
    nll = 0.5 * (torch.log(2 * torch.pi * v) + (target - mean) ** 2 / v)
    return float(nll.mean().item())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=24, help="Feature dimension (no intercept).")
    p.add_argument("--n-test", type=int, default=4000)
    p.add_argument(
        "--shots",
        type=str,
        default="4,8,16,32,64",
        help="Comma-separated training sizes (each run uses an independent draw).",
    )
    p.add_argument("--noise", type=float, default=0.4, help="Homoscedastic noise std.")
    p.add_argument("--ridge-alpha", type=float, default=0.25, help="Ridge L2 strength.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    dtype = torch.float64
    sigma2 = args.noise**2
    tau = args.ridge_alpha / sigma2
    shot_sizes = [int(s.strip()) for s in args.shots.split(",") if s.strip()]

    w_true = torch.randn(args.dim, 1, device=device, dtype=dtype)
    phi_test = torch.randn(args.n_test, args.dim, device=device, dtype=dtype)
    y_test = phi_test @ w_true + args.noise * torch.randn(
        args.n_test, 1, device=device, dtype=dtype
    )

    print(
        f"{'n_shot':>8}  {'rmse_blr':>10}  {'rmse_ridge':>12}  "
        f"{'nll_blr':>10}  {'nll_ridge_oracle':>16}"
    )
    for n in shot_sizes:
        phi_tr = torch.randn(n, args.dim, device=device, dtype=dtype)
        y_tr = phi_tr @ w_true + args.noise * torch.randn(n, 1, device=device, dtype=dtype)

        head = BayesianLinearHead(
            in_features=args.dim,
            fit_intercept=False,
            prior_precision=float(tau),
            noise_variance=float(sigma2),
        ).fit(phi_tr, y_tr)

        w_ridge = _ridge_weights(phi_tr, y_tr, float(args.ridge_alpha))
        pred_blr = head.predict(phi_test, return_std=True)
        mean_blr = pred_blr["mean"]
        mean_ridge = phi_test @ w_ridge

        rmse_b = _rmse(mean_blr, y_test)
        rmse_r = _rmse(mean_ridge, y_test)
        nll_b = _gaussian_nll(y_test, mean_blr, pred_blr["variance"])
        nll_r = _gaussian_nll(y_test, mean_ridge, torch.full_like(mean_ridge, sigma2))

        print(f"{n:8d}  {rmse_b:10.5f}  {rmse_r:12.5f}  {nll_b:10.4f}  {nll_r:16.4f}")


if __name__ == "__main__":
    main()
