"""
Synthetic exactness demo: conjugate Bayesian linear head on fixed features.

Fits ``BayesianLinearHead``, compares batch ``fit`` to two-step ``RecursiveBayesianHead``
(``forgetting_factor=1``), reports posterior error vs a known weight vector, and prints
a quick Gaussian predictive-interval coverage check on held-out points.
"""

from __future__ import annotations

import argparse

import torch

from torchregress.test_time import BayesianLinearHead, RecursiveBayesianHead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    d = args.dim
    w_true = torch.randn(d)

    x_train = torch.randn(args.n_train, d)
    y_train = (x_train @ w_true).unsqueeze(-1) + args.noise * torch.randn(args.n_train, 1)
    x_test = torch.randn(args.n_test, d)
    y_test = (x_test @ w_true).unsqueeze(-1) + args.noise * torch.randn(args.n_test, 1)

    noise_var = args.noise**2
    cfg = dict(
        in_features=d,
        fit_intercept=False,
        prior_precision=1e-2,
        noise_variance=noise_var,
    )

    batch = BayesianLinearHead(**cfg).fit(x_train, y_train)
    rec = RecursiveBayesianHead(**cfg, forgetting_factor=1.0)
    mid = args.n_train // 2
    rec.partial_fit(x_train[:mid], y_train[:mid])
    rec.partial_fit(x_train[mid:], y_train[mid:])

    max_post_diff = (batch.posterior_mean - rec.posterior_mean).abs().max().item()
    w_err = (batch.posterior_mean[0] - w_true).norm().item()
    print("max |posterior_mean_batch - posterior_mean_recursive|:", round(max_post_diff, 8))
    print("||posterior_mean - w_true||_2:", round(w_err, 4))

    pred = batch.predict(x_test, return_std=True, include_noise=True)
    mean, std = pred["mean"], pred["std"]
    z = (y_test - mean) / std.clamp(min=1e-8)
    coverage_95 = ((z.abs() <= 1.96).float().mean()).item()
    print("empirical 95% Gaussian interval coverage (held-out):", round(coverage_95, 3))


if __name__ == "__main__":
    main()
