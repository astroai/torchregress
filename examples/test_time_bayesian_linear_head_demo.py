"""
Conjugate Bayesian linear head on fixed features: exactness check and predictive adapter.

Modes (``--mode``):

``exactness``
    Fits ``BayesianLinearHead``, compares batch ``fit`` to two-step
    ``RecursiveBayesianHead`` (``forgetting_factor=1``), reports posterior error vs a
    known weight vector, and prints a Gaussian predictive-interval coverage check on
    held-out points. Defaults: ``--n-train 200 --n-test 500 --dim 5 --noise 0.35``.

``adapter``
    A thin wrapper over ``BayesianLinearHead`` that satisfies
    ``torchregress.test_time.SupportsPredictiveBatch``, reusing BLR predictive moments
    for adaptation pipelines that consume ``PredictiveBatch``.
    Defaults: ``--n-train 96 --n-test 24 --dim 4``.

``all`` (default)
    Runs ``exactness`` then ``adapter``, each with its own defaults unless a size flag
    is given explicitly (then it applies to both). Each mode seeds itself.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from torchregress.prediction import PredictiveBatch
from torchregress.test_time import BayesianLinearHead, RecursiveBayesianHead

MODES = ("exactness", "adapter")
MODE_DEFAULTS: dict[str, dict[str, int]] = {
    "exactness": {"n_train": 200, "n_test": 500, "dim": 5},
    "adapter": {"n_train": 96, "n_test": 24, "dim": 4},
}


def run_exactness(*, n_train: int, n_test: int, dim: int, noise: float, seed: int) -> None:
    """Batch vs recursive posterior agreement, weight error and held-out coverage."""
    torch.manual_seed(seed)
    d = dim
    w_true = torch.randn(d)

    x_train = torch.randn(n_train, d)
    y_train = (x_train @ w_true).unsqueeze(-1) + noise * torch.randn(n_train, 1)
    x_test = torch.randn(n_test, d)
    y_test = (x_test @ w_true).unsqueeze(-1) + noise * torch.randn(n_test, 1)

    noise_var = noise**2
    cfg = dict(
        in_features=d,
        fit_intercept=False,
        prior_precision=1e-2,
        noise_variance=noise_var,
    )

    batch = BayesianLinearHead(**cfg).fit(x_train, y_train)
    rec = RecursiveBayesianHead(**cfg, forgetting_factor=1.0)
    mid = n_train // 2
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


class BLRPredictiveAdapter:
    """Minimal adapter implementing SupportsPredictiveBatch."""

    def __init__(self, head: BayesianLinearHead) -> None:
        self.head = head

    def predict_distribution(self, X: np.ndarray, **kwargs: object) -> PredictiveBatch:
        include_noise = bool(kwargs.get("include_noise", True))
        return self.head.predictive_batch(X, include_noise=include_noise)


def _make_toy(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d, 1)).astype(np.float32)
    y = x @ w + 0.15 * rng.normal(size=(n, 1)).astype(np.float32)
    return x, y


def run_adapter(*, n_train: int, n_test: int, dim: int, seed: int) -> None:
    """Fit a BLR head and query it through the PredictiveBatch adapter."""
    x_train, y_train = _make_toy(n_train, dim, seed)
    x_test, y_test = _make_toy(n_test, dim, seed + 1)

    head = BayesianLinearHead(
        in_features=dim,
        out_features=1,
        fit_intercept=True,
        prior_precision=1.0,
        noise_variance=0.2**2,
    ).fit(x_train, y_train)

    adapter = BLRPredictiveAdapter(head)

    pb = adapter.predict_distribution(x_test, include_noise=True)
    rmse = float(np.sqrt(np.mean((pb.mean.detach().cpu().numpy() - y_test) ** 2)))
    mean_std = float(pb.std.detach().mean().item())
    print(f"BLR adapter RMSE={rmse:.4f}, mean predictive std={mean_std:.4f}")
    print("PredictiveBatch extra keys:", sorted(pb.extra.keys()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--n-train", type=int, default=None, help="default depends on mode")
    parser.add_argument("--n-test", type=int, default=None, help="default depends on mode")
    parser.add_argument("--dim", type=int, default=None, help="default depends on mode")
    parser.add_argument("--noise", type=float, default=0.35, help="exactness mode noise std")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    modes = MODES if args.mode == "all" else (args.mode,)
    for mode in modes:
        if len(modes) > 1:
            print(f"=== mode: {mode} ===")
        sizes = {
            key: getattr(args, key) if getattr(args, key) is not None else default
            for key, default in MODE_DEFAULTS[mode].items()
        }
        if mode == "exactness":
            run_exactness(**sizes, noise=args.noise, seed=args.seed)
        else:
            run_adapter(**sizes, seed=args.seed)


if __name__ == "__main__":
    main()
