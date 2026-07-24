"""
Thin SupportsPredictiveBatch adapter over BayesianLinearHead.

Demonstrates how a small wrapper can satisfy
`torchregress.test_time.SupportsPredictiveBatch` while reusing BLR predictive
moments for adaptation pipelines that consume `PredictiveBatch`.
"""

from __future__ import annotations

import argparse

import numpy as np

from torchregress.prediction import PredictiveBatch
from torchregress.test_time import BayesianLinearHead


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=96)
    p.add_argument("--n-test", type=int, default=24)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    x_train, y_train = _make_toy(args.n_train, args.dim, args.seed)
    x_test, y_test = _make_toy(args.n_test, args.dim, args.seed + 1)

    head = BayesianLinearHead(
        in_features=args.dim,
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


if __name__ == "__main__":
    main()
