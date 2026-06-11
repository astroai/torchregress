"""NumPy helpers shared by test-time adaptation modules."""

from __future__ import annotations

import numpy as np


def subsample_rows(
    X: np.ndarray,
    max_rows: int | None,
    *,
    random_state: int | None,
) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or X.shape[0] <= int(max_rows):
        return X
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=int(max_rows), replace=False)
    return X[np.sort(idx.astype(np.int64, copy=False))]


def winsorize(X: np.ndarray, clip_quantile: float | None) -> np.ndarray:
    if clip_quantile is None:
        return X
    q = float(clip_quantile)
    if not 0.0 <= q < 0.5:
        raise ValueError("clip_quantile must be in [0, 0.5)")
    if q == 0.0:
        return X
    lo = np.quantile(X, q, axis=0)
    hi = np.quantile(X, 1.0 - q, axis=0)
    return np.clip(X, lo[None, :], hi[None, :])
