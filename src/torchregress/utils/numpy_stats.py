"""NumPy helpers shared by test-time adaptation modules."""

from __future__ import annotations

import numpy as np


def subsample_rows(
    X: np.ndarray,
    max_rows: int | None,
    *,
    random_state: int | None,
) -> np.ndarray:
    """
    Subsample rows of a NumPy array to a maximum limit.

    Parameters
    ----------
    X : np.ndarray
        The input array.
    max_rows : Optional[int]
        The maximum number of rows to return. If None, <= 0, or greater
        than the number of rows in X, the original array is returned.
    random_state : Optional[int]
        Random seed for the row selection.

    Returns
    -------
    np.ndarray
        The subsampled array with sorted indices to preserve sequence.
    """
    if max_rows is None or max_rows <= 0 or X.shape[0] <= int(max_rows):
        return X
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=int(max_rows), replace=False)
    return X[np.sort(idx.astype(np.int64, copy=False))]


def winsorize(X: np.ndarray, clip_quantile: float | None) -> np.ndarray:
    """
    Clip array values along axis 0 to specified lower and upper quantiles.

    Parameters
    ----------
    X : np.ndarray
        The input array.
    clip_quantile : Optional[float]
        The quantile threshold to clip values. Must be in [0, 0.5).
        If None or 0.0, the original array is returned.

    Returns
    -------
    np.ndarray
        The winsorized array with extreme values clipped.
    """
    if clip_quantile is None:
        return X
    q = float(clip_quantile)
    if not 0.0 <= q < 0.5:
        raise ValueError("clip_quantile must be in [0, 0.5)")
    if q == 0.0:
        return X
    lo = np.quantile(X, q, axis=0)
    hi = np.quantile(X, 1.0 - q, axis=0)
    return np.clip(X, lo, hi)
