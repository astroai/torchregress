"""Edge-case unit tests for torchregress.utils.numpy_stats.

The basic happy-path tests live in tests/test_time/test_calibration.py.
This file covers boundary conditions, dimensionality, and numerical edges.
"""

from __future__ import annotations

import numpy as np

from torchregress.utils.numpy_stats import subsample_rows, winsorize

# ── subsample_rows ────────────────────────────────────────────────────────────


def test_subsample_rows_1d_input() -> None:
    """1-D array subsampling returns a 1-D result with sorted positions."""
    X = np.arange(20, dtype=float)
    result = subsample_rows(X, 5, random_state=42)
    assert result.ndim == 1
    assert len(result) == 5
    assert np.all(np.diff(result) > 0)


def test_subsample_rows_empty_input() -> None:
    """Empty (0-row) input returns unchanged."""
    X = np.empty((0, 3))
    result = subsample_rows(X, 5, random_state=42)
    assert result.shape == (0, 3)


def test_subsample_rows_single_row() -> None:
    """max_rows >= n_rows on single-row input returns the row unchanged."""
    X = np.array([[7.0, 13.0]])
    result = subsample_rows(X, 5, random_state=42)
    np.testing.assert_array_equal(result, X)


def test_subsample_rows_max_rows_one() -> None:
    """max_rows=1 returns exactly one row."""
    X = np.arange(30, dtype=float).reshape(10, 3)
    result = subsample_rows(X, 1, random_state=42)
    assert result.shape == (1, 3)
    assert result.dtype == X.dtype


def test_subsample_rows_random_state_none() -> None:
    """random_state=None uses entropy-based seeding (still returns valid shape)."""
    X = np.arange(40, dtype=float).reshape(20, 2)
    result = subsample_rows(X, 4, random_state=None)
    assert result.shape == (4, 2)
    # Rows must be a subset (check membership)
    for row in result:
        assert any(np.array_equal(row, x_row) for x_row in X)


def test_subsample_rows_3d_input() -> None:
    """3-D input preserves trailing dimensions beyond rows."""
    X = np.arange(60, dtype=float).reshape(10, 3, 2)
    result = subsample_rows(X, 4, random_state=42)
    assert result.shape == (4, 3, 2)


# ── winsorize ─────────────────────────────────────────────────────────────────


def test_winsorize_1d_input() -> None:
    """1-D array winsorization works along the sole axis."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 10, size=100).astype(float)
    X[-5:] = 1e6  # extreme outliers
    result = winsorize(X, 0.1)
    assert result.shape == X.shape
    assert np.max(result) < 1e6
    assert np.min(result) > -1e6
    assert not np.any(np.isnan(result))


def test_winsorize_all_equal_values() -> None:
    """When all values are equal, winsorize is a no-op."""
    X = np.full((10, 3), 5.0)
    result = winsorize(X, 0.25)
    np.testing.assert_array_equal(result, X)


def test_winsorize_near_limit_quantile() -> None:
    """q=0.49 (just under the 0.5 limit) does not raise and produces output."""
    X = np.arange(20, dtype=float).reshape(5, 4)
    result = winsorize(X, 0.49)
    assert result.shape == X.shape
    # Only the very extremes should be touched
    assert not np.any(np.isnan(result))


def test_winsorize_single_element() -> None:
    """Single-element array is unchanged by winsorization."""
    X = np.array([[3.14]])
    result = winsorize(X, 0.1)
    np.testing.assert_array_equal(result, X)


def test_winsorize_never_introduces_nan() -> None:
    """Winsorize output is NaN-free for any valid input."""
    X = np.arange(100, dtype=float).reshape(25, 4)
    for q in [0.0, 0.05, 0.25, 0.499999]:
        result = winsorize(X, q)
        assert not np.any(np.isnan(result))


def test_winsorize_3d_input() -> None:
    """3-D input preserves trailing dimensions."""
    X = np.arange(60, dtype=float).reshape(10, 3, 2)
    result = winsorize(X, 0.2)
    assert result.shape == (10, 3, 2)


def test_winsorize_clip_quantile_edge() -> None:
    """q extremely small but non-zero still clips some values."""
    X = np.array([[0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [50.0, 50.0]])
    # q=0.1 → lo = 0, hi = 100 for both cols; middle row stays
    result = winsorize(X, 0.01)
    assert result.shape == X.shape
    lo = np.quantile(X, 0.01, axis=0)
    hi = np.quantile(X, 0.99, axis=0)
    assert np.all(result >= lo[None, :])
    assert np.all(result <= hi[None, :])
