import numpy as np
import pytest

from torchregress.test_time.subspace import (
    FeatureStatNormalizer,
    SignificantSubspaceAligner,
    _clip_scale_ratio,
    _feature_significance,
)
from torchregress.utils.numpy_stats import subsample_rows as _subsample_rows
from torchregress.utils.numpy_stats import winsorize as _winsorize


def test_subsample_rows():
    X = np.arange(10).reshape(5, 2)
    # No max_rows
    assert np.array_equal(_subsample_rows(X, None, random_state=42), X)
    # max_rows <= 0
    assert np.array_equal(_subsample_rows(X, 0, random_state=42), X)
    assert np.array_equal(_subsample_rows(X, -1, random_state=42), X)
    # max_rows >= X.shape[0]
    assert np.array_equal(_subsample_rows(X, 5, random_state=42), X)
    assert np.array_equal(_subsample_rows(X, 10, random_state=42), X)

    # max_rows < X.shape[0]
    subsampled = _subsample_rows(X, 3, random_state=42)
    assert subsampled.shape == (3, 2)
    # Verify rows are from original X and keep order
    is_in_X = [any(np.array_equal(row, x_row) for x_row in X) for row in subsampled]
    assert all(is_in_X)
    # Because of sort, values should be increasing
    assert np.all(np.diff(subsampled[:, 0]) > 0)


def test_winsorize():
    X = np.arange(100, dtype=float).reshape(100, 1)

    # None or 0
    assert np.array_equal(_winsorize(X, None), X)
    assert np.array_equal(_winsorize(X, 0.0), X)

    # Valid clip
    winsorized = _winsorize(X, 0.1)
    assert winsorized.shape == (100, 1)
    assert np.min(winsorized) == pytest.approx(9.9)
    assert np.max(winsorized) == pytest.approx(89.1)

    # Invalid clip
    with pytest.raises(ValueError, match="clip_quantile must be in \\[0, 0.5\\)"):
        _winsorize(X, 0.5)
    with pytest.raises(ValueError, match="clip_quantile must be in \\[0, 0.5\\)"):
        _winsorize(X, -0.1)


def test_feature_significance():
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])

    # Without y (returns std dev)
    sig = _feature_significance(X, None, 1e-6)
    assert np.allclose(sig, [1.11803399, 2.23606798])

    # With y (perfectly correlated with col 0, inversely correlated with col 1)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    # x_centered = [[-1.5, -3], [-0.5, -1], [0.5, 1], [1.5, 3]]
    # y_centered = [-1.5, -0.5, 0.5, 1.5]
    sig_with_y = _feature_significance(X, y, 1e-6)
    # Absolute correlation is 1 for both cols
    assert np.allclose(sig_with_y, [1.0, 1.0])

    # Zero variance feature should be clipped to eps
    X_zero_var = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0], [1.0, 8.0]])
    sig_zero = _feature_significance(X_zero_var, y, 1e-2)
    assert sig_zero[0] == pytest.approx(1e-2)


def test_clip_scale_ratio():
    ratio = np.array([0.05, 0.5, 1.0, 2.0, 20.0])

    assert np.array_equal(_clip_scale_ratio(ratio, None), ratio)

    clipped = _clip_scale_ratio(ratio, 10.0)
    assert np.allclose(clipped, [0.1, 0.5, 1.0, 2.0, 10.0])

    with pytest.raises(ValueError, match="max_scale_ratio must be greater than 1"):
        _clip_scale_ratio(ratio, 1.0)
    with pytest.raises(ValueError, match="max_scale_ratio must be greater than 1"):
        _clip_scale_ratio(ratio, 0.5)


def test_feature_stat_normalizer():
    X_source = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    X_target = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]])

    # Fit
    norm = FeatureStatNormalizer(eps=1e-6)
    with pytest.raises(RuntimeError, match="call fit\\(\\) before transform\\(\\)"):
        norm.transform(X_target)

    norm.fit(X_source)
    assert np.allclose(norm.source_mean_, [2.5, 25.0])
    assert np.allclose(norm.source_std_, [1.11803399, 11.18033989])

    # Transform
    # Target mean: [25.0, 250.0], Target std: [11.18033989, 111.80339887]
    # Scale ratio will be [0.1, 0.1]
    # (X_target - tgt_mean) * scale_ratio + src_mean
    # First row: ([10, 100] - [25, 250]) * 0.1 + [2.5, 25] = [-1.5, -15] + [2.5, 25] = [1.0, 10.0]
    transformed = norm.transform(X_target)
    assert np.allclose(transformed, X_source)


def test_significant_subspace_aligner():
    rng = np.random.default_rng(42)
    # Create simple 2D source and shifted/scaled target
    X_source = rng.normal(loc=[0, 0], scale=[1, 2], size=(100, 2))
    y_source = X_source[:, 0] * 2 + X_source[:, 1] * 0.5 + rng.normal(0, 0.1, size=100)

    # Target is shifted and scaled
    X_target = X_source * np.array([2.0, 0.5]) + np.array([10.0, -5.0])

    # Errors on invalid inputs
    aligner = SignificantSubspaceAligner()
    with pytest.raises(ValueError, match="X_source must be 2D"):
        aligner.fit(np.zeros(10))
    with pytest.raises(ValueError, match="y_source must match X_source rows"):
        aligner.fit(X_source, np.zeros(10))
    with pytest.raises(RuntimeError, match="call fit\\(\\) before transform\\(\\)"):
        aligner.transform(X_target)

    # Fit and transform with y
    transformed_with_y = aligner.fit_transform(X_source, X_target, y_source)
    assert transformed_with_y.shape == X_target.shape

    # State validation
    assert aligner.state_ is not None
    assert aligner.state_.rank >= 1

    # Fit without y
    aligner_no_y = SignificantSubspaceAligner()
    transformed_no_y = aligner_no_y.fit_transform(X_source, X_target)
    assert transformed_no_y.shape == X_target.shape

    # Explicit rank
    aligner_rank_1 = SignificantSubspaceAligner(rank=1)
    aligner_rank_1.fit(X_source, y_source)
    assert aligner_rank_1.state_.rank == 1
