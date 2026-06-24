import numpy as np
import pytest

from torchregress.test_time.calibration import RepresentationShiftInflator
from torchregress.utils.numpy_stats import subsample_rows as _subsample_rows
from torchregress.utils.numpy_stats import winsorize as _winsorize


def test_subsample_rows_returns_original_if_max_rows_none_or_invalid():
    X = np.arange(10).reshape(5, 2)
    # max_rows is None
    np.testing.assert_array_equal(_subsample_rows(X, None, random_state=42), X)
    # max_rows <= 0
    np.testing.assert_array_equal(_subsample_rows(X, 0, random_state=42), X)
    np.testing.assert_array_equal(_subsample_rows(X, -5, random_state=42), X)
    # max_rows >= X.shape[0]
    np.testing.assert_array_equal(_subsample_rows(X, 5, random_state=42), X)
    np.testing.assert_array_equal(_subsample_rows(X, 10, random_state=42), X)


def test_subsample_rows_subsamples_correctly():
    X = np.arange(20).reshape(10, 2)
    subsampled = _subsample_rows(X, 3, random_state=42)
    assert subsampled.shape == (3, 2)
    # Check that rows are actually from X
    for row in subsampled:
        assert any(np.array_equal(row, x_row) for x_row in X)


def test_subsample_rows_preserves_order():
    X = np.arange(20).reshape(10, 2)
    # The first column is strictly increasing
    subsampled = _subsample_rows(X, 4, random_state=42)
    # Check that the order is preserved (values should still be increasing)
    assert np.all(np.diff(subsampled[:, 0]) > 0)


def test_subsample_rows_random_state_reproducibility():
    X = np.arange(100).reshape(50, 2)
    out1 = _subsample_rows(X, 10, random_state=123)
    out2 = _subsample_rows(X, 10, random_state=123)
    np.testing.assert_array_equal(out1, out2)

    # Different random state should likely yield different results
    out3 = _subsample_rows(X, 10, random_state=456)
    assert not np.array_equal(out1, out3)


def test_winsorize_returns_original_if_clip_quantile_none():
    X = np.arange(10).reshape(5, 2)
    np.testing.assert_array_equal(_winsorize(X, None), X)


def test_winsorize_invalid_quantile_raises_value_error():
    X = np.arange(10).reshape(5, 2)
    with pytest.raises(ValueError, match="clip_quantile must be in \\[0, 0.5\\)"):
        _winsorize(X, -0.1)
    with pytest.raises(ValueError, match="clip_quantile must be in \\[0, 0.5\\)"):
        _winsorize(X, 0.5)
    with pytest.raises(ValueError, match="clip_quantile must be in \\[0, 0.5\\)"):
        _winsorize(X, 1.0)


def test_winsorize_quantile_zero_returns_original():
    X = np.arange(10).reshape(5, 2)
    np.testing.assert_array_equal(_winsorize(X, 0.0), X)


def test_winsorize_clips_correctly():
    # Array with clear quantiles
    # Columns are separate so we can test axis=0 behavior
    # 10 rows: 0 to 9
    X = np.arange(20, dtype=float).reshape(10, 2)
    X[:, 1] = X[:, 1] * 2  # scale second column differently

    # q=0.2 means:
    # col 0: 0,1,2...9 -> 20% is 1.8, 80% is 7.2
    # So 0, 1 -> 1.8, 8, 9 -> 7.2
    winsorized = _winsorize(X, 0.2)

    lo = np.quantile(X, 0.2, axis=0)
    hi = np.quantile(X, 0.8, axis=0)

    expected = np.clip(X, lo[None, :], hi[None, :])
    np.testing.assert_array_almost_equal(winsorized, expected)

    # Check that it actually did clip
    assert np.all(winsorized >= lo[None, :])
    assert np.all(winsorized <= hi[None, :])
    assert not np.array_equal(X, winsorized)


def test_calibrator_initialization():
    calibrator = RepresentationShiftInflator(
        base_temperature=1.5,
        slope=2.0,
        max_temperature=10.0,
        source_sample_size=100,
        random_state=42,
        clip_quantile=0.1,
        eps=1e-5,
    )
    assert calibrator.base_temperature == 1.5
    assert calibrator.slope == 2.0
    assert calibrator.max_temperature == 10.0
    assert calibrator.source_sample_size == 100
    assert calibrator.random_state == 42
    assert calibrator.clip_quantile == 0.1
    assert calibrator.eps == 1e-5
    assert calibrator.source_mean_ is None
    assert calibrator.source_var_ is None
    assert calibrator.reference_scale_ is None


def test_calibrator_fit():
    rng = np.random.default_rng(42)
    source_reps = rng.normal(loc=5.0, scale=2.0, size=(100, 3))

    calibrator = RepresentationShiftInflator()
    calibrator.fit(source_reps)

    assert calibrator.source_mean_ is not None
    assert calibrator.source_var_ is not None
    assert calibrator.reference_scale_ is not None

    assert calibrator.source_mean_.shape == (3,)
    assert calibrator.source_var_.shape == (3,)
    assert isinstance(calibrator.reference_scale_, float)

    # Check that variance is bounded by eps
    assert np.all(calibrator.source_var_ >= calibrator.eps)


def test_calibrator_methods_raise_error_before_fit():
    calibrator = RepresentationShiftInflator()
    target_reps = np.ones((10, 3))

    with pytest.raises(RuntimeError, match="call fit\\(\\) before computing shift scores"):
        calibrator.shift_scores(target_reps)

    with pytest.raises(RuntimeError, match="call fit\\(\\) before computing shift scores"):
        calibrator.temperatures(target_reps)


def test_calibrator_shift_scores():
    source_reps = np.array([[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]])
    # mean is [0, 0], var is [1, 1]

    calibrator = RepresentationShiftInflator(eps=1e-6)
    calibrator.fit(source_reps)

    target_reps = np.array(
        [
            [0.0, 0.0],  # At mean, dist = 0
            [1.0, 0.0],  # dist = sqrt(1^2 / 1 + 0^2 / 1) = 1
            [1.0, 1.0],  # dist = sqrt(1^2 / 1 + 1^2 / 1) = sqrt(2)
        ]
    )

    scores = calibrator.shift_scores(target_reps)
    assert scores.shape == (3,)
    np.testing.assert_array_almost_equal(scores, [0.0, 1.0, np.sqrt(2)])


def test_calibrator_temperatures():

    calibrator = RepresentationShiftInflator(base_temperature=1.0, max_temperature=5.0, slope=1.0)
    # Inject stats manually to easily test temperature calculation
    calibrator.source_mean_ = np.array([0.0, 0.0])
    calibrator.source_var_ = np.array([1.0, 1.0])
    calibrator.reference_scale_ = 2.0  # mock ref scale

    # Temps = 1.0 * (1.0 + 1.0 * score / 2.0)
    # if score = 0 -> temp = 1.0
    # if score = 2.0 -> temp = 2.0
    # if score = 10.0 -> temp = 6.0 (clipped to 5.0)

    target_reps = np.array(
        [
            [0.0, 0.0],  # score = 0
            [2.0, 0.0],  # score = 2
            [10.0, 0.0],  # score = 10
        ]
    )

    temps = calibrator.temperatures(target_reps)
    np.testing.assert_array_almost_equal(temps, [1.0, 2.0, 5.0])


def test_calibrator_calibrate_probabilities():
    calibrator = RepresentationShiftInflator()
    calibrator.source_mean_ = np.array([0.0])
    calibrator.source_var_ = np.array([1.0])
    calibrator.reference_scale_ = 1.0

    probs = np.array([[0.8, 0.2], [0.9, 0.1]])

    target_reps = np.array([[0.0], [0.0]])  # Should yield temperature 1.0

    calibrated_probs = calibrator.calibrate_probabilities(probs, target_reps)

    # Temperature is 1.0, probabilities should remain largely unchanged
    np.testing.assert_array_almost_equal(calibrated_probs, probs)

    # Sum to 1 check
    np.testing.assert_array_almost_equal(calibrated_probs.sum(axis=1), [1.0, 1.0])

    # With higher temperature, probabilities should become more uniform
    target_reps_far = np.array([[10.0], [10.0]])  # Will yield max temp 5.0
    calibrated_probs_far = calibrator.calibrate_probabilities(probs, target_reps_far)

    assert calibrated_probs_far[0, 0] < 0.8
    assert calibrated_probs_far[0, 1] > 0.2
    assert calibrated_probs_far[1, 0] < 0.9
    assert calibrated_probs_far[1, 1] > 0.1

    # Still sum to 1
    np.testing.assert_array_almost_equal(calibrated_probs_far.sum(axis=1), [1.0, 1.0])


def test_calibrator_calibrate_std():
    calibrator = RepresentationShiftInflator()
    calibrator.source_mean_ = np.array([0.0])
    calibrator.source_var_ = np.array([1.0])
    calibrator.reference_scale_ = 1.0

    stds = np.array([1.0, 2.0])

    target_reps = np.array([[0.0], [0.0]])  # temp = 1.0
    calibrated_stds = calibrator.calibrate_std(stds, target_reps)
    np.testing.assert_array_almost_equal(calibrated_stds, stds)

    target_reps_far = np.array([[10.0], [10.0]])  # temp = 5.0
    calibrated_stds_far = calibrator.calibrate_std(stds, target_reps_far)
    np.testing.assert_array_almost_equal(calibrated_stds_far, [5.0, 10.0])
