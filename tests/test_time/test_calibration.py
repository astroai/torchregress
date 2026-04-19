import numpy as np
import pytest

from torchregress.test_time.calibration import (
    RepresentationShiftCalibrator,
    _subsample_rows,
    _winsorize,
)


def test_subsample_rows():
    X = np.arange(10).reshape(5, 2)

    # max_rows is None
    out = _subsample_rows(X, None, random_state=42)
    np.testing.assert_array_equal(out, X)

    # max_rows <= 0
    out = _subsample_rows(X, 0, random_state=42)
    np.testing.assert_array_equal(out, X)

    # max_rows >= X.shape[0]
    out = _subsample_rows(X, 5, random_state=42)
    np.testing.assert_array_equal(out, X)
    out = _subsample_rows(X, 10, random_state=42)
    np.testing.assert_array_equal(out, X)

    # max_rows < X.shape[0]
    out1 = _subsample_rows(X, 3, random_state=42)
    assert out1.shape == (3, 2)
    # Check rows are a subset and sorted in original order
    for row in out1:
        assert any(np.array_equal(row, x_row) for x_row in X)

    # Check reproducibility
    out2 = _subsample_rows(X, 3, random_state=42)
    np.testing.assert_array_equal(out1, out2)


def test_winsorize():
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
    ])

    # clip_quantile is None
    out = _winsorize(X, None)
    np.testing.assert_array_equal(out, X)

    # clip_quantile is 0.0
    out = _winsorize(X, 0.0)
    np.testing.assert_array_equal(out, X)

    # invalid clip_quantile
    with pytest.raises(ValueError, match="clip_quantile must be in"):
        _winsorize(X, -0.1)
    with pytest.raises(ValueError, match="clip_quantile must be in"):
        _winsorize(X, 0.5)

    # clip_quantile is 0.2
    out = _winsorize(X, 0.2)
    expected = np.array([
        [1.8, 18.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [4.2, 42.0],
    ])
    np.testing.assert_allclose(out, expected)


def test_representation_shift_calibrator_fit():
    calibrator = RepresentationShiftCalibrator(random_state=42, clip_quantile=0.1)
    X = np.random.RandomState(0).randn(100, 5)

    # check not fitted error
    with pytest.raises(RuntimeError, match="call fit"):
        calibrator._squared_mahalanobis(X)

    calibrator.fit(X)
    assert calibrator.source_mean_ is not None
    assert calibrator.source_var_ is not None
    assert calibrator.reference_scale_ is not None

    assert calibrator.source_mean_.shape == (5,)
    assert calibrator.source_var_.shape == (5,)
    assert isinstance(calibrator.reference_scale_, float)


def test_representation_shift_calibrator_shift_scores():
    calibrator = RepresentationShiftCalibrator(random_state=42)
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, -2.0]
    ])
    calibrator.fit(X)

    scores = calibrator.shift_scores(X)
    assert scores.shape == (3,)
    assert np.all(scores >= 0)


def test_representation_shift_calibrator_temperatures():
    calibrator = RepresentationShiftCalibrator(
        base_temperature=1.0,
        slope=2.0,
        max_temperature=4.0,
        random_state=42
    )
    X = np.random.RandomState(0).randn(100, 2)
    calibrator.fit(X)

    target = np.array([
        [0.0, 0.0],     # Close to mean
        [10.0, 10.0],   # Far from mean
        [100.0, 100.0], # Very far from mean
    ])

    temps = calibrator.temperatures(target)
    assert temps.shape == (3,)
    assert temps[0] >= 1.0
    assert temps[1] > temps[0]
    assert temps[2] == 4.0  # Should be clipped to max_temperature


def test_representation_shift_calibrator_calibrate_probabilities():
    calibrator = RepresentationShiftCalibrator(random_state=42)
    X = np.random.RandomState(0).randn(100, 2)
    calibrator.fit(X)

    target = np.array([
        [0.0, 0.0],
        [10.0, 10.0],
    ])
    probs = np.array([
        [0.1, 0.9],
        [0.1, 0.9],
    ])

    calibrated_probs = calibrator.calibrate_probabilities(probs, target)
    assert calibrated_probs.shape == (2, 2)
    np.testing.assert_allclose(calibrated_probs.sum(axis=1), 1.0)

    # The one further away should have a higher temperature, making probabilities closer to uniform
    assert calibrated_probs[1, 0] > calibrated_probs[0, 0]
    assert calibrated_probs[1, 1] < calibrated_probs[0, 1]


def test_representation_shift_calibrator_calibrate_std():
    calibrator = RepresentationShiftCalibrator(random_state=42)
    X = np.random.RandomState(0).randn(100, 2)
    calibrator.fit(X)

    target = np.array([
        [0.0, 0.0],
        [10.0, 10.0],
    ])
    stds = np.array([1.0, 1.0])

    calibrated_stds = calibrator.calibrate_std(stds, target)
    assert calibrated_stds.shape == (2,)
    # Further away should have a larger std
    assert calibrated_stds[1] > calibrated_stds[0]
