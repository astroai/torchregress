from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.calibration.semicp import SemiConformalCalibrator


def test_fit_and_unweighted_thresholds() -> None:
    scores = np.array([1.5, 0.5, 2.5, 3.5, 1.0])
    calibrator = SemiConformalCalibrator()
    calibrator.fit(scores)

    # Sorted scores should be: [0.5, 1.0, 1.5, 2.5, 3.5]
    assert np.allclose(calibrator.scores_cal_, [0.5, 1.0, 1.5, 2.5, 3.5])
    assert np.allclose(calibrator.weights_cal_, [1.0, 1.0, 1.0, 1.0, 1.0])

    # Unweighted: target weight = 1.0. For alpha = 0.2 (80% coverage):
    # Denominator = 5 + 1 = 6.
    # Cumulative probability sum >= 1 - 0.2 = 0.8.
    # p_i = 1/6. Cumsum: [1/6, 2/6, 3/6, 4/6, 5/6] -> [0.166, 0.333, 0.5, 0.667, 0.833]
    # Smallest index with cumsum >= 0.8 is index 4 (value 3.5).
    q = calibrator.compute_thresholds(1.0, alpha=0.2)
    assert np.isclose(q, 3.5)

    # For alpha = 0.4 (60% coverage):
    # Cumsum >= 0.6 is index 3 (value 2.5).
    q_60 = calibrator.compute_thresholds(1.0, alpha=0.4)
    assert np.isclose(q_60, 2.5)


def test_weighted_conformal_thresholds() -> None:
    scores = np.array([0.5, 1.5, 2.5])
    weights = np.array([2.0, 1.0, 0.5])  # More weight on low scores
    calibrator = SemiConformalCalibrator()
    calibrator.fit(scores, weights)

    # Total cal weights = 3.5
    # For a target point with weight = 0.5 and alpha = 0.3 (70% coverage):
    # Denominator = 4.0.
    # p = [2/4, 1/4, 0.5/4] = [0.5, 0.25, 0.125].
    # Cumsum: [0.5, 0.75, 0.875].
    # Smallest index with cumsum >= 0.7 is index 1 (value 1.5).
    q = calibrator.compute_thresholds(0.5, alpha=0.3)
    assert np.isclose(q, 1.5)

    # Verify batch computation of thresholds
    target_weights = np.array([0.5, 2.5])
    q_batch = calibrator.compute_thresholds(target_weights, alpha=0.3)
    assert q_batch.shape == (2,)
    # For weight 2.5: Denominator = 6.0.
    # p = [2/6, 1/6, 0.5/6] = [0.333, 0.167, 0.083].
    # Cumsum: [0.333, 0.5, 0.583]. None is >= 0.7, fallback to max score (2.5)
    assert np.isclose(q_batch[0], 1.5)
    assert np.isclose(q_batch[1], 2.5)


def test_calibrate_interval_numpy_and_torch() -> None:
    scores = np.array([0.5, 1.0, 1.5])
    calibrator = SemiConformalCalibrator()
    calibrator.fit(scores)

    pred_lower = np.array([10.0, 20.0])
    pred_upper = np.array([12.0, 22.0])
    weights = np.array([1.0, 1.0])

    # Uncalibrated: [10, 12] and [20, 22]
    # For alpha = 0.5 (50% coverage). Denominator = 4.
    # p = [0.25, 0.25, 0.25]. Cumsum: [0.25, 0.5, 0.75].
    # Index 1 (value 1.0) is the first >= 0.5. So threshold is 1.0.
    lower_cal, upper_cal = calibrator.calibrate_interval(pred_lower, pred_upper, weights, alpha=0.5)
    assert np.allclose(lower_cal, [9.0, 19.0])
    assert np.allclose(upper_cal, [13.0, 23.0])

    # Test with torch tensors
    pred_lower_t = torch.tensor([10.0, 20.0], dtype=torch.float32)
    pred_upper_t = torch.tensor([12.0, 22.0], dtype=torch.float32)
    weights_t = torch.tensor([1.0, 1.0], dtype=torch.float32)

    lower_t, upper_t = calibrator.calibrate_interval(
        pred_lower_t, pred_upper_t, weights_t, alpha=0.5
    )
    assert isinstance(lower_t, torch.Tensor)
    assert isinstance(upper_t, torch.Tensor)
    assert torch.allclose(lower_t, torch.tensor([9.0, 19.0]))
    assert torch.allclose(upper_t, torch.tensor([13.0, 23.0]))


def test_invalid_parameters() -> None:
    calibrator = SemiConformalCalibrator()

    # Call before fit
    with pytest.raises(RuntimeError):
        calibrator.compute_thresholds(1.0)

    # Mismatched shapes
    with pytest.raises(ValueError):
        calibrator.fit([1.0, 2.0], [1.0])

    # Invalid alpha
    calibrator.fit([1.0, 2.0])
    with pytest.raises(ValueError):
        calibrator.compute_thresholds(1.0, alpha=0.0)
    with pytest.raises(ValueError):
        calibrator.compute_thresholds(1.0, alpha=1.0)
