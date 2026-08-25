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
    # Finite-sample level q_adj = ceil((5+1)*0.8)/5 = 1.0 (TR-MET-12).
    # Denominator = 5 + 1 = 6; p_i = 1/6.
    # Cumsum: [1/6, 2/6, 3/6, 4/6, 5/6] — never reaches 1.0,
    # so the threshold falls back to the largest score (3.5).
    q = calibrator.compute_thresholds(1.0, alpha=0.2)
    assert np.isclose(q, 3.5)

    # For alpha = 0.4 (60% coverage): q_adj = ceil(6*0.6)/5 = 0.8.
    # Smallest index with cumulative probability >= 0.8 is index 4 (value 3.5);
    # the old uncorrected level 1 - alpha = 0.6 picked index 3 (2.5), which
    # under-covered (TR-MET-12).
    q_60 = calibrator.compute_thresholds(1.0, alpha=0.4)
    assert np.isclose(q_60, 3.5)


def test_weighted_conformal_thresholds() -> None:
    scores = np.array([0.5, 1.5, 2.5])
    weights = np.array([2.0, 1.0, 0.5])  # More weight on low scores
    calibrator = SemiConformalCalibrator()
    calibrator.fit(scores, weights)

    # Finite-sample level for n=3, alpha=0.3: q_adj = ceil(4*0.7)/3 = 1.0
    # (TR-MET-12); with the target pseudo-weight the cumulative curve never
    # reaches it, so thresholds fall back to the largest score.
    #
    # For a target point with weight = 0.5:
    # Denominator = 4.0. p = [2/4, 1/4, 0.5/4]; cumsum [0.5, 0.75, 0.875].
    q = calibrator.compute_thresholds(0.5, alpha=0.3)
    assert np.isclose(q, 2.5)

    # Verify batch computation of thresholds
    target_weights = np.array([0.5, 2.5])
    q_batch = calibrator.compute_thresholds(target_weights, alpha=0.3)
    assert q_batch.shape == (2,)
    # For weight 2.5: Denominator = 6.0; cumsum [0.333, 0.5, 0.583].
    # Neither target weight's curve reaches q_adj = 1.0 -> max score for both.
    assert np.isclose(q_batch[0], 2.5)
    assert np.isclose(q_batch[1], 2.5)


def test_weighted_path_finite_sample_level() -> None:
    """Weighted path must use the (n+1)-corrected mass, not 1 - alpha (TR-MET-12)."""
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
    weights = torch.tensor([1.0, 1.0, 1.0, 3.0])
    calibrator = SemiConformalCalibrator().fit(scores, weights)
    # sum_w_cal = 6; w_tgt = 0 -> denom = 6; normalized cum:
    # [1/6, 2/6, 3/6, 1.0]. q_adj = ceil(5*0.5)/4 = 3/4 = 0.75
    # -> smallest index with cum >= 0.75 is index 3 (score 4.0).
    # The uncorrected level 1 - alpha = 0.5 would pick index 2 (score 3.0),
    # which has no finite-sample guarantee.
    q = calibrator.compute_thresholds(0.0, alpha=0.5)
    assert np.isclose(q, 4.0)


def test_calibrate_interval_numpy_and_torch() -> None:
    scores = np.array([0.5, 1.0, 1.5])
    calibrator = SemiConformalCalibrator()
    calibrator.fit(scores)

    pred_lower = np.array([10.0, 20.0])
    pred_upper = np.array([12.0, 22.0])
    weights = np.array([1.0, 1.0])

    # Uncalibrated: [10, 12] and [20, 22]
    # For alpha = 0.5 (50% coverage): finite-sample level
    # q_adj = ceil((3+1)*0.5)/3 = 2/3 (TR-MET-12). Denominator = 4;
    # p = [0.25, 0.25, 0.25]. Cumsum: [0.25, 0.5, 0.75].
    # Index 2 (value 1.5) is the first >= 2/3 (the uncorrected level 0.5
    # picked index 1 = 1.0). Threshold is 1.5.
    lower_cal, upper_cal = calibrator.calibrate_interval(pred_lower, pred_upper, weights, alpha=0.5)
    assert np.allclose(lower_cal, [8.5, 18.5])
    assert np.allclose(upper_cal, [13.5, 23.5])

    # Test with torch tensors
    pred_lower_t = torch.tensor([10.0, 20.0], dtype=torch.float32)
    pred_upper_t = torch.tensor([12.0, 22.0], dtype=torch.float32)
    weights_t = torch.tensor([1.0, 1.0], dtype=torch.float32)

    lower_t, upper_t = calibrator.calibrate_interval(
        pred_lower_t, pred_upper_t, weights_t, alpha=0.5
    )
    assert isinstance(lower_t, torch.Tensor)
    assert isinstance(upper_t, torch.Tensor)
    assert torch.allclose(lower_t, torch.tensor([8.5, 18.5]))
    assert torch.allclose(upper_t, torch.tensor([13.5, 23.5]))


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

    # Negative weights
    with pytest.raises(ValueError):
        calibrator.fit([1.0, 2.0], [-1.0, 1.0])

    with pytest.raises(ValueError):
        calibrator.compute_thresholds(-1.0)

    with pytest.raises(ValueError):
        calibrator.compute_thresholds(np.array([-1.0, 1.0]))
