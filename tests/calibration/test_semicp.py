from __future__ import annotations

import math

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

    # Target weight = 1.0 joins the normalization denominator (augmented
    # distribution): denom = 5 + 1 = 6; p_i = 1/6; cumsum [1/6 .. 5/6].
    # For alpha = 0.2 (80% coverage): finite-sample level ceil(6*0.8)/6 = 5/6,
    # applied ONCE (TR-COR-06); smallest index reaching it is index 4 -> 3.5.
    q = calibrator.compute_thresholds(1.0, alpha=0.2)
    assert np.isclose(q, 3.5)

    # For alpha = 0.4 (60% coverage): level ceil(6*0.6)/6 = 4/6; smallest index
    # with cumulative probability >= 4/6 is index 3 (value 2.5). The old code
    # additionally inflated the level to ceil(3.6)/5 = 0.8 and fell back to 3.5.
    q_60 = calibrator.compute_thresholds(1.0, alpha=0.4)
    assert np.isclose(q_60, 2.5)


def test_weighted_conformal_thresholds() -> None:
    scores = np.array([0.5, 1.5, 2.5])
    weights = np.array([2.0, 1.0, 0.5])  # More weight on low scores
    calibrator = SemiConformalCalibrator()
    calibrator.fit(scores, weights)

    # Finite-sample level on the augmented distribution for n=3, alpha=0.3:
    # ceil(4*0.7)/4 = 3/4 = 0.75 (TR-COR-06: correction applied once).
    #
    # For a target point with weight = 0.5:
    # Denominator = 4.0. p = [2/4, 1/4, 0.5/4]; cumsum [0.5, 0.75, 0.875];
    # smallest index with cum >= 0.75 is index 1 (score 1.5).
    q = calibrator.compute_thresholds(0.5, alpha=0.3)
    assert np.isclose(q, 1.5)

    # Verify batch computation of thresholds
    target_weights = np.array([0.5, 2.5])
    q_batch = calibrator.compute_thresholds(target_weights, alpha=0.3)
    assert q_batch.shape == (2,)
    # For weight 2.5: denominator = 6.0; cumsum [0.333, 0.5, 0.583] never
    # reaches 0.75 -> max-score fallback; weight 0.5 picks score 1.5.
    assert np.isclose(q_batch[0], 1.5)
    assert np.isclose(q_batch[1], 2.5)


def test_weighted_path_finite_sample_level() -> None:
    """Weighted path must use the (n+1)-corrected mass, not 1 - alpha (TR-MET-12)."""
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
    weights = torch.tensor([1.0, 1.0, 1.0, 3.0])
    calibrator = SemiConformalCalibrator().fit(scores, weights)
    # sum_w_cal = 6; w_tgt = 0 -> denom = 6; augmented cum:
    # [1/6, 2/6, 3/6, 1.0]. Level = ceil(5*0.5)/5 = 3/5 = 0.6 (TR-COR-06,
    # applied once) -> smallest index with cum >= 0.6 is index 3 (score 4.0);
    # the uncorrected level 1 - alpha = 0.5 would pick index 2 (score 3.0).
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
    # For alpha = 0.5 (50% coverage): finite-sample level on the augmented
    # distribution = ceil((3+1)*0.5)/4 = 0.5 (TR-COR-06, applied once).
    # Denominator = 4; p = [0.25, 0.25, 0.25]. Cumsum: [0.25, 0.5, 0.75].
    # Index 1 (value 1.0) is the first >= 0.5. Threshold is 1.0.
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

    # Negative weights
    with pytest.raises(ValueError):
        calibrator.fit([1.0, 2.0], [-1.0, 1.0])

    with pytest.raises(ValueError):
        calibrator.compute_thresholds(-1.0)

    with pytest.raises(ValueError):
        calibrator.compute_thresholds(np.array([-1.0, 1.0]))


def test_fully_labeled_limit_matches_order_statistic() -> None:
    """w_target -> 0 reduces to the ceil((n+1)(1-alpha))-th order statistic."""
    generator = torch.Generator().manual_seed(11)
    scores = torch.randn(40, generator=generator)
    calibrator = SemiConformalCalibrator().fit(scores)
    n = scores.shape[0]
    sorted_scores = torch.sort(scores).values
    for alpha in (0.05, 0.1, 0.25):
        k = min(math.ceil((n + 1) * (1.0 - alpha)), n)
        expected = float(sorted_scores[k - 1])
        assert np.isclose(float(calibrator.compute_thresholds(0.0, alpha=alpha)), expected)
        # Tiny positive target weight exercises the general weighted path and
        # must stay on the same order statistic (correction applied once).
        tiny = float(calibrator.compute_thresholds(1e-9, alpha=alpha))
        assert abs(tiny - expected) <= 1e-6 * max(1.0, abs(expected))


def test_threshold_sandwiched_between_split_and_old_inflated() -> None:
    """Regression guard: split threshold <= corrected <= old double-inflated."""
    generator = torch.Generator().manual_seed(13)
    scores = torch.sort(torch.randn(100, generator=generator)).values
    calibrator = SemiConformalCalibrator().fit(scores)
    n = scores.shape[0]
    ranks = torch.arange(1, n + 1, dtype=torch.float64)
    for alpha in (0.05, 0.1, 0.2):
        k = min(math.ceil((n + 1) * (1.0 - alpha)), n)
        split_q = float(scores[k - 1])
        for w_tgt in (0.0, 1.0, 5.0, 20.0):
            q_new = float(calibrator.compute_thresholds(w_tgt, alpha=alpha))
            # Old double-inflated path: level k/n over masses 1/(n + w_tgt).
            reach = (ranks / (n + w_tgt)) >= min(k / n, 1.0)
            hit = torch.nonzero(reach)
            q_old = float(scores[hit[0]]) if hit.numel() > 0 else float(scores[-1])
            assert split_q <= q_new <= q_old


def test_coverage_simulation_moderate_label_rate() -> None:
    """1000-trial exchangeable simulation: coverage within nominal +/- 3 sigma."""
    generator = torch.Generator().manual_seed(2024)
    n_cal, alpha, trials = 200, 0.1, 1000
    hits = 0
    for _ in range(trials):
        cal_scores = torch.randn(n_cal, generator=generator)
        calibrator = SemiConformalCalibrator().fit(cal_scores)
        # Moderate label rate: target pseudo-weight comparable to a few
        # labeled calibration points.
        threshold = float(calibrator.compute_thresholds(2.0, alpha=alpha))
        test_score = float(torch.randn(1, generator=generator))
        hits += test_score <= threshold
    coverage = hits / trials
    sigma = math.sqrt(alpha * (1.0 - alpha) / trials)
    assert abs(coverage - (1.0 - alpha)) <= 3.0 * sigma, (
        f"coverage {coverage:.3f} outside nominal {1.0 - alpha:.3f} +/- 3 sigma"
    )
