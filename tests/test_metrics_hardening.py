import pytest
import torch

from torchregress.metrics import (
    ContinuousRankedProbabilityScore,
    ExpectedCalibrationError,
    MedianAbsoluteDeviation,
    PredictionIntervalCoverageProbability,
    continuous_ranked_probability_score,
    expected_calibration_error,
    median_absolute_deviation,
    prediction_interval_coverage_probability,
)


def test_metrics_consistency_mad():
    """Test consistency between class and functional MAD."""
    preds = torch.randn(100)
    target = torch.randn(100)

    metric = MedianAbsoluteDeviation()
    metric.update(preds, target)
    score_class = metric.compute()

    score_func = median_absolute_deviation(preds, target)

    assert torch.isclose(score_class, torch.tensor(score_func))


def test_metrics_consistency_ece():
    """Test consistency between class and functional ECE."""
    y_true = torch.randn(100)

    # Create fake quantile predictions
    # Quantiles 0.1, 0.5, 0.9
    q10 = y_true - 1.0
    q50 = y_true
    q90 = y_true + 1.0

    y_pred_quantiles = {0.1: q10, 0.5: q50, 0.9: q90}

    metric = ExpectedCalibrationError()
    metric.update(y_pred_quantiles, y_true)
    results_class = metric.compute()

    results_func = expected_calibration_error(y_pred_quantiles, y_true)

    # Check all keys
    for key in results_class:
        assert torch.isclose(results_class[key], torch.tensor(results_func[key])), (
            f"Mismatch in {key}"
        )


def test_metrics_stateful_behavior():
    """Test standard torchmetrics state management."""
    preds1 = torch.randn(10)
    target1 = torch.randn(10)

    preds2 = torch.randn(10)
    target2 = torch.randn(10)

    metric = MedianAbsoluteDeviation()

    # Batch 1
    metric.update(preds1, target1)

    # Batch 2
    metric.update(preds2, target2)
    final_res = metric.compute()

    # Combined
    all_preds = torch.cat([preds1, preds2])
    all_target = torch.cat([target1, target2])
    expected = median_absolute_deviation(all_preds, all_target)

    assert torch.isclose(final_res, torch.tensor(expected))

    # Reset
    metric.reset()
    # After reset, state should be cleared.
    # For ConcatenationMetric (which likely MAD inherits from implicitly via Metric),
    # it might hold empty list or tensor.
    # Let's verify updating again works
    metric.update(preds1, target1)
    res_after_reset = metric.compute()
    expected_after_reset = median_absolute_deviation(preds1, target1)
    assert torch.isclose(res_after_reset, torch.tensor(expected_after_reset))


def test_metrics_input_validation():
    """Test that metrics raise errors on shape mismatch."""
    metric = MedianAbsoluteDeviation()
    preds = torch.randn(10)
    target = torch.randn(11)  # Mismatch

    with pytest.raises(ValueError):
        metric.update(preds, target)


def test_interval_coverage_consistency():
    """Test prediction interval coverage."""
    # Lower/Upper bounds
    lower = torch.randn(100)
    upper = lower + torch.abs(torch.randn(100))  # Ensure upper > lower
    target = (lower + upper) / 2  # Targets in middle

    metric = PredictionIntervalCoverageProbability()
    metric.update(lower, upper, target)
    score = metric.compute()

    # All targets are inside, so coverage should be 1.0
    assert torch.isclose(score, torch.tensor(1.0))

    # Functional
    score_func = prediction_interval_coverage_probability(lower, upper, target)
    assert torch.isclose(score, torch.tensor(score_func))


def test_crps_consistency():
    """Test CRPS consistency."""
    y_true = torch.randn(100)

    # Create fake quantile predictions
    q10 = y_true - 1.0
    q50 = y_true
    q90 = y_true + 1.0

    y_pred_quantiles = {0.1: q10, 0.5: q50, 0.9: q90}

    metric = ContinuousRankedProbabilityScore()
    metric.update(y_pred_quantiles, y_true)
    score = metric.compute()

    score_func = continuous_ranked_probability_score(y_pred_quantiles, y_true)
    assert torch.isclose(score, torch.tensor(float(score_func)))
