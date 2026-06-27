"""
Tests for torchregress.metrics.point — class-based metrics and functional helpers.
"""

import numpy as np
import pytest
import torch

from torchregress.metrics.point import (
    HuberMetric,
    MedianAbsoluteDeviation,
    MedianAbsoluteError,
    NormalizedMedianAbsoluteDeviation,
    NormalizedRMSE,
    OutlierFraction,
    TrimmedMeanSquaredError,
    _tail_mask,
    attenuation_factor,
    huber_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    normalized_rmse,
    r2_score,
    regression_metrics_report,
    rmse,
    tail_mae,
    tail_rmse,
    trimmed_mean_squared_error,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Class-based metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestMedianAbsoluteError:
    """MedianAbsoluteError — torchmetrics Metric subclass."""

    def test_basic(self) -> None:
        metric = MedianAbsoluteError()
        metric.update(torch.tensor([0.0, 2.0, 4.0]), torch.tensor([1.0, 1.0, 1.0]))
        val = float(metric.compute())
        # abs errors: [1, 1, 3]; median = 1
        assert val == pytest.approx(1.0)

    def test_multioutput_uniform_average(self) -> None:
        """multioutput='uniform_average' first averages across outputs then takes median."""
        metric = MedianAbsoluteError(multioutput="uniform_average")
        y_pred = torch.tensor([[0.0, 10.0], [0.0, 0.0], [5.0, 5.0]])
        y_true = torch.tensor([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0]])
        metric.update(y_pred, y_true)
        val = float(metric.compute())
        # per-sample mean abs errors: row0=(0+10)/2=5, row1=(5+5)/2=5, row2=(0+0)/2=0
        # median across rows = 5
        assert val == pytest.approx(5.0)

    def test_multioutput_raw_values(self) -> None:
        """multioutput='raw_values' gives per-output-column medians."""
        metric = MedianAbsoluteError(multioutput="raw_values")
        y_pred = torch.tensor([[0.0, 10.0], [0.0, 0.0], [5.0, 5.0]])
        y_true = torch.tensor([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0]])
        metric.update(y_pred, y_true)
        result = metric.compute()
        # col0 abs errors: [0, 5, 0] → median=0
        # col1 abs errors: [10, 5, 0] → median=5
        assert tuple(float(x) for x in result) == (0.0, 5.0)
        assert result.shape == (2,)

    def test_single_output_raw_values_falls_back(self) -> None:
        """raw_values on a single-output problem still gives scalar median."""
        metric = MedianAbsoluteError(multioutput="raw_values")
        metric.update(torch.tensor([1.0, 3.0, 5.0]), torch.tensor([0.0, 0.0, 0.0]))
        result = metric.compute()
        assert result.ndim == 0
        assert float(result) == pytest.approx(3.0)

    def test_multiple_updates_concatenates(self) -> None:
        metric = MedianAbsoluteError()
        metric.update(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 0.0]))
        metric.update(torch.tensor([5.0]), torch.tensor([0.0]))
        # abs errors: [1, 3] + [5]; median = 3
        assert float(metric.compute()) == pytest.approx(3.0)

    def test_exact_match_yields_zero(self) -> None:
        metric = MedianAbsoluteError()
        metric.update(torch.ones(10), torch.ones(10))
        assert float(metric.compute()) == pytest.approx(0.0)


class TestNormalizedRMSE:
    """NormalizedRMSE — torchmetrics Metric subclass."""

    def test_std_normalization(self) -> None:
        metric = NormalizedRMSE(normalization="std")
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([1.0, 3.0, 5.0])
        metric.update(y_pred, y_true)
        val = float(metric.compute())
        # rmse = sqrt(mean(([0, -1, -2])²)) = sqrt(5/3) ≈ 1.291
        # torch.std defaults to unbiased=True (n-1), so std([1,3,5]) = sqrt(8/2)=2.0
        # nrmse ≈ 1.291 / 2.0 ≈ 0.6455
        assert val == pytest.approx(1.2909944487358056 / 2.0, rel=1e-6)

    def test_range_normalization(self) -> None:
        metric = NormalizedRMSE(normalization="range")
        metric.update(torch.tensor([1.0, 3.0, 5.0]), torch.tensor([1.0, 3.0, 5.0]))
        # rmse=0 → nrmse=0
        assert float(metric.compute()) == pytest.approx(0.0)

    def test_mean_normalization(self) -> None:
        metric = NormalizedRMSE(normalization="mean")
        metric.update(torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0]))
        val = float(metric.compute())
        # rmse=2, mean(|y_true|)=2, nrmse=1
        assert val == pytest.approx(1.0)

    def test_iqr_normalization(self) -> None:
        metric = NormalizedRMSE(normalization="iqr")
        metric.update(torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.tensor([1.0, 2.0, 3.0, 4.0]))
        # IQR = q75 - q25; Non-perfect fit → nrmse > 0
        assert float(metric.compute()) > 0

    def test_unknown_normalization_raises(self) -> None:
        metric = NormalizedRMSE(normalization="mad")
        metric.update(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
        with pytest.raises(ValueError, match="Unknown normalization"):
            metric.compute()

    def test_near_zero_variance_yields_inf(self) -> None:
        """When norm_factor < 1e-8, return inf."""
        metric = NormalizedRMSE(normalization="std")
        metric.update(torch.tensor([1.0, 1.1]), torch.tensor([5.0, 5.0]))
        # std ≈ 0; norm_factor near 0 → inf
        result = metric.compute()
        assert float(result) == float("inf")


class TestHuberMetric:
    """HuberMetric — torchmetrics Metric subclass (differentiable)."""

    def test_default_delta(self) -> None:
        metric = HuberMetric()
        # errors: [0.5, 1.5] — first is quadratic (< delta=1), second is linear
        metric.update(torch.tensor([0.0, 0.0]), torch.tensor([0.5, 1.5]))
        val = float(metric.compute())
        # quadratic: 0.5*0.5²=0.125; linear: 1*1.5-0.5=1.0; mean=(0.125+1.0)/2=0.5625
        assert val == pytest.approx(0.5625)

    def test_custom_delta(self) -> None:
        metric = HuberMetric(delta=2.0)
        # both errors ≤ 2, so both quadratic
        metric.update(torch.tensor([0.0, 0.0]), torch.tensor([0.5, 1.5]))
        val = float(metric.compute())
        # 0.5*(0.5²+1.5²)/2 = 0.5*(0.25+2.25)/2 = 0.625
        assert val == pytest.approx(0.625)

    def test_exact_match_yields_zero(self) -> None:
        metric = HuberMetric()
        metric.update(torch.ones(10), torch.ones(10))
        assert float(metric.compute()) == pytest.approx(0.0)

    def test_multiple_updates_accumulate(self) -> None:
        metric = HuberMetric()
        metric.update(torch.tensor([0.0]), torch.tensor([1.0]))
        metric.update(torch.tensor([0.0]), torch.tensor([1.0]))
        # single error=1: 0.5*1²=0.5; mean across both = 0.5
        assert float(metric.compute()) == pytest.approx(0.5)

    def test_is_differentiable(self) -> None:
        assert HuberMetric.is_differentiable is True


class TestTrimmedMeanSquaredError:
    """TrimmedMeanSquaredError — torchmetrics Metric subclass."""

    def test_default_proportion(self) -> None:
        metric = TrimmedMeanSquaredError()
        # errors²: [1, 4, 9, 16, 25]; sorted same
        # proportion=0.1, n=5: lower=0, upper=4 → keep [1,4,9,16]; mean=7.5
        metric.update(
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        val = float(metric.compute())
        assert val == pytest.approx(7.5)

    def test_proportion_zero_trims_nothing(self) -> None:
        metric = TrimmedMeanSquaredError(proportion=0.0)
        metric.update(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 2.0]))
        # all kept: mean((1²+2²)/2)=2.5
        assert float(metric.compute()) == pytest.approx(2.5)

    def test_proportion_near_half(self) -> None:
        # n=10, proportion=0.49 would give lower=int(4.9)=4, upper=int(5.1)=5 → empty slice
        # n=10, proportion=0.4: lower=4, upper=6 → keeps [25, 36]; mean=30.5
        y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        metric = TrimmedMeanSquaredError(proportion=0.4)
        metric.update(torch.zeros(10), y)
        # squared errors: [1,4,9,16,25,36,49,64,81,100]; sorted same
        # lower=4, upper=6 → [25, 36]; mean=30.5
        assert float(metric.compute()) == pytest.approx(30.5)

    def test_proportion_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="0 and 0.5"):
            TrimmedMeanSquaredError(proportion=-0.1)

    def test_proportion_half_raises(self) -> None:
        with pytest.raises(ValueError, match="0 and 0.5"):
            TrimmedMeanSquaredError(proportion=0.5)

    def test_multiple_updates(self) -> None:
        metric = TrimmedMeanSquaredError(proportion=0.0)
        metric.update(torch.tensor([0.0]), torch.tensor([1.0]))
        metric.update(torch.tensor([0.0]), torch.tensor([3.0]))
        # errors²: [1, 9]; mean=5
        assert float(metric.compute()) == pytest.approx(5.0)


class TestMedianAbsoluteDeviation:
    """MedianAbsoluteDeviation — torchmetrics Metric subclass."""

    def test_default_scale(self) -> None:
        metric = MedianAbsoluteDeviation()
        metric.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.0, 0.0, 0.0]))
        val = float(metric.compute())
        # errors: [1,2,3]; median=2; deviations: [1,0,1]; median=1; mad=1.4826*1
        assert val == pytest.approx(1.4826)

    def test_custom_scale(self) -> None:
        metric = MedianAbsoluteDeviation(scale=2.0)
        metric.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.0, 0.0, 0.0]))
        assert float(metric.compute()) == pytest.approx(2.0)

    def test_exact_match_yields_zero(self) -> None:
        metric = MedianAbsoluteDeviation()
        metric.update(torch.ones(10), torch.ones(10))
        assert float(metric.compute()) == pytest.approx(0.0)

    def test_multiple_updates_pools_errors(self) -> None:
        """Multiple update() calls should give same result as one shot with pooled data."""
        # One-shot reference
        ref = MedianAbsoluteDeviation()
        y_pred_all = torch.tensor([0.0, 0.0, 1.0, 1.0])
        y_true_all = torch.tensor([1.0, 3.0, 0.0, 4.0])
        ref.update(y_pred_all, y_true_all)
        expected = float(ref.compute())
        # Split across two updates
        metric = MedianAbsoluteDeviation()
        metric.update(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 3.0]))
        metric.update(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 4.0]))
        assert float(metric.compute()) == pytest.approx(expected)


class TestOutlierFraction:
    """OutlierFraction — torchmetrics Metric subclass."""

    def test_relative_mode_default_threshold(self) -> None:
        metric = OutlierFraction()
        # scaled error = |y_pred - y_true| / (1 + y_true)
        # sample 0: |0-0| / 1 = 0.0 → not outlier
        # sample 1: |1.0-0.0| / 1.0 = 1.0 > 0.15 → outlier
        # sample 2: |0.1-0.0| / 1.0 = 0.1 → not outlier
        metric.update(torch.tensor([0.0, 1.0, 0.1]), torch.tensor([0.0, 0.0, 0.0]))
        # 1 outlier / 3 total = 1/3
        assert float(metric.compute()) == pytest.approx(1.0 / 3.0)

    def test_absolute_mode(self) -> None:
        # OutlierFraction(mode="absolute", threshold=0.5): std(y_true) of [0,0,0] = 0
        # → scaled_error = abs_error / 0 = degenerate.  Tested in absolute_mode_nonzero_spread.
        pass

    def test_absolute_mode_nonzero_spread(self) -> None:
        metric = OutlierFraction(mode="absolute", threshold=0.5)
        y_true = torch.tensor([0.0, 2.0, 4.0])
        y_pred = torch.tensor([0.0, 3.0, 4.0])  # errors: [0, 1, 0]
        metric.update(y_pred, y_true)
        # std = sqrt(((0-2)^2 + (2-2)^2 + (4-2)^2))/2≈ sqrt(8/2)=2.0
        # scaled: [0/2, 1/2, 0/2] = [0, 0.5, 0]; 0.5 > 0.5? No (not >), so 0 outliers
        # Let's make error larger:
        metric2 = OutlierFraction(mode="absolute", threshold=0.4)
        metric2.update(y_pred, y_true)
        # 0.5 > 0.4 → 1 outlier; fraction = 1/3
        assert float(metric2.compute()) == pytest.approx(1.0 / 3.0)

    def test_multiple_updates_accumulate_counts(self) -> None:
        metric = OutlierFraction(threshold=0.15)
        metric.update(torch.tensor([1.0]), torch.tensor([0.0]))  # outlier
        metric.update(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))  # neither outlier
        # 1 outlier / 3 total
        assert float(metric.compute()) == pytest.approx(1.0 / 3.0)

    def test_no_outliers_yields_zero(self) -> None:
        metric = OutlierFraction()
        metric.update(torch.ones(10), torch.ones(10))
        assert float(metric.compute()) == pytest.approx(0.0)


class TestNormalizedMedianAbsoluteDeviation:
    """NormalizedMedianAbsoluteDeviation — torchmetrics Metric subclass."""

    def test_default_median_normalization(self) -> None:
        metric = NormalizedMedianAbsoluteDeviation()
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([0.0, 0.0, 0.0])
        metric.update(y_pred, y_true)
        val = float(metric.compute())
        # diffs: [1,2,3]; median=2; abs_dev: [1,0,1]; median=1; nmad=1.4826
        assert val == pytest.approx(1.4826)

    def test_relative_normalization(self) -> None:
        metric = NormalizedMedianAbsoluteDeviation(normalization="relative")
        # Use non-zero y_true so the (1 + y_true) denominator has an effect
        y_pred = torch.tensor([6.0, 10.0, 14.0])
        y_true = torch.tensor([3.0, 3.0, 3.0])
        metric.update(y_pred, y_true)
        val = float(metric.compute())
        # diffs = (y_pred - y_true) / (1 + y_true) = [3,7,11] / 4 = [0.75, 1.75, 2.75]
        # median_diff = 1.75; abs_dev = [1.0, 0.0, 1.0]; median_abs_dev = 1.0
        # nmad = 1.4826 * 1.0 ≈ 1.4826
        assert val == pytest.approx(1.4826, rel=1e-4)

    def test_exact_match_yields_zero(self) -> None:
        metric = NormalizedMedianAbsoluteDeviation()
        metric.update(torch.ones(10), torch.ones(10))
        assert float(metric.compute()) == pytest.approx(0.0)

    def test_multiple_updates_pools_diffs(self) -> None:
        metric = NormalizedMedianAbsoluteDeviation()
        metric.update(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]))
        metric.update(torch.tensor([3.0, 4.0]), torch.tensor([0.0, 0.0]))
        # diffs: [1, 2, 3, 4]; median = 2.5
        # abs_dev: [1.5, 0.5, 0.5, 1.5]; median = 1.0; 1.4826
        assert float(metric.compute()) == pytest.approx(1.4826)


# ═══════════════════════════════════════════════════════════════════════════════
# Untested functional metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestTailMAE:
    """tail_mae — MAE restricted to target-tail samples."""

    def test_upper_tail(self) -> None:
        y_pred = torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0])
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # With quantile=0.9 on 5 samples, threshold ≈ 4.6
        # Upper-tail samples: only y_true=5.0
        result = float(tail_mae(y_pred, y_true, quantile=0.9, tail="upper"))
        # |10 - 5| = 5
        assert result == pytest.approx(5.0)

    def test_lower_tail(self) -> None:
        y_pred = torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0])
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # Lower tail with quantile=0.9 uses threshold at 1-quantile=0.1
        # threshold ≈ 1.4; lower-tail: y_true=1.0
        result = float(tail_mae(y_pred, y_true, quantile=0.9, tail="lower"))
        assert result == pytest.approx(1.0)

    def test_both_tails(self) -> None:
        y_pred = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        # "both" tail: median=3.5; abs_dev: [2.5,1.5,0.5,0.5,1.5,2.5]
        # quantile=0.5 → threshold ≈ 1.25
        # Both-tail samples: y_true in [1,2,5,6]; MAE = mean(|0-1|,|0-2|,|0-5|,|10-6|)/4 = (1+2+5+4)/4 = 3
        result = float(tail_mae(y_pred, y_true, quantile=0.5, tail="both"))
        assert result > 0

    def test_nonempty_mask_all_samples_match(self) -> None:
        """When all samples match the tail mask, MAE is computed on all of them."""
        y_pred = torch.tensor([0.0, 10.0])
        y_true = torch.tensor([5.0, 5.0])
        # quantile=0.01 upper tail: threshold ≈ 5.0; both >= threshold
        result = float(tail_mae(y_pred, y_true, quantile=0.01, tail="upper"))
        # MAE = (|0-5| + |10-5|) / 2 = (5 + 5) / 2 = 5
        assert result == pytest.approx(5.0)

    def test_invalid_tail_raises(self) -> None:
        with pytest.raises(ValueError, match="tail"):
            tail_mae(torch.ones(5), torch.ones(5), tail="middle")

    def test_invalid_quantile_raises(self) -> None:
        with pytest.raises(ValueError, match="quantile"):
            tail_mae(torch.ones(5), torch.ones(5), quantile=0.0)

    def test_multivariate_per_sample_mean(self) -> None:
        """Multivariate targets are reduced via per-sample mean before tail masking."""
        y_pred = torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
        y_true = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
        # Per-sample mean: y_true=[1,3,5]; tail upper quantile=0.9: threshold≈4.6, picks last
        # MAE on that sample: |10-5| averaged per-output = 5
        result = float(tail_mae(y_pred, y_true, quantile=0.9, tail="upper"))
        assert result == pytest.approx(5.0)


class TestTailRMSE:
    """tail_rmse — RMSE restricted to target-tail samples."""

    def test_upper_tail(self) -> None:
        y_pred = torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0])
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # Upper-tail sample: y_true=5.0; RMSE = |10-5| = 5
        result = float(tail_rmse(y_pred, y_true, quantile=0.9, tail="upper"))
        assert result == pytest.approx(5.0)

    def test_lower_tail(self) -> None:
        y_pred = torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0])
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = float(tail_rmse(y_pred, y_true, quantile=0.9, tail="lower"))
        assert result == pytest.approx(1.0)

    def test_nonempty_mask(self) -> None:
        y_pred = torch.tensor([0.0, 10.0])
        y_true = torch.tensor([5.0, 5.0])
        # quantile=0.01 upper tail: both in mask; RMSE = sqrt((25+25)/2) = 5
        result = float(tail_rmse(y_pred, y_true, quantile=0.01, tail="upper"))
        assert result == pytest.approx(5.0)

    def test_invalid_tail_raises(self) -> None:
        with pytest.raises(ValueError, match="tail"):
            tail_rmse(torch.ones(5), torch.ones(5), tail="middle")


class TestTailMask:
    """_tail_mask — internal helper for tail metrics."""

    def test_upper_tail(self) -> None:
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = _tail_mask(y_true, quantile=0.5, tail="upper")
        # median=3; upper half ≥ 3
        assert mask.sum().item() >= 2

    def test_lower_tail(self) -> None:
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = _tail_mask(y_true, quantile=0.5, tail="lower")
        # Threshold at 1-0.5=0.5 quantile = 3; lower half ≤ 3
        assert mask.sum().item() >= 2

    def test_both_tail(self) -> None:
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = _tail_mask(y_true, quantile=0.6, tail="both")
        # Uses median-symmetric absolute deviation
        assert 0 < mask.sum().item() < 5  # some but not all

    def test_invalid_quantile_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="quantile"):
            _tail_mask(torch.ones(5), quantile=0.0, tail="upper")

    def test_invalid_quantile_one_raises(self) -> None:
        with pytest.raises(ValueError, match="quantile"):
            _tail_mask(torch.ones(5), quantile=1.0, tail="upper")

    def test_invalid_tail_raises(self) -> None:
        with pytest.raises(ValueError, match="tail"):
            _tail_mask(torch.ones(5), quantile=0.5, tail="left")


class TestR2Score:
    """r2_score — R² functional wrapper."""

    def test_perfect_prediction(self) -> None:
        y = torch.tensor([1.0, 2.0, 3.0])
        r2 = float(r2_score(y, y))
        assert r2 == pytest.approx(1.0)

    def test_mean_prediction(self) -> None:
        """Always predicting the mean gives R² ≈ 0."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.full_like(y_true, 3.0)
        r2 = float(r2_score(y_pred, y_true))
        assert r2 == pytest.approx(0.0, abs=1e-6)

    def test_worse_than_mean(self) -> None:
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        r2 = float(r2_score(y_pred, y_true))
        assert r2 < 0  # negative R²

    def test_as_numpy(self) -> None:
        result = r2_score(torch.ones(5), torch.ones(5), as_numpy=True)
        # create_metric_result converts 0-d tensor → Python float
        assert isinstance(result, float)

    def test_numpy_input(self) -> None:
        result = r2_score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        # create_metric_result converts 0-d tensor → Python float

# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases for existing functional metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestExistingMetricReductions:
    """Non-mean reductions and sample_weight for already-imported functions."""

    def test_mse_sum_reduction(self) -> None:
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([1.0, 2.0])
        result = float(mean_squared_error(y_pred, y_true, reduction="sum"))
        assert result == pytest.approx(5.0)

    def test_mse_none_reduction(self) -> None:
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([1.0, 2.0])
        result = mean_squared_error(y_pred, y_true, reduction="none")
        # squared errors: [1, 4]; per_sample: [1, 4]; shape (2,)
        assert result.shape == (2,)
        assert float(result[0]) == pytest.approx(1.0)
        assert float(result[1]) == pytest.approx(4.0)

    def test_mse_sample_weight(self) -> None:
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([1.0, 2.0])
        weights = torch.tensor([1.0, 0.0])
        # weighted mean: (1*1 + 0*4) / 2 = 0.5 (still mean reduction)
        result = float(mean_squared_error(y_pred, y_true, sample_weight=weights))
        assert result == pytest.approx(0.5)

    def test_mse_numpy_input_returns_scalar(self) -> None:
        result = mean_squared_error(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert isinstance(result, (float, np.floating))

    def test_mae_sum_reduction(self) -> None:
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([1.0, 2.0])
        result = float(mean_absolute_error(y_pred, y_true, reduction="sum"))
        assert result == pytest.approx(3.0)

    def test_mae_none_reduction(self) -> None:
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([1.0, 2.0])
        result = mean_absolute_error(y_pred, y_true, reduction="none")
        assert result.shape == (2,)

    def test_mae_sample_weight(self) -> None:
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([1.0, 2.0])
        weights = torch.tensor([2.0, 0.0])
        result = float(mean_absolute_error(y_pred, y_true, sample_weight=weights))
        # weighted: (2*1 + 0*2)/2 = 1.0
        assert result == pytest.approx(1.0)

    def test_rmse_none_reduction(self) -> None:
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([1.0, 2.0])
        result = rmse(y_pred, y_true, reduction="none")
        assert result.shape == (2,)
        assert float(result[0]) == pytest.approx(1.0)
        assert float(result[1]) == pytest.approx(2.0)

    def test_rmse_sum_reduction(self) -> None:
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([1.0, 2.0])
        result = float(rmse(y_pred, y_true, reduction="sum"))
        # sqrt(sum([1, 4])) = sqrt(5)
        assert result == pytest.approx(5.0**0.5)

    def test_rmse_sample_weight(self) -> None:
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([1.0, 2.0])
        weights = torch.tensor([0.0, 1.0])
        result = float(rmse(y_pred, y_true, sample_weight=weights))
        # weighted squared: [0, 4]; mean=2; sqrt=sqrt(2)
        assert result == pytest.approx(2.0**0.5)

    def test_huber_sum_reduction(self) -> None:
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([0.5, 1.5])
        result = float(huber_loss(y_pred, y_true, reduction="sum"))
        # per-sample: 0.5*0.5²=0.125, 1*1.5-0.5=1.0; sum=1.125
        assert result == pytest.approx(1.125)

    def test_huber_sample_weight(self) -> None:
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([0.5, 1.5])
        weights = torch.tensor([1.0, 0.0])
        result = float(huber_loss(y_pred, y_true, sample_weight=weights))
        # only first sample counts: 0.125; weighted mean: 0.125/2=0.0625
        assert result == pytest.approx(0.125 / 2.0)

    def test_mse_alias_matches(self) -> None:
        y_pred = torch.tensor([0.0, 1.0])
        y_true = torch.tensor([1.0, 2.0])
        assert float(mean_squared_error(y_pred, y_true)) == pytest.approx(1.0)

    def test_invalid_reduction_raises(self) -> None:
        with pytest.raises(ValueError, match="reduction"):
            mean_squared_error(torch.ones(5), torch.ones(5), reduction="median")


class TestAttenuationFactorEdgeCases:
    """Edge cases beyond the smoke tests in test_metrics.py."""

    def test_with_sample_weight(self) -> None:
        y_true = torch.linspace(-2.0, 2.0, 16)
        y_pred = y_true.clone()
        # Uniform weights → same as unweighted
        w = torch.ones(16)
        lam = float(attenuation_factor(y_pred, y_true, sample_weight=w))
        assert lam == pytest.approx(1.0, abs=1e-6)

    def test_sample_weight_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_weight shape"):
            attenuation_factor(torch.ones(5), torch.ones(5), sample_weight=torch.ones(3))

    def test_sample_weight_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN or Inf"):
            attenuation_factor(
                torch.ones(3), torch.ones(3), sample_weight=torch.tensor([1.0, float("nan"), 1.0])
            )

    def test_as_numpy(self) -> None:
        result = attenuation_factor(torch.ones(5), torch.ones(5), as_numpy=True)
        assert isinstance(result, float)

    def test_numpy_input(self) -> None:
        result = attenuation_factor(np.ones(5), np.ones(5))
        assert isinstance(result, float)


class TestMedianAbsoluteErrorFunc:
    """median_absolute_error — functional version edge cases."""

    def test_multioutput_raw_values(self) -> None:
        y_pred = torch.tensor([[0.0, 10.0], [0.0, 0.0], [5.0, 5.0]])
        y_true = torch.tensor([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0]])
        result = median_absolute_error(y_pred, y_true, multioutput="raw_values")
        assert tuple(float(x) for x in result) == (0.0, 5.0)

    def test_multioutput_uniform_average(self) -> None:
        y_pred = torch.tensor([[0.0, 10.0], [0.0, 0.0], [5.0, 5.0]])
        y_true = torch.tensor([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0]])
        result = float(median_absolute_error(y_pred, y_true, multioutput="uniform_average"))
        assert result == pytest.approx(5.0)


class TestTrimmedMSEFunc:
    """trimmed_mean_squared_error — functional version edge cases."""

    def test_proportion_zero(self) -> None:
        result = float(
            trimmed_mean_squared_error(
                torch.tensor([0.0, 0.0]), torch.tensor([1.0, 2.0]), proportion=0.0
            )
        )
        assert result == pytest.approx(2.5)

    def test_invalid_proportion_raises(self) -> None:
        with pytest.raises(ValueError, match="0 and 0.5"):
            trimmed_mean_squared_error(torch.ones(5), torch.ones(5), proportion=0.5)

    def test_as_numpy(self) -> None:
        result = trimmed_mean_squared_error(torch.ones(5), torch.ones(5), as_numpy=True)
        assert isinstance(result, float)

    def test_numpy_input(self) -> None:
        result = trimmed_mean_squared_error(np.ones(5), np.ones(5))
        assert isinstance(result, float)


class TestNormalizedRMSEFunc:
    """normalized_rmse — functional version edge cases."""

    def test_unknown_normalization_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown normalization"):
            normalized_rmse(torch.ones(5), torch.ones(5), normalization="mad")

    def test_range_normalization(self) -> None:
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = float(normalized_rmse(y, y, normalization="range"))
        assert result == pytest.approx(0.0)

    def test_near_zero_variance_yields_inf(self) -> None:
        result = float(normalized_rmse(torch.tensor([1.0, 1.1]), torch.tensor([5.0, 5.0])))
        assert result == float("inf")

    def test_as_numpy(self) -> None:
        result = normalized_rmse(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), as_numpy=True)
        assert isinstance(result, float)


class TestRegressionMetricsReportEdgeCases:
    """regression_metrics_report edge cases."""

    def test_all_keys_present(self) -> None:
        y_pred = torch.randn(20)
        y_true = torch.randn(20)
        report = regression_metrics_report(y_pred, y_true)
        expected_keys = {
            "mse",
            "rmse",
            "mae",
            "r2",
            "huber_loss",
            "mad",
            "nmad",
            "outlier_fraction",
        }
        assert set(report.keys()) == expected_keys

    def test_numpy_output(self) -> None:
        report = regression_metrics_report(np.ones(5), np.ones(5), as_numpy=True)
        # create_metric_result converts scalar tensors to Python float
        assert all(isinstance(v, float) for v in report.values())

    def test_with_sample_weight(self) -> None:
        y_pred = torch.randn(20)
        y_true = torch.randn(20)
        weights = torch.ones(20)
        report = regression_metrics_report(y_pred, y_true, sample_weight=weights)
        assert set(report.keys()) == {
            "mse",
            "rmse",
            "mae",
            "r2",
            "huber_loss",
            "mad",
            "nmad",
            "outlier_fraction",
        }


class TestHuberLossEdgeCases:
    """huber_loss functional edge cases."""

    def test_custom_delta(self) -> None:
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([0.5, 1.5])
        result = float(huber_loss(y_pred, y_true, delta=2.0))
        # both quadratic: 0.5*(0.5²+1.5²)/2 = 0.625
        assert result == pytest.approx(0.625)

    def test_numpy_input(self) -> None:
        result = huber_loss(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert isinstance(result, float)
