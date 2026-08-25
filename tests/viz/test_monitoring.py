"""
Unit tests for torchregress.viz.monitoring.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from torchregress.viz.monitoring import (
    _add_early_stopping_annotations,
    _create_lr_plot,
    _filter_lr_find_data,
    _find_early_stopping_point,
    _format_metric_value,
    _plot_early_stopping_markers,
    _plot_single_metric,
    _smooth_losses,
    _suggest_learning_rate,
    plot_early_stopping,
    plot_learning_curves,
    plot_lr_find_results,
    plot_validation_metrics,
)

# ═══════════════════════════════════════════════════════════════════════════════
# _format_metric_value
# ═══════════════════════════════════════════════════════════════════════════════


class TestFormatMetricValue:
    def test_scientific_large(self) -> None:
        """Large values use scientific notation when enabled."""
        result = _format_metric_value(12345.0, scientific_notation=True)
        assert "e" in result

    def test_scientific_small(self) -> None:
        """Tiny values use scientific notation when enabled."""
        result = _format_metric_value(0.00005, scientific_notation=True)
        assert "e" in result

    def test_no_scientific_large(self) -> None:
        """scientific_notation=False formats normally."""
        result = _format_metric_value(12345.0, scientific_notation=False)
        assert "e" not in result

    def test_no_scientific_small(self) -> None:
        """scientific_notation=False formats tiny values normally."""
        result = _format_metric_value(0.00005, scientific_notation=False)
        assert "e" not in result

    def test_very_small(self) -> None:
        """Very small positive value returns 4 decimal places."""
        result = _format_metric_value(0.005, scientific_notation=False)
        assert result == "0.0050"

    def test_medium_small(self) -> None:
        """0.05 => 3 decimal places (0.05 < 0.1 triggers .3f)."""
        result = _format_metric_value(0.05, scientific_notation=False)
        assert result == "0.050"

    def test_medium_large(self) -> None:
        """0.5 => 2 decimal places (0.5 < 1 triggers .2f)."""
        result = _format_metric_value(0.5, scientific_notation=False)
        assert result == "0.50"

    def test_integer_less_than_10(self) -> None:
        """5.0 < 10 => always .1f format, returns '5.0'."""
        result = _format_metric_value(5.0, scientific_notation=False)
        assert result == "5.0"

    def test_non_integer_less_than_10(self) -> None:
        """Non-integer < 10 => 1 decimal place."""
        result = _format_metric_value(5.3, scientific_notation=False)
        assert result == "5.3"


# ═══════════════════════════════════════════════════════════════════════════════
# _find_early_stopping_point
# ═══════════════════════════════════════════════════════════════════════════════


class TestFindEarlyStoppingPoint:
    def test_no_stop_no_improvement(self) -> None:
        """When losses never improve, best is first epoch, stops at end."""
        losses = [10.0, 10.5, 11.0, 12.0]
        best_epoch, best_loss, stop_epoch = _find_early_stopping_point(
            losses, patience=3, delta=0.0
        )
        assert best_epoch == 1
        assert best_loss == 10.0
        assert stop_epoch == 4

    def test_improvement_resets_counter(self) -> None:
        """Improvement resets the patience counter."""
        losses = [10.0, 9.0, 8.0, 7.0]
        best_epoch, best_loss, stop_epoch = _find_early_stopping_point(
            losses, patience=2, delta=0.0
        )
        assert best_epoch == 4
        assert best_loss == 7.0
        assert stop_epoch == 4

    def test_early_stop_triggered(self) -> None:
        """Patience exceeded triggers early stop."""
        losses = [10.0, 9.0, 9.5, 10.0, 10.5]
        best_epoch, best_loss, stop_epoch = _find_early_stopping_point(
            losses, patience=2, delta=0.0
        )
        assert best_epoch == 2
        assert best_loss == 9.0
        assert stop_epoch == 4

    def test_delta_margin(self) -> None:
        """Small improvements < delta don't count."""
        losses = [10.0, 9.99, 9.98, 9.97, 10.0]
        best_epoch, best_loss, stop_epoch = _find_early_stopping_point(
            losses, patience=2, delta=0.1
        )
        assert best_epoch == 1
        assert best_loss == 10.0
        assert stop_epoch == 3

    def test_empty_list(self) -> None:
        """Empty loss list returns zeros."""
        best_epoch, best_loss, stop_epoch = _find_early_stopping_point([], patience=2, delta=0.0)
        assert best_epoch == 0
        assert stop_epoch == 0


# ═══════════════════════════════════════════════════════════════════════════════
# _smooth_losses
# ═══════════════════════════════════════════════════════════════════════════════


class TestSmoothLosses:
    def test_no_smoothing(self) -> None:
        """smoothing <= 0 returns original."""
        losses = np.array([1.0, 2.0, 3.0, 4.0])
        result = _smooth_losses(losses, smoothing=0.0)
        assert np.array_equal(result, losses)

    def test_smoothing_reduces_noise(self) -> None:
        """Smoothing reduces variance."""
        np.random.seed(42)
        losses = np.random.randn(100) * 2 + 5.0
        result = _smooth_losses(losses, smoothing=0.2)
        assert result.std() < losses.std()

    def test_ema_first_value_preserved(self) -> None:
        """EMA keeps s_0 = x_0; later values are recency-weighted blends (TR-VIZ-06..21)."""
        losses = np.array([100.0, 1.0, 1.0, 1.0, 1.0, 100.0])
        result = _smooth_losses(losses, smoothing=0.3)
        assert result[0] == pytest.approx(losses[0])
        assert result[-1] != pytest.approx(losses[-1])

    def test_single_element(self) -> None:
        """Single element array is returned unchanged (s_0 = x_0)."""
        losses = np.array([5.0])
        result = _smooth_losses(losses, smoothing=0.3)
        assert result.shape == losses.shape
        assert np.isfinite(result).all()
        assert result[0] == pytest.approx(5.0)


# ═══════════════════════════════════════════════════════════════════════════════
# _filter_lr_find_data
# ═══════════════════════════════════════════════════════════════════════════════


class TestFilterLRFindData:
    def test_removes_inf(self) -> None:
        """Inf values are removed."""
        lrs = np.array([1e-5, 1e-4, 1e-3, 1e-2])
        losses = np.array([10.0, np.inf, 5.0, 2.0])
        smooth = losses.copy()
        lrs_f, losses_f, smooth_f = _filter_lr_find_data(lrs, losses, smooth)
        assert len(lrs_f) == 3
        assert np.isfinite(losses_f).all()

    def test_removes_nan(self) -> None:
        """NaN values are removed."""
        lrs = np.array([1e-5, 1e-4, 1e-3])
        losses = np.array([10.0, np.nan, 5.0])
        smooth = losses.copy()
        lrs_f, losses_f, smooth_f = _filter_lr_find_data(lrs, losses, smooth)
        assert len(lrs_f) == 2
        assert np.isfinite(losses_f).all()


# ═══════════════════════════════════════════════════════════════════════════════
# _suggest_learning_rate
# ═══════════════════════════════════════════════════════════════════════════════


class TestSuggestLearningRate:
    def test_valley_method(self) -> None:
        """valley method finds gradient sign change."""
        lrs = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        # Losses decreasing then increasing
        smooth = np.array([10.0, 5.0, 2.0, 8.0, 20.0])
        suggestion = _suggest_learning_rate(lrs, smooth, "valley")
        assert suggestion is not None
        assert suggestion > 0

    def test_steepest_method(self) -> None:
        """steepest method finds steepest descent."""
        lrs = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        smooth = np.array([10.0, 9.0, 8.0, 2.0, 1.5])
        suggestion = _suggest_learning_rate(lrs, smooth, "steepest")
        assert suggestion is not None
        assert suggestion > 0

    def test_minimum_method(self) -> None:
        """minimum method returns LR at min loss, without the old x0.1 fudge (TR-VIZ-06..21)."""
        lrs = np.array([1e-5, 1e-4, 1e-3, 1e-2])
        smooth = np.array([10.0, 5.0, 2.0, 8.0])
        suggestion = _suggest_learning_rate(lrs, smooth, "minimum")
        assert suggestion is not None
        # Suggested verbatim at the valley minimum (lrs[2]); no x0.1 shrinkage.
        assert suggestion == pytest.approx(1e-3)

    def test_empty_array(self) -> None:
        """Empty array returns None."""
        suggestion = _suggest_learning_rate(np.array([]), np.array([]), "valley")
        assert suggestion is None


class TestPlotSingleMetric:
    def test_basic(self) -> None:
        """Produces a plot on the given axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _plot_single_metric(ax, "loss", [1, 2, 3], [10.0, 8.0, 6.0], highlight_best=True)
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_with_error_bars(self) -> None:
        """Error bars are added when provided."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _plot_single_metric(
            ax,
            "accuracy",
            [1, 2, 3],
            [0.9, 0.85, 0.95],
            highlight_best=True,
            yerr=[0.01, 0.02, 0.01],
        )
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# _plot_early_stopping_markers / _add_early_stopping_annotations
# ═══════════════════════════════════════════════════════════════════════════════


class TestEarlyStoppingMarkers:
    def test_basic_markers(self) -> None:
        """Markers are added to axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # stop_epoch < n_epochs so early-stop line is also drawn
        _plot_early_stopping_markers(ax, best_epoch=3, stop_epoch=5, n_epochs=10, patience=2)
        # Should have vertical lines from best epoch and stop epoch
        assert len(ax.lines) >= 2
        plt.close(fig)

    def test_no_early_stop_when_stop_at_end(self) -> None:
        """No early stop marker when stopped at last epoch."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _plot_early_stopping_markers(ax, best_epoch=5, stop_epoch=5, n_epochs=5, patience=10)
        # Only best epoch line, no early stop line
        plt.close(fig)


class TestEarlyStoppingAnnotations:
    def test_with_stop(self) -> None:
        """Annotations include stop info when stopped early."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _add_early_stopping_annotations(
            ax, best_epoch=3, best_val_loss=0.5, stop_epoch=5, n_epochs=10, patience=2, delta=0.0
        )
        plt.close(fig)

    def test_without_stop(self) -> None:
        """Annotations without early stop."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _add_early_stopping_annotations(
            ax, best_epoch=3, best_val_loss=0.5, stop_epoch=10, n_epochs=10, patience=5, delta=0.0
        )
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# plot_learning_curves
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlotLearningCurves:
    def test_empty_history_returns_figure(self) -> None:
        """Empty train history returns a figure with warning text."""
        fig = plot_learning_curves({}, return_figure=True)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_values_returns_figure(self) -> None:
        """All-empty value lists return figure."""
        fig = plot_learning_curves({"loss": []}, return_figure=True)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_history_no_return(self) -> None:
        """Empty history without return_figure returns None."""
        result = plot_learning_curves({}, return_figure=False)
        assert result is None

    def test_single_metric_train_only(self) -> None:
        """Single metric with only training data."""
        train = {"loss": [10.0, 8.0, 6.0, 5.0]}
        fig = plot_learning_curves(train, return_figure=True)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_validation(self) -> None:
        """Training + validation curves."""
        train = {"loss": [10.0, 8.0, 6.0, 5.0]}
        val = {"loss": [9.0, 7.0, 6.5, 6.0]}
        fig = plot_learning_curves(train, val, return_figure=True, show_annotations=False)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_smoothing_applied(self) -> None:
        """Smoothing parameter is applied."""
        train = {"loss": [10.0, 8.0, 6.0, 5.0]}
        fig = plot_learning_curves(train, smoothing=0.3, return_figure=True)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_log_scale(self) -> None:
        """log_scale enables log y-axis."""
        train = {"loss": [0.01, 0.008, 0.005, 0.003]}
        fig = plot_learning_curves(train, log_scale=["loss"], return_figure=True)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_no_grid(self) -> None:
        """show_grid=False disables grid."""
        train = {"loss": [10.0, 8.0, 6.0]}
        fig = plot_learning_curves(train, show_grid=False, return_figure=True)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_multiple_metrics(self) -> None:
        """Multiple metrics produce subplots."""
        train = {"loss": [10, 8, 6], "mae": [2.0, 1.5, 1.0]}
        val = {"loss": [9, 7, 6.5], "mae": [1.8, 1.4, 1.2]}
        fig = plot_learning_curves(train, val, return_figure=True)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_nan_values(self) -> None:
        """NaN values are filtered out."""
        train = {"loss": [10.0, float("nan"), 6.0, 5.0]}
        fig = plot_learning_curves(train, return_figure=True)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_metrics_to_plot_subset(self) -> None:
        """Only specified metrics are plotted."""
        train = {"loss": [10, 8, 6], "mae": [2.0, 1.5, 1.0]}
        fig = plot_learning_curves(train, metrics_to_plot=["loss"], return_figure=True)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_no_valid_metrics_raises(self) -> None:
        """No metrics found in history raises ValueError."""
        with pytest.raises(ValueError, match="No valid metrics found"):
            plot_learning_curves(
                {"loss": [1.0]}, metrics_to_plot=["nonexistent"], return_figure=True
            )


# ═══════════════════════════════════════════════════════════════════════════════
# plot_validation_metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlotValidationMetrics:
    def test_single_metric(self) -> None:
        """Single metric produces a figure."""
        fig = plot_validation_metrics(
            epochs=[1, 2, 3],
            metrics={"loss": [10.0, 8.0, 6.0]},
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_error_bars(self) -> None:
        """Error bars are displayed when provided."""
        fig = plot_validation_metrics(
            epochs=[1, 2, 3],
            metrics={"loss": [10.0, 8.0, 6.0], "mae": [2.0, 1.5, 1.0]},
            error_bars={"loss": [0.5, 0.3, 0.2]},
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_no_highlight(self) -> None:
        """highlight_best=False doesn't add best markers."""
        fig = plot_validation_metrics(
            epochs=[1, 2, 3],
            metrics={"accuracy": [0.9, 0.85, 0.95]},
            highlight_best=False,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# plot_early_stopping
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlotEarlyStopping:
    def test_returns_figure(self) -> None:
        """Returns a Figure when return_figure=True."""
        train_losses = [10.0, 8.0, 7.0, 7.5, 8.0]
        val_losses = [12.0, 9.0, 8.5, 9.0, 10.0]
        result = plot_early_stopping(train_losses, val_losses, patience=2, return_figure=True)
        assert isinstance(result, Figure)
        import matplotlib.pyplot as plt

        plt.close(result)

    def test_no_early_stop_when_improving(self) -> None:
        """When val loss keeps improving, no early stop."""
        train_losses = [10.0, 8.0, 6.0, 4.0, 2.0]
        val_losses = [9.0, 7.0, 5.0, 3.0, 1.0]
        result = plot_early_stopping(train_losses, val_losses, patience=3, return_figure=True)
        assert isinstance(result, Figure)
        import matplotlib.pyplot as plt

        plt.close(result)

    def test_no_return(self) -> None:
        """Without return_figure, returns None."""
        result = plot_early_stopping([10.0, 8.0], [12.0, 9.0], return_figure=False)
        assert result is None

    def test_with_delta(self) -> None:
        """Delta parameter prevents minor improvements from counting."""
        train_losses = [10.0, 9.99, 9.98, 10.0, 10.5]
        val_losses = [11.0, 10.99, 10.98, 11.0, 11.5]
        result = plot_early_stopping(
            train_losses, val_losses, patience=2, delta=0.1, return_figure=True
        )
        assert isinstance(result, Figure)
        import matplotlib.pyplot as plt

        plt.close(result)


# ═══════════════════════════════════════════════════════════════════════════════
# _create_lr_plot
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateLRPlot:
    def test_with_suggested_lr(self) -> None:
        """Figure is created with suggested LR line."""
        fig = _create_lr_plot(
            np.array([1e-5, 1e-4, 1e-3, 1e-2]),
            np.array([10.0, 5.0, 2.0, 8.0]),
            np.array([10.0, 5.0, 2.0, 8.0]),
            suggested_lr=5e-4,
            figsize=(10, 6),
            title="Test LR",
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_without_suggested_lr(self) -> None:
        """Figure without suggested LR still works."""
        fig = _create_lr_plot(
            np.array([1e-5, 1e-4, 1e-3]),
            np.array([10.0, 5.0, 2.0]),
            np.array([10.0, 5.0, 2.0]),
            suggested_lr=None,
            figsize=(10, 6),
            title="Test LR",
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_arrays(self) -> None:
        """Empty arrays still produce a figure."""
        fig = _create_lr_plot(
            np.array([]),
            np.array([]),
            np.array([]),
            suggested_lr=None,
            figsize=(10, 6),
            title="Empty",
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# plot_lr_find_results
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlotLRFindResults:
    def test_returns_figure_and_lr(self) -> None:
        """return_figure=True returns (figure, suggested_lr)."""
        lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        losses = [10.0, 5.0, 2.0, 8.0, 20.0]
        result = plot_lr_find_results(lrs, losses, smoothing=0.1, return_figure=True)
        assert result is not None
        fig, suggested_lr = result
        assert isinstance(fig, Figure)
        assert isinstance(suggested_lr, float)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_valley_method(self) -> None:
        """valley method finds the minimum before divergence."""
        lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        losses = [10.0, 5.0, 2.0, 8.0, 20.0]
        result = plot_lr_find_results(lrs, losses, suggestion_method="valley", return_figure=True)
        assert result is not None
        _, suggested_lr = result
        assert suggested_lr > 0
        import matplotlib.pyplot as plt

        plt.close(result[0])

    def test_steepest_method(self) -> None:
        """steepest method finds steepest descent."""
        lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        losses = [10.0, 9.0, 8.0, 2.0, 1.5]
        result = plot_lr_find_results(lrs, losses, suggestion_method="steepest", return_figure=True)
        assert result is not None
        _, suggested_lr = result
        assert suggested_lr > 0
        import matplotlib.pyplot as plt

        plt.close(result[0])

    def test_minimum_method(self) -> None:
        """minimum method finds absolute minimum."""
        lrs = [1e-5, 1e-4, 1e-3, 1e-2]
        losses = [10.0, 5.0, 2.0, 8.0]
        result = plot_lr_find_results(lrs, losses, suggestion_method="minimum", return_figure=True)
        assert result is not None
        _, suggested_lr = result
        assert suggested_lr > 0
        import matplotlib.pyplot as plt

        plt.close(result[0])

    def test_no_return(self) -> None:
        """Without return_figure, returns None."""
        result = plot_lr_find_results([1e-5, 1e-4], [10.0, 5.0], return_figure=False)
        assert result is None
