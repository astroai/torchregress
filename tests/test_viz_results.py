import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from torchregress.viz.results import (
    plot_feature_importance,
    plot_model_ensemble_contributions,
    plot_parameter_sensitivity,
    plot_performance_comparison,
)


class TestPlotPerformanceComparison:
    """Test performance comparison visualization."""

    def setup_method(self):
        """Setup test data."""
        self.metrics = {
            "Model A": {
                "rmse": 0.5,
                "r2": 0.8,
                "mae": 0.4,
                "latency": 10.0,
            },
            "Model B": {
                "rmse": 0.4,
                "r2": 0.85,
                "mae": 0.35,
                "latency": 25.0,
            },
            "Model C": {
                "rmse": 0.6,
                "r2": 0.75,
                "mae": 0.5,
                "latency": 5.0,
                "extra_metric": 1.0,
            },
        }

    def teardown_method(self):
        """Close all figures."""
        plt.close("all")

    def test_basic_bar_plot(self):
        """Test basic bar plot rendering."""
        fig = plot_performance_comparison(
            self.metrics,
            return_figure=True,
            plot_type="bar",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_metrics_to_include(self):
        """Test specifying explicit metrics to include."""
        fig = plot_performance_comparison(
            self.metrics,
            metrics_to_include=["rmse", "r2"],
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        # Should only have 2 metric groups in the plot
        plt.close(fig)

    def test_higher_is_better_logic(self):
        """Test custom higher_is_better logic and sorting."""
        higher_is_better = {
            "rmse": False,
            "r2": True,
            "mae": False,
            "latency": False,
        }
        fig = plot_performance_comparison(
            self.metrics,
            higher_is_better=higher_is_better,
            sort_by="rmse",
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_radar_plot(self):
        """Test radar plot rendering."""
        fig = plot_performance_comparison(
            self.metrics,
            plot_type="radar",
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_heatmap_plot(self):
        """Test heatmap plot rendering."""
        fig = plot_performance_comparison(
            self.metrics,
            plot_type="heatmap",
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_no_common_metrics_error(self):
        """Test error when no common metrics exist."""
        bad_metrics = {
            "Model A": {"metric1": 1.0},
            "Model B": {"metric2": 2.0},
        }
        with pytest.raises(ValueError, match="No common metrics found across all models"):
            plot_performance_comparison(bad_metrics)

    def test_invalid_plot_type(self):
        """Test error with invalid plot type."""
        with pytest.raises(ValueError, match="Unknown plot_type: invalid"):
            plot_performance_comparison(self.metrics, plot_type="invalid")

    def test_provided_axes(self):
        """Test providing pre-existing axes for bar plot."""
        fig, ax = plt.subplots()
        returned_fig = plot_performance_comparison(
            self.metrics,
            ax=ax,
            plot_type="bar",
            return_figure=True,
        )
        assert returned_fig is fig
        plt.close(fig)


class TestPlotParameterSensitivity:
    """Test parameter sensitivity visualization."""

    def setup_method(self):
        """Setup test data."""
        self.parameter_values = {
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [16, 32, 64],
        }
        self.metric_values = {
            "rmse": [0.5, 0.4, 0.6],
            "r2": [0.7, 0.8, 0.6],
        }

    def teardown_method(self):
        """Close all figures."""
        plt.close("all")

    def test_line_plot(self):
        """Test basic line plot rendering."""
        fig = plot_parameter_sensitivity(
            self.parameter_values,
            self.metric_values,
            plot_type="line",
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_bar_plot(self):
        """Test bar plot rendering."""
        fig = plot_parameter_sensitivity(
            self.parameter_values,
            self.metric_values,
            plot_type="bar",
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_single_metric(self):
        """Test single metric case."""
        fig = plot_parameter_sensitivity(
            self.parameter_values,
            {"rmse": [0.5, 0.4, 0.6]},
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_no_return_figure(self, monkeypatch):
        """Test return_figure=False"""
        monkeypatch.setattr(plt, "show", lambda: None)
        res = plot_parameter_sensitivity(
            self.parameter_values,
            self.metric_values,
            return_figure=False,
        )
        assert res is None


class TestPlotFeatureImportance:
    """Test feature importance visualization."""

    def setup_method(self):
        """Setup test data."""
        self.feature_names = ["feature1", "feature2", "feature3"]
        self.importance_values = [0.1, 0.8, 0.3]
        self.importance_errors = [0.05, 0.1, 0.02]

    def teardown_method(self):
        """Close all figures."""
        plt.close("all")

    def test_horizontal_plot(self):
        """Test horizontal plot rendering."""
        fig = plot_feature_importance(
            self.feature_names,
            self.importance_values,
            horizontal=True,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_vertical_plot(self):
        """Test vertical plot rendering."""
        fig = plot_feature_importance(
            self.feature_names,
            self.importance_values,
            horizontal=False,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_errors(self):
        """Test rendering with errors."""
        fig = plot_feature_importance(
            self.feature_names,
            self.importance_values,
            importance_errors=self.importance_errors,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_top_n(self):
        """Test limiting to top N features."""
        fig = plot_feature_importance(
            self.feature_names,
            self.importance_values,
            top_n=2,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_tensor_inputs(self):
        """Test rendering with tensor inputs."""
        import torch

        fig = plot_feature_importance(
            self.feature_names,
            torch.tensor(self.importance_values),
            importance_errors=torch.tensor(self.importance_errors),
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_provided_axes(self):
        """Test providing axes."""
        fig, ax = plt.subplots()
        returned_fig = plot_feature_importance(
            self.feature_names,
            self.importance_values,
            ax=ax,
            return_figure=True,
        )
        assert returned_fig is fig
        plt.close(fig)

    def test_no_return_figure(self, monkeypatch):
        """Test return_figure=False"""
        monkeypatch.setattr(plt, "show", lambda: None)
        res = plot_feature_importance(
            self.feature_names,
            self.importance_values,
            return_figure=False,
        )
        assert res is None


class TestPlotModelEnsembleContributions:
    """Test model ensemble contributions visualization."""

    def setup_method(self):
        """Setup test data."""
        import numpy as np

        self.predictions = {
            "Model A": np.array([1.0, 2.0, 3.0]),
            "Model B": np.array([1.5, 2.5, 3.5]),
        }
        self.ensemble_prediction = np.array([1.25, 2.25, 3.25])
        self.model_weights = {"Model A": 0.5, "Model B": 0.5}

    def teardown_method(self):
        """Close all figures."""
        plt.close("all")

    def test_basic_plot(self):
        """Test basic plot rendering."""
        fig = plot_model_ensemble_contributions(
            self.predictions,
            self.ensemble_prediction,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_weights(self):
        """Test rendering with weights."""
        fig = plot_model_ensemble_contributions(
            self.predictions,
            self.ensemble_prediction,
            model_weights=self.model_weights,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_provided_axes(self):
        """Test providing axes."""
        fig, ax = plt.subplots()
        returned_fig = plot_model_ensemble_contributions(
            self.predictions,
            self.ensemble_prediction,
            ax=ax,
            return_figure=True,
        )
        assert returned_fig is fig
        plt.close(fig)

    def test_no_return_figure(self, monkeypatch):
        """Test return_figure=False"""
        monkeypatch.setattr(plt, "show", lambda: None)
        res = plot_model_ensemble_contributions(
            self.predictions,
            self.ensemble_prediction,
            return_figure=False,
        )
        assert res is None


def test_plot_performance_comparison_no_return_figure(monkeypatch):
    """Test return_figure=False for all plot types."""
    metrics = {
        "Model A": {"rmse": 0.5, "r2": 0.8},
        "Model B": {"rmse": 0.4, "r2": 0.85},
    }
    monkeypatch.setattr(plt, "show", lambda: None)

    # Radar and Heatmap no longer fail because we don't change plt params globally during heatmaps/radars
    for plot_type in ["bar", "radar", "heatmap"]:
        res = plot_performance_comparison(
            metrics,
            plot_type=plot_type,
            return_figure=False,
        )
        assert res is None
        plt.close("all")


def test_plot_performance_bar_latex_success(monkeypatch):
    """Test latex rendering block in bar plot without failing due to missing system latex."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/latex")
    # Actually mock rcParams update to avoid invoking latex subprocess if font not found
    monkeypatch.setattr(plt.rcParams, "update", lambda x: None)

    metrics = {
        "Model A": {"rmse": 0.5, "r2": 0.8, "mae": 0.4, "other": 1.0},
    }
    fig = plot_performance_comparison(metrics, plot_type="bar", return_figure=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_performance_bar_latex_exception(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/latex")

    # Force exception inside try block
    def mock_update(*args, **kwargs):
        raise RuntimeError("simulated")

    monkeypatch.setattr(plt.rcParams, "update", mock_update)

    metrics = {
        "Model A": {"rmse": 0.5},
    }
    fig = plot_performance_comparison(metrics, plot_type="bar", return_figure=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_performance_bar_axes_and_not_return_figure(monkeypatch):
    """Test plot_performance_bar with provided axes and return_figure=False."""
    monkeypatch.setattr(plt, "show", lambda: None)
    metrics = {"Model A": {"rmse": 0.5}}
    fig, ax = plt.subplots()
    res = plot_performance_comparison(metrics, ax=ax, plot_type="bar", return_figure=False)
    assert res is None
    plt.close(fig)


def test_plot_performance_radar_higher_is_better_false():
    """Test radar plot with higher_is_better=False."""
    metrics = {
        "Model A": {"mse": 0.1, "mae": 0.2},
        "Model B": {"mse": 0.2, "mae": 0.1},
    }
    higher_is_better = {"mse": False, "mae": False}
    fig = plot_performance_comparison(
        metrics, plot_type="radar", higher_is_better=higher_is_better, return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_performance_heatmap_higher_is_better_false():
    """Test heatmap plot with higher_is_better=False."""
    metrics = {
        "Model A": {"mse": 0.1, "mae": 0.2},
        "Model B": {"mse": 0.2, "mae": 0.1},
    }
    higher_is_better = {"mse": False, "mae": False}
    fig = plot_performance_comparison(
        metrics, plot_type="heatmap", higher_is_better=higher_is_better, return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_parameter_sensitivity_lower_is_better():
    """Test plot_parameter_sensitivity with lower is better metrics."""
    parameter_values = {"p1": [1, 2, 3]}
    metric_values = {"error": [0.5, 0.4, 0.6]}
    fig = plot_parameter_sensitivity(parameter_values, metric_values, return_figure=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_parameter_sensitivity_bar_plot_colors():
    parameter_values = {"p1": [1, 2, 3]}
    metric_values = {"acc": [0.5, 0.4, 0.6]}
    fig = plot_parameter_sensitivity(
        parameter_values, metric_values, plot_type="bar", return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_parameter_sensitivity_mismatched_lengths():
    parameter_values = {"p1": [1, 2]}
    metric_values = {"acc": [0.5, 0.4, 0.6]}
    fig = plot_parameter_sensitivity(parameter_values, metric_values, return_figure=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_parameter_sensitivity_non_numeric_params():
    parameter_values = {"p1": ["a", "b", "c"]}
    metric_values = {"acc": [0.5, 0.4, 0.6]}
    fig = plot_parameter_sensitivity(parameter_values, metric_values, return_figure=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_feature_importance_horizontal_top_n_descending():
    feature_names = ["f1", "f2", "f3", "f4"]
    importance_values = [0.1, 0.4, 0.2, 0.3]
    fig = plot_feature_importance(
        feature_names, importance_values, horizontal=False, top_n=2, return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_feature_importance_horizontal_false_top_n_none():
    feature_names = ["f1", "f2", "f3", "f4"]
    importance_values = [0.1, 0.4, 0.2, 0.3]
    fig = plot_feature_importance(
        feature_names, importance_values, horizontal=False, return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_feature_importance_axes_none_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    feature_names = ["f1"]
    importance_values = [0.1]
    res = plot_feature_importance(feature_names, importance_values, ax=None, return_figure=False)
    assert res is None
    plt.close("all")


def test_plot_model_ensemble_contributions_no_x_provided_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    import numpy as np

    predictions = {"Model A": np.array([1.0])}
    ensemble = np.array([1.0])
    res = plot_model_ensemble_contributions(predictions, ensemble, x=None, return_figure=False)
    assert res is None
    plt.close("all")


def test_plot_performance_bar_extremes():
    """Test values < 0.01 and >= 1000 in bar plot."""
    metrics = {
        "Model A": {"metric1": 0.005, "metric2": 1500.0},
    }
    fig = plot_performance_comparison(metrics, plot_type="bar", return_figure=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_performance_radar_no_best_values():
    """Test radar plot with highlight_best=False."""
    metrics = {"Model A": {"mse": 0.1}}
    fig = plot_performance_comparison(
        metrics, plot_type="radar", highlight_best=False, return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_performance_heatmap_no_best_values():
    """Test heatmap plot with highlight_best=False."""
    metrics = {"Model A": {"mse": 0.1}}
    fig = plot_performance_comparison(
        metrics, plot_type="heatmap", highlight_best=False, return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_parameter_sensitivity_line_plot_no_highlight_best():
    """Test plot_parameter_sensitivity without highlight_best."""
    parameter_values = {"p1": [1, 2, 3]}
    metric_values = {"acc": [0.5, 0.4, 0.6]}
    fig = plot_parameter_sensitivity(
        parameter_values, metric_values, plot_type="line", highlight_best=False, return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_parameter_sensitivity_bar_plot_no_highlight_best():
    """Test plot_parameter_sensitivity without highlight_best."""
    parameter_values = {"p1": [1, 2, 3]}
    metric_values = {"acc": [0.5, 0.4, 0.6]}
    fig = plot_parameter_sensitivity(
        parameter_values, metric_values, plot_type="bar", highlight_best=False, return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_feature_importance_no_show_values():
    """Test plot_feature_importance with show_values=False."""
    feature_names = ["f1"]
    importance_values = [0.1]
    fig = plot_feature_importance(
        feature_names, importance_values, show_values=False, return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)

    fig = plot_feature_importance(
        feature_names, importance_values, horizontal=False, show_values=False, return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_model_ensemble_contributions_no_model_weights():
    """Test plot_model_ensemble_contributions without model_weights."""
    import numpy as np

    predictions = {"Model A": np.array([1.0])}
    ensemble = np.array([1.0])
    fig = plot_model_ensemble_contributions(
        predictions, ensemble, model_weights=None, return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_performance_bar_medium_value():
    """Test values between 0.01 and 0.1."""
    metrics = {
        "Model A": {"metric1": 0.05},
    }
    fig = plot_performance_comparison(metrics, plot_type="bar", return_figure=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_performance_bar_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    metrics = {
        "Model A": {"metric1": 1.0},
    }
    res = plot_performance_comparison(metrics, plot_type="bar", return_figure=False)
    assert res is None


def test_plot_performance_heatmap_non_float_values():
    """Test heatmap string value."""
    import numpy as np

    metrics = {"Model A": {"mse": np.array([1])[0]}}
    fig = plot_performance_comparison(metrics, plot_type="heatmap", return_figure=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_feature_importance_top_n_with_errors():
    """Test top_n with horizontal=True and horizontal=False with errors."""
    feature_names = ["f1", "f2", "f3"]
    importance_values = [0.1, 0.4, 0.2]
    importance_errors = [0.01, 0.04, 0.02]
    fig = plot_feature_importance(
        feature_names,
        importance_values,
        importance_errors=importance_errors,
        top_n=2,
        horizontal=True,
        return_figure=True,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)
    fig = plot_feature_importance(
        feature_names,
        importance_values,
        importance_errors=importance_errors,
        top_n=2,
        horizontal=False,
        return_figure=True,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_feature_importance_show(monkeypatch):
    """Test feature importance ax=None and return_figure=False."""
    monkeypatch.setattr(plt, "show", lambda: None)
    feature_names = ["f1"]
    importance_values = [0.1]
    res = plot_feature_importance(feature_names, importance_values, ax=None, return_figure=False)
    assert res is None


def test_plot_model_ensemble_contributions_show(monkeypatch):
    """Test ensemble contributions ax=None and return_figure=False."""
    monkeypatch.setattr(plt, "show", lambda: None)
    import numpy as np

    predictions = {"Model A": np.array([1.0])}
    ensemble = np.array([1.0])
    res = plot_model_ensemble_contributions(predictions, ensemble, ax=None, return_figure=False)
    assert res is None


def test_plot_performance_bar_show_ax_none(monkeypatch):
    """Test plot_performance_bar with ax=None and return_figure=False."""
    monkeypatch.setattr(plt, "show", lambda: None)
    metrics = {"Model A": {"rmse": 0.5}}
    res = plot_performance_comparison(metrics, ax=None, plot_type="bar", return_figure=False)
    assert res is None


def test_plot_feature_importance_show_ax_none(monkeypatch):
    """Test plot_feature_importance with ax=None and return_figure=False."""
    monkeypatch.setattr(plt, "show", lambda: None)
    res = plot_feature_importance(["f1"], [0.1], ax=None, return_figure=False)
    assert res is None


def test_plot_model_ensemble_contributions_show_ax_none(monkeypatch):
    """Test plot_model_ensemble_contributions with ax=None and return_figure=False."""
    import numpy as np

    monkeypatch.setattr(plt, "show", lambda: None)
    res = plot_model_ensemble_contributions(
        {"M": np.array([1])}, np.array([1]), ax=None, return_figure=False
    )
    assert res is None


def test_plot_performance_bar_show_ax_provided_return_figure_false(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    metrics = {"Model A": {"rmse": 0.5}}
    fig, ax = plt.subplots()
    res = plot_performance_comparison(metrics, plot_type="bar", ax=ax, return_figure=False)
    assert res is None
    plt.close(fig)


def test_plot_feature_importance_show_ax_provided_return_figure_false(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig, ax = plt.subplots()
    res = plot_feature_importance(["f1"], [0.1], ax=ax, return_figure=False)
    assert res is None
    plt.close(fig)


def test_plot_model_ensemble_contributions_show_ax_provided_return_figure_false(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    import numpy as np

    fig, ax = plt.subplots()
    res = plot_model_ensemble_contributions(
        {"M": np.array([1])}, np.array([1]), ax=ax, return_figure=False
    )
    assert res is None
    plt.close(fig)


def test_plot_performance_radar_plt_show_called(monkeypatch):
    """Test plot_performance_comparison radar where plt.show() is called."""
    import torchregress.viz.results

    called = False

    def mock_show(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(torchregress.viz.results.plt, "show", mock_show)
    metrics = {"Model A": {"rmse": 0.5}}
    plot_performance_comparison(metrics, plot_type="radar", return_figure=False)
    assert called


def test_plot_feature_importance_plt_show_called(monkeypatch):
    """Test plot_feature_importance where plt.show() is called."""
    import torchregress.viz.results

    called = False

    def mock_show(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(torchregress.viz.results.plt, "show", mock_show)
    plot_feature_importance(["f1"], [0.1], ax=None, return_figure=False)
    assert called


def test_plot_model_ensemble_contributions_plt_show_called(monkeypatch):
    """Test plot_model_ensemble_contributions where plt.show() is called."""
    import numpy as np

    import torchregress.viz.results

    called = False

    def mock_show(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(torchregress.viz.results.plt, "show", mock_show)
    plot_model_ensemble_contributions(
        {"M": np.array([1])}, np.array([1]), ax=None, return_figure=False
    )
    assert called


def test_plot_parameter_sensitivity_plt_show_called(monkeypatch):
    import torchregress.viz.results

    called = False

    def mock_show(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(torchregress.viz.results.plt, "show", mock_show)
    parameter_values = {"p1": [1, 2, 3]}
    metric_values = {"acc": [0.5, 0.4, 0.6]}
    plot_parameter_sensitivity(
        parameter_values, metric_values, plot_type="line", return_figure=False
    )
    assert called


def test_plot_performance_heatmap_plt_show_called(monkeypatch):
    import torchregress.viz.results

    called = False

    def mock_show(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(torchregress.viz.results.plt, "show", mock_show)
    metrics = {"Model A": {"rmse": 0.5}}
    plot_performance_comparison(metrics, plot_type="heatmap", return_figure=False)
    assert called


def test_plot_performance_bar_plt_show_called(monkeypatch):
    import torchregress.viz.results

    called = False

    def mock_show(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(torchregress.viz.results.plt, "show", mock_show)
    metrics = {"Model A": {"rmse": 0.5}}
    plot_performance_comparison(metrics, plot_type="bar", return_figure=False)
    assert called
