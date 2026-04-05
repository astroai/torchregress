import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from torchregress.viz.results import plot_performance_comparison


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
