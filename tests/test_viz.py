from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from torchregress.viz.diagnostic import (
    _add_residual_trend,
    _filter_residual_data,
    _plot_residual_scatter,
    plot_calibration_curve,
    plot_distribution_comparison,
    plot_prediction_intervals,
    plot_qq_plot,
    plot_reliability_diagram,
    plot_residual_histogram,
    plot_residuals,
)
from torchregress.viz.utils import add_zero_line, create_grid_figure


class TestVizDiagnostic:
    """Test visualization diagnostic functions."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        torch.manual_seed(42)
        self.n_samples = 100
        self.y_true = np.random.randn(self.n_samples)
        self.y_pred = self.y_true + np.random.randn(self.n_samples) * 0.1
        self.y_lower = self.y_pred - 0.2
        self.y_upper = self.y_pred + 0.2

    def teardown_method(self):
        """Close all figures."""
        plt.close("all")

    def test_filter_residual_data(self):
        """Test _filter_residual_data helper."""
        y_pred = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 100.0])
        residuals = np.array([0.1, 0.2, 0.3, np.inf, 0.5, 100.0])

        # Test NaN/Inf filtering
        filtered_pred, filtered_res = _filter_residual_data(
            y_pred, residuals, clip_outliers=False, downsample=False
        )
        assert len(filtered_pred) == 4
        assert len(filtered_res) == 4
        assert np.all(np.isfinite(filtered_pred))
        assert np.all(np.isfinite(filtered_res))

        # Test outlier clipping
        filtered_pred_clip, filtered_res_clip = _filter_residual_data(
            y_pred, residuals, clip_outliers=True, clip_percentile=90.0, downsample=False
        )
        # 10% clip on 4 items removes min and max
        assert len(filtered_pred_clip) == 2
        assert len(filtered_res_clip) == 2
        assert 100.0 not in filtered_res_clip

        # Test downsampling
        np.random.seed(42)
        large_pred = np.arange(1000)
        large_res = np.random.randn(1000)
        downsampled_pred, downsampled_res = _filter_residual_data(
            large_pred, large_res, downsample=True, max_points=100
        )
        assert len(downsampled_pred) == 100
        assert len(downsampled_res) == 100

    def test_plot_residual_scatter(self):
        """Test _plot_residual_scatter helper."""
        fig, ax = plt.subplots()
        y_pred = np.array([1.0, 2.0, 3.0])
        residuals = np.array([0.1, -0.1, 0.0])

        _plot_residual_scatter(ax, y_pred, residuals, alpha=0.5, color="blue")

        # Check if scatter plot was created
        assert len(ax.collections) > 0
        plt.close(fig)

        # Test large dataset (hexbin)
        fig, ax = plt.subplots()
        large_pred = np.arange(1500)
        large_res = np.random.randn(1500)

        _plot_residual_scatter(ax, large_pred, large_res, alpha=0.5, color="blue")

        # Check if hexbin was created
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_add_residual_trend(self):
        """Test _add_residual_trend helper."""
        fig, ax = plt.subplots()
        y_pred = np.array([1.0, 2.0, 3.0])
        residuals = np.array([0.1, 0.2, 0.3])

        _add_residual_trend(ax, y_pred, residuals, trend_color="red")

        # Check if trend line was plotted
        lines = ax.get_lines()
        assert len(lines) == 1
        assert lines[0].get_color() == "red"
        assert lines[0].get_linestyle() == "--"
        plt.close(fig)

        # Test exception handling (e.g. identical x values causes error/warning)
        fig, ax = plt.subplots()
        # In modern numpy/matplotlib this might still plot a line if polyfit warns,
        # but let's test with empty arrays to guarantee it fails gracefully
        bad_pred = np.array([1.0])
        bad_res = np.array([0.1])

        # Should not raise (and not plot because len(y_pred) <= 1)
        _add_residual_trend(ax, bad_pred, bad_res, trend_color="red")
        assert len(ax.get_lines()) == 0

        # Test polyfit exception with inf
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig2, ax2 = plt.subplots()
            # To force an exception in polyfit, we need something that causes np.linalg.lstsq to fail
            # Forcing ValueError by using arrays of different lengths
            bad_pred2 = np.array([1.0, 2.0])
            bad_res2 = np.array([0.1, 0.2, 0.3])

            _add_residual_trend(ax2, bad_pred2, bad_res2, trend_color="red")
            assert len(ax2.get_lines()) == 0
            plt.close(fig2)
        plt.close(fig)

    def test_plot_residuals(self):
        """Test plot_residuals."""
        fig = plot_residuals(
            self.y_pred,
            self.y_true,
            return_figure=True,
            show_trend=True,
            show_zero_line=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_prediction_intervals(self):
        """Test plot_prediction_intervals."""
        fig = plot_prediction_intervals(
            self.y_pred,
            self.y_lower,
            self.y_upper,
            y_true=self.y_true,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_qq_plot(self):
        """Test plot_qq_plot."""
        fig = plot_qq_plot(self.y_pred, self.y_true, return_figure=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_residual_histogram(self):
        """Test plot_residual_histogram."""
        fig = plot_residual_histogram(self.y_pred, self.y_true, return_figure=True, show_kde=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_calibration_curve(self):
        """Test plot_calibration_curve."""
        # Need binary data for calibration curve (or probabilities)
        y_true_bin = np.random.randint(0, 2, self.n_samples)
        y_pred_probs = np.random.rand(self.n_samples)

        fig = plot_calibration_curve(y_pred_probs, y_true_bin, return_figure=True, add_hist=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_reliability_diagram(self):
        """Test plot_reliability_diagram."""
        # Need quantiles
        y_pred_quantiles = {
            0.1: self.y_pred - 0.5,
            0.5: self.y_pred,
            0.9: self.y_pred + 0.5,
        }
        fig = plot_reliability_diagram(y_pred_quantiles, self.y_true, return_figure=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_distribution_comparison(self):
        """Test plot_distribution_comparison."""
        # Need predicted samples [n_samples, batch_size]
        n_pred_samples = 50
        predicted_samples = np.random.randn(n_pred_samples, self.n_samples)

        fig = plot_distribution_comparison(
            predicted_samples,
            self.y_true,
            n_samples_to_show=2,
            return_figure=True,
            plot_type="histogram",  # Avoid KDE issues in test env
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestVizUtils:
    """Test visualization utility functions."""

    def test_add_zero_line_invalid_axis(self):
        """Test add_zero_line raises ValueError on invalid axis."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="axis must be 'x' or 'y', got z"):
            add_zero_line(ax, axis="z")
        plt.close(fig)

    def test_add_zero_line_valid_axis(self):
        """Test add_zero_line works with valid axes."""
        fig, ax = plt.subplots()
        # Should not raise any exceptions
        add_zero_line(ax, axis="x")
        add_zero_line(ax, axis="y")
        plt.close(fig)

    @patch("matplotlib.pyplot.subplots")
    def test_create_grid_figure_auto_dims(self, mock_subplots):
        """Test create_grid_figure automatically determines dimensions correctly."""
        # Setup mock
        mock_fig = MagicMock(spec=Figure)

        # When plt.subplots is called, we return a 2D array of mocked Axes
        def side_effect(nrows, ncols, **kwargs):
            # Create a properly dimensioned list of lists first, then convert to numpy object array
            # so flatten() works as expected.
            axes_list = []
            for r in range(nrows):
                row = []
                for c in range(ncols):
                    row.append(MagicMock())
                axes_list.append(row)

            mock_axes = np.empty((nrows, ncols), dtype=object)
            for r in range(nrows):
                for c in range(ncols):
                    mock_axes[r, c] = axes_list[r][c]
            return mock_fig, mock_axes

        mock_subplots.side_effect = side_effect

        # Test 1: n_plots = 4
        # expected: ncols = 3, nrows = ceil(4/3) = 2
        fig, axes = create_grid_figure(n_plots=4)
        mock_subplots.assert_called_with(
            2, 3, figsize=(12, 8), sharex=False, sharey=False, squeeze=False
        )
        assert len(axes) == 4

        # Test 2: n_plots = 2
        # expected: ncols = 2, nrows = ceil(2/2) = 1
        fig, axes = create_grid_figure(n_plots=2)
        mock_subplots.assert_called_with(
            1, 2, figsize=(12, 8), sharex=False, sharey=False, squeeze=False
        )
        assert len(axes) == 2

    @patch("matplotlib.pyplot.subplots")
    def test_create_grid_figure_provided_dims(self, mock_subplots):
        """Test create_grid_figure uses provided dimensions."""
        mock_fig = MagicMock(spec=Figure)

        def side_effect(nrows, ncols, **kwargs):
            mock_axes = np.empty((nrows, ncols), dtype=object)
            for r in range(nrows):
                for c in range(ncols):
                    mock_axes[r, c] = MagicMock()
            return mock_fig, mock_axes

        mock_subplots.side_effect = side_effect

        # Test with nrows provided
        # expected: nrows = 2, ncols = ceil(4/2) = 2
        fig, axes = create_grid_figure(n_plots=4, nrows=2)
        mock_subplots.assert_called_with(
            2, 2, figsize=(12, 8), sharex=False, sharey=False, squeeze=False
        )

        # Test with ncols provided
        # expected: ncols = 2, nrows = ceil(4/2) = 2
        fig, axes = create_grid_figure(n_plots=4, ncols=2)
        mock_subplots.assert_called_with(
            2, 2, figsize=(12, 8), sharex=False, sharey=False, squeeze=False
        )

        # Test with both provided
        fig, axes = create_grid_figure(n_plots=4, nrows=3, ncols=3)
        mock_subplots.assert_called_with(
            3, 3, figsize=(12, 8), sharex=False, sharey=False, squeeze=False
        )

    @patch("matplotlib.pyplot.subplots")
    def test_create_grid_figure_hide_unused(self, mock_subplots):
        """Test create_grid_figure hides unused subplots."""
        mock_fig = MagicMock(spec=Figure)

        # Create persistent mock axes to verify calls later
        mock_ax_list = [MagicMock() for _ in range(6)]

        def side_effect(nrows, ncols, **kwargs):
            # Shape it to 2x3 for n_plots=4 default
            mock_axes = np.empty((nrows, ncols), dtype=object)
            idx = 0
            for r in range(nrows):
                for c in range(ncols):
                    if idx < len(mock_ax_list):
                        mock_axes[r, c] = mock_ax_list[idx]
                        idx += 1
            return mock_fig, mock_axes

        mock_subplots.side_effect = side_effect

        # Test with 4 plots in a 2x3 grid (6 axes total)
        fig, axes = create_grid_figure(n_plots=4)

        # Verify first 4 axes are returned
        assert len(axes) == 4

        # Verify the 5th and 6th axes are hidden
        mock_ax_list[4].set_visible.assert_called_once_with(False)
        mock_ax_list[5].set_visible.assert_called_once_with(False)

        # Verify first 4 are NOT hidden
        for i in range(4):
            mock_ax_list[i].set_visible.assert_not_called()


def test_plot_binned_metrics_helpers():
    import matplotlib.pyplot as plt
    import numpy as np

    from torchregress.viz.diagnostic import _compute_binned_metrics, _render_binned_metrics_plot

    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    y_pred_std = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 20)
    y_true = np.array([1.1, 1.9, 3.2, 3.8, 5.1] * 20)

    metrics = _compute_binned_metrics(y_pred, y_pred_std, y_true, n_bins=2)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0
    first_key = list(metrics.keys())[0]
    assert "rmse" in metrics[first_key]
    assert "n_samples" in metrics[first_key]

    fig, ax = _render_binned_metrics_plot(
        binned_metrics=metrics,
        metric="rmse",
        figsize=(8, 4),
        title="Test Plot",
        color="red",
        ax=None,
    )

    assert ax.get_title() == "Test Plot"
    assert ax.get_ylabel() == "RMSE"
    assert len(ax.patches) == len(metrics)  # 1 bar per bin
    plt.close(fig)
