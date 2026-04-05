import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

import pytest
from torchregress.viz.diagnostic import (
    plot_calibration_curve,
    plot_distribution_comparison,
    plot_prediction_intervals,
    plot_qq_plot,
    plot_reliability_diagram,
    plot_residual_histogram,
    plot_residuals,
)
from torchregress.viz.utils import add_zero_line


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
