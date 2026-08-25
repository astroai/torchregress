import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torchregress.viz.monitoring import (
    _add_early_stopping_annotations,
    _find_early_stopping_point,
    _format_metric_value,
    _plot_early_stopping_markers,
    _smooth_losses,
    plot_early_stopping,
    plot_lr_find_results,
)


class TestMonitoringVizRefactored:
    """Test refactored early stopping visualization functions."""

    def test_find_early_stopping_point(self):
        """Test _find_early_stopping_point helper logic."""
        val_losses = [1.0, 0.9, 0.8, 0.85, 0.9, 0.95]
        patience = 2
        delta = 0.0

        best_epoch, best_val_loss, stop_epoch = _find_early_stopping_point(
            val_losses, patience, delta
        )

        # best val loss is 0.8 at index 2 (epoch 3)
        assert best_epoch == 3
        assert best_val_loss == 0.8

        # Patience is 2, so it stops after 2 more epochs: index 3, index 4
        # stop_epoch is index 4 + 1 = 5
        assert stop_epoch == 5

    def test_plot_early_stopping_markers(self):
        """Test _plot_early_stopping_markers helper."""
        fig, ax = plt.subplots()
        _plot_early_stopping_markers(ax, best_epoch=3, stop_epoch=5, n_epochs=6, patience=2)

        # Check that vertical lines are drawn
        lines = ax.get_lines()
        assert len(lines) == 2  # One for best epoch, one for stop epoch
        assert lines[0].get_xdata()[0] == 3
        assert lines[1].get_xdata()[0] == 5

        # Check the shaded region
        patches = ax.patches
        assert len(patches) > 0

        plt.close(fig)

    def test_add_early_stopping_annotations(self):
        """Test _add_early_stopping_annotations helper."""
        fig, ax = plt.subplots()

        _add_early_stopping_annotations(
            ax, best_epoch=3, best_val_loss=0.8, stop_epoch=5, n_epochs=6, patience=2, delta=0.0
        )

        texts = ax.texts
        # One annotate call for best value, plus those added via add_annotations helper
        assert len(texts) >= 1
        assert "Best:" in texts[0].get_text()

        plt.close(fig)

    def test_plot_early_stopping(self):
        """Test the refactored plot_early_stopping function."""
        train_losses = [1.2, 1.0, 0.9, 0.8, 0.75, 0.7]
        val_losses = [1.0, 0.9, 0.8, 0.85, 0.9, 0.95]

        fig = plot_early_stopping(
            train_losses=train_losses,
            val_losses=val_losses,
            patience=2,
            return_figure=True,
        )

        assert isinstance(fig, Figure)
        plt.close(fig)


class TestMonitoringHelpers:
    def test_format_metric_value(self) -> None:
        assert _format_metric_value(0.0003) == "3.00e-04"
        assert _format_metric_value(0.05, scientific_notation=False) == "0.050"
        assert _format_metric_value(0.5, scientific_notation=False) == "0.50"
        assert _format_metric_value(5.0, scientific_notation=False) == "5.0"
        assert _format_metric_value(12.0, scientific_notation=False) == "12"

    def test_smooth_losses_ema_recurrence(self) -> None:
        """_smooth_losses follows the EMA recurrence s_t = a*x_t + (1-a)*s_{t-1} (TR-VIZ-06..21)."""
        losses = np.linspace(1.0, 0.4, 20, dtype=float)
        smoothed = _smooth_losses(losses, smoothing=0.25)
        assert smoothed.shape == losses.shape
        assert np.isfinite(smoothed).all()
        # Recurrence identity, seeded with s_0 = x_0
        expected = np.empty_like(losses)
        s = losses[0]
        for i, x in enumerate(losses):
            s = 0.25 * x + 0.75 * s
            expected[i] = s
        assert np.allclose(smoothed, expected)

    def test_plot_lr_find_results(self) -> None:
        lrs = np.logspace(-4, -1, 20).tolist()
        losses = [1.0 / (lr * 100.0) + 0.1 for lr in lrs]
        fig, suggested_lr = plot_lr_find_results(
            lrs,
            losses,
            smoothing=0.05,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        assert np.isfinite(suggested_lr)
        plt.close(fig)
