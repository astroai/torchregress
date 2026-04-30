import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from torchregress.viz.monitoring import (
    _add_early_stopping_annotations,
    _find_early_stopping_point,
    _plot_early_stopping_markers,
    plot_early_stopping,
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
