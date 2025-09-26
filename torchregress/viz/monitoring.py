"""
Training monitoring plots for regression models.

This module provides visualization utilities for monitoring model training
progress, validation metrics, and early stopping.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torchregress.viz.utils import add_annotations


def plot_learning_curves(
    train_history: Dict[str, List[float]],
    val_history: Optional[Dict[str, List[float]]] = None,
    metrics_to_plot: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    n_cols: int = 2,
    smoothing: float = 0.0,
    title: str = "Learning Curves",
    use_grid: bool = True,
    log_scale: Optional[List[str]] = None,
    color_train: str = "blue",
    color_val: str = "orange",
    return_figure: bool = False,
    scientific_notation: bool = True,
    show_annotations: bool = True,
) -> Optional[Figure]:
    """
    Plot learning curves from training and validation metrics history.

    Args:
        train_history: Dictionary mapping metric names to lists of values (training)
        val_history: Dictionary mapping metric names to lists of validation values
        metrics_to_plot: List of metric names to plot (defaults to all metrics)
        figsize: Figure size (width, height)
        n_cols: Number of columns in the grid
        smoothing: Smoothing factor for the curves (0.0 = no smoothing, 0.9 = high smoothing)
        title: Overall figure title
        use_grid: Whether to add grid to the plots
        log_scale: List of metrics to display with log scale
        color_train: Color for training curves
        color_val: Color for validation curves
        return_figure: If True, return figure object instead of displaying
        scientific_notation: Whether to use scientific notation for very small/large numbers
        show_annotations: Whether to show best value annotations

    Returns:
        Matplotlib Figure object if return_figure=True
    """
    # Check for empty data
    if not train_history or all(len(v) == 0 for v in train_history.values()):
        print("Warning: Empty training history, cannot create learning curves")
        if return_figure:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No training data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig
        return None

    # Determine which metrics to plot
    if metrics_to_plot is None:
        metrics_to_plot = list(train_history.keys())

    # Filter metrics to those that exist in the history
    metrics_to_plot = [m for m in metrics_to_plot if m in train_history]

    if not metrics_to_plot:
        raise ValueError("No valid metrics found to plot")

    # Create figure and axes
    n_metrics = len(metrics_to_plot)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single subplot case
    if n_metrics == 1:
        axes = np.array([axes])

    # Ensure axes is always a 1D array
    axes = axes.flatten()

    # Function to apply simple exponential smoothing
    def smooth(values, alpha=0.1):
        if alpha <= 0:
            return values
        smoothed = []
        last = values[0]
        for point in values:
            smoothed_val = alpha * last + (1 - alpha) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    # Format number for display
    def format_number(x):
        if scientific_notation:
            if abs(x) < 0.001 or abs(x) >= 1000:
                return f"{x:.2e}"

        # Standard formatting based on magnitude
        if abs(x) < 0.01:
            return f"{x:.4f}"
        elif abs(x) < 0.1:
            return f"{x:.3f}"
        elif abs(x) < 1:
            return f"{x:.2f}"
        elif abs(x) < 10:
            return f"{x:.1f}"
        else:
            return f"{x:.0f}" if x == int(x) else f"{x:.1f}"

    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]

        # Handle NaN or Inf values
        train_values = np.array(train_history[metric])
        valid_idx = np.isfinite(train_values)
        if not np.all(valid_idx):
            print(f"Warning: {np.sum(~valid_idx)} non-finite values removed from training {metric}")
            epochs = np.arange(1, len(train_values) + 1)
            train_values = train_values[valid_idx]
            epochs = epochs[valid_idx]
        else:
            epochs = np.arange(1, len(train_values) + 1)

        # Apply smoothing if requested
        if smoothing > 0 and len(train_values) > 1:
            train_smoothed = smooth(train_values, smoothing)
        else:
            train_smoothed = train_values

        # Plot training curve
        ax.plot(epochs, train_smoothed, label=f"Train {metric}", color=color_train, marker=None)

        # Plot validation curve if available
        if val_history is not None and metric in val_history:
            val_values = np.array(val_history[metric])
            val_epochs = np.arange(1, len(val_values) + 1)

            # Handle NaN or Inf values
            val_valid_idx = np.isfinite(val_values)
            if not np.all(val_valid_idx):
                print(
                    f"Warning: {np.sum(~val_valid_idx)} non-finite values removed from validation {metric}"
                )
                val_values = val_values[val_valid_idx]
                val_epochs = val_epochs[val_valid_idx]

            if smoothing > 0 and len(val_values) > 1:
                val_smoothed = smooth(val_values, smoothing)
            else:
                val_smoothed = val_values

            ax.plot(
                val_epochs,
                val_smoothed,
                label=f"Val {metric}",
                color=color_val,
                marker="o",
                markersize=4,
            )

        # Set plot styling
        ax.set_title(f"{metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend(loc="best")

        # Add grid
        if use_grid:
            ax.grid(True, alpha=0.3)

        # Use log scale if specified
        if log_scale is not None and metric in log_scale:
            ax.set_yscale("log")

        # Find best values and annotate them
        if show_annotations and len(train_values) > 0:
            is_loss = any(
                term in metric.lower() for term in ["loss", "error", "mae", "mse", "rmse"]
            )

            best_train_idx = np.argmin(train_values) if is_loss else np.argmax(train_values)
            best_train = train_values[best_train_idx]
            annotations = {"Best train": format_number(best_train)}

            if val_history is not None and metric in val_history and len(val_values) > 0:
                val_best_idx = np.argmin(val_values) if is_loss else np.argmax(val_values)
                best_val = val_values[val_best_idx]
                best_epoch = val_epochs[val_best_idx]
                annotations["Best val"] = format_number(best_val)
                annotations["Best epoch"] = int(best_epoch)

            add_annotations(ax, annotations, loc="upper right")

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if return_figure:
        return fig
    else:
        plt.show()
        return None


def plot_validation_metrics(
    epochs: List[int],
    metrics: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 6),
    n_cols: int = 3,
    title: str = "Validation Metrics",
    highlight_best: bool = True,
    error_bars: Optional[Dict[str, List[float]]] = None,
    return_figure: bool = False,
) -> Optional[Figure]:
    """
    Plot validation metrics across epochs with optional error bars.

    Args:
        epochs: List of epoch numbers
        metrics: Dictionary mapping metric names to lists of values
        figsize: Figure size (width, height)
        n_cols: Number of columns in the grid
        title: Overall figure title
        highlight_best: Whether to highlight the best value for each metric
        error_bars: Dictionary mapping metric names to error values (std dev)
        return_figure: If True, return figure object instead of displaying

    Returns:
        Matplotlib Figure object if return_figure=True
    """
    # Create figure and axes
    n_metrics = len(metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single subplot case
    if n_metrics == 1:
        axes = np.array([axes])

    # Ensure axes is always a 1D array
    axes = axes.flatten()

    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]

        # Get error bars if available
        yerr = None
        if error_bars is not None and metric_name in error_bars:
            yerr = error_bars[metric_name]

        # Plot metric values
        ax.errorbar(epochs, values, yerr=yerr, marker="o", linestyle="-")

        # Highlight best value if requested
        if highlight_best:
            # Determine if lower or higher is better
            is_loss = "loss" in metric_name.lower() or "error" in metric_name.lower()

            if is_loss:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)

            best_epoch = epochs[best_idx]
            best_value = values[best_idx]

            # Highlight best point
            ax.scatter(
                [best_epoch],
                [best_value],
                color="red",
                s=100,
                marker="*",
                label=f"Best: {best_value:.4f} (epoch {best_epoch})",
            )

            # Add text annotation
            ax.annotate(
                f"{best_value:.4f}",
                xy=(best_epoch, best_value),
                xytext=(10, 0),
                textcoords="offset points",
                color="red",
            )

        # Set labels and title
        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)

        # Add legend if best value was highlighted
        if highlight_best:
            ax.legend(loc="best")

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if return_figure:
        return fig
    else:
        plt.show()
        return None


def plot_early_stopping(
    train_losses: List[float],
    val_losses: List[float],
    patience: int = 10,
    delta: float = 0.0,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Early Stopping Analysis",
    return_figure: bool = False,
) -> Optional[Figure]:
    """
    Visualize early stopping behavior.

    Args:
        train_losses: List of training loss values
        val_losses: List of validation loss values
        patience: Patience parameter used
        delta: Minimum change to qualify as improvement
        figsize: Figure size (width, height)
        title: Plot title
        return_figure: If True, return figure object instead of displaying

    Returns:
        Matplotlib Figure object if return_figure=True
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot loss curves
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, label="Training Loss", color="blue")
    ax.plot(epochs, val_losses, label="Validation Loss", color="orange")

    # Detect early stopping point
    best_val_loss = float("inf")
    best_epoch = 0
    counter = 0
    stop_epoch = len(val_losses)

    for i, val_loss in enumerate(val_losses):
        if val_loss < best_val_loss - delta:
            best_val_loss = val_loss
            best_epoch = i + 1
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                stop_epoch = i + 1
                break

    # Highlight best and stopping points
    ax.axvline(
        x=best_epoch, color="green", linestyle="--", label=f"Best Model (epoch {best_epoch})"
    )

    if stop_epoch < len(val_losses):
        ax.axvline(
            x=stop_epoch, color="red", linestyle="-", label=f"Early Stop (epoch {stop_epoch})"
        )

    # Fill the waiting period
    waiting_start = best_epoch
    waiting_end = min(stop_epoch, len(val_losses))
    ax.axvspan(
        waiting_start,
        waiting_end,
        alpha=0.2,
        color="red",
        label=f"Patience Window ({patience} epochs)",
    )

    # Set labels, title and legend
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate(
        f"Best: {best_val_loss:.4f}",
        xy=(best_epoch, best_val_loss),
        xytext=(10, -20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="green"),
        color="green",
    )

    annotations = {
        "Best epoch": best_epoch,
        "Best val loss": best_val_loss,
        "Patience": patience,
        "Delta": delta,
    }

    if stop_epoch < len(val_losses):
        annotations["Stopped at"] = stop_epoch
        annotations["Training completed"] = f"{stop_epoch}/{len(val_losses)} epochs"

    add_annotations(ax, annotations, loc="upper left")

    plt.tight_layout()

    if return_figure:
        return fig
    else:
        plt.show()
        return None


def plot_lr_find_results(
    learning_rates: List[float],
    losses: List[float],
    smoothing: float = 0.05,
    suggestion_method: str = "valley",
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Learning Rate Finder Results",
    return_figure: bool = False,
) -> Optional[Tuple[Figure, float]]:
    """
    Plot learning rate finder results and suggest optimal learning rate.

    Args:
        learning_rates: List of learning rates tested
        losses: Corresponding loss values
        smoothing: Smoothing factor for the loss curve
        suggestion_method: Method to suggest learning rate ('valley', 'steepest', or 'minimum')
        figsize: Figure size (width, height)
        title: Plot title
        return_figure: If True, return (figure, suggested_lr) instead of displaying

    Returns:
        Tuple of (Figure, suggested_lr) if return_figure=True
    """
    # Convert to numpy arrays
    lrs = np.array(learning_rates)
    losses = np.array(losses)

    # Apply smoothing using moving average
    if smoothing > 0:
        weights = np.ones(int(1 / smoothing))
        weights = weights / weights.sum()
        smooth_losses = np.convolve(losses, weights, mode="same")
        # Fix boundaries
        smooth_losses[: int(1 / smoothing)] = losses[: int(1 / smoothing)]
        smooth_losses[-int(1 / smoothing) :] = losses[-int(1 / smoothing) :]
    else:
        smooth_losses = losses

    # Remove inf/nan values
    valid_idx = np.isfinite(smooth_losses)
    lrs = lrs[valid_idx]
    losses = losses[valid_idx]
    smooth_losses = smooth_losses[valid_idx]

    # Skip data points where loss is too high at the beginning
    start_idx = 0
    for i in range(len(losses) - 1):
        if i > 10 and losses[i] > 3 * losses[i + 1]:
            start_idx = i + 1
        else:
            break

    lrs = lrs[start_idx:]
    losses = losses[start_idx:]
    smooth_losses = smooth_losses[start_idx:]

    # Calculate gradients
    gradients = np.gradient(smooth_losses, np.log10(lrs))

    # Suggest learning rate
    suggested_lr = None
    suggested_idx = None

    if suggestion_method == "valley":
        # Find point where gradient starts to increase sharply
        for i in range(len(gradients) - 1):
            if i > 2 and gradients[i] < 0 and gradients[i + 1] > 0:
                suggested_idx = i
                break

        if suggested_idx is None and len(lrs) > 0:
            # Fallback to minimum point
            suggested_idx = np.argmin(smooth_losses)

    elif suggestion_method == "steepest":
        # Find steepest downward slope
        min_gradient_idx = np.argmin(gradients)
        if min_gradient_idx > 0 and min_gradient_idx < len(lrs) - 1:
            suggested_idx = min_gradient_idx

    elif suggestion_method == "minimum":
        # Simply use the minimum point
        suggested_idx = np.argmin(smooth_losses)

    if suggested_idx is not None:
        suggested_lr = lrs[suggested_idx]

        # Suggest slightly lower LR for better generalization
        if suggested_idx > 0:
            suggested_lr = suggested_lr * 0.1

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot raw and smoothed losses
    ax.plot(lrs, losses, "o", alpha=0.4, label="Raw loss")
    ax.plot(lrs, smooth_losses, "-", label="Smoothed loss")

    # Mark suggested learning rate if found
    if suggested_lr is not None:
        ax.axvline(
            x=suggested_lr, color="red", linestyle="--", label=f"Suggested LR: {suggested_lr:.1e}"
        )

    # Set scales, labels, and title
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add annotations
    if len(lrs) > 0:
        min_lr = lrs[0]
        max_lr = lrs[-1]
        min_loss = np.min(losses)

        annotations = {
            "Min LR": f"{min_lr:.1e}",
            "Max LR": f"{max_lr:.1e}",
            "Min Loss": f"{min_loss:.4f}",
        }

        if suggested_lr is not None:
            annotations["Suggested LR"] = f"{suggested_lr:.1e}"

        add_annotations(ax, annotations, loc="upper right")

    plt.tight_layout()

    if return_figure:
        return fig, suggested_lr
    else:
        plt.show()
        return None
