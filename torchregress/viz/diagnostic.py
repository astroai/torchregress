"""
Diagnostic plotting utilities for regression and uncertainty quantification.
"""

from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from torchregress.metrics.calibration import expected_calibration_error
from torchregress.metrics.utils import convert_to_tensor, validate_inputs
from torchregress.viz.utils import add_annotations, add_identity_line, add_zero_line


def plot_reliability_diagram(
    y_pred_quantiles: Dict[float, Union[torch.Tensor, np.ndarray]],
    y_true: Union[torch.Tensor, np.ndarray],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Reliability Diagram",
    show_diagonal: bool = True,
    show_grid: bool = True,
    color: str = "blue",
    marker: str = "o",
    markersize: int = 8,
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Plot a reliability diagram for quantile predictions.

    A reliability diagram plots the predicted quantiles against the
    empirical proportions of observations below each quantile.

    Args:
        y_pred_quantiles: Dictionary mapping quantile levels to predictions
        y_true: Ground truth values
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        show_diagonal: Whether to show diagonal line (perfect calibration)
        show_grid: Whether to show grid
        color: Line color
        marker: Marker style
        markersize: Size of markers
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    # Calculate calibration metrics
    calibration_metrics = expected_calibration_error(
        y_pred_quantiles, y_true, return_diagnostics=True
    )

    # Extract data for plotting
    expected = calibration_metrics["expected_proportions"]
    actual = calibration_metrics["actual_proportions"]
    mace = calibration_metrics["mean_absolute_calibration_error"]

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot data points
    ax.plot(
        expected,
        actual,
        marker=marker,
        markersize=markersize,
        linestyle="-",
        color=color,
        label=f"MACE: {mace:.4f}",
    )

    # Show diagonal line for perfect calibration
    if show_diagonal:
        add_identity_line(ax, label="Perfectly Calibrated")

    ax.set_xlabel("Expected proportion")
    ax.set_ylabel("Observed proportion")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title(title)
    ax.legend(loc="best")

    if show_grid:
        ax.grid(True, alpha=0.3)

    if return_figure:
        return fig
    elif ax is None:  # Only show if we created the figure here
        plt.tight_layout()
        plt.show()

    return None


def plot_residuals(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Residual Plot",
    xlabel: str = "Predicted Values",
    ylabel: str = "Residuals",
    color: str = "blue",
    alpha: float = 0.6,
    show_zero_line: bool = True,
    show_trend: bool = True,
    trend_color: str = "red",
    clip_outliers: bool = False,
    clip_percentile: float = 99.0,
    downsample: bool = False,
    max_points: int = 5000,
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Plot residuals (y_true - y_pred) against predicted values.

    This plot is useful for diagnosing heteroscedasticity, non-linearity,
    and other issues in regression models.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        color: Color for scatter points
        alpha: Transparency of scatter points
        show_zero_line: Whether to show a horizontal line at y=0
        show_trend: Whether to show a trend line for residuals
        trend_color: Color for trend line
        clip_outliers: Whether to clip outliers for better visualization
        clip_percentile: Percentile to clip outliers (if clip_outliers is True)
        downsample: Whether to downsample large datasets
        max_points: Maximum number of points to plot if downsampling
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    y_pred = convert_to_tensor(y_pred).detach().cpu().numpy()
    y_true = convert_to_tensor(y_true).detach().cpu().numpy()
    validate_inputs(torch.tensor(y_pred), torch.tensor(y_true))

    # Flatten arrays if multi-dimensional
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # Calculate residuals
    residuals = y_true - y_pred

    # Handle NaN and Inf values
    valid_idx = np.isfinite(residuals) & np.isfinite(y_pred)
    if not np.all(valid_idx):
        print(f"Warning: {np.sum(~valid_idx)} non-finite values removed from residual plot")
        y_pred = y_pred[valid_idx]
        residuals = residuals[valid_idx]

    # Clip outliers if requested
    if clip_outliers and len(residuals) > 0:
        lower = np.percentile(residuals, 100 - clip_percentile)
        upper = np.percentile(residuals, clip_percentile)
        clip_idx = (residuals >= lower) & (residuals <= upper)
        y_pred = y_pred[clip_idx]
        residuals = residuals[clip_idx]

    # Downsample large datasets if requested
    if downsample and len(residuals) > max_points:
        idx = np.random.choice(len(residuals), max_points, replace=False)
        y_pred = y_pred[idx]
        residuals = residuals[idx]

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Scatter plot of residuals vs predictions
    if len(y_pred) > 1000:
        # Use hexbin for large datasets
        hb = ax.hexbin(y_pred, residuals, gridsize=50, cmap="Blues", bins="log")
        plt.colorbar(hb, ax=ax, label="log10(count)")
    else:
        ax.scatter(y_pred, residuals, alpha=alpha, color=color, edgecolor="none")

    # Add horizontal line at y=0 for reference
    if show_zero_line:
        add_zero_line(ax, axis="y", label="Perfect Prediction")

    # Add trend line using polynomial fit
    if show_trend and len(y_pred) > 1:
        try:
            z = np.polyfit(y_pred, residuals, 1)
            p = np.poly1d(z)
            ax.plot(
                np.sort(y_pred),
                p(np.sort(y_pred)),
                color=trend_color,
                linestyle="--",
                label=f"Trend: y={z[0]:.3f}x{z[1]:+.3f}",
            )
        except Exception as e:
            print(f"Warning: Could not fit trend line: {e}")

    # Show legend if there are labeled elements
    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if return_figure:
        return fig
    elif ax is None:  # Only show if we created the figure here
        plt.tight_layout()
        plt.show()

    return None


def plot_prediction_intervals(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_lower: Union[torch.Tensor, np.ndarray],
    y_upper: Union[torch.Tensor, np.ndarray],
    y_true: Optional[Union[torch.Tensor, np.ndarray]] = None,
    x: Optional[Union[torch.Tensor, np.ndarray]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Prediction Intervals",
    xlabel: str = "Sample Index",
    ylabel: str = "Value",
    color_pred: str = "blue",
    color_interval: str = "lightblue",
    color_true: str = "black",
    alpha: float = 0.3,
    sorted_by_pred: bool = False,
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Plot prediction intervals along with predicted and true values.

    Args:
        y_pred: Predicted values (mean or median predictions)
        y_lower: Lower bounds of prediction intervals
        y_upper: Upper bounds of prediction intervals
        y_true: Optional ground truth values
        x: Optional x-coordinates (e.g., time points or feature values)
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        color_pred: Color for predicted values line
        color_interval: Color for prediction interval
        color_true: Color for ground truth values
        alpha: Transparency of prediction interval
        sorted_by_pred: If True, sort all values by predicted values
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    y_pred = convert_to_tensor(y_pred).detach().cpu().numpy()
    y_lower = convert_to_tensor(y_lower).detach().cpu().numpy()
    y_upper = convert_to_tensor(y_upper).detach().cpu().numpy()

    # Flatten arrays if multi-dimensional
    y_pred = y_pred.reshape(-1)
    y_lower = y_lower.reshape(-1)
    y_upper = y_upper.reshape(-1)

    # Create x values if not provided
    if x is None:
        x = np.arange(len(y_pred))
    else:
        x = convert_to_tensor(x).detach().cpu().numpy().reshape(-1)

    # Process true values if provided
    if y_true is not None:
        y_true = convert_to_tensor(y_true).detach().cpu().numpy().reshape(-1)

    # Sort by predicted values if requested
    if sorted_by_pred:
        sort_idx = np.argsort(y_pred)
        y_pred = y_pred[sort_idx]
        y_lower = y_lower[sort_idx]
        y_upper = y_upper[sort_idx]
        x = x[sort_idx]
        if y_true is not None:
            y_true = y_true[sort_idx]

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot prediction intervals
    ax.fill_between(
        x, y_lower, y_upper, alpha=alpha, color=color_interval, label="Prediction Interval"
    )

    # Plot predicted values
    ax.plot(x, y_pred, color=color_pred, label="Predicted")

    # Plot true values if provided
    if y_true is not None:
        ax.scatter(x, y_true, color=color_true, s=10, alpha=0.6, label="True")

    # Calculate coverage if true values provided
    if y_true is not None:
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper)) * 100
        ax.set_title(f"{title} (Coverage: {coverage:.1f}%)")
    else:
        ax.set_title(title)

    # Add labels and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if return_figure:
        return fig
    elif ax is None:  # Only show if we created the figure here
        plt.tight_layout()
        plt.show()

    return None


def plot_qq_plot(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    figsize: Tuple[int, int] = (8, 8),
    title: str = "Q-Q Plot",
    xlabel: str = "Theoretical Quantiles",
    ylabel: str = "Sample Quantiles",
    color: str = "blue",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Create a quantile-quantile (Q-Q) plot to assess normality of residuals.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        color: Color for scatter points
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    y_pred = convert_to_tensor(y_pred).detach().cpu().numpy()
    y_true = convert_to_tensor(y_true).detach().cpu().numpy()

    # Calculate residuals
    residuals = (y_true - y_pred).flatten()

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Generate theoretical quantiles from standard normal distribution
    n = len(residuals)
    quantiles = np.arange(1, n + 1) / (n + 1)
    theoretical_quantiles = np.quantile(np.random.normal(0, 1, n), quantiles)

    # Standardize residuals
    std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    # Sort the standardized residuals
    std_residuals = np.sort(std_residuals)

    # Create Q-Q plot
    ax.scatter(theoretical_quantiles, std_residuals, color=color, alpha=0.6)

    # Add identity line
    add_identity_line(ax, label="Normal")

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if return_figure:
        return fig
    elif ax is None:  # Only show if we created the figure here
        plt.tight_layout()
        plt.show()

    return None


def plot_residual_histogram(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Residual Histogram",
    xlabel: str = "Residual Value",
    ylabel: str = "Frequency",
    show_kde: bool = True,
    color: str = "skyblue",
    kde_color: str = "navy",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Plot histogram of residuals with optional density curve.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        bins: Number of histogram bins
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        show_kde: Whether to show kernel density estimate
        color: Color for histogram bars
        kde_color: Color for density curve
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    y_pred = convert_to_tensor(y_pred).detach().cpu().numpy()
    y_true = convert_to_tensor(y_true).detach().cpu().numpy()

    # Calculate residuals
    residuals = (y_true - y_pred).flatten()

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot histogram
    _, bin_edges, _ = ax.hist(residuals, bins=bins, color=color, alpha=0.7, density=True)

    # Add KDE if requested
    if show_kde:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(residuals)
            x_range = np.linspace(min(residuals), max(residuals), 1000)
            ax.plot(x_range, kde(x_range), color=kde_color, linewidth=2, label="Density")

            # Add normal distribution for comparison
            from scipy.stats import norm

            mu, std = norm.fit(residuals)
            ax.plot(
                x_range,
                norm.pdf(x_range, mu, std),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Normal (μ={mu:.2f}, σ={std:.2f})",
            )
            ax.legend()
        except ImportError:
            pass  # Skip KDE if scipy not available

    # Add vertical line at zero
    add_zero_line(ax, axis="x")

    # Add statistics as text box
    mean = np.mean(residuals)
    std = np.std(residuals)
    annotations = {
        "Mean": mean,
        "Std": std,
    }
    add_annotations(ax, annotations, loc="upper right")

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if return_figure:
        return fig
    elif ax is None:  # Only show if we created the figure here
        plt.tight_layout()
        plt.show()

    return None


def plot_distribution_comparison(
    predicted_samples: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    n_samples_to_show: int = 5,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Predicted Distributions vs. True Values",
    xlabel: str = "Value",
    ylabel: str = "Density",
    plot_type: str = "kde",
    color_samples: str = "blue",
    color_true: str = "red",
    alpha: float = 0.1,
    credible_interval: float = 0.95,
    max_samples_per_plot: int = 1000,
    return_figure: bool = False,
) -> Optional[Figure]:
    """
    Plot predicted distributions against true values for selected samples.

    Args:
        predicted_samples: Samples from predicted distributions [n_samples, batch_size]
        y_true: Ground truth values [batch_size]
        n_samples_to_show: Number of examples to display
        figsize: Figure size (width, height)
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        plot_type: Plot type ('kde', 'histogram', or 'both')
        color_samples: Color for predicted distribution
        color_true: Color for true values
        alpha: Transparency for individual KDE curves
        credible_interval: Credible interval for shading (e.g., 0.95 for 95% CI)
        max_samples_per_plot: Maximum number of samples to use per distribution (for scalability)
        return_figure: If True, return figure object instead of displaying

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    pred_samples = convert_to_tensor(predicted_samples).detach().cpu().numpy()
    y_true = convert_to_tensor(y_true).detach().cpu().numpy().flatten()

    # Verify shape of inputs
    if len(pred_samples.shape) != 2:
        raise ValueError(
            f"predicted_samples should have shape [n_samples, batch_size], got {pred_samples.shape}"
        )

    # Select random indices to display
    batch_size = min(pred_samples.shape[1], len(y_true))
    if batch_size < n_samples_to_show:
        n_samples_to_show = batch_size
        print(f"Only {batch_size} samples available, showing all of them")

    indices = np.random.choice(batch_size, n_samples_to_show, replace=False)

    # Create figure with subplots
    fig, axes = plt.subplots(1, len(indices), figsize=figsize, sharey=True)

    # Handle single subplot case
    if len(indices) == 1:
        axes = [axes]

    # For KDE plotting
    try:
        from scipy.stats import gaussian_kde

        has_kde = True
    except ImportError:
        has_kde = False
        if plot_type == "kde" or plot_type == "both":
            print("scipy not available, using histogram instead of KDE")
            plot_type = "histogram"

    for i, idx in enumerate(indices):
        ax = axes[i]
        samples = pred_samples[:, idx]
        true_value = y_true[idx]

        # Remove any NaN or Inf values
        valid_samples = samples[np.isfinite(samples)]
        if len(valid_samples) < len(samples):
            removed = len(samples) - len(valid_samples)
            print(f"Warning: {removed} non-finite values removed from samples")

        # Make sure we have data to plot
        if len(valid_samples) == 0:
            ax.text(0.5, 0.5, "No valid samples", ha="center", va="center")
            continue

        samples = valid_samples

        # Compute credible interval
        lower_ci = np.percentile(samples, (1 - credible_interval) * 100 / 2)
        upper_ci = np.percentile(samples, 100 - (1 - credible_interval) * 100 / 2)
        mean_pred = np.mean(samples)
        median_pred = np.median(samples)

        # Plot the predicted distribution
        if plot_type in ["kde", "both"] and has_kde and len(samples) >= 3:
            try:
                # Create KDE
                kde = gaussian_kde(samples)
                x_range = np.linspace(min(samples), max(samples), 1000)
                ax.plot(x_range, kde(x_range), color=color_samples, linewidth=2, label="Predicted")

                # Add credible interval shading
                x_ci = x_range[(x_range >= lower_ci) & (x_range <= upper_ci)]
                if len(x_ci) > 0:
                    y_ci = kde(x_ci)
                    ax.fill_between(
                        x_ci,
                        0,
                        y_ci,
                        color=color_samples,
                        alpha=0.2,
                        label=f"{int(credible_interval*100)}% CI",
                    )

                # Add sample curves with low alpha for uncertainty visualization
                if len(samples) <= 100:  # Only if we have a reasonable number of samples
                    for sample in samples:
                        ax.axvline(x=sample, color=color_samples, alpha=alpha, linewidth=1)
            except Exception as e:
                print(f"Warning: KDE failed: {e}")
                # Fall back to histogram
                ax.hist(
                    samples,
                    bins=min(20, len(samples) // 5 + 1),
                    alpha=0.3,
                    color=color_samples,
                    density=True,
                )

        if plot_type in ["histogram", "both"] or (plot_type == "kde" and not has_kde):
            # Plot histogram
            ax.hist(
                samples,
                bins=min(20, len(samples) // 5 + 1),
                alpha=0.3,
                color=color_samples,
                density=True,
            )

        # Add true value as vertical line
        if np.isfinite(true_value):
            ax.axvline(x=true_value, color=color_true, linewidth=2, label="True Value")

            # Add text showing true value
            ax.text(
                true_value,
                0.01,
                f"{true_value:.2f}",
                color=color_true,
                ha="center",
                va="bottom",
                rotation=90,
                fontweight="bold",
            )

        # Add additional vertical lines for mean and median
        ax.axvline(x=mean_pred, color="green", linestyle="--", linewidth=1.5, label="Mean")
        ax.axvline(x=median_pred, color="purple", linestyle=":", linewidth=1.5, label="Median")

        # Add CI boundaries
        ax.axvline(
            x=lower_ci, color="orange", linestyle="-.", linewidth=1, alpha=0.7, label="CI Bounds"
        )
        ax.axvline(x=upper_ci, color="orange", linestyle="-.", linewidth=1, alpha=0.7)

        # Add metrics as text
        if np.isfinite(true_value):
            error = mean_pred - true_value
            in_ci = lower_ci <= true_value <= upper_ci
            ci_text = "in CI" if in_ci else "outside CI"

            annotations = {
                "Mean": f"{mean_pred:.2f}",
                "Median": f"{median_pred:.2f}",
                "Error": f"{error:.2f}",
                "True": f"{true_value:.2f} ({ci_text})",
            }
            add_annotations(ax, annotations, loc="upper left")

        # Add labels and legend
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(f"Sample {idx}")

    # Add common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=3,
        fancybox=True,
        shadow=True,
    )

    # Add overall title
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.2)

    if return_figure:
        return fig
    else:
        plt.show()

    return None


def plot_calibration_curve(
    y_pred_probs: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Calibration Curve",
    xlabel: str = "Mean Predicted Probability",
    ylabel: str = "Observed Frequency",
    color: str = "blue",
    add_hist: bool = True,
    hist_height: float = 0.1,
    hist_alpha: float = 0.3,
    hist_color: Optional[str] = None,
    return_figure: bool = False,
    return_diagnostics: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Union[Optional[Figure], Tuple[Optional[Figure], Dict[str, Any]]]:
    """
    Plot calibration curve (reliability diagram) for probabilistic predictions.

    Args:
        y_pred_probs: Predicted probabilities [batch_size]
        y_true: Binary ground truth values (0 or 1) [batch_size]
        n_bins: Number of bins
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        color: Color for the calibration curve
        add_hist: Whether to add histogram of prediction distribution at bottom
        hist_height: Maximum height of histogram bars as proportion of plot
        hist_alpha: Alpha transparency for histogram
        hist_color: Color for histogram (defaults to same as curve color)
        return_figure: If True, return figure object instead of displaying
        return_diagnostics: If True, return dictionary with calibration metrics
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
        If return_diagnostics=True, returns (figure, diagnostics_dict) tuple
    """
    y_pred_probs = convert_to_tensor(y_pred_probs).detach().cpu().numpy().flatten()
    y_true = convert_to_tensor(y_true).detach().cpu().numpy().flatten()

    # Remove NaN or Inf values
    valid_idx = np.isfinite(y_pred_probs) & np.isfinite(y_true)
    if not np.all(valid_idx):
        print(f"Warning: {np.sum(~valid_idx)} non-finite values removed from calibration data")
        y_true = y_true[valid_idx]
        y_pred_probs = y_pred_probs[valid_idx]

    # Ensure valid prediction probabilities
    if np.min(y_pred_probs) < 0 or np.max(y_pred_probs) > 1:
        min_prob = np.min(y_pred_probs)
        max_prob = np.max(y_pred_probs)
        print(
            f"Warning: Predicted probabilities outside [0, 1] range: "
            f"min={min_prob}, max={max_prob}"
        )
        y_pred_probs = np.clip(y_pred_probs, 0, 1)

    # Create bins and find bin edges
    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binids = np.digitize(y_pred_probs, bins) - 1

    # Calculate mean predicted probability and observed frequency in each bin
    bin_sums = np.bincount(binids, weights=y_pred_probs, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    bin_counts = np.bincount(binids, minlength=n_bins)

    # Avoid division by zero
    nonzero = bin_counts > 0

    # Calculate mean predicted probability and observed frequency
    prob_true = np.zeros(len(bins) - 1)
    prob_pred = np.zeros(len(bins) - 1)
    prob_true[nonzero] = bin_true[nonzero] / bin_counts[nonzero]
    prob_pred[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot calibration curve
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color=color, label="Calibration Curve")

    # Add diagonal reference line
    add_identity_line(ax, label="Perfectly Calibrated")

    # Calculate calibration metrics
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    rmsce = np.sqrt(np.mean((prob_true - prob_pred) ** 2))
    max_calib_error = np.max(np.abs(prob_true - prob_pred))

    # Add histogram of predicted probabilities as a barplot at the bottom
    if add_hist:
        # Use same color as calibration curve if not specified
        if hist_color is None:
            hist_color = color

        # Calculate histogram heights
        hist = np.bincount(binids, minlength=len(bins) - 1) / len(binids)
        scaled_hist = hist * hist_height

        # Create histogram bars
        for i in range(len(scaled_hist)):
            ax.bar(
                bin_centers[i],
                scaled_hist[i],
                width=(1 / n_bins),
                bottom=-scaled_hist[i],
                align="center",
                alpha=hist_alpha,
                color=hist_color,
                label="Prediction Dist." if i == 0 else None,
            )

        # Adjust ylimit to accommodate the histograms at the bottom
        ax.set_ylim(-hist_height, 1.0)
    else:
        ax.set_ylim(0, 1.0)

    # Set x-axis range
    ax.set_xlim(0, 1.0)

    # Add text box with calibration metrics
    annotations = {
        "Mean Calibration Error": calibration_error,
        "RMSCE": rmsce,
        "Max Calibration Error": max_calib_error,
    }
    add_annotations(ax, annotations, loc="upper left")

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Create diagnostics dictionary
    diagnostics = {
        "mean_calibration_error": calibration_error,
        "root_mean_squared_calibration_error": rmsce,
        "max_calibration_error": max_calib_error,
        "bin_counts": bin_counts,
        "predicted_probs": prob_pred,
        "observed_freqs": prob_true,
    }

    if return_figure and return_diagnostics:
        return fig, diagnostics
    elif return_diagnostics:
        if ax is None:
            plt.tight_layout()
            plt.show()
        return diagnostics
    elif return_figure:
        return fig
    elif ax is None:  # Only show if we created the figure here
        plt.tight_layout()
        plt.show()

    return None
