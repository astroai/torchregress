"""
Diagnostic plotting utilities for regression and uncertainty quantification.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union, cast

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
        fig = cast(Figure, ax.figure)

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
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))
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


def _filter_residual_data(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    clip_outliers: bool = False,
    clip_percentile: float = 99.0,
    downsample: bool = False,
    max_points: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to filter, clip, and downsample residual data."""
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

    return y_pred, residuals


def _plot_residual_scatter(
    ax: plt.Axes,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    alpha: float,
    color: str,
) -> None:
    """Helper to plot residual scatter or hexbin for large datasets."""
    if len(y_pred) > 1000:
        # Use hexbin for large datasets
        hb = ax.hexbin(y_pred, residuals, gridsize=50, cmap="Blues", bins="log")
        plt.colorbar(hb, ax=ax, label="log10(count)")
    else:
        ax.scatter(y_pred, residuals, alpha=alpha, color=color, edgecolor="none")


def _compute_uncertainty_stats(
    y_pred: np.ndarray,
    y_pred_std: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Helper to compute absolute errors and Spearman correlation."""
    from scipy import stats

    abs_errors = np.abs(y_pred - y_true)
    correlation, p_value = stats.spearmanr(y_pred_std, abs_errors)
    return abs_errors, correlation, p_value


def _subsample_scatter_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to subsample data points for plotting performance."""
    if len(x_data) > max_points:
        idx = np.random.choice(len(x_data), max_points, replace=False)
        return x_data[idx], y_data[idx]
    return x_data, y_data


def _add_uncertainty_trend(
    ax: plt.Axes,
    y_pred_std: np.ndarray,
    abs_errors: np.ndarray,
    correlation: float,
    show_correlation: bool,
) -> None:
    """Helper to fit and plot a linear trend line for uncertainty vs error."""
    if len(y_pred_std) > 1:
        z = np.polyfit(y_pred_std, abs_errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(y_pred_std.min(), y_pred_std.max(), 100)
        label = f"Trend (ρ={correlation:.3f})" if show_correlation else "Trend"
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=label)
        ax.legend()


def _add_residual_trend(
    ax: plt.Axes,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    trend_color: str,
) -> None:
    """Helper to fit and plot a polynomial trend line."""
    if len(y_pred) > 1:
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


def plot_residuals(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred_std: Optional[Union[torch.Tensor, np.ndarray]] = None,
    y_true_std: Optional[Union[torch.Tensor, np.ndarray]] = None,
    censoring_indicator: Optional[Union[torch.Tensor, np.ndarray]] = None,
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
    and other issues in regression models. It automatically adapts to show
    standardized residuals if prediction or target uncertainties are provided,
    and visualizes censored observations if censoring indicators are present.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        y_pred_std: Predicted standard deviation (aleatoric/predictive uncertainty)
        y_true_std: Ground-truth target uncertainty (measurement noise)
        censoring_indicator: Binary indicators where 1 = observed, 0 = censored (right-censored)
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
    y_pred_np: np.ndarray = convert_to_tensor(y_pred).detach().cpu().numpy().flatten()
    y_true_np: np.ndarray = convert_to_tensor(y_true).detach().cpu().numpy().flatten()
    validate_inputs(torch.tensor(y_pred_np), torch.tensor(y_true_np))

    # Compute total uncertainty for standardization if stds are provided
    std_total: Optional[np.ndarray] = None
    if y_pred_std is not None or y_true_std is not None:
        var_pred: np.ndarray = np.array(0.0)
        var_true: np.ndarray = np.array(0.0)
        if y_pred_std is not None:
            var_pred = convert_to_tensor(y_pred_std).detach().cpu().numpy().flatten() ** 2
        if y_true_std is not None:
            var_true = convert_to_tensor(y_true_std).detach().cpu().numpy().flatten() ** 2
        std_total = np.sqrt(var_pred + var_true)
        # Avoid division by zero
        std_total = np.where(std_total == 0.0, 1e-8, std_total)

    if std_total is not None:
        residuals_np: np.ndarray = (y_true_np - y_pred_np) / std_total
        if ylabel == "Residuals":
            ylabel = "Standardized Residuals"
    else:
        residuals_np = y_true_np - y_pred_np

    if censoring_indicator is not None:
        censoring_indicator = (
            convert_to_tensor(censoring_indicator).detach().cpu().numpy().flatten()
        )

    # Filter and clean datasets aligned
    valid_idx = np.isfinite(residuals_np) & np.isfinite(y_pred_np)
    if std_total is not None:
        valid_idx = valid_idx & np.isfinite(std_total)

    y_pred_np = y_pred_np[valid_idx]
    residuals_np = residuals_np[valid_idx]
    if std_total is not None:
        std_total = std_total[valid_idx]
    if censoring_indicator is not None:
        censoring_indicator = censoring_indicator[valid_idx]

    if clip_outliers and len(residuals_np) > 0:
        lower = np.percentile(residuals_np, 100 - clip_percentile)
        upper = np.percentile(residuals_np, clip_percentile)
        clip_idx = (residuals_np >= lower) & (residuals_np <= upper)
        y_pred_np = y_pred_np[clip_idx]
        residuals_np = residuals_np[clip_idx]
        if std_total is not None:
            std_total = std_total[clip_idx]
        if censoring_indicator is not None:
            censoring_indicator = censoring_indicator[clip_idx]

    if downsample and len(residuals_np) > max_points:
        idx = np.random.choice(len(residuals_np), max_points, replace=False)
        y_pred_np = y_pred_np[idx]
        residuals_np = residuals_np[idx]
        if std_total is not None:
            std_total = std_total[idx]
        if censoring_indicator is not None:
            censoring_indicator = censoring_indicator[idx]

    created_fig = ax is None

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Plot points
    if censoring_indicator is not None:
        obs_idx = censoring_indicator == 1
        cens_idx = censoring_indicator == 0
        if np.sum(obs_idx) > 0:
            ax.scatter(
                y_pred_np[obs_idx],
                residuals_np[obs_idx],
                alpha=alpha,
                color=color,
                marker="o",
                edgecolor="none",
                label="Observed",
            )
        if np.sum(cens_idx) > 0:
            ax.scatter(
                y_pred_np[cens_idx],
                residuals_np[cens_idx],
                alpha=alpha,
                color="orange",
                marker="^",
                edgecolor="none",
                label="Censored (Right)",
            )
    else:
        _plot_residual_scatter(ax, y_pred_np, residuals_np, alpha, color)

    # Add reference bands for standardized residuals
    if std_total is not None:
        ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5, label="±1σ")
        ax.axhline(y=-1, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=2, color="gray", linestyle="-.", alpha=0.5, label="±2σ")
        ax.axhline(y=-2, color="gray", linestyle="-.", alpha=0.5)

    # Add horizontal line at y=0 for reference
    if show_zero_line:
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.8, label="Zero Bias")

    # Add trend line using polynomial fit
    if show_trend:
        _add_residual_trend(ax, y_pred_np, residuals_np, trend_color)

    # Show legend if there are labeled elements
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")

    # Add labels and title
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if return_figure:
        return fig
    elif created_fig:  # Only show if we created the figure here
        plt.tight_layout()
        plt.show()

    return None


def _prepare_interval_data(
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    x: Optional[np.ndarray],
    y_true: Optional[np.ndarray],
    sorted_by_pred: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Prepare and sort data for prediction interval plots."""
    # Flatten arrays
    y_pred = y_pred.reshape(-1)
    y_lower = y_lower.reshape(-1)
    y_upper = y_upper.reshape(-1)

    if x is None:
        x = np.arange(len(y_pred))
    else:
        x = x.reshape(-1)

    if y_true is not None:
        y_true = y_true.reshape(-1)

    if sorted_by_pred:
        sort_idx = np.argsort(y_pred)
        y_pred = y_pred[sort_idx]
        y_lower = y_lower[sort_idx]
        y_upper = y_upper[sort_idx]
        x = x[sort_idx]
        if y_true is not None:
            y_true = y_true[sort_idx]

    return x, y_pred, y_lower, y_upper, y_true


def _add_interval_elements(
    ax: plt.Axes,
    x: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    y_true: Optional[np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
    color_pred: str,
    color_interval: str,
    color_true: str,
    alpha: float,
) -> None:
    """Add lines, fill, scatter points, and formatting to prediction interval axes."""
    ax.fill_between(
        x, y_lower, y_upper, alpha=alpha, color=color_interval, label="Prediction Interval"
    )
    ax.plot(x, y_pred, color=color_pred, label="Predicted")

    if y_true is not None:
        ax.scatter(x, y_true, color=color_true, s=10, alpha=0.6, label="True")
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper)) * 100
        ax.set_title(f"{title} (Coverage: {coverage:.1f}%)")
    else:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


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
    y_pred_np = convert_to_tensor(y_pred).detach().cpu().numpy()
    y_lower_np = convert_to_tensor(y_lower).detach().cpu().numpy()
    y_upper_np = convert_to_tensor(y_upper).detach().cpu().numpy()

    x_np = None
    if x is not None:
        x_np = convert_to_tensor(x).detach().cpu().numpy()

    y_true_np = None
    if y_true is not None:
        y_true_np = convert_to_tensor(y_true).detach().cpu().numpy()

    x_plt, y_pred_plt, y_lower_plt, y_upper_plt, y_true_plt = _prepare_interval_data(
        y_pred=y_pred_np,
        y_lower=y_lower_np,
        y_upper=y_upper_np,
        x=x_np,
        y_true=y_true_np,
        sorted_by_pred=sorted_by_pred,
    )

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    _add_interval_elements(
        ax=ax,
        x=x_plt,
        y_pred=y_pred_plt,
        y_lower=y_lower_plt,
        y_upper=y_upper_plt,
        y_true=y_true_plt,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        color_pred=color_pred,
        color_interval=color_interval,
        color_true=color_true,
        alpha=alpha,
    )

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

    from scipy import stats  # ponytail: for norm.ppf theoretical quantiles

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Generate theoretical quantiles from standard normal distribution
    n = len(residuals)
    quantiles = np.arange(1, n + 1) / (n + 1)
    theoretical_quantiles = stats.norm.ppf(quantiles)

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


def _add_residual_density_curves(ax: plt.Axes, residuals: np.ndarray, kde_color: str) -> None:
    """Add KDE and normal distribution fit to residual histogram."""
    try:
        from scipy.stats import gaussian_kde  # type: ignore[import-untyped]

        kde = gaussian_kde(residuals)
        x_range = np.linspace(min(residuals), max(residuals), 1000)
        ax.plot(x_range, kde(x_range), color=kde_color, linewidth=2, label="Density")

        # Add normal distribution for comparison
        from scipy.stats import norm  # type: ignore[import-untyped]

        mu, std = norm.fit(residuals)
        ax.plot(
            x_range,
            norm.pdf(x_range, mu, std),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Normal (μ={mu:.2f}, σ={std:.2f})",
        )
    except ImportError:
        pass  # Skip KDE if scipy not available


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
        fig = cast(Figure, ax.figure)

    # Plot histogram
    _, bin_edges, _ = ax.hist(residuals, bins=bins, color=color, alpha=0.7, density=True)

    # Add KDE if requested
    if show_kde:
        _add_residual_density_curves(ax, residuals, kde_color)

    # Add vertical line at zero
    add_zero_line(ax, axis="x", label="Perfect Prediction")

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

    # Show legend if there are labeled elements
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")

    if return_figure:
        return fig
    elif ax is None:  # Only show if we created the figure here
        plt.tight_layout()
        plt.show()

    return None


def _plot_kde_distribution(
    ax: plt.Axes,
    samples: np.ndarray,
    color_samples: str,
    credible_interval: float,
    lower_ci: float,
    upper_ci: float,
    alpha: float,
) -> bool:
    """Helper function to plot KDE for a distribution. Returns True if successful."""
    try:
        from scipy.stats import gaussian_kde  # type: ignore[import-untyped]

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
                label=f"{int(credible_interval * 100)}% CI",
            )

        # Add sample curves with low alpha for uncertainty visualization
        if len(samples) <= 100:  # Only if we have a reasonable number of samples
            for sample in samples:
                ax.axvline(x=sample, color=color_samples, alpha=alpha, linewidth=1)
        return True
    except Exception as e:
        print(f"Warning: KDE failed: {e}")
        return False


def _plot_histogram_distribution(
    ax: plt.Axes,
    samples: np.ndarray,
    color_samples: str,
) -> None:
    """Helper function to plot histogram for a distribution."""
    ax.hist(
        samples,
        bins=min(20, len(samples) // 5 + 1),
        alpha=0.3,
        color=color_samples,
        density=True,
    )


def _add_distribution_statistics(
    ax: plt.Axes,
    mean_pred: float,
    median_pred: float,
    lower_ci: float,
    upper_ci: float,
    true_value: float,
    color_true: str,
) -> None:
    """Helper function to add statistical markers and text to a distribution plot."""
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
            transform=ax.get_xaxis_transform(),
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


def _plot_single_distribution(
    ax: plt.Axes,
    samples: np.ndarray,
    true_value: float,
    idx: int,
    is_first: bool,
    credible_interval: float,
    plot_type: str,
    has_kde: bool,
    color_samples: str,
    color_true: str,
    alpha: float,
    xlabel: str,
    ylabel: str,
) -> None:
    """Helper function to plot a single distribution comparison."""
    # Remove any NaN or Inf values
    valid_samples = samples[np.isfinite(samples)]
    if len(valid_samples) < len(samples):
        removed = len(samples) - len(valid_samples)
        print(f"Warning: {removed} non-finite values removed from samples")

    # Make sure we have data to plot
    if len(valid_samples) == 0:
        ax.text(
            0.5,
            0.5,
            "No valid samples",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    samples = valid_samples

    # Compute credible interval
    lower_ci = float(np.percentile(samples, (1 - credible_interval) * 100 / 2))
    upper_ci = float(np.percentile(samples, 100 - (1 - credible_interval) * 100 / 2))
    mean_pred = float(np.mean(samples))
    median_pred = float(np.median(samples))

    # Plot the predicted distribution
    if plot_type in ["kde", "both"] and has_kde and len(samples) >= 3:
        success = _plot_kde_distribution(
            ax, samples, color_samples, credible_interval, lower_ci, upper_ci, alpha
        )
        if not success:
            _plot_histogram_distribution(ax, samples, color_samples)

    if plot_type in ["histogram", "both"] or (plot_type == "kde" and not has_kde):
        _plot_histogram_distribution(ax, samples, color_samples)

    _add_distribution_statistics(
        ax, mean_pred, median_pred, lower_ci, upper_ci, true_value, color_true
    )

    # Add labels and legend
    ax.set_xlabel(xlabel)
    if is_first:
        ax.set_ylabel(ylabel)
    ax.set_title(f"Sample {idx}")


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
        import scipy.stats  # type: ignore[import-untyped]

        has_kde = hasattr(scipy.stats, "gaussian_kde")
    except ImportError:
        has_kde = False

    if not has_kde and plot_type in ["kde", "both"]:
        print("scipy not available, using histogram instead of KDE")
        plot_type = "histogram"

    for i, idx in enumerate(indices):
        _plot_single_distribution(
            ax=axes[i],
            samples=pred_samples[:, idx],
            true_value=y_true[idx],
            idx=int(idx),
            is_first=(i == 0),
            credible_interval=credible_interval,
            plot_type=plot_type,
            has_kde=has_kde,
            color_samples=color_samples,
            color_true=color_true,
            alpha=alpha,
            xlabel=xlabel,
            ylabel=ylabel,
        )

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


def _clean_calibration_data(
    y_pred_probs: np.ndarray, y_true: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to clean calibration data by removing NaNs and clipping to [0, 1]."""
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
            f"Warning: Predicted probabilities outside [0, 1] range: min={min_prob}, max={max_prob}"
        )
        y_pred_probs = np.clip(y_pred_probs, 0, 1)

    return y_pred_probs, y_true


def _calculate_calibration_bins(
    y_pred_probs: np.ndarray, y_true: np.ndarray, n_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper to calculate bins, centers, and probabilities for calibration curves."""
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

    return bins, bin_centers, binids, prob_true, prob_pred, bin_counts


def _add_calibration_histogram(
    ax: plt.Axes,
    bins: np.ndarray,
    bin_centers: np.ndarray,
    binids: np.ndarray,
    hist_height: float,
    hist_alpha: float,
    hist_color: str,
    n_bins: int,
) -> None:
    """Helper to add a histogram of predicted probabilities to the calibration plot."""
    # Add a zero line to separate the calibration curve from the histogram
    add_zero_line(ax, axis="y", color="black", linestyle="-", alpha=0.3)

    # Calculate histogram heights
    hist = np.bincount(binids, minlength=len(bins) - 1) / len(binids)
    scaled_hist = hist * hist_height

    # Create histogram bars
    for i, h in enumerate(scaled_hist):
        ax.bar(
            bin_centers[i],
            h,
            width=(1 / n_bins),
            bottom=-h,
            align="center",
            alpha=hist_alpha,
            color=hist_color,
            label="Prediction Dist." if i == 0 else None,
        )


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
) -> Union[Optional[Figure], Dict[str, Any], Tuple[Optional[Figure], Dict[str, Any]]]:
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

    y_pred_probs, y_true = _clean_calibration_data(y_pred_probs, y_true)

    bins, bin_centers, binids, prob_true, prob_pred, bin_counts = _calculate_calibration_bins(
        y_pred_probs, y_true, n_bins
    )

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

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

        _add_calibration_histogram(
            ax, bins, bin_centers, binids, hist_height, hist_alpha, hist_color, n_bins
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
        return cast(Dict[str, Any], diagnostics)
    elif return_figure:
        return fig
    elif ax is None:  # Only show if we created the figure here
        plt.tight_layout()
        plt.show()

    return None


def plot_pit_histogram(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_pred_std: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    n_bins: int = 20,
    figsize: Tuple[int, int] = (8, 5),
    title: str = "PIT Histogram",
    color: str = "steelblue",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Plot Probability Integral Transform (PIT) histogram for Gaussian predictions.

    For a well-calibrated probabilistic model with Gaussian predictive distributions,
    the PIT values should be uniformly distributed. Deviations indicate miscalibration:
    - U-shaped: underconfident (variances too large)
    - Inverse U-shaped: overconfident (variances too small)
    - Skewed: biased predictions

    Args:
        y_pred: Predicted mean values
        y_pred_std: Predicted standard deviations
        y_true: Ground truth values
        n_bins: Number of histogram bins
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        color: Color for the histogram bars
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object

    Example:
        >>> plot_pit_histogram(preds, pred_stds, targets, return_figure=True)
    """
    from scipy import stats  # type: ignore[import-untyped]

    y_pred = convert_to_tensor(y_pred).detach().cpu().numpy().flatten()
    y_pred_std = convert_to_tensor(y_pred_std).detach().cpu().numpy().flatten()
    y_true = convert_to_tensor(y_true).detach().cpu().numpy().flatten()

    # Compute PIT: CDF of target under predicted Gaussian
    z_scores = (y_true - y_pred) / (y_pred_std + 1e-8)
    pit_values = stats.norm.cdf(z_scores)

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Plot histogram
    ax.hist(
        pit_values,
        bins=n_bins,
        density=True,
        alpha=0.7,
        color=color,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add reference line for uniform distribution
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.8, label="Uniform (ideal)")

    # Add statistics
    ks_stat, ks_pval = stats.kstest(pit_values, "uniform")
    annotations = {
        "KS statistic": ks_stat,
        "p-value": ks_pval,
    }
    add_annotations(ax, annotations, loc="upper right")

    ax.set_xlabel("PIT Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    if return_figure:
        return fig
    elif ax is None:
        plt.tight_layout()
        plt.show()

    return None


def plot_uncertainty_vs_error(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_pred_std: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    aleatoric_var: Optional[Union[torch.Tensor, np.ndarray]] = None,
    epistemic_var: Optional[Union[torch.Tensor, np.ndarray]] = None,
    sort_by: str = "uncertainty",
    feature_vals: Optional[Union[torch.Tensor, np.ndarray]] = None,
    figsize: Tuple[int, int] = (8, 8),
    title: str = "Uncertainty vs Error",
    color: str = "steelblue",
    max_points: int = 2000,
    show_trend: bool = True,
    show_correlation: bool = True,
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Union[Figure, float, Tuple[Figure, float]]]:
    """
    Plot predicted uncertainty vs absolute error.

    A good uncertainty estimator should show positive correlation:
    when the model predicts high uncertainty, errors should actually be larger.
    If aleatoric and epistemic variance arrays are provided, it also plots
    the uncertainty decomposition stack.

    Args:
        y_pred: Predicted mean values
        y_pred_std: Predicted standard deviations
        y_true: Ground truth values
        aleatoric_var: Aleatoric variance component
        epistemic_var: Epistemic variance component
        sort_by: Metric/dimension to sort samples by in decomposition
            ('uncertainty', 'error', 'target', 'feature')
        feature_vals: Feature values to sort by if sort_by='feature'
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        color: Color for scatter points
        max_points: Maximum number of points to plot (for performance)
        show_trend: Whether to show linear trend line
        show_correlation: Whether to show Spearman correlation in legend
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
        If show_correlation=True, also returns Spearman correlation coefficient
    """
    y_pred = convert_to_tensor(y_pred).detach().cpu().numpy().flatten()
    y_pred_std = convert_to_tensor(y_pred_std).detach().cpu().numpy().flatten()
    y_true = convert_to_tensor(y_true).detach().cpu().numpy().flatten()

    abs_errors, correlation, p_value = _compute_uncertainty_stats(y_pred, y_pred_std, y_true)
    y_pred_std_plot, abs_errors_plot = _subsample_scatter_data(y_pred_std, abs_errors, max_points)

    created_fig = ax is None

    if aleatoric_var is not None and epistemic_var is not None:
        aleatoric_np: np.ndarray = convert_to_tensor(aleatoric_var).detach().cpu().numpy().flatten()
        epistemic_np: np.ndarray = convert_to_tensor(epistemic_var).detach().cpu().numpy().flatten()
        if feature_vals is not None:
            feature_vals = convert_to_tensor(feature_vals).detach().cpu().numpy().flatten()

        if created_fig:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 1.8, figsize[1]))
        else:
            assert ax is not None
            fig = cast(Figure, ax.figure)
            ax1 = None
            ax2 = ax
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = cast(Figure, ax.figure)
        ax1 = ax
        ax2 = None

    # Plot Uncertainty vs Error
    if ax1 is not None:
        ax1.scatter(y_pred_std_plot, abs_errors_plot, alpha=0.3, s=10, color=color)
        if show_trend:
            _add_uncertainty_trend(
                ax1, y_pred_std_plot, abs_errors_plot, correlation, show_correlation
            )
        annotations = {
            "Spearman ρ": correlation,
            "p-value": f"{p_value:.2e}",
        }
        add_annotations(ax1, annotations, loc="upper left")
        ax1.set_xlabel("Predicted Uncertainty (σ)", fontweight="bold")
        ax1.set_ylabel("Absolute Error |ŷ - y|", fontweight="bold")
        ax1.set_title(title, fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

    # Plot Uncertainty Decomposition Stack
    if ax2 is not None:
        # Determine sorting index
        if sort_by == "uncertainty":
            sort_idx = np.argsort(aleatoric_np + epistemic_np)
        elif sort_by == "error":
            sort_idx = np.argsort(np.abs(y_true - y_pred))
        elif sort_by == "target":
            sort_idx = np.argsort(y_true)
        elif sort_by == "feature" and feature_vals is not None:
            sort_idx = np.argsort(feature_vals)
        else:
            sort_idx = np.argsort(aleatoric_np + epistemic_np)

        ale_sorted: np.ndarray = aleatoric_np[sort_idx]
        epi_sorted: np.ndarray = epistemic_np[sort_idx]

        ale_smoothed: np.ndarray
        epi_smoothed: np.ndarray
        x_vals: np.ndarray

        if len(ale_sorted) > 100:
            window = max(5, len(ale_sorted) // 40)
            ale_smoothed = np.convolve(ale_sorted, np.ones(window) / window, mode="valid")
            epi_smoothed = np.convolve(epi_sorted, np.ones(window) / window, mode="valid")
            x_vals = np.linspace(0.0, 100.0, len(ale_smoothed))
            x_label = f"Samples (Percentile, sorted by {sort_by})"
        else:
            ale_smoothed = ale_sorted
            epi_smoothed = epi_sorted
            x_vals = np.arange(len(ale_smoothed), dtype=float)
            x_label = f"Samples (sorted by {sort_by})"

        ax2.stackplot(
            x_vals,
            ale_smoothed,
            epi_smoothed,
            labels=["Aleatoric Uncertainty", "Epistemic Uncertainty"],
            colors=["#F08080", "#87CEFA"],
            alpha=0.8,
        )
        ax2.set_xlabel(x_label, fontweight="bold")
        ax2.set_ylabel("Variance Component", fontweight="bold")
        ax2.set_title("Uncertainty Decomposition Stack", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

    if return_figure:
        if show_correlation:
            return fig, float(correlation)
        return fig
    elif created_fig:
        plt.tight_layout()
        plt.show()

    if show_correlation:
        return float(correlation)
    return None


def _compute_binned_metrics(
    y_pred: np.ndarray,
    y_pred_std: np.ndarray,
    y_true: np.ndarray,
    n_bins: int,
) -> Dict[str, Dict[str, float]]:
    """Compute evaluation metrics across bins of the target variable."""
    from scipy import stats

    bin_edges = np.quantile(y_true, np.linspace(0, 1, n_bins + 1))
    binned_metrics: Dict[str, Dict[str, float]] = {}

    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_true >= low) & (y_true < high)

        if i == len(bin_edges) - 2:  # Include right edge for last bin
            mask = (y_true >= low) & (y_true <= high)

        if mask.sum() < 10:  # Skip bins with too few samples
            continue

        bin_preds = y_pred[mask]
        bin_stds = y_pred_std[mask]
        bin_targets = y_true[mask]

        # Compute metrics
        errors = bin_preds - bin_targets
        abs_errors = np.abs(errors)

        # RMSE
        rmse = np.sqrt(np.mean(errors**2))

        # MAE
        mae = np.mean(abs_errors)

        # Bias
        bias_val = np.mean(errors)

        # NMAD (normalized median absolute deviation)
        delta_z_norm = errors / (1 + bin_targets)
        nmad = 1.48 * np.median(np.abs(delta_z_norm - np.median(delta_z_norm)))

        # PICP at 95%
        z = stats.norm.ppf(0.975)
        lower = bin_preds - z * bin_stds
        upper = bin_preds + z * bin_stds
        picp = np.mean((bin_targets >= lower) & (bin_targets <= upper))

        # Mean prediction interval width
        mpiw = np.mean(upper - lower)

        bin_name = f"[{low:.3f}, {high:.3f})"
        binned_metrics[bin_name] = {
            "n_samples": int(mask.sum()),
            "rmse": float(rmse),
            "mae": float(mae),
            "bias": float(bias_val),
            "nmad": float(nmad),
            "picp_95": float(picp),
            "mpiw_95": float(mpiw),
        }

    return binned_metrics


def _render_binned_metrics_plot(
    binned_metrics: Dict[str, Dict[str, float]],
    metric: str,
    figsize: Tuple[int, int],
    title: Optional[str],
    color: str,
    ax: Optional[plt.Axes],
) -> Tuple[Figure, plt.Axes]:
    """Render the binned metrics bar plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    bins = list(binned_metrics.keys())
    values = [binned_metrics[b][metric] for b in bins]
    n_samples = [binned_metrics[b]["n_samples"] for b in bins]

    x = np.arange(len(bins))
    bars = ax.bar(x, values, alpha=0.7, color=color)

    # Add sample counts as text
    for bar, n in zip(bars, n_samples):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45, ha="right")
    ax.set_xlabel("Target Bin")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} by Target Bin")
    ax.grid(True, alpha=0.3, axis="y")

    return fig, ax


def plot_binned_metrics(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_pred_std: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    n_bins: int = 5,
    metric: str = "rmse",
    figsize: Tuple[int, int] = (10, 5),
    title: Optional[str] = None,
    color: str = "steelblue",
    return_figure: bool = False,
    return_metrics: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[
    Union[Figure, Dict[str, Dict[str, float]], Tuple[Figure, Dict[str, Dict[str, float]]]]
]:
    """
    Compute and plot metrics in bins of the target variable.

    This reveals whether model performance degrades in certain regions,
    particularly at the tails of the distribution.

    Args:
        y_pred: Predicted mean values
        y_pred_std: Predicted standard deviations
        y_true: Ground truth values
        n_bins: Number of bins
        metric: Which metric to plot ('rmse', 'mae', 'bias', 'nmad', 'picp_95', 'mpiw_95')
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title (auto-generated if None)
        color: Color for the bars
        return_figure: If True, return figure object instead of displaying
        return_metrics: If True, return the binned metrics dictionary
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
        If return_metrics=True, returns dict of metrics per bin

    Example:
        >>> metrics = plot_binned_metrics(preds, stds, targets, return_metrics=True)
        >>> print(metrics)
    """
    y_pred_np = convert_to_tensor(y_pred).detach().cpu().numpy().flatten()
    y_pred_std_np = convert_to_tensor(y_pred_std).detach().cpu().numpy().flatten()
    y_true_np = convert_to_tensor(y_true).detach().cpu().numpy().flatten()

    binned_metrics = _compute_binned_metrics(y_pred_np, y_pred_std_np, y_true_np, n_bins)

    if return_metrics and not return_figure:
        return binned_metrics

    created_ax = ax is None
    fig, ax = _render_binned_metrics_plot(binned_metrics, metric, figsize, title, color, ax)

    plt.tight_layout()

    if return_figure:
        if return_metrics:
            return fig, binned_metrics
        return fig
    elif created_ax:
        plt.show()

    if return_metrics:
        return binned_metrics
    return None


def plot_gaussian_reliability_diagram(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_pred_std: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    n_levels: int = 10,
    figsize: Tuple[int, int] = (6, 6),
    title: str = "Reliability Diagram",
    color: str = "steelblue",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Plot reliability diagram for Gaussian predictive distributions.

    A well-calibrated model should have observed coverage matching expected coverage
    at all confidence levels. Points on the diagonal indicate perfect calibration.

    This is different from the quantile-based plot_reliability_diagram - this version
    is designed for models that output (mean, std) predictions assuming Gaussian.

    Args:
        y_pred: Predicted mean values
        y_pred_std: Predicted standard deviations
        y_true: Ground truth values
        n_levels: Number of confidence levels to evaluate
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        color: Color for the plot line
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object

    Example:
        >>> plot_gaussian_reliability_diagram(preds, stds, targets, return_figure=True)
    """
    from scipy import stats

    y_pred = convert_to_tensor(y_pred).detach().cpu().numpy().flatten()
    y_pred_std = convert_to_tensor(y_pred_std).detach().cpu().numpy().flatten()
    y_true = convert_to_tensor(y_true).detach().cpu().numpy().flatten()

    # Confidence levels to check
    confidence_levels = np.linspace(0.1, 0.99, n_levels)

    expected_coverage = []
    observed_coverage = []

    for conf in confidence_levels:
        # For Gaussian: z-score for given confidence level
        z = stats.norm.ppf((1 + conf) / 2)

        # Compute interval bounds
        lower = y_pred - z * y_pred_std
        upper = y_pred + z * y_pred_std

        # Check coverage
        covered = (y_true >= lower) & (y_true <= upper)
        observed = np.mean(covered)

        expected_coverage.append(conf)
        observed_coverage.append(observed)

    expected_coverage_arr = np.array(expected_coverage)
    observed_coverage_arr = np.array(observed_coverage)

    # Compute calibration error
    mace = np.mean(np.abs(expected_coverage_arr - observed_coverage_arr))

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Plot diagonal (perfect calibration)
    add_identity_line(ax, label="Perfectly Calibrated")

    # Plot calibration curve
    ax.plot(
        expected_coverage_arr,
        observed_coverage_arr,
        "o-",
        color=color,
        markersize=6,
        label=f"Model (MACE={mace:.4f})",
    )

    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Observed Coverage")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if return_figure:
        return fig
    elif ax is None:
        plt.tight_layout()
        plt.show()

    return None


def plot_target_density_error_overlap(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    n_bins: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Target Density vs. Error Overlap",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Overlays the target empirical distribution (using KDE or histogram)
    and the local mean absolute error in bins of target values.
    Useful for checking rare-target or imbalanced regression performance.

    Args:
        y_true: Ground truth target values
        y_pred: Predicted target values
        n_bins: Number of bins to group the target values into
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    from scipy import stats

    y_true = convert_to_tensor(y_true).detach().cpu().numpy().flatten()
    y_pred = convert_to_tensor(y_pred).detach().cpu().numpy().flatten()
    validate_inputs(torch.tensor(y_pred), torch.tensor(y_true))

    abs_errors = np.abs(y_true - y_pred)

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Plot empirical density of y_true (left axis)
    ax.set_xlabel("Target Variable (y)", fontweight="bold")
    ax.set_ylabel("Empirical Target Density", color="steelblue", fontweight="bold")

    # Use kernel density estimate for smooth target density representation
    kde_x = np.linspace(y_true.min(), y_true.max(), 200)
    try:
        kde = stats.gaussian_kde(y_true)
        kde_y = kde(kde_x)
        ax.plot(kde_x, kde_y, color="steelblue", linewidth=2.5, label="Target Density (KDE)")
        ax.fill_between(kde_x, 0, kde_y, color="steelblue", alpha=0.15)
    except Exception:
        # Fallback to histogram if KDE fails (e.g. singular covariance)
        ax.hist(
            y_true,
            bins=n_bins,
            density=True,
            color="steelblue",
            alpha=0.3,
            edgecolor="black",
            label="Target Density",
        )

    ax.tick_params(axis="y", labelcolor="steelblue")

    # Create twin axis for mean absolute error (right axis)
    ax_err = ax.twinx()
    ax_err.set_ylabel("Mean Absolute Error (MAE)", color="crimson", fontweight="bold")

    # Calculate local error in bins
    bin_edges = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_maes = []

    for i in range(n_bins):
        in_bin = (y_true >= bin_edges[i]) & (y_true < bin_edges[i + 1])
        # Include upper edge in the last bin
        if i == n_bins - 1:
            in_bin = in_bin | (y_true == bin_edges[i + 1])

        if np.sum(in_bin) > 0:
            bin_maes.append(np.mean(abs_errors[in_bin]))
        else:
            bin_maes.append(np.nan)

    bin_maes_np: np.ndarray = np.array(bin_maes)

    # Plot local error curve
    ax_err.plot(
        bin_centers,
        bin_maes_np,
        "o-",
        color="crimson",
        linewidth=2.5,
        markersize=6,
        label="Local MAE",
    )
    ax_err.tick_params(axis="y", labelcolor="crimson")

    # Combine legends
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax_err.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if return_figure:
        return fig
    elif created_fig:
        plt.tight_layout()
        plt.show()
    return None


def plot_conditional_density_slices(
    density_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_slices: Union[torch.Tensor, np.ndarray],
    y_grid: Union[torch.Tensor, np.ndarray],
    y_true_slices: Optional[Union[torch.Tensor, np.ndarray]] = None,
    figsize: Tuple[int, int] = (12, 4),
    title: str = "Conditional Predictive Density Slices",
    return_figure: bool = False,
) -> Optional[Figure]:
    """
    Plots 1D conditional density p(y | x) for selected representative features values.
    Helpful for diagnosing mixture density networks, normalizing flows, and multi-modal models.

    Args:
        density_fn: Callable that accepts (x_slice, y_grid_val) and returns probability densities.
                    x_slice should be shape [D] and y_grid_val should be shape [M].
                    Returns density of shape [M].
        x_slices: [N, D] array of feature slice values representing N distinct evaluation cases.
        y_grid: [M] array of target values to evaluate the density on.
        y_true_slices: Optional [N] true target values corresponding to x_slices.
        figsize: Figure size (width, height)
        title: Plot title
        return_figure: If True, return figure object
    """
    x_slices = convert_to_tensor(x_slices).detach().cpu().numpy()
    y_grid = convert_to_tensor(y_grid).detach().cpu().numpy().flatten()
    if y_true_slices is not None:
        y_true_slices = convert_to_tensor(y_true_slices).detach().cpu().numpy().flatten()

    n_slices = len(x_slices)

    # Create grid of subplots (1 row, n_slices columns)
    fig, axes = plt.subplots(1, n_slices, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for i in range(n_slices):
        x_val = x_slices[i]

        # Evaluate density over the grid
        densities = density_fn(x_val, y_grid)

        ax = axes_flat[i]
        ax.plot(y_grid, densities, color="darkblue", linewidth=2.0)
        ax.fill_between(y_grid, 0, densities, color="darkblue", alpha=0.15)

        if y_true_slices is not None:
            true_val = y_true_slices[i]
            ax.axvline(
                x=true_val, color="crimson", linestyle="--", linewidth=1.5, label="True Target"
            )
            ax.legend(loc="upper right")

        ax.set_title(f"Case {i + 1}", fontweight="bold")
        ax.set_xlabel("Target Value (y)", fontweight="bold")
        if i == 0:
            ax.set_ylabel("Predictive Density p(y|x)", fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")

    if return_figure:
        return fig
    else:
        plt.tight_layout()
        plt.show()
    return None


def plot_censored_survival_curves(
    predicted_survival: Union[torch.Tensor, np.ndarray],
    time_grid: Union[torch.Tensor, np.ndarray],
    observed_times: Union[torch.Tensor, np.ndarray],
    censoring_indicators: Union[torch.Tensor, np.ndarray],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Predicted vs. Empirical Survival Functions",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Overlays the mean predicted survival probability curve S(t | x) with a native Kaplan-Meier
    empirical estimator of the observed and censored times. Useful for survival regression.

    Args:
        predicted_survival: [N, T] array of predicted survival probabilities at each grid time step.
        time_grid: [T] time values corresponding to the T steps.
        observed_times: [N] observed times (event or censoring times).
        censoring_indicators: [N] event indicators (1 = event observed, 0 = censored).
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        return_figure: If True, return figure object
        ax: Optional matplotlib axes
    """
    predicted_survival = convert_to_tensor(predicted_survival).detach().cpu().numpy()
    time_grid = convert_to_tensor(time_grid).detach().cpu().numpy().flatten()
    observed_times = convert_to_tensor(observed_times).detach().cpu().numpy().flatten()
    censoring_indicators = convert_to_tensor(censoring_indicators).detach().cpu().numpy().flatten()

    # 1. Native Kaplan-Meier Estimator computation
    # Sort distinct observed times
    unique_times = np.sort(np.unique(observed_times))
    km_times = [0.0]
    km_survival = [1.0]

    current_survival = 1.0
    for t in unique_times:
        if t <= 0:
            continue
        # Count deaths (events) at exactly t
        d = np.sum((observed_times == t) & (censoring_indicators == 1))
        # Count number at risk (observed time >= t)
        n = np.sum(observed_times >= t)

        if n > 0:
            current_survival *= 1.0 - d / n

        km_times.append(t)
        km_survival.append(current_survival)

    km_times_np: np.ndarray = np.array(km_times)
    km_survival_np: np.ndarray = np.array(km_survival)

    # 2. Compute Mean and Std of predicted survival curves
    mean_predicted = np.mean(predicted_survival, axis=0)
    std_predicted = np.std(predicted_survival, axis=0)

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Plot predicted survival (mean + shade)
    ax.plot(time_grid, mean_predicted, color="navy", linewidth=2.5, label="Mean Predicted Survival")
    ax.fill_between(
        time_grid,
        np.clip(mean_predicted - std_predicted, 0, 1),
        np.clip(mean_predicted + std_predicted, 0, 1),
        color="navy",
        alpha=0.15,
        label="Predicted Spread (±1σ)",
    )

    # Plot empirical Kaplan-Meier curve using step plotting
    ax.step(
        km_times_np,
        km_survival_np,
        where="post",
        color="darkorange",
        linewidth=2.5,
        label="Empirical (Kaplan-Meier)",
    )

    ax.set_xlabel("Time (t)", fontweight="bold")
    ax.set_ylabel("Survival Probability S(t)", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if return_figure:
        return fig
    elif created_fig:
        plt.tight_layout()
        plt.show()
    return None
