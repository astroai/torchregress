"""
Results visualization tools for regression models.

This module provides visualization utilities for presenting
and comparing regression model results.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

# Import visualization utilities
from torchregress.viz.utils import create_color_palette


def plot_performance_comparison(
    metrics: Dict[str, Dict[str, Union[float, np.ndarray]]],
    highlight_best: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Model Performance Comparison",
    metrics_to_include: Optional[List[str]] = None,
    higher_is_better: Optional[Dict[str, bool]] = None,
    sort_by: Optional[str] = None,
    color_palette: str = "tab10",
    plot_type: str = "bar",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Create a comparison plot for multiple models across different metrics.

    Args:
        metrics: Dictionary mapping model names to dictionaries of metrics
        highlight_best: Whether to highlight the best model for each metric
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        metrics_to_include: List of metrics to include (defaults to all common metrics)
        higher_is_better: Dictionary specifying for each metric if higher values are better
        sort_by: Metric name to sort models by
        color_palette: Name of color palette to use
        plot_type: Type of plot ('bar', 'radar', or 'heatmap')
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting (only used for bar plot)

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    # Extract common metrics across all models if not specified
    if metrics_to_include is None:
        metrics_to_include = list(
            set.intersection(*[set(model_metrics.keys()) for model_metrics in metrics.values()])
        )
        metrics_to_include = sorted(metrics_to_include)

    # Filter to only include metrics that exist for all models
    metrics_to_include = [
        m
        for m in metrics_to_include
        if all(m in model_metrics for model_metrics in metrics.values())
    ]

    if not metrics_to_include:
        raise ValueError("No common metrics found across all models")

    # Determine whether higher or lower is better for each metric
    if higher_is_better is None:
        higher_is_better = {}
        for metric in metrics_to_include:
            # Default: higher is better unless metric contains 'error', 'loss', or 'mae'
            is_higher_better = not any(
                term in metric.lower() for term in ["error", "loss", "mae", "mse", "rmse", "mape"]
            )
            higher_is_better[metric] = is_higher_better

    # Build numpy arrays for easier manipulation
    model_names = list(metrics.keys())
    n_models = len(model_names)
    n_metrics = len(metrics_to_include)

    # Build values array (n_models, n_metrics)
    values_arr = np.zeros((n_models, n_metrics))
    for j, metric in enumerate(metrics_to_include):
        for i, model in enumerate(model_names):
            values_arr[i, j] = float(metrics[model][metric])

    # Sort by specified metric if requested
    if sort_by is not None and sort_by in metrics_to_include:
        sort_col = metrics_to_include.index(sort_by)
        ascending = not higher_is_better.get(sort_by, True)
        order = np.argsort(values_arr[:, sort_col])
        if not ascending:
            order = order[::-1]
        values_arr = values_arr[order]
        model_names = [model_names[i] for i in order]

    # Store metric names and values as list-based structures (was DataFrame)
    metric_names_list = list(metrics_to_include)

    # Determine best model for each metric if highlighting is requested
    best_values = {}
    if highlight_best:
        for j, metric in enumerate(metrics_to_include):
            if higher_is_better.get(metric, True):
                best_values[metric] = float(values_arr[:, j].max())
            else:
                best_values[metric] = float(values_arr[:, j].min())

    # Determine colors for models
    colors = cast(List[Any], create_color_palette(len(model_names), palette_name=color_palette))

    # Create plot based on specified type
    if plot_type == "bar":
        return _plot_performance_bar(
            model_names,
            metric_names_list,
            values_arr,
            best_values,
            figsize,
            title,
            higher_is_better,
            colors,
            return_figure,
            ax,
        )
    elif plot_type == "radar":
        return _plot_performance_radar(
            model_names,
            metric_names_list,
            values_arr,
            best_values,
            figsize,
            title,
            higher_is_better,
            colors,
            return_figure,
        )
    elif plot_type == "heatmap":
        return _plot_performance_heatmap(
            model_names,
            metric_names_list,
            values_arr,
            best_values,
            figsize,
            title,
            higher_is_better,
            return_figure,
        )
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")


def _add_performance_bars(
    ax: plt.Axes,
    model_names: list[str],
    metric_names: list[str],
    values_arr: np.ndarray,
    best_values: Dict[str, float],
    colors: List,
    bar_width: float,
    indices: np.ndarray,
) -> None:
    """Helper to add bars and value annotations."""
    for i, model_name in enumerate(model_names):
        values = values_arr[i]
        bars = ax.bar(
            indices + i * bar_width,
            values,
            bar_width,
            label=model_name,
            color=colors[i % len(colors)],
        )

        # Highlight best model for each metric
        if best_values:
            for j, metric in enumerate(metric_names):
                if np.isclose(values_arr[i, j], best_values[metric]):
                    bars[j].set_edgecolor("black")
                    bars[j].set_linewidth(2)

        # Add value labels on top of bars
        for j, v in enumerate(values):
            # Format value based on magnitude
            if abs(v) < 0.01 or abs(v) >= 1000:
                value_str = f"{v:.2e}"
            elif abs(v) < 0.1:
                value_str = f"{v:.3f}"
            else:
                value_str = f"{v:.2f}"

            ax.text(
                indices[j] + i * bar_width,
                v * 1.01,
                value_str,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )


def _format_performance_axes(
    ax: plt.Axes,
    metric_names: list[str],
    title: str,
    indices: np.ndarray,
    bar_width: float,
    n_models: int,
    higher_is_better: Dict[str, bool],
) -> None:
    """Helper to format axes and labels."""
    # Add labels, title and legend
    ax.set_xlabel("Metric", fontweight="bold")
    ax.set_ylabel("Value", fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(indices + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=45, ha="right", fontweight="bold")

    # Skip LaTeX rendering if not available or fails
    # Use rc_context to avoid globally mutating rcParams
    try:
        import shutil

        if shutil.which("latex") is not None:
            with plt.rc_context(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                }
            ):
                # Replace common metric names with LaTeX
                metric_labels = []
                for metric in metric_names:
                    if "rmse" in metric.lower():
                        metric_labels.append(r"$\mathrm{RMSE}$")
                    elif "mae" in metric.lower():
                        metric_labels.append(r"$\mathrm{MAE}$")
                    elif "r2" in metric.lower():
                        metric_labels.append(r"$R^2$")
                    else:
                        metric_labels.append(metric)
                ax.set_xticklabels(metric_labels, rotation=45, ha="right")
    except (ImportError, RuntimeError, Exception):
        pass  # Skip LaTeX rendering if not available

    ax.legend(loc="best", frameon=True, fancybox=True, framealpha=0.9)

    # Add grid
    ax.grid(True, axis="y", alpha=0.3)

    # Add annotations for which direction is better
    for i, metric in enumerate(metric_names):
        direction = "↑" if higher_is_better.get(metric, True) else "↓"
        ax.annotate(
            direction,
            xy=(i, 0),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=12,
        )


def _plot_performance_bar(
    model_names: list[str],
    metric_names: list[str],
    values_arr: np.ndarray,
    best_values: Dict[str, float],
    figsize: Tuple[int, int],
    title: str,
    higher_is_better: Dict[str, bool],
    colors: List,
    return_figure: bool,
    ax: Optional[plt.Axes],
) -> Optional[Figure]:
    """Helper function for bar plot comparison."""
    # Create plot if no axes provided
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Number of models and metrics
    n_models = len(model_names)
    n_metrics = len(metric_names)

    # Set width of bars
    bar_width = 0.8 / n_models

    # Set positions of bars on x-axis
    indices = np.arange(n_metrics)

    _add_performance_bars(
        ax, model_names, metric_names, values_arr, best_values, colors, bar_width, indices
    )
    _format_performance_axes(
        ax, metric_names, title, indices, bar_width, n_models, higher_is_better
    )

    plt.tight_layout()

    if return_figure:
        return fig
    elif created_fig:  # Only show if we created the figure here
        plt.show()

    return None


def _plot_performance_radar(
    model_names: list[str],
    metric_names: list[str],
    values_arr: np.ndarray,
    best_values: Dict[str, float],
    figsize: Tuple[int, int],
    title: str,
    higher_is_better: Dict[str, bool],
    colors: List,
    return_figure: bool,
) -> Optional[Figure]:
    """Helper function for radar plot comparison."""
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # Number of metrics (variables)
    n_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()

    # Add the last angle to close the polygon
    angles += angles[:1]

    # Normalize metrics to range [0, 1] for radar plot
    norm_values = np.zeros_like(values_arr)
    for j in range(n_metrics):
        col = values_arr[:, j]
        min_val = float(col.min())
        max_val = float(col.max())
        metric = metric_names[j]
        if higher_is_better.get(metric, True):
            norm_values[:, j] = (col - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        else:
            norm_values[:, j] = (
                1.0 - (col - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            )

    # Plot each model
    for i, model_name in enumerate(model_names):
        values = norm_values[i].tolist()
        values += values[:1]  # Close the polygon

        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=model_name)
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

    # Set labels and ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [
            f"{metric}\n({'↑' if higher_is_better.get(metric, True) else '↓'})"
            for metric in metric_names
        ]
    )

    # Set y-ticks
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color="grey", size=8)
    cast(Any, ax).set_rlabel_position(0)

    # Add title and legend
    plt.title(title, size=14, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    if return_figure:
        return fig
    else:
        plt.show()
        return None


def _plot_performance_heatmap(
    model_names: list[str],
    metric_names: list[str],
    values_arr: np.ndarray,
    best_values: Dict[str, float],
    figsize: Tuple[int, int],
    title: str,
    higher_is_better: Dict[str, bool],
    return_figure: bool,
) -> Optional[Figure]:
    """Helper function for heatmap plot comparison."""
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize metrics to range [0, 1] for heatmap
    n_metrics = len(metric_names)
    norm_values = np.zeros_like(values_arr)
    for j in range(n_metrics):
        col = values_arr[:, j]
        min_val = float(col.min())
        max_val = float(col.max())
        metric = metric_names[j]
        if higher_is_better.get(metric, True):
            norm_values[:, j] = (col - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        else:
            norm_values[:, j] = (
                1.0 - (col - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            )

    # Create heatmap
    im = ax.imshow(norm_values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Performance (normalized)", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(
        [
            f"{metric}\n({'↑' if higher_is_better.get(metric, True) else '↓'})"
            for metric in metric_names
        ]
    )
    ax.set_yticklabels(model_names)

    # Rotate x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data and add annotations with actual values
    for i in range(len(model_names)):
        for j in range(n_metrics):
            metric_value = values_arr[i, j]
            is_best = best_values and np.isclose(metric_value, best_values[metric_names[j]])

            # Format text based on value type
            if isinstance(metric_value, float):
                text = f"{metric_value:.4f}"
            else:
                text = str(metric_value)

            # Add bold font for best values
            if is_best:
                text = f"$\\bf{{{text}}}$"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black" if norm_values[i, j] > 0.5 else "white",
            )

    # Add title
    ax.set_title(title)

    plt.tight_layout()

    if return_figure:
        return fig
    else:
        plt.show()
        return None


def plot_parameter_sensitivity(
    parameter_values: Dict[str, List[Union[float, int, str]]],
    metric_values: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Parameter Sensitivity Analysis",
    n_cols: int = 2,
    higher_is_better: Optional[Dict[str, bool]] = None,
    highlight_best: bool = True,
    plot_type: str = "line",
    color_palette: str = "viridis",
    return_figure: bool = False,
) -> Optional[Figure]:
    """
    Create a parameter sensitivity plot to analyze how parameters affect model performance.

    Args:
        parameter_values: Dictionary mapping parameter names to lists of values tested
        metric_values: Dictionary mapping metric names to lists of resulting values
        figsize: Figure size (width, height)
        title: Plot title
        n_cols: Number of columns in the grid
        higher_is_better: Dictionary specifying for each metric if higher values are better
        highlight_best: Whether to highlight the best parameter value for each metric
        plot_type: Type of plot ('line' or 'bar')
        color_palette: Name of color palette to use
        return_figure: If True, return figure object instead of displaying

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    # Determine if higher or lower is better for each metric
    if higher_is_better is None:
        higher_is_better = {}
        for metric in metric_values:
            # Default: higher is better unless metric contains 'error', 'loss', or 'mae'
            is_higher_better = not any(
                term in metric.lower() for term in ["error", "loss", "mae", "mse", "rmse", "mape"]
            )
            higher_is_better[metric] = is_higher_better

    # Create grid of subplots
    n_metrics = len(metric_values)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single subplot case
    if n_metrics == 1:
        axes = np.array([axes])

    # Ensure axes is always a 1D array
    axes = axes.flatten()

    # Plot each parameter's effect on each metric
    for i, (metric, values) in enumerate(metric_values.items()):
        ax = axes[i]

        for param_name, param_values in parameter_values.items():
            # Skip if number of parameter values doesn't match metric values
            if len(param_values) != len(values):
                continue

            # Sort values by parameter if parameter is numeric
            if all(isinstance(p, (int, float)) for p in param_values):
                sorted_order = [int(j) for j in np.argsort(param_values)]
                sorted_params = [param_values[j] for j in sorted_order]
                sorted_values = [values[j] for j in sorted_order]
            else:
                sorted_params = param_values
                sorted_values = values

            # Determine best value
            if higher_is_better.get(metric, True):
                best_idx = np.argmax(sorted_values)
            else:
                best_idx = np.argmin(sorted_values)

            # Get colors
            if plot_type == "line":
                # For line plot, use a single color
                pass  # Use default color cycle
            else:
                # For bar plot, use a color gradient
                cmap = plt.get_cmap(color_palette)
                colors = [cmap(i / len(sorted_params)) for i in range(len(sorted_params))]

            # Create plot based on type
            if plot_type == "line":
                ax.plot(sorted_params, sorted_values, "o-", label=param_name)

                # Highlight best value
                if highlight_best:
                    ax.scatter(
                        [sorted_params[best_idx]],
                        [sorted_values[best_idx]],
                        color="red",
                        s=100,
                        zorder=10,
                        label=f"Best: {sorted_params[best_idx]}",
                    )
            else:  # bar plot
                x_pos = np.arange(len(sorted_params))
                bars = ax.bar(x_pos, sorted_values, color=colors)
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(p) for p in sorted_params], rotation=45)

                # Highlight best value
                if highlight_best:
                    bars[best_idx].set_edgecolor("red")
                    bars[best_idx].set_linewidth(2)
                    bars[best_idx].set_hatch("//")

        # Set title and labels
        is_higher_better = higher_is_better.get(metric, True)
        direction = "↑" if is_higher_better else "↓"
        ax.set_title(f"{metric} {direction}")

        if plot_type == "line":
            ax.set_xlabel("Parameter Value")
            ax.set_ylabel("Metric Value")
            ax.legend(loc="best")
        else:  # bar plot
            ax.set_xlabel(param_name)
            ax.set_ylabel("Metric Value")

        # Add grid
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    # Set overall title
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if return_figure:
        return fig
    else:
        plt.show()
        return None


def _normalize_importance_inputs(
    importance_values: Union[np.ndarray, List[float], torch.Tensor],
    importance_errors: Optional[Union[np.ndarray, List[float], torch.Tensor]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Helper to convert importance inputs to numpy arrays."""
    if isinstance(importance_values, torch.Tensor):
        importance_values = importance_values.detach().cpu().numpy()
    importance_values_np = np.array(importance_values)

    importance_errors_np = None
    if importance_errors is not None:
        if isinstance(importance_errors, torch.Tensor):
            importance_errors = importance_errors.detach().cpu().numpy()
        importance_errors_np = np.array(importance_errors)

    return importance_values_np, importance_errors_np


def _sort_and_limit_features(
    feature_names: List[str],
    importance_values: np.ndarray,
    importance_errors: Optional[np.ndarray],
    sort_values: bool,
    horizontal: bool,
    top_n: Optional[int],
) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
    """Helper to sort and limit feature importance values."""
    if sort_values:
        idx = np.argsort(importance_values)
        if not horizontal:  # For vertical, we want descending order
            idx = idx[::-1]

        feature_names = [feature_names[i] for i in idx]
        importance_values = importance_values[idx]

        if importance_errors is not None:
            importance_errors = importance_errors[idx]

    if top_n is not None and top_n < len(feature_names):
        if horizontal:  # For horizontal, we want highest values at the top
            feature_names = feature_names[-top_n:]
            importance_values = importance_values[-top_n:]
            if importance_errors is not None:
                importance_errors = importance_errors[-top_n:]
        else:  # For vertical, we've already sorted in descending order
            feature_names = feature_names[:top_n]
            importance_values = importance_values[:top_n]
            if importance_errors is not None:
                importance_errors = importance_errors[:top_n]

    return feature_names, importance_values, importance_errors


def _plot_importance_bars(
    ax: plt.Axes,
    feature_names: List[str],
    importance_values: np.ndarray,
    importance_errors: Optional[np.ndarray],
    horizontal: bool,
    color: str,
    error_color: str,
    show_values: bool,
) -> None:
    """Helper to plot the bars for feature importance."""
    y_pos = np.arange(len(feature_names))

    if horizontal:
        bars = ax.barh(
            y_pos,
            importance_values,
            color=color,
            xerr=importance_errors,
            error_kw=dict(ecolor=error_color),
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Highest values at the top

        if show_values:
            for bar in bars:
                width = bar.get_width()
                label_position = max(width * 1.05, 0.01)
                ax.text(
                    label_position, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center"
                )
    else:
        bars = ax.bar(
            y_pos,
            importance_values,
            color=color,
            yerr=importance_errors,
            error_kw=dict(ecolor=error_color),
        )
        ax.set_xticks(y_pos)
        ax.set_xticklabels(feature_names, rotation=45, ha="right")

        if show_values:
            for bar in bars:
                height = bar.get_height()
                label_position = height * 1.05
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_position,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )


def plot_feature_importance(
    feature_names: List[str],
    importance_values: Union[np.ndarray, List[float], torch.Tensor],
    importance_errors: Optional[Union[np.ndarray, List[float], torch.Tensor]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Feature Importance",
    color: str = "skyblue",
    error_color: str = "black",
    sort_values: bool = True,
    horizontal: bool = True,
    top_n: Optional[int] = None,
    return_figure: bool = False,
    show_values: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Create a feature importance plot to visualize which features are most important.

    Args:
        feature_names: Names of features
        importance_values: Importance score for each feature
        importance_errors: Optional error/uncertainty for importance values
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        color: Bar color
        error_color: Error bar color
        sort_values: Whether to sort features by importance
        horizontal: Whether to create a horizontal bar chart
        top_n: Optionally limit to top N features
        return_figure: If True, return figure object instead of displaying
        show_values: Whether to display importance values on bars
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    # Convert to numpy array for easier manipulation
    importance_values_np, importance_errors_np = _normalize_importance_inputs(
        importance_values, importance_errors
    )

    # Sort and limit features
    feature_names, importance_values_np, importance_errors_np = _sort_and_limit_features(
        feature_names, importance_values_np, importance_errors_np, sort_values, horizontal, top_n
    )

    # Create plot if no axes provided
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Create bars
    _plot_importance_bars(
        ax,
        feature_names,
        importance_values_np,
        importance_errors_np,
        horizontal,
        color,
        error_color,
        show_values,
    )

    # Set labels and title
    if horizontal:
        ax.set_xlabel("Importance")
    else:
        ax.set_ylabel("Importance")

    ax.set_title(title)

    # Add grid
    if horizontal:
        ax.grid(True, axis="x", alpha=0.3)
    else:
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if return_figure:
        return fig
    elif created_fig:  # Only show if we created the figure here
        plt.show()

    return None


def plot_model_ensemble_contributions(
    predictions: Dict[str, np.ndarray],
    ensemble_prediction: np.ndarray,
    model_weights: Optional[Dict[str, float]] = None,
    x: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Model Ensemble Contributions",
    xlabel: str = "Sample Index",
    ylabel: str = "Prediction",
    color_palette: str = "tab10",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Visualize how different models contribute to an ensemble prediction.

    Args:
        predictions: Dictionary mapping model names to their predictions
        ensemble_prediction: The final ensemble prediction
        model_weights: Optional dictionary of weights for each model
        x: Optional x-coordinates (e.g., time points or feature values)
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        color_palette: Name of color palette to use
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    # Create plot if no axes provided
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Create x values if not provided
    if x is None:
        x = np.arange(len(ensemble_prediction))

    # Get colors
    colors = cast(List[Any], create_color_palette(len(predictions), palette_name=color_palette))

    # Plot individual model predictions
    for i, (model_name, preds) in enumerate(predictions.items()):
        weight = model_weights.get(model_name, 1.0) if model_weights else 1.0
        label = f"{model_name} (w={weight:.2f})" if model_weights else model_name
        ax.plot(x, preds, color=colors[i], alpha=0.5, linestyle="--", label=label)

    # Plot ensemble prediction
    ax.plot(x, ensemble_prediction, color="black", linewidth=2, label="Ensemble")

    # Add labels, title and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if return_figure:
        return fig
    elif created_fig:  # Only show if we created the figure here
        plt.show()

    return None


def plot_simex_extrapolation(
    lambda_values: Union[torch.Tensor, np.ndarray],
    simulated_values: Union[torch.Tensor, np.ndarray],
    extrapolator: Any,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "SIMEX Extrapolation Diagnostics",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Plots SIMEX simulation points and the fitted extrapolation curve back to lambda = -1.
    Useful for diagnosing measurement error (Error-in-Variables) correction.

    Args:
        lambda_values: [M] array of added noise multipliers (typically >= 0).
        simulated_values: [M] or [M, P] array of simulated metrics/coefficient values.
        extrapolator: A callable function or fitted extrapolator model
            (e.g. poly1d) that takes lambda and returns prediction.
        figsize: Figure size (width, height)
        title: Plot title
        return_figure: If True, return figure object
        ax: Optional matplotlib axes
    """
    from torchregress.metrics.utils import convert_to_tensor

    lambda_values = convert_to_tensor(lambda_values).detach().cpu().numpy().flatten()
    simulated_values = convert_to_tensor(simulated_values).detach().cpu().numpy()

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Plot simulation points
    if len(simulated_values.shape) > 1 and simulated_values.shape[1] > 1:
        for j in range(simulated_values.shape[1]):
            ax.scatter(
                lambda_values, simulated_values[:, j], alpha=0.8, marker="o", label=f"Param {j}"
            )
    else:
        ax.scatter(
            lambda_values,
            simulated_values.flatten(),
            color="navy",
            s=50,
            zorder=5,
            label="Simulated Metrics",
        )

    # Extrapolation curve
    lambdas_fine = np.linspace(-1.0, max(lambda_values) * 1.1, 200)

    try:
        if hasattr(extrapolator, "predict"):
            pred_fine = extrapolator.predict(lambdas_fine.reshape(-1, 1)).flatten()
            pred_minus_1 = extrapolator.predict(np.array([[-1.0]])).item()
        elif callable(extrapolator):
            pred_fine = extrapolator(lambdas_fine).flatten()
            pred_minus_1 = extrapolator(-1.0)
        else:
            pred_fine = np.polyval(extrapolator, lambdas_fine).flatten()
            pred_minus_1 = np.polyval(extrapolator, -1.0)

        ax.plot(
            lambdas_fine,
            pred_fine,
            color="crimson",
            linestyle="-",
            linewidth=2.5,
            label="Fitted Extrapolation",
        )
        ax.scatter(
            [-1.0],
            [pred_minus_1],
            color="forestgreen",
            marker="*",
            s=250,
            zorder=6,
            label=f"Corrected Estimate: {pred_minus_1:.4f}",
        )
    except Exception as e:
        print(f"Warning: Could not plot extrapolation curve: {e}")

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="Naive / Observed Noise")
    ax.axvline(x=-1, color="forestgreen", linestyle="--", alpha=0.5, label="Theoretical Zero Noise")

    ax.set_xlabel("Noise Multiplier (λ)", fontweight="bold")
    ax.set_ylabel("Parameter Estimate / Metric Value", fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if return_figure:
        return fig
    elif created_fig:
        plt.tight_layout()
        plt.show()
    return None


def plot_risk_coverage_curve(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    rejection_scores: Union[torch.Tensor, np.ndarray],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Risk-Coverage Selective Prediction Curve",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Plots validation risk (MSE/MAE) vs. coverage (percentage of remaining dataset kept).
    Allows evaluating out-of-distribution scores or uncertainty estimators for selective prediction.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        rejection_scores: Rejection score for each sample (higher score = reject first).
                          Typically predicted uncertainty std, typicality, or entropy score.
        figsize: Figure size (width, height)
        title: Plot title
        return_figure: If True, return figure
        ax: Optional axes
    """
    from torchregress.metrics.utils import convert_to_tensor

    y_true = convert_to_tensor(y_true).detach().cpu().numpy().flatten()
    y_pred = convert_to_tensor(y_pred).detach().cpu().numpy().flatten()
    rejection_scores = convert_to_tensor(rejection_scores).detach().cpu().numpy().flatten()

    errors = (y_true - y_pred) ** 2

    # Sort samples by rejection scores ascending
    sort_idx = np.argsort(rejection_scores)
    sorted_errors = errors[sort_idx]

    n_samples = len(y_true)
    coverages = np.linspace(1 / n_samples, 1.0, n_samples)

    # Cumulative mean risk for model-based rejection
    model_risk = np.cumsum(sorted_errors) / np.arange(1, n_samples + 1)

    # Random rejection (constant expected risk)
    random_risk = np.full(n_samples, np.mean(errors))

    # Oracle rejection (sorting by actual error)
    oracle_idx = np.argsort(errors)
    oracle_sorted_errors = errors[oracle_idx]
    oracle_risk = np.cumsum(oracle_sorted_errors) / np.arange(1, n_samples + 1)

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    ax.plot(coverages, model_risk, color="navy", linewidth=2.5, label="Model Rejection Policy")
    ax.plot(
        coverages,
        random_risk,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Random Rejection",
    )
    ax.plot(
        coverages,
        oracle_risk,
        color="forestgreen",
        linestyle=":",
        linewidth=2.0,
        label="Oracle (Ideal)",
    )

    aurcc_model = np.mean(model_risk)
    aurcc_random = np.mean(random_risk)
    aurcc_oracle = np.mean(oracle_risk)

    # Re-import add_annotations here to ensure it's available in this scope
    from torchregress.viz.utils import add_annotations

    annotations = {
        "AURCC Model": aurcc_model,
        "AURCC Random": aurcc_random,
        "AURCC Oracle": aurcc_oracle,
    }

    add_annotations(ax, annotations, loc="upper right")

    ax.set_xlabel("Coverage (Fraction of data kept)", fontweight="bold")
    ax.set_ylabel("Remaining Risk (MSE)", fontweight="bold")
    ax.set_xlim(0, 1.02)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    if return_figure:
        return fig
    elif created_fig:
        plt.tight_layout()
        plt.show()
    return None


def plot_causal_uplift_qini(
    uplift_scores: Union[torch.Tensor, np.ndarray],
    treatment: Union[torch.Tensor, np.ndarray],
    y_obs: Union[torch.Tensor, np.ndarray],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Causal Uplift Qini Curve",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Plots the Qini curve showing cumulative uplift per treated fraction.
    Useful for evaluating doubly robust causal uplift models.

    Args:
        uplift_scores: Predicted individual treatment effect (ITE) or uplift score.
                       Higher score means the treatment is predicted to have larger positive effect.
        treatment: Binary treatment indicator (1 = treated, 0 = control)
        y_obs: Observed outcome
        figsize: Figure size (width, height)
        title: Plot title
        return_figure: If True, return figure
        ax: Optional axes
    """
    from torchregress.metrics.utils import convert_to_tensor

    uplift_scores = convert_to_tensor(uplift_scores).detach().cpu().numpy().flatten()
    treatment = convert_to_tensor(treatment).detach().cpu().numpy().flatten().astype(int)
    y_obs = convert_to_tensor(y_obs).detach().cpu().numpy().flatten()

    sort_idx = np.argsort(uplift_scores)[::-1]
    treatment_sorted = treatment[sort_idx]
    y_sorted = y_obs[sort_idx]

    n_samples = len(uplift_scores)
    qini_x = np.arange(1, n_samples + 1) / n_samples

    cum_treated = np.cumsum(treatment_sorted)
    cum_control = np.arange(1, n_samples + 1) - cum_treated

    cum_y_treated = np.cumsum(y_sorted * treatment_sorted)
    cum_y_control = np.cumsum(y_sorted * (1 - treatment_sorted))

    with np.errstate(divide="ignore", invalid="ignore"):
        qini_y = cum_y_treated - cum_y_control * (
            cum_treated / np.where(cum_control == 0, 1.0, cum_control)
        )

    qini_x = np.insert(qini_x, 0, 0.0)
    qini_y = np.insert(qini_y, 0, 0.0)

    random_y = qini_x * qini_y[-1]

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    ax.plot(qini_x, qini_y, color="navy", linewidth=2.5, label="Model Policy")
    ax.plot(qini_x, random_y, color="gray", linestyle="--", linewidth=1.5, label="Random Policy")

    # Use np.trapezoid (NumPy 2.0+) or fallback to np.trapz / scipy.integrate.trapezoid
    if hasattr(np, "trapezoid"):
        qini_area = np.trapezoid(qini_y - random_y, qini_x)
    else:
        try:
            qini_area = np.trapz(qini_y - random_y, qini_x)
        except AttributeError:
            from scipy.integrate import trapezoid

            qini_area = trapezoid(qini_y - random_y, qini_x)

    # Re-import add_annotations here to ensure it's available in this scope
    from torchregress.viz.utils import add_annotations

    annotations = {
        "Qini Area Metric": qini_area,
    }
    add_annotations(ax, annotations, loc="lower right")

    ax.set_xlabel("Fraction of Population Treated (ranked by uplift)", fontweight="bold")
    ax.set_ylabel("Cumulative Uplift (Incremental Outcome)", fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    if return_figure:
        return fig
    elif created_fig:
        plt.tight_layout()
        plt.show()
    return None
