"""
Results visualization tools for regression models.

This module provides visualization utilities for presenting
and comparing regression model results.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

# Import visualization utilities
from torchregress.viz.utils import create_color_palette

# Set random seed for reproducibility
np.random.seed(42)


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

    # Create DataFrame for easier manipulation
    model_names = list(metrics.keys())
    data = {}
    for metric in metrics_to_include:
        data[metric] = [metrics[model][metric] for model in model_names]

    df = pd.DataFrame(data, index=model_names)

    # Sort by specified metric if requested
    if sort_by is not None and sort_by in metrics_to_include:
        ascending = not higher_is_better.get(sort_by, True)
        df = df.sort_values(by=sort_by, ascending=ascending)

    # Determine best model for each metric if highlighting is requested
    best_values = {}
    if highlight_best:
        for metric in metrics_to_include:
            if higher_is_better.get(metric, True):
                best_values[metric] = df[metric].max()
            else:
                best_values[metric] = df[metric].min()

    # Determine colors for models
    colors = cast(List[Any], create_color_palette(len(df.index), palette_name=color_palette))

    # Create plot based on specified type
    if plot_type == "bar":
        return _plot_performance_bar(
            df, best_values, figsize, title, higher_is_better, colors, return_figure, ax
        )
    elif plot_type == "radar":
        return _plot_performance_radar(
            df, best_values, figsize, title, higher_is_better, colors, return_figure
        )
    elif plot_type == "heatmap":
        return _plot_performance_heatmap(
            df, best_values, figsize, title, higher_is_better, return_figure
        )
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")


def _plot_performance_bar(
    df: pd.DataFrame,
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
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Number of models and metrics
    n_models = len(df.index)
    n_metrics = len(df.columns)

    # Set width of bars
    bar_width = 0.8 / n_models

    # Set positions of bars on x-axis
    indices = np.arange(n_metrics)

    # Create bars for each model
    for i, model_name in enumerate(df.index):
        values = df.loc[model_name].values
        bars = ax.bar(
            indices + i * bar_width,
            values,
            bar_width,
            label=model_name,
            color=colors[i % len(colors)],
        )

        # Highlight best model for each metric
        if best_values:
            for j, metric in enumerate(df.columns):
                if np.isclose(df.loc[model_name, metric], best_values[metric]):
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

    # Add labels, title and legend
    ax.set_xlabel("Metric", fontweight="bold")
    ax.set_ylabel("Value", fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(indices + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontweight="bold")

    # Use LaTeX for proper math rendering if available
    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )
        # Replace common metric names with LaTeX
        metric_labels = []
        for metric in df.columns:
            if "rmse" in metric.lower():
                metric_labels.append(r"$\mathrm{RMSE}$")
            elif "mae" in metric.lower():
                metric_labels.append(r"$\mathrm{MAE}$")
            elif "r2" in metric.lower():
                metric_labels.append(r"$R^2$")
            else:
                metric_labels.append(metric)
        ax.set_xticklabels(metric_labels, rotation=45, ha="right")
    except (ImportError, RuntimeError):
        pass  # Skip LaTeX rendering if not available

    ax.legend(loc="best", frameon=True, fancybox=True, framealpha=0.9)

    # Add grid
    ax.grid(True, axis="y", alpha=0.3)

    # Add annotations for which direction is better
    for i, metric in enumerate(df.columns):
        direction = "↑" if higher_is_better.get(metric, True) else "↓"
        ax.annotate(
            direction,
            xy=(i, 0),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=12,
        )

    plt.tight_layout()

    if return_figure:
        return fig
    elif ax is None:  # Only show if we created the figure here
        plt.show()

    return None


def _plot_performance_radar(
    df: pd.DataFrame,
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
    n_metrics = len(df.columns)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()

    # Add the last angle to close the polygon
    angles += angles[:1]

    # Normalize metrics to range [0, 1] for radar plot
    df_normalized = pd.DataFrame(index=df.index)
    for metric in df.columns:
        min_val = df[metric].min()
        max_val = df[metric].max()

        if higher_is_better.get(metric, True):
            # Higher is better: higher values should be further from center
            df_normalized[metric] = (
                (df[metric] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            )
        else:
            # Lower is better: lower values should be further from center
            df_normalized[metric] = (
                1 - (df[metric] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            )

    # Plot each model
    for i, model_name in enumerate(df.index):
        values = df_normalized.loc[model_name].values.tolist()
        values += values[:1]  # Close the polygon

        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=model_name)
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

    # Set labels and ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [
            f"{metric}\n({'↑' if higher_is_better.get(metric, True) else '↓'})"
            for metric in df.columns
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
    df: pd.DataFrame,
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
    df_normalized = pd.DataFrame(index=df.index)
    for metric in df.columns:
        min_val = df[metric].min()
        max_val = df[metric].max()

        if higher_is_better.get(metric, True):
            # Higher is better: higher values = better
            df_normalized[metric] = (
                (df[metric] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            )
        else:
            # Lower is better: lower values = better
            df_normalized[metric] = (
                1 - (df[metric] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            )

    # Create heatmap
    im = ax.imshow(df_normalized.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Performance (normalized)", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(
        [
            f"{metric}\n({'↑' if higher_is_better.get(metric, True) else '↓'})"
            for metric in df.columns
        ]
    )
    ax.set_yticklabels(df.index)

    # Rotate x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data and add annotations with actual values
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            metric_value = df.iloc[i, j]
            is_best = best_values and np.isclose(metric_value, best_values[df.columns[j]])

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
                color="black" if df_normalized.iloc[i, j] > 0.5 else "white",
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
    if isinstance(importance_values, torch.Tensor):
        importance_values = importance_values.detach().cpu().numpy()
    importance_values = np.array(importance_values)

    if importance_errors is not None:
        if isinstance(importance_errors, torch.Tensor):
            importance_errors = importance_errors.detach().cpu().numpy()
        importance_errors = np.array(importance_errors)

    # Sort by importance if requested
    if sort_values:
        idx = np.argsort(importance_values)
        if not horizontal:  # For vertical, we want descending order
            idx = idx[::-1]

        feature_names = [feature_names[i] for i in idx]
        importance_values = importance_values[idx]

        if importance_errors is not None:
            importance_errors = importance_errors[idx]

    # Limit to top N features if specified
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

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Set positions for bars
    y_pos = np.arange(len(feature_names))

    # Create bars
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

        # Add values on bars if requested
        if show_values:
            for i, bar in enumerate(bars):
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

        # Add values on bars if requested
        if show_values:
            for i, bar in enumerate(bars):
                height = bar.get_height()
                label_position = height * 1.05
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_position,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
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
    elif ax is None:  # Only show if we created the figure here
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
    if ax is None:
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
    elif ax is None:  # Only show if we created the figure here
        plt.show()

    return None
