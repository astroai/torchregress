"""
Common visualization utilities for regression plots.

This module provides helper functions for consistent styling and
common operations used across different visualization functions.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, cast

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - hit only without the viz extra
    raise ImportError(
        "torchregress.viz requires matplotlib; pip install 'torchregress[viz]'"
    ) from exc

import numpy as np
from matplotlib.figure import Figure

# Metric-name terms whose lower value indicates a better model (TR-VIZ-01).
LOWER_IS_BETTER_TERMS = (
    "error",
    "loss",
    "mae",
    "mse",
    "rmse",
    "mape",
    "nll",
    "crps",
    "pinball",
    "brier",
    "log_loss",
    "ece",
    "mce",
    "deviance",
)


def is_lower_better(metric_name: str) -> bool:
    """Return True when ``metric_name`` contains a lower-is-better term."""
    name = metric_name.lower()
    return any(term in name for term in LOWER_IS_BETTER_TERMS)


_VALID_STYLES = ("default", "whitegrid", "darkgrid", "ticks", "minimal")


def set_style(
    style: str = "default",
    context: str = "notebook",
    font_scale: float = 1.0,
    rc: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Set consistent styling for matplotlib plots.

    Args:
        style: Style name ('default', 'whitegrid', 'darkgrid', 'ticks', or 'minimal')
        context: Context name ('paper', 'notebook', 'talk', or 'poster')
        font_scale: Scale factor for font sizes
        rc: Dictionary of rc parameter mappings to override

    Raises:
        ValueError: If ``style`` is not one of the recognized names.
    """
    # ponytail: matplotlib styling, no seaborn dependency.
    if style not in _VALID_STYLES:
        raise ValueError(f"Unknown style {style!r}; valid styles: {', '.join(_VALID_STYLES)}")

    if style in ("whitegrid", "default"):
        plt.style.use("seaborn-v0_8-whitegrid")
    elif style == "darkgrid":
        plt.style.use("seaborn-v0_8-darkgrid")
    elif style == "ticks":
        plt.style.use("seaborn-v0_8-ticks")
    else:  # minimal: its own spines-off, grid-off look
        plt.style.use("seaborn-v0_8-white")
        mpl.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": False,
            }
        )

    # Set font sizes based on context
    if context == "paper":
        mpl.rcParams.update({"font.size": 8 * font_scale})
    elif context == "notebook":
        mpl.rcParams.update({"font.size": 10 * font_scale})
    elif context == "talk":
        mpl.rcParams.update({"font.size": 14 * font_scale})
    elif context == "poster":
        mpl.rcParams.update({"font.size": 18 * font_scale})

    # Apply custom rc parameters if provided
    if rc is not None:
        mpl.rcParams.update(cast(Any, rc))


def create_grid_figure(
    n_plots: int,
    figsize: Tuple[int, int] = (12, 8),
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    sharex: bool = False,
    sharey: bool = False,
) -> Tuple[Figure, List[plt.Axes]]:
    """
    Create a figure with a grid of subplots.

    Args:
        n_plots: Number of plots needed
        figsize: Figure size (width, height)
        nrows: Number of rows (optional, determined automatically if None)
        ncols: Number of columns (optional, determined automatically if None)
        sharex: Whether to share x-axes among subplots
        sharey: Whether to share y-axes among subplots

    Returns:
        Figure and list of axes
    """
    # Determine grid dimensions if not specified
    if nrows is None and ncols is None:
        # Calculate a reasonable grid shape
        ncols = min(3, n_plots)
        nrows = int(np.ceil(n_plots / ncols))
    elif nrows is None:
        assert ncols is not None
        nrows = int(np.ceil(n_plots / ncols))
    elif ncols is None:
        assert nrows is not None
        ncols = int(np.ceil(n_plots / nrows))
    assert nrows is not None
    assert ncols is not None

    # Create figure and axes
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,  # Always return 2D array of axes
    )

    # Flatten axes array for easier access
    axes_flat = axes.flatten()

    # Hide unused subplots
    for i in range(n_plots, nrows * ncols):
        axes_flat[i].set_visible(False)

    return fig, cast(List[plt.Axes], axes_flat[:n_plots].tolist())


def add_identity_line(
    ax: plt.Axes,
    color: str = "gray",
    linestyle: str = "--",
    alpha: float = 0.8,
    label: Optional[str] = None,
) -> None:
    """
    Add identity (y=x) line to a plot.

    Args:
        ax: Matplotlib axes object
        color: Line color
        linestyle: Line style
        alpha: Transparency
        label: Label for legend
    """
    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Find common range
    min_val = max(xlim[0], ylim[0])
    max_val = min(xlim[1], ylim[1])

    # Ensure valid range (handle case where ranges don't overlap)
    if min_val >= max_val:
        min_val = min(xlim[0], ylim[0])
        max_val = max(xlim[1], ylim[1])

    # Plot identity line
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=color,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
    )

    # Restore original limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def add_zero_line(
    ax: plt.Axes,
    axis: str = "y",
    color: str = "gray",
    linestyle: str = "--",
    alpha: float = 0.8,
    label: Optional[str] = None,
) -> None:
    """
    Add a zero reference line to a plot.

    Args:
        ax: Matplotlib axes object
        axis: Which axis to add the zero line to ('x' or 'y')
        color: Line color
        linestyle: Line style
        alpha: Transparency
        label: Label for legend
    """
    if axis == "y":
        ax.axhline(y=0, color=color, linestyle=linestyle, alpha=alpha, label=label)
    elif axis == "x":
        ax.axvline(x=0, color=color, linestyle=linestyle, alpha=alpha, label=label)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis}")


def save_figure(
    fig: Figure,
    filename: str,
    directory: str = "./figures",
    formats: Tuple[str, ...] = ("png", "pdf"),
    dpi: int = 300,
    transparent: bool = False,
    bbox_inches: str = "tight",
) -> List[str]:
    """
    Save a figure in multiple formats.

    Args:
        fig: Matplotlib figure
        filename: Base filename (without extension)
        directory: Output directory
        formats: Formats to save (default: ``("png", "pdf")``)
        dpi: Resolution for raster formats
        transparent: Whether to use transparent background
        bbox_inches: Bounding box setting

    Returns:
        Paths of the files that were written.
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Save in each format
    written: List[str] = []
    for fmt in formats:
        output_path = os.path.join(directory, f"{filename}.{fmt}")
        fig.savefig(
            output_path, format=fmt, dpi=dpi, transparent=transparent, bbox_inches=bbox_inches
        )
        written.append(output_path)
        print(f"Saved figure to {output_path}")

    return written


def add_annotations(
    ax: plt.Axes,
    annotations: Dict[str, Any],
    loc: str = "upper right",
    fontsize: int = 10,
    frameon: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    Add annotations (e.g., metrics) to a plot.

    Args:
        ax: Matplotlib axes
        annotations: Dictionary of annotation name-value pairs
        loc: Location on plot ('upper right', 'upper left', etc.)
        fontsize: Font size for annotations
        frameon: Whether to draw a frame around the annotations
        title: Optional title for the annotations box
    """
    # Convert annotations to strings
    annotation_strings = []
    if title:
        annotation_strings.append(title)

    for name, value in annotations.items():
        if isinstance(value, float):
            annotation_strings.append(f"{name}: {value:.4f}")
        else:
            annotation_strings.append(f"{name}: {value}")

    # Add text box with annotations
    ax.annotate(
        "\n".join(annotation_strings),
        xy=(0.02, 0.98) if "left" in loc else (0.98, 0.98),
        xycoords="axes fraction",
        ha="left" if "left" in loc else "right",
        va="top",
        fontsize=fontsize,
        bbox=(
            dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray")
            if frameon
            else None
        ),
    )


def create_color_palette(
    n_colors: int, palette_name: str = "tab10", as_hex: bool = False, as_cmap: bool = False
) -> Any:
    """
    Create a color palette for consistent colors across plots.

    Args:
        n_colors: Number of colors needed
        palette_name: Name of colormap or palette to use
        as_hex: Return colors as hex strings instead of RGB tuples
        as_cmap: Return a colormap object instead of color list

    Returns:
        List of colors or colormap
    """
    # ponytail: matplotlib colormaps, no seaborn dependency.
    cmap = plt.get_cmap(palette_name)

    if as_cmap:
        return cmap

    def _sample(i: int) -> Any:
        # Qualitative (ListedColormap) palettes have discrete slots; sample them
        # directly instead of interpolating between neighbouring colours (TR-VIZ-06..21).
        if isinstance(cmap, mpl.colors.ListedColormap):
            return cmap(i % cmap.N)
        if n_colors == 1:
            return cmap(0.0)
        return cmap(i / (n_colors - 1))

    if as_hex:
        return [mpl.colors.rgb2hex(_sample(i)[:3]) for i in range(n_colors)]

    return [_sample(i) for i in range(n_colors)]


def enable_latex_rendering(enable: bool = True) -> bool:
    """
    Enable or disable LaTeX rendering for mathematical symbols in plots.

    Args:
        enable: Whether to enable LaTeX rendering

    Returns:
        Whether LaTeX rendering was successfully enabled
    """
    if not enable:
        # Disable LaTeX rendering
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            }
        )
        return False

    # Try to enable LaTeX rendering
    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
        )
        # Test if LaTeX works by rendering a simple formula
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, r"$\alpha + \beta = \gamma$")
        fig.canvas.draw()
        plt.close(fig)
        return True
    except Exception:
        # If LaTeX fails, fall back to standard rendering
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            }
        )
        print("Warning: LaTeX rendering failed, using standard math rendering")
        return False


def format_metric_label(metric_name: str, use_latex: bool = True) -> str:
    """
    Format metric name with proper mathematical notation.

    Args:
        metric_name: Name of the metric
        use_latex: Whether to use LaTeX formatting if available

    Returns:
        Formatted metric label
    """
    if not use_latex:
        return metric_name

    # Common metric name mappings to LaTeX
    latex_mappings = {
        "mse": r"$\mathrm{MSE}$",
        "rmse": r"$\mathrm{RMSE}$",
        "mae": r"$\mathrm{MAE}$",
        "r2": r"$R^2$",
        "mean_absolute_error": r"$\mathrm{MAE}$",
        "mean_squared_error": r"$\mathrm{MSE}$",
        "root_mean_squared_error": r"$\mathrm{RMSE}$",
        "median_absolute_error": r"$\mathrm{MedAE}$",
        "explained_variance": r"$\mathrm{Explained~Variance}$",
        "max_error": r"$\mathrm{Max~Error}$",
        "mean_absolute_percentage_error": r"$\mathrm{MAPE}$",
        "mape": r"$\mathrm{MAPE}$",
        "nll": r"$\mathrm{NLL}$",
        "negative_log_likelihood": r"$\mathrm{NLL}$",
        "calibration_error": r"$\mathrm{Calibration~Error}$",
        "sharpness": r"$\mathrm{Sharpness}$",
        "pinball_loss": r"$\mathrm{Pinball~Loss}$",
        "crps": r"$\mathrm{CRPS}$",
        "continuous_ranked_probability_score": r"$\mathrm{CRPS}$",
    }

    # Check for exact matches
    if metric_name.lower() in latex_mappings:
        return latex_mappings[metric_name.lower()]

    # Check for partial matches
    for key, value in latex_mappings.items():
        if key in metric_name.lower():
            return value

    # If no match found, return the original name
    return metric_name
