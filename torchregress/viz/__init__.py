"""
Visualization utilities for regression models.

This module provides visualization functions for diagnostic, monitoring,
and results presentation.

Style Guide for Visualization Functions:
---------------------------------------

1. Parameters:
   - All functions should accept at minimum:
     - Main data arguments specific to the plot
     - figsize: Tuple[int, int] for figure size
     - title: str for plot title
     - return_figure: bool to control return behavior
     - ax: Optional[plt.Axes] to support existing axes

2. Colors:
   - Use sensible defaults from standard matplotlib color names
   - Allow customization via parameters
   - Support colorblind-friendly palettes (viridis, plasma, etc.)

3. Aesthetics:
   - Use bold fonts for titles and labels
   - Set meaningful axis limits
   - Add grid lines for readability
   - Include legends when multiple data series are plotted

4. Return Behavior:
   - If return_figure=True: Return the Figure object
   - If ax is provided: Plot on existing axes, don't show or return
   - Otherwise: Display the plot and return None

5. Error Handling:
   - Validate inputs and provide helpful error messages
   - Handle edge cases (empty data, NaN/Inf values)
   - Include appropriate warnings when approximations are made
"""

# Import diagnostic plotting utilities
from torchregress.viz.diagnostic import (
    plot_reliability_diagram,
    plot_residuals,
    plot_prediction_intervals,
    plot_qq_plot,
    plot_residual_histogram,
    plot_distribution_comparison,
    plot_calibration_curve,
)

# Import training monitoring plots
from torchregress.viz.monitoring import (
    plot_learning_curves,
    plot_validation_metrics,
    plot_early_stopping,
    plot_lr_find_results,
)

# Import results visualization tools
from torchregress.viz.results import (
    plot_performance_comparison,
    plot_parameter_sensitivity,
    plot_feature_importance,
    plot_model_ensemble_contributions,
)

# Import common utility functions
from torchregress.viz.utils import (
    set_style,
    create_grid_figure,
    add_identity_line,
    save_figure,
    add_annotations,
    create_color_palette,
    enable_latex_rendering,
    format_metric_label,
)

__all__ = [
    "plot_reliability_diagram",
    "plot_residuals",
    "plot_prediction_intervals",
    "plot_qq_plot",
    "plot_residual_histogram",
    "plot_distribution_comparison",
    "plot_calibration_curve",
    "plot_learning_curves",
    "plot_validation_metrics",
    "plot_early_stopping",
    "plot_lr_find_results",
    "plot_performance_comparison",
    "plot_parameter_sensitivity",
    "plot_feature_importance",
    "plot_model_ensemble_contributions",
    "set_style",
    "create_grid_figure",
    "add_identity_line",
    "save_figure",
    "add_annotations",
    "create_color_palette",
    "enable_latex_rendering",
    "format_metric_label",
]
