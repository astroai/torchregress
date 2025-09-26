"""
Metrics for evaluating regression models.

This module provides metrics for both point prediction and probabilistic regression models,
with a consistent API where prediction inputs precede target inputs.
"""

# Import point prediction metrics
# Import calibration metrics
from torchregress.metrics.calibration import (
    calibration_metrics_report,
    expected_calibration_error,
    marginal_calibration_error,
)

# Import distribution prediction metrics
from torchregress.metrics.distribution import (
    continuous_ranked_probability_score,
    distribution_metrics_report,
    energy_score,
    probability_integral_transform,
)

# Ensemble metrics
from torchregress.metrics.ensemble import (
    ensemble_interval_bounds,
    ensemble_interval_metrics,
    ensemble_statistics,
    gaussian_nll_ensemble,
    uncertainty_decomposition,
)

# Import interval metrics
from torchregress.metrics.interval import (
    interval_metrics_report,
    interval_score,
    prediction_interval_coverage_probability,
)

# Multivariate metrics
from torchregress.metrics.multivariate import (
    multivariate_mae,
    multivariate_rmse,
)

# Import OOD metrics
from torchregress.metrics.ood import (
    entropy_score,
    kernel_density_score,
    mahalanobis_distance,
    ood_metrics_report,
    typicality_score,
)
from torchregress.metrics.point import (
    explained_variance_score,
    huber_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    # Basic metrics
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_deviation,
    median_absolute_error,
    normalized_median_absolute_deviation,
    normalized_rmse,
    outlier_fraction,
    r2_score,
    regression_metrics_report,
    trimmed_mean_squared_error,
)

# Update __all__
__all__ = []
__all__.extend(
    [
        # Calibration metrics
        "calibration_metrics_report",
        "expected_calibration_error",
        "marginal_calibration_error",
        # Distribution prediction metrics
        "continuous_ranked_probability_score",
        "distribution_metrics_report",
        "energy_score",
        "probability_integral_transform",
        # Ensemble metrics
        "ensemble_interval_bounds",
        "ensemble_interval_metrics",
        "ensemble_statistics",
        "gaussian_nll_ensemble",
        "uncertainty_decomposition",
        # Interval metrics
        "interval_metrics_report",
        "interval_score",
        "prediction_interval_coverage_probability",
        # Multivariate metrics
        "multivariate_mae",
        "multivariate_rmse",
        # OOD metrics
        "entropy_score",
        "kernel_density_score",
        "mahalanobis_distance",
        "ood_metrics_report",
        "typicality_score",
        # Point prediction metrics
        "explained_variance_score",
        "huber_loss",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
        "mean_squared_log_error",
        "median_absolute_deviation",
        "median_absolute_error",
        "normalized_median_absolute_deviation",
        "normalized_rmse",
        "outlier_fraction",
        "r2_score",
        "regression_metrics_report",
        "trimmed_mean_squared_error",
    ]
)
