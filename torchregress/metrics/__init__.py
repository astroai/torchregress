"""
Metrics for evaluating regression models.

This module provides metrics for both point prediction and probabilistic regression models,
with a consistent API where prediction inputs precede target inputs.
"""

# Import point prediction metrics
from torchregress.metrics.point import (
    # Basic metrics
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    median_absolute_error,
    normalized_rmse,
    huber_loss,
    trimmed_mean_squared_error,
    median_absolute_deviation,
    normalized_median_absolute_deviation,
    outlier_fraction,
    regression_metrics_report,
)

# Import distribution prediction metrics
from torchregress.metrics.distribution import (
    probability_integral_transform,
    continuous_ranked_probability_score,
    energy_score,
    distribution_metrics_report,
)

# Import calibration metrics
from torchregress.metrics.calibration import (
    expected_calibration_error,
    marginal_calibration_error,
    calibration_metrics_report,
)

# Import interval metrics
from torchregress.metrics.interval import (
    interval_score,
    prediction_interval_coverage_probability,
    interval_metrics_report,
)

# Import OOD metrics
from torchregress.metrics.ood import (
    mahalanobis_distance,
    typicality_score,
    entropy_score,
    kernel_density_score,
    ood_metrics_report,
)
