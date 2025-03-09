"""
Metrics for evaluating regression models.
"""

# Import point prediction metrics
from torchregression.metrics.point import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    mean_squared_log_error,
    normalized_rmse,
    # Robust metrics
    huber_loss,
    trimmed_mean_squared_error,
    median_absolute_deviation,
    normalized_median_absolute_deviation,
    # Outlier metrics
    outlier_fraction,
    # Report generator
    regression_metrics_report
)

# Import distribution prediction metrics
from torchregression.metrics.distribution import (
    probability_integral_transform,
    continuous_ranked_probability_score,
    energy_score,
    distribution_metrics_report
)

# Import calibration metrics
from torchregression.metrics.calibration import (
    expected_calibration_error,
    calibration_error,
    marginal_calibration_error,
    calibration_metrics_report
)

# Import interval metrics
from torchregression.metrics.interval import (
    interval_score,
    prediction_interval_coverage_probability,
    winkler_score,
    interval_metrics_report
)

# Import OOD metrics
from torchregression.metrics.ood import (
    mahalanobis_distance,
    typicality_score,
    entropy_score,
    kernel_density_score,
    ood_metrics_report
)

# For backward compatibility
quantile_score = continuous_ranked_probability_score
