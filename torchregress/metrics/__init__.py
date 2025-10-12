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

# Convenience aliases for common metrics
mse = mean_squared_error
mae = mean_absolute_error
mape = mean_absolute_percentage_error
picp = prediction_interval_coverage_probability
ece = expected_calibration_error
mce = marginal_calibration_error
crps = continuous_ranked_probability_score
pit = probability_integral_transform


# Convenience functions
def rmse(y_pred, y_true, **kwargs):
    """
    Root Mean Squared Error (convenience function).

    Args:
        y_pred: Predicted values
        y_true: True values
        **kwargs: Additional arguments passed to mean_squared_error

    Returns:
        RMSE value

    Examples:
        >>> import torch
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])
        >>> y_true = torch.tensor([1.1, 2.1, 2.9])
        >>> error = rmse(y_pred, y_true)
    """
    import torch

    return torch.sqrt(mean_squared_error(y_pred, y_true, **kwargs))


def mpiw(lower, upper):
    """
    Mean Prediction Interval Width (convenience function).

    Args:
        lower: Lower bounds of prediction intervals
        upper: Upper bounds of prediction intervals

    Returns:
        Mean interval width

    Examples:
        >>> import torch
        >>> lower = torch.tensor([1.0, 2.0, 3.0])
        >>> upper = torch.tensor([2.0, 3.0, 4.0])
        >>> width = mpiw(lower, upper)
    """
    import torch

    return torch.mean(upper - lower)


def gaussian_nll(mean, target, var, reduction="mean"):
    """
    Gaussian Negative Log-Likelihood (convenience function).

    Computes the negative log-likelihood assuming a Gaussian distribution
    with predicted mean and variance.

    Args:
        mean: Predicted means
        target: True values
        var: Predicted variances
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Returns:
        NLL value

    Examples:
        >>> import torch
        >>> mean = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([1.1, 2.1, 2.9])
        >>> var = torch.tensor([0.1, 0.1, 0.1])
        >>> nll = gaussian_nll(mean, target, var)
    """
    import math

    import torch

    nll = 0.5 * (torch.log(var) + (target - mean) ** 2 / var + math.log(2 * math.pi))

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    else:
        return nll


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
        # Convenience aliases
        "mse",
        "mae",
        "mape",
        "rmse",
        "picp",
        "mpiw",
        "ece",
        "mce",
        "crps",
        "pit",
        "gaussian_nll",
    ]
)
