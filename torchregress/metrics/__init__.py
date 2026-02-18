"""
Metrics for evaluating regression models.
"""

from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from .calibration import (
    ExpectedCalibrationError,
    MarginalCalibrationError,
    bias,
    calibration_metrics_report,
    expected_calibration_error,
    marginal_calibration_error,
)
from .decision import (
    RejectionPolicy,
    RiskCoverageCurve,
    risk_coverage_curve,
)
from .distribution import (
    ContinuousRankedProbabilityScore,
    EnergyScore,
    continuous_ranked_probability_score,
    distribution_metrics_report,
    energy_score,
    probability_integral_transform,
)
from .ensemble import (
    EnsembleIntervalMetrics,
    GaussianNLLEnsemble,
    ensemble_mean,
    ensemble_statistics,
    ensemble_std,
    ensemble_variance_decomposition,
    uncertainty_decomposition,
)
from .interval import (
    IntervalScore,
    MeanPredictionIntervalWidth,
    PredictionIntervalCoverageProbability,
    interval_metrics_report,
    interval_score,
    prediction_interval_coverage_probability,
)
from .multivariate import (
    MultivariateMAE,
    MultivariateRMSE,
)
from .ood import (
    EntropyScore,
    KernelDensityScore,
    MahalanobisDistance,
    TypicalityScore,
    entropy_score,
    kernel_density_score,
    mahalanobis_distance,
    ood_metrics_report,
    typicality_score,
)
from .point import (
    HuberMetric,
    MedianAbsoluteDeviation,
    MedianAbsoluteError,
    NormalizedMedianAbsoluteDeviation,
    NormalizedRMSE,
    OutlierFraction,
    TrimmedMeanSquaredError,
    huber_loss,
    mae,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_deviation,
    median_absolute_error,
    mse,
    normalized_rmse,
    regression_metrics_report,
    rmse,
    trimmed_mean_squared_error,
)

__all__ = [
    # Calibration
    "ExpectedCalibrationError",
    "MarginalCalibrationError",
    "bias",
    "expected_calibration_error",
    "marginal_calibration_error",
    "calibration_metrics_report",
    # Distribution
    "ContinuousRankedProbabilityScore",
    "EnergyScore",
    "continuous_ranked_probability_score",
    "energy_score",
    "probability_integral_transform",
    "distribution_metrics_report",
    # Decision
    "RiskCoverageCurve",
    "RejectionPolicy",
    "risk_coverage_curve",
    # Ensemble
    "GaussianNLLEnsemble",
    "EnsembleIntervalMetrics",
    "ensemble_mean",
    "ensemble_std",
    "ensemble_statistics",
    "ensemble_variance_decomposition",
    "uncertainty_decomposition",
    # Interval
    "IntervalScore",
    "MeanPredictionIntervalWidth",
    "PredictionIntervalCoverageProbability",
    "interval_score",
    "prediction_interval_coverage_probability",
    "interval_metrics_report",
    # Multivariate
    "MultivariateMAE",
    "MultivariateRMSE",
    # OOD
    "EntropyScore",
    "KernelDensityScore",
    "MahalanobisDistance",
    "TypicalityScore",
    "entropy_score",
    "kernel_density_score",
    "mahalanobis_distance",
    "typicality_score",
    "ood_metrics_report",
    # Point
    "HuberMetric",
    "MedianAbsoluteError",
    "MedianAbsoluteDeviation",
    "NormalizedMedianAbsoluteDeviation",
    "NormalizedRMSE",
    "OutlierFraction",
    "TrimmedMeanSquaredError",
    "mse",
    "mean_squared_error",
    "mae",
    "mean_absolute_error",
    "median_absolute_error",
    "huber_loss",
    "rmse",
    "trimmed_mean_squared_error",
    "median_absolute_deviation",
    "normalized_rmse",
    "regression_metrics_report",
    # Torchmetrics direct exports
    "MeanSquaredError",
    "MeanAbsoluteError",
    "R2Score",
]
