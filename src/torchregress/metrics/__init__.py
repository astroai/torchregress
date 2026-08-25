"""
Metrics for evaluating regression models.
"""

# torchmetrics classes (MeanSquaredError, MeanAbsoluteError, R2Score) are no longer
# re-exported.  Import them directly from ``torchmetrics`` instead.

from torchmetrics import (  # noqa: F401  # kept for soft backward compatibility; do not use in new code
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)

from .calibration import (
    ExpectedCalibrationError,
    MarginalCalibrationError,
    bias,
    calibration_metrics_report,
    calibration_score,
    expected_calibration_error,
    marginal_calibration_error,
)
from .censored import (
    censoring_rate,
    concordance_index,
    interval_overlap_rate,
    observed_mae,
)
from .decision import (
    RejectionPolicy,
    RiskCoverageCurve,
    risk_coverage_curve,
)
from .distribution import (
    ContinuousRankedProbabilityScore,
    DawidSebastianiScore,
    EnergyScore,
    PinballMetric,
    VarioScore,
    WassersteinGaussian,
    conditional_density_estimation_loss,
    continuous_ranked_probability_score,
    crps_from_samples,
    crps_gaussian,
    distribution_metrics_report,
    dss_score,
    energy_score,
    gaussian_nll,
    highest_posterior_density_coverage,
    highest_posterior_density_level,
    kolmogorov_smirnov_uniform_statistic,
    pinball_loss,
    pinball_metric,
    probability_integral_transform,
    vario_score,
    wasserstein_gaussian_p2,
)
from .ensemble import (
    EnsembleIntervalMetrics,
    GaussianNLLEnsemble,
    ensemble_interval_bounds,
    ensemble_interval_metrics,
    ensemble_mean,
    ensemble_statistics,
    ensemble_std,
    ensemble_variance_decomposition,
    gaussian_nll_ensemble,
    uncertainty_decomposition,
)
from .interval import (
    IntervalScore,
    MeanPredictionIntervalWidth,
    PredictionIntervalCoverageProbability,
    Sharpness,
    interval_metrics_report,
    interval_score,
    prediction_interval_coverage,
    prediction_interval_coverage_probability,
    sharpness,
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
from .ordinal import (
    mean_absolute_class_error,
    ordinal_accuracy,
    quadratic_weighted_kappa,
)
from .point import (
    HuberMetric,
    MedianAbsoluteDeviation,
    MedianAbsoluteError,
    NormalizedMedianAbsoluteDeviation,
    NormalizedRMSE,
    OutlierFraction,
    TrimmedMeanSquaredError,
    attenuation_factor,
    huber_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_deviation,
    median_absolute_error,
    normalized_rmse,
    r2_score,
    regression_metrics_report,
    rmse,
    tail_mae,
    tail_rmse,
    trimmed_mean_squared_error,
)
from .tac import (
    TaskAgnosticCorrelations,
    task_agnostic_correlations,
)
from .uncertain import (
    consistency_error,
    noisy_target_gaussian_nll,
    pseudo_label_acceptance_rate,
    uncertain_gt_metrics_report,
)

__all__ = [
    # Calibration
    "ExpectedCalibrationError",
    "MarginalCalibrationError",
    "bias",
    "expected_calibration_error",
    "marginal_calibration_error",
    "calibration_score",
    "calibration_metrics_report",
    # Censored
    "censoring_rate",
    "observed_mae",
    "concordance_index",
    "interval_overlap_rate",
    # Distribution
    "ContinuousRankedProbabilityScore",
    "DawidSebastianiScore",
    "EnergyScore",
    "PinballMetric",
    "VarioScore",
    "WassersteinGaussian",
    "conditional_density_estimation_loss",
    "continuous_ranked_probability_score",
    "crps_from_samples",
    "dss_score",
    "energy_score",
    "gaussian_nll",
    "crps_gaussian",
    "highest_posterior_density_coverage",
    "highest_posterior_density_level",
    "kolmogorov_smirnov_uniform_statistic",
    "pinball_loss",
    "pinball_metric",
    "probability_integral_transform",
    "vario_score",
    "wasserstein_gaussian_p2",
    "distribution_metrics_report",
    # Decision
    "RiskCoverageCurve",
    "RejectionPolicy",
    "risk_coverage_curve",
    # Ensemble
    "GaussianNLLEnsemble",
    "EnsembleIntervalMetrics",
    "gaussian_nll_ensemble",
    "ensemble_interval_bounds",
    "ensemble_interval_metrics",
    "ensemble_mean",
    "ensemble_std",
    "ensemble_statistics",
    "ensemble_variance_decomposition",
    "uncertainty_decomposition",
    # Interval
    "IntervalScore",
    "MeanPredictionIntervalWidth",
    "PredictionIntervalCoverageProbability",
    "Sharpness",
    "interval_score",
    "sharpness",
    "prediction_interval_coverage",
    "prediction_interval_coverage_probability",
    "interval_metrics_report",
    # Multivariate
    "MultivariateMAE",
    "MultivariateRMSE",
    "TaskAgnosticCorrelations",
    "task_agnostic_correlations",
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
    # Ordinal
    "ordinal_accuracy",
    "mean_absolute_class_error",
    "quadratic_weighted_kappa",
    # Point
    "HuberMetric",
    "MedianAbsoluteError",
    "MedianAbsoluteDeviation",
    "NormalizedMedianAbsoluteDeviation",
    "NormalizedRMSE",
    "OutlierFraction",
    "TrimmedMeanSquaredError",
    "attenuation_factor",
    "mean_squared_error",
    "mean_absolute_error",
    "median_absolute_error",
    "huber_loss",
    "rmse",
    "r2_score",
    "trimmed_mean_squared_error",
    "median_absolute_deviation",
    "normalized_rmse",
    "tail_mae",
    "tail_rmse",
    "regression_metrics_report",
    # Uncertain/weak ground-truth metrics
    "noisy_target_gaussian_nll",
    "consistency_error",
    "pseudo_label_acceptance_rate",
    "uncertain_gt_metrics_report",
    # Torchmetrics direct exports — deprecated; import directly from torchmetrics
    # "MeanSquaredError",
    # "MeanAbsoluteError",
    # "R2Score",
]
