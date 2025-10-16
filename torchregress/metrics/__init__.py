"""
Metrics for evaluating regression models.
"""

from .calibration import (
    ExpectedCalibrationError,
    MarginalCalibrationError,
)
from .distribution import (
    ContinuousRankedProbabilityScore,
    EnergyScore,
)
from .ensemble import (
    GaussianNLLEnsemble,
    EnsembleIntervalMetrics,
)
from .interval import (
    IntervalScore,
    PredictionIntervalCoverageProbability,
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
)
from .point import (
    HuberLoss,
    MedianAbsoluteError,
    MedianAbsoluteDeviation,
    NormalizedMedianAbsoluteDeviation,
    NormalizedRMSE,
    OutlierFraction,
    TrimmedMeanSquaredError,
)

__all__ = [
    # Calibration
    "ExpectedCalibrationError",
    "MarginalCalibrationError",
    # Distribution
    "ContinuousRankedProbabilityScore",
    "EnergyScore",
    # Ensemble
    "GaussianNLLEnsemble",
    "EnsembleIntervalMetrics",
    # Interval
    "IntervalScore",
    "PredictionIntervalCoverageProbability",
    # Multivariate
    "MultivariateMAE",
    "MultivariateRMSE",
    # OOD
    "EntropyScore",
    "KernelDensityScore",
    "MahalanobisDistance",
    "TypicalityScore",
    # Point
    "HuberLoss",
    "MedianAbsoluteError",
    "MedianAbsoluteDeviation",
    "NormalizedMedianAbsoluteDeviation",
    "NormalizedRMSE",
    "OutlierFraction",
    "TrimmedMeanSquaredError",
]