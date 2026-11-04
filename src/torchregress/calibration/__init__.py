"""Unified calibration surface for torchregress.

This package groups three related concerns:

- **posthoc** — validation-set transforms for means, variances, and PIT values
- **metrics** — calibration scoring rules (ECE, marginal error, reports)
- **shift** — representation-shift temperature scaling at test time

EIV measurement-error calibration lives in ``torchregress.algorithms.rc.RegressionCalibration``.
"""

from .metrics import (
    ExpectedCalibrationError,
    MarginalCalibrationError,
    bias,
    calibration_metrics_report,
    calibration_score,
    expected_calibration_error,
    marginal_calibration_error,
)
from .multisource import (
    CommonScaleSplineHead,
    ConditionBiasHead,
    RelativeSurveySplineHead,
    bspline_basis_with_derivative,
)
from .posthoc import (
    IsotonicMeanCalibrator,
    PITCalibrator,
    VarianceRestoringCalibrator,
    VarianceTemperatureScaler,
)
from .semicp import SemiConformalCalibrator
from .shift import BinnedLabelShiftEstimator, RepresentationShiftInflator

__all__ = [
    "CommonScaleSplineHead",
    "ConditionBiasHead",
    "RelativeSurveySplineHead",
    "bspline_basis_with_derivative",
    # posthoc transforms
    "VarianceTemperatureScaler",
    "IsotonicMeanCalibrator",
    "PITCalibrator",
    "VarianceRestoringCalibrator",
    # conformal calibration under shift
    "SemiConformalCalibrator",
    # calibration metrics
    "ExpectedCalibrationError",
    "MarginalCalibrationError",
    "bias",
    "calibration_metrics_report",
    "calibration_score",
    "expected_calibration_error",
    "marginal_calibration_error",
    # shift-aware test-time calibration
    "RepresentationShiftInflator",
    "BinnedLabelShiftEstimator",
]
