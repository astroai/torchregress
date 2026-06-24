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
from .posthoc import IsotonicMeanCalibrator, PITCalibrator, VarianceTemperatureScaler
from .semicp import SemiConformalCalibrator
from .shift import BinnedLabelShiftEstimator, RepresentationShiftInflator

__all__ = [
    # posthoc transforms
    "VarianceTemperatureScaler",
    "IsotonicMeanCalibrator",
    "PITCalibrator",
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
