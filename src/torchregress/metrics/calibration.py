"""Backward-compatible shim; canonical implementation is ``torchregress.calibration.metrics``."""

from torchregress.calibration.metrics import (
    ExpectedCalibrationError,
    MarginalCalibrationError,
    bias,
    calibration_metrics_report,
    calibration_score,
    expected_calibration_error,
    marginal_calibration_error,
)

__all__ = [
    "ExpectedCalibrationError",
    "MarginalCalibrationError",
    "bias",
    "calibration_metrics_report",
    "calibration_score",
    "expected_calibration_error",
    "marginal_calibration_error",
]
