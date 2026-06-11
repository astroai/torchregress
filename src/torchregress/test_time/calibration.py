"""Backward-compatible shim; canonical implementation is ``torchregress.calibration.shift``."""

from torchregress.calibration.shift import RepresentationShiftCalibrator

__all__ = ["RepresentationShiftCalibrator"]
