"""Unified calibration package surface."""

from __future__ import annotations

import torchregress.calibration as calibration


def test_calibration_package_exports_posthoc_metrics_and_shift() -> None:
    for symbol in calibration.__all__:
        assert hasattr(calibration, symbol), symbol

    assert calibration.VarianceTemperatureScaler is not None
    assert calibration.expected_calibration_error is not None
    assert calibration.RepresentationShiftCalibrator is not None
