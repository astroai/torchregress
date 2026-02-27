# Constraints + Calibration Comparison

Script: `examples/constraints_calibration_comparison.py`

Compares a raw baseline against post-hoc calibrated + constrained outputs using:

- `VarianceTemperatureScaler`
- `IsotonicMeanCalibrator`
- `PITCalibrator`
- `NonCrossingSort` and `BoundedHead` (plus smoke usage of `NonNegativeHead`, `SimplexHead`, `SpectralNormWrapper`)

Reported metrics:

- `MAE`
- `NLL`
- `PITChi2`
- `CrossingRate`
- `BoundViolation`
- `train_s`, `eval_s`

## Run

```bash
uv run python examples/constraints_calibration_comparison.py
```

## Summary Artifact

```bash
uv run python examples/constraints_calibration_comparison.py \
  --summary-json-path reports/example_summaries/constraints_calibration_comparison_full.json
```
