# TransferZ Conformal Photo-z Comparison

Script: `examples/photoz_transferz_conformal_comparison.py`

This is the dedicated real-data conformal benchmark for the public `TransferZ` release.

Split semantics are preserved:

- `TRAINING` -> predictor fit
- `VALIDATION` -> post-hoc temperature / variance calibration
- `CONFORMAL` -> conformal calibration
- `TESTING` -> final evaluation

## Methods

- `NativeQuantile90`
- `NativeGaussian90`
- `SplitConformal`
- `CQR`
- `DensityConformal` (validation-tuned bandwidth)
- `PrevalenceAdjustedCP` (validation-tuned group structure)
- `MonteCarloConformal`
- `R2CConformal`

## Reported metrics

- `Coverage90`
- `Width90`
- `IntervalScore90`
- `NMAD`
- `CatastrophicRate`
- `HighZ_MAE`
- `HighZCoverage90`
- `HighZWidth90`
- `train_s`, `eval_s`

## Run

If normalized `TransferZ` splits are already present:

```bash
uv run python examples/photoz_transferz_conformal_comparison.py \
  --train-dataset-path data/transferz/normalized/transferz_train_photoz.csv \
  --cal-dataset-path data/transferz/normalized/transferz_cal_photoz.csv \
  --conformal-dataset-path data/transferz/normalized/transferz_conformal_photoz.csv \
  --test-dataset-path data/transferz/normalized/transferz_test_photoz.csv \
  --require-real-data
```

Or run through the end-to-end pipeline:

```bash
uv run python tools/photoz_transferz_pipeline.py \
  --profile full \
  --download-if-missing
```

## Summary Artifact

The `TransferZ` pipeline writes this example into:

```bash
reports/example_summaries/transferz/photoz_transferz_conformal_comparison_full.json
```

## Notes

!!! info
    `DensityConformal` and `PrevalenceAdjustedCP` are tuned on `VALIDATION` for high-z interval efficiency, then recalibrated on the reserved `CONFORMAL` split before `TESTING`.

!!! warning
    `TransferZ` targets are COSMOS2020-derived photo-z values, not spectroscopic redshifts. Coverage conclusions are still useful for public tabular benchmarking, but they are not a substitute for a spectroscopic external-validity benchmark.
