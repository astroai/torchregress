# Photo-z NNC-CRPS Comparison (Examples-Only)

Script: `examples/photoz_nnc_crps_rail_comparison.py`

This example runs an ordered-bin photo-z comparison aligned with a regression-as-classification flow.

Included methods:

- `BinnedCE`
- `BinnedCE+TempScaling`
- `SoftBinnedCE`
- `SoftBinnedCE+Pseudo`
- `SoftCumulativeLink`
- `OrderedBinCRPS`
- `OrderedBinCRPS+TempScaling`
- `GaussianNLL`
- `MultiQuantileLoss`

The ordered-bin CRPS utilities are intentionally implemented in `examples/photoz_binned_utils.py` (not core API).
The soft-target ordered-bin rows reuse the core ordinal losses:

- `OrdinalCrossEntropyLoss`
- `CumulativeLinkLoss`

Soft targets are built from spectroscopic-redshift uncertainty, and the pseudo-label row uses a partial-label teacher-student track.

## Run

```bash
uv run python examples/photoz_nnc_crps_rail_comparison.py
```

## Summary Artifact

```bash
uv run python examples/photoz_nnc_crps_rail_comparison.py \
  --summary-json-path reports/example_summaries/photoz_nnc_crps_rail_comparison_full.json
```

Metrics in the summary rows include:

- `NMAD`, `CatastrophicRate`, `HighZ_MAE`
- `CRPS`, `PDF_NLL`, `PITChi2`
- `NativeCov90`, `NativeWidth90`
- `train_s`, `eval_s`, `calibrate_s`
- `LabeledFraction`, `PseudoAcceptRate` for the soft pseudo-label row

For manifest-driven RAIL data collection + merged baseline comparison, run:

```bash
uv run python tools/photoz_rail_pipeline.py --preset nnc_crps --profile full
```
