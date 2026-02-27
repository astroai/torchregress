# Photo-z NNC-CRPS Comparison (Examples-Only)

Script: `examples/photoz_nnc_crps_rail_comparison.py`

This example runs an ordered-bin photo-z comparison aligned with a regression-as-classification flow.

Included methods:

- `BinnedCE`
- `BinnedCE+TempScaling`
- `OrderedBinCRPS`
- `OrderedBinCRPS+TempScaling`
- `GaussianNLL`
- `MultiQuantileLoss`

The ordered-bin CRPS utilities are intentionally implemented in `examples/photoz_binned_utils.py` (not core API).

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
