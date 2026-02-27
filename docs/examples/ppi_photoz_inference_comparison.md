# Prediction-Powered Inference (Photo-z Style)

Script: `examples/ppi_photoz_inference_comparison.py`

This example compares labeled-only confidence intervals against
prediction-powered inference (PPI) intervals for:

- population mean
- high quantile (`q90` by default)

It uses a photo-z style setup: small labeled set + large prediction-only set.

## Run

```bash
uv run python examples/ppi_photoz_inference_comparison.py
```

## Summary Artifact

```bash
uv run python examples/ppi_photoz_inference_comparison.py \
  --summary-json-path reports/example_summaries/ppi_photoz_inference_comparison_full.json
```

The output rows report:

- `Estimate`
- `AbsError`
- `CIWidth`
- `CoversTruth`

