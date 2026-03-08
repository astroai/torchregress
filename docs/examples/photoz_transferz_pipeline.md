# Photo-z TransferZ Pipeline

Tool: `tools/photoz_transferz_pipeline.py`

Use this when you want a public tabular real-data photo-z benchmark with released `TRAINING`, `VALIDATION`, `TESTING`, and `CONFORMAL` splits instead of the older single-file fallback flow.

## What it does

1. Downloads the public `TransferZ` Zenodo release.
2. Normalizes the released CSV files into the canonical `torchregress` photo-z schema.
3. Preserves the released split semantics:
   - `TRAINING` -> train
   - `VALIDATION` -> standard benchmark calibration / post-hoc scaling
   - `TESTING` -> test
   - `CONFORMAL` -> dedicated conformal calibration
4. Runs `tools/photoz_benchmark_suite.py` in real-data-only mode on train/validation/test.
5. Runs `examples/photoz_transferz_conformal_comparison.py` on train/validation/conformal/test.

## Run it

```bash
uv run python tools/photoz_transferz_pipeline.py \
  --profile full \
  --download-if-missing
```

If you want to refresh the raw and normalized files:

```bash
uv run python tools/photoz_transferz_pipeline.py \
  --profile full \
  --force-download
```

## Outputs

- `reports/example_summaries/photoz_transferz_pipeline_latest.json`
- `reports/example_summaries/transferz/photoz_transferz_pipeline_latest.json`
- `reports/example_summaries/transferz/photoz_transferz_suite_latest.json`
- `reports/example_summaries/transferz/photoz_transferz_suite_latest.md`
- `reports/example_summaries/transferz/photoz_transferz_conformal_comparison_<profile>.json`
- `data/transferz/raw/transferz_*.csv`
- `data/transferz/normalized/transferz_*_photoz.csv`

!!! tip
    The `TransferZ` pipeline writes into `reports/example_summaries/transferz/` on purpose so it does not overwrite the governance baseline artifacts used by the synthetic/audit threshold checks.

## Notes

!!! info
    `TransferZ` targets are COSMOS2020-derived photo-z values, not spectroscopic redshifts. They are still useful for tabular benchmark coverage of calibration, domain shift, imbalance overlays, and semi-supervised protocols.

!!! tip
    The pipeline now uses `CONFORMAL` explicitly in the dedicated conformal photo-z benchmark, while keeping the standard regression and ordered-bin tracks on the original train/validation/test protocol.
