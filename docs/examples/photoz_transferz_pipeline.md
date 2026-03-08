# Photo-z TransferZ Pipeline

Tool: `tools/photoz_transferz_pipeline.py`

Use this when you want a public tabular real-data photo-z benchmark with released `TRAINING`, `VALIDATION`, `TESTING`, and `CONFORMAL` splits instead of the older single-file fallback flow.

## What it does

1. Downloads the public `TransferZ` Zenodo release.
2. Normalizes the released CSV files into the canonical `torchregress` photo-z schema.
3. Preserves the released split semantics:
   - `TRAINING` -> train
   - `VALIDATION` -> calibration
   - `TESTING` -> test
   - `CONFORMAL` -> reserved for future conformal-specific benchmarking
4. Runs `tools/photoz_benchmark_suite.py` in real-data-only mode on those splits.

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
- `data/transferz/raw/transferz_*.csv`
- `data/transferz/normalized/transferz_*_photoz.csv`

!!! tip
    The `TransferZ` pipeline writes into `reports/example_summaries/transferz/` on purpose so it does not overwrite the governance baseline artifacts used by the synthetic/audit threshold checks.

## Notes

!!! info
    `TransferZ` targets are COSMOS2020-derived photo-z values, not spectroscopic redshifts. They are still useful for tabular benchmark coverage of calibration, domain shift, imbalance overlays, and semi-supervised protocols.

!!! warning
    The current benchmark maps `VALIDATION` to the benchmark calibration split and leaves `CONFORMAL` unused. That is deliberate. The conformal split should be consumed in a dedicated conformal benchmark tranche rather than folded into generic model selection.
