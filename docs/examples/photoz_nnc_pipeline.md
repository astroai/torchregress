# Photo-z NNC End-to-End Pipeline

Tool: `tools/photoz_nnc_pipeline.py`

This is the one-command NNC path:

1. download a raw NNC-style catalog, or use an existing one
2. normalize it into the canonical `torchregress` photo-z frame
3. run the real-data photo-z benchmark tracks
4. emit JSON summaries plus a ranked markdown report

## What it runs

- `examples/photoz_benchmark_comparison.py`
- `examples/photoz_nnc_crps_rail_comparison.py`

The synthetic-only PPI track is intentionally skipped.

## Supported raw formats

- `.csv`
- `.json`
- `.jsonl`
- `.pkl` / `.pickle`
- `.fits` if `astropy` is installed

## Run it

If you already have a raw NNC catalog:

```bash
uv run python tools/photoz_nnc_pipeline.py \
  --raw-catalog data/nnc_crps/catalogs/your_catalog.csv \
  --profile full
```

If you want the tool to download the raw catalog first:

```bash
uv run python tools/photoz_nnc_pipeline.py \
  --download-if-missing \
  --profile full
```

## Outputs

- `reports/example_summaries/photoz_nnc_pipeline_latest.json`
- `reports/example_summaries/photoz_nnc_suite_latest.json`
- `reports/example_summaries/photoz_nnc_suite_latest.md`
- `data/nnc_crps/nnc_photoz_real.csv`

## Column inference

The normalization step tries to infer:

- target redshift from columns such as `spec_z`, `z_spec`, `redshift`, `z_true`
- target error from columns such as `spec_z_err`, `redshift_err`, `z_true_err`
- photometric bands from common magnitude or flux naming patterns

If inference is wrong, override it:

```bash
uv run python tools/photoz_nnc_pipeline.py \
  --raw-catalog data/nnc_crps/catalogs/your_catalog.csv \
  --target-col redshift \
  --target-err-col redshift_err \
  --profile full
```

## FITS note

!!! warning
    Raw FITS catalogs require `astropy`. Without it, the pipeline fails with an explicit dependency error.
