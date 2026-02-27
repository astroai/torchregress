# Photo-z RAIL Baseline Comparison

Tool: `tools/photoz_rail_compare.py`

This adapter merges:

- torchregress photo-z summary artifacts
- RAIL baseline outputs

into one consolidated `comparison_example_summary` artifact.

Core RAIL baseline set:

- `flexzboost`
- `pzflow`
- `delight`
- `bpz`
- `lephare` optional

## Manifest-Based Parity

Manifest file:

- `data/rail/rail_photoz_manifest.json`
- Templates (versioned):
  - `configs/photoz/rail_photoz_manifest.template.json`
  - `configs/photoz/nnc_crps_photoz_manifest.template.json`

The manifest now supports automatic collection metadata:

- `dataset_files`: train/test/calibration asset paths + optional URLs
- `baseline_payloads`: per-method baseline payload paths + optional URLs
- `checksum_policy`: auto-updated SHA256 values for dataset assets

In `paper-parity` mode (default), the tool enforces:

- dataset id match
- split id match
- required core baseline methods present

## Run

```bash
uv run python tools/photoz_rail_compare.py \
  --manifest data/rail/rail_photoz_manifest.json \
  --torchregress-summary reports/example_summaries/photoz_nnc_crps_rail_comparison_full.json \
  --rail-inputs /path/to/flexzboost.json /path/to/pzflow.json /path/to/delight.json /path/to/bpz.json \
  --output reports/example_summaries/photoz_rail_baseline_comparison_full.json
```

Disable strict parity checks if needed:

```bash
uv run python tools/photoz_rail_compare.py \
  --no-paper-parity \
  ...
```

## Collect Assets Automatically

After filling manifest URLs once, collect datasets + baseline payloads automatically:

```bash
uv run python tools/photoz_rail_assets.py \
  --preset rail \
  --manifest data/rail/rail_photoz_manifest.json \
  --report reports/example_summaries/photoz_rail_materialization_latest.json
```

Without editing the manifest file, you can also inject URLs/paths directly:

```bash
uv run python tools/photoz_rail_assets.py \
  --dataset-url train_catalog_sha256=https://example.org/train.parquet \
  --dataset-url test_catalog_sha256=https://example.org/test.parquet \
  --dataset-url calibration_catalog_sha256=https://example.org/cal.parquet \
  --baseline-url flexzboost=https://example.org/flexzboost.json \
  --baseline-url pzflow=https://example.org/pzflow.json \
  --baseline-url delight=https://example.org/delight.json \
  --baseline-url bpz=https://example.org/bpz.json
```

## Integrated Pipeline Run

You can produce both torchregress photo-z summaries and the merged RAIL comparison
in one run:

```bash
uv run python tools/photoz_rail_pipeline.py \
  --preset rail \
  --manifest data/rail/rail_photoz_manifest.json \
  --profile full \
  --output-dir reports/example_summaries
```

Single-command run without manifest editing:

```bash
uv run python tools/photoz_rail_pipeline.py \
  --preset nnc_crps \
  --profile full \
  --dataset-url train_catalog_sha256=https://example.org/train.parquet \
  --dataset-url test_catalog_sha256=https://example.org/test.parquet \
  --dataset-url calibration_catalog_sha256=https://example.org/cal.parquet \
  --baseline-url flexzboost=https://example.org/flexzboost.json \
  --baseline-url pzflow=https://example.org/pzflow.json \
  --baseline-url delight=https://example.org/delight.json \
  --baseline-url bpz=https://example.org/bpz.json
```

This writes:

- `reports/example_summaries/photoz_nnc_crps_rail_comparison_full.json`
- `reports/example_summaries/photoz_rail_baseline_comparison_full.json`
- `reports/example_summaries/photoz_rail_materialization_latest.json`
- `reports/example_summaries/photoz_rail_pipeline_latest.json`
