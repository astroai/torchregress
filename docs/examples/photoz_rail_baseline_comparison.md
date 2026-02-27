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
