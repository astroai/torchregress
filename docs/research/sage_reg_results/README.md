# SAGE / SPT paper result trees (committed artifacts)

This directory holds **dated experiment outputs** referenced from `papers/neurips_sage_reg/` and `papers/neurips_spt_reg/`.

## What is tracked in git

Per `.gitignore`, only small **result artifacts** are versioned:

- `*.json`, `*.csv`, `*.md`, `*.png`

## What stays local only

- **Raw caches**, e.g. OpenML Year as `**/openml_year.csv` (hundreds of MB). Materialize with `tools/materialize_openml_year.py` or pass `--year-cache-path` to runners.
- **Binary datasets**: `*.parquet`, `*.zip`, `*.npy`, CatBoost `*.cbm`, checkpoints, etc.

## SPT-Reg full runs

Canonical **SPT** one-shot outputs also live under `reports/neurips_spt_reg/runs/<date>/neurips_spt_reg_full/` (same extension policy), e.g. `METRICS.md`, `spt_paper_report.json`, and `full/year_competing_methods_full.json`.

## Entrypoints

- SAGE full refresh: `scripts/run_neurips_sage_reg_full.py`
- SPT full refresh: `scripts/run_neurips_spt_reg_full.py`
- Joint tabular bundle: `scripts/run_tabular_paper_bundle.sh`
