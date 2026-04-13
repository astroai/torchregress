# SPT-Reg reproducibility map

This file links each manuscript artifact to generated outputs on the branch.

## Library entrypoints

| Component | Location |
|-----------|----------|
| Shift-factored transport | `torchregress/test_time/transport.py` (`ShiftFactoredPredictiveTransport`, `ShiftFactoredTransportConfig`, …) |
| Predictive container | `torchregress/prediction.py` (`PredictiveBatch`) |

## Planned main artifacts

- Main synthetic table: `reports/neurips_spt_reg/synthetic_competing_methods_<profile>.json`
  - generator: `examples/spt_reg_synthetic_comparison.py`
  - includes decomposed rows such as `SPTTransportGaussian` (adaptation without conformal), `RawConformalMDN`, `SPTTransportMDN`, and full `SPTReg*` pipelines
- Main small-tabular table: `reports/neurips_spt_reg/tabular_competing_methods_<profile>.json`
  - generator: `examples/spt_reg_realdata_comparison.py`
- Main large-tabular table: `reports/neurips_spt_reg/year_competing_methods_<profile>.json`
  - generator: `examples/spt_reg_year_comparison.py`
- Optional photo-z compatibility table in `torchregress`: `reports/neurips_spt_reg/photoz_competing_methods_<profile>.json`
  - generator: `examples/spt_reg_photoz_comparison.py`
  - render only with `--include-photoz`
- Calibration/coverage figure: `reports/neurips_spt_reg/fig_calibration_coverage.png`
- PPI efficiency figure: `reports/neurips_spt_reg/fig_ppi_efficiency.png`
- Artifact manifest: `reports/neurips_spt_reg/artifact_manifest_latest.json`
- Full paper run tree (dated): `reports/neurips_spt_reg/runs/<date>/neurips_spt_reg_full/` from `scripts/run_neurips_spt_reg_full.py` (`neurips_spt_reg_full_manifest.json`, `spt_paper_report.json`, `METRICS.md`)

## Notes

- **Committed artifacts:** `reports/neurips_spt_reg/**/*.json` (plus `README.md`, small `year_local_dataset_*.csv` stubs, `runs/…/METRICS.md`, etc.) and `docs/research/sage_reg_results/**/*.{json,csv,md,png}` hold paper-scale **results**. The **OpenML Year** table at paths like `docs/research/sage_reg_results/.../openml_year.csv` is a **local cache** (gitignored there); materialize with download flags or `tools/materialize_openml_year.py`. See `docs/research/sage_reg_results/README.md`.
- **One command (full + audit + aggregation):**

```bash
uv run python scripts/run_neurips_spt_reg_full.py
```

Quick smoke (small renderer profile; the **extra-large OpenML** track is skipped in `--quick` to keep CI fast):

```bash
uv run python scripts/run_neurips_spt_reg_full.py --quick
```

**Default full run:** extra-large OpenML regression (**diamonds**, id **42225**) and **Shifts** README materialization are **on**; photo-z stays off unless `--include-photoz`. Opt out with `--skip-large-tabular` (alias `--skip-yolanda`) / `--skip-shifts`. Tunables: `--large-tabular-cache`, `--large-tabular-openml-id`, `--large-tabular-max-rows`. `--skip-stage-a-sweep` skips the Stage-A sweep.

**Both tracks in one shell pass** (same flags forwarded to SPT then SAGE; do not pass `--run-root`):

```bash
./scripts/run_neurips_paper_bundle.sh
./scripts/run_neurips_paper_bundle.sh --quick
```

After a full run, summarize key Gaussian rows (raw vs. weighted conformal vs. SPT-Reg):

```bash
uv run python tools/analyze_neurips_spt_reg_run.py \
  --run-root reports/neurips_spt_reg/runs/<date>/neurips_spt_reg_full
```

- Ad hoc render entrypoint (single output directory):

```bash
uv run python tools/render_spt_reg_paper_artifacts.py --profile smoke
```

### Trustworthy OpenML Year (large-tabular track)

Default renderer behavior still writes **synthetic** `year_local_dataset_<profile>.csv` unless you pass **`--year-cache-path`** or **`--year-dataset-path`**.

**Recommended: one command for profile `full` on real OpenML Year** (downloads into the cache path if missing):

```bash
uv run python tools/render_spt_reg_paper_artifacts.py \
  --profile full \
  --year-cache-path docs/research/sage_reg_results/2026-04-10/openml_year.csv \
  --year-allow-download \
  --output-dir reports/neurips_spt_reg
```

After the CSV exists, you can drop **`--year-allow-download`**. Check **`artifact_manifest_latest.json`** (or your `--report` path): field **`year_track_data`** records how the Year track was sourced.

**Optional:** run only the Year benchmark and a custom JSON path:

```bash
uv run python examples/spt_reg_year_comparison.py \
  --allow-download \
  --cache-path docs/research/sage_reg_results/2026-04-10/openml_year.csv \
  --summary-json-path reports/neurips_spt_reg/year_openml_only.json
```

That uses **default** `SPTRegYearConfig` pool sizes (already OpenML-scale). For **`full`** renderer budgets specifically, prefer the **`render_spt_reg_paper_artifacts.py`** command above so `n_*` match `PROFILE_CONFIGS["full"]["year"]`.

### Extra-large OpenML regression (default: diamonds **42225**)

The NeurIPS full runner writes this track under **`large_tabular/`** (OpenML **42225**, ggplot2 **diamonds**, ~54k rows × 9 numeric features, **price → `target`**). It is **larger than Year** in the manuscript bundle and has been **stable** under `sklearn.datasets.fetch_openml` in practice (unlike Yolanda **42705**, which often hits MD5 drift).

Renderer example:

```bash
uv run python tools/render_spt_reg_paper_artifacts.py \
  --profile full \
  --year-openml-data-id 42225 \
  --year-max-dataset-rows 60000 \
  --year-cache-path data/paper/openml_large_tabular_diamonds.parquet \
  --year-allow-download \
  --output-dir reports/neurips_spt_reg
```

Direct benchmark (same knobs on the example script):

```bash
uv run python examples/spt_reg_year_comparison.py \
  --openml-data-id 42225 \
  --max-dataset-rows 60000 \
  --allow-download \
  --cache-path data/paper/openml_large_tabular_diamonds.parquet \
  --summary-json-path reports/neurips_spt_reg/diamonds_competing_methods_full.json
```

The first download can be slow; with **`--cache-path`**, later runs read the materialized table only.

If sklearn raises **MD5 checksum** for any id, the Year comparison scripts fall back to **`torchregress.utils.openml_relaxed`**. For a **pinned** table:

```bash
uv run python tools/materialize_openml_large_tabular.py \
  --cache-path data/paper/openml_large_tabular_diamonds.parquet \
  --max-rows 60000
```

You can point **`--large-tabular-openml-id`** at another regression dataset (e.g. King County house sales **42731**); sanity-check targets/features for leakage before claiming results.

**SAGE-Reg (semi-sup vs sup):** [papers/neurips_sage_reg/reproducibility.md](../neurips_sage_reg/reproducibility.md).

- Optional photo-z compatibility render:

```bash
uv run python tools/render_spt_reg_paper_artifacts.py --profile smoke --include-photoz
```

- Profiles currently supported: `smoke`, `audit`, and `full`.
- **Important:** `render_spt_reg_paper_artifacts.py` does **not** download OpenML Year by default. For the “large tabular” slot it materializes **`reports/neurips_spt_reg/year_local_dataset_<profile>.csv`**, a **synthetic** Gaussian linear CSV whose **row counts** match the profile’s `n_source` / `n_target_*` budgets (for `full`, that is 4096 + 8192 + 2048 + 4096 rows — “OpenML-scale” **sizes**, not the real Year feature distribution). For **actual** OpenML tables, pass **`--year-cache-path`** / **`--year-dataset-path`** / **`--year-openml-data-id`**, or run `examples/spt_reg_year_comparison.py` with the matching flags and **`--summary-json-path`** yourself.
- For ad-hoc scaling from defaults: `examples/spt_reg_year_comparison.py --scale-split-factor K`.

- Shared dataset/source registry: `papers/neurips_dataset_registry.md`
- Keep filenames stable after they are cited in the paper.
- The default render path covers one synthetic SPT benchmark, diabetes real-data, and the **synthetic large-tabular** Year-shaped CSV above (until you replace it with a real-data path).

## Sister track

SAGE-Reg artifact map: [papers/neurips_sage_reg/reproducibility.md](../neurips_sage_reg/reproducibility.md).
