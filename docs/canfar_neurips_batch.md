# CANFAR batch: full NeurIPS SAGE-Reg + overnight scope

This document describes how to reproduce the evidence surface of [`scripts/run_neurips_sage_reg_full.py`](../scripts/run_neurips_sage_reg_full.py) and the extras from [`scripts/run_overnight_ssl_full_comparison.sh`](../scripts/run_overnight_ssl_full_comparison.sh) on **Skaha headless** sessions, using **VOSpace** for inputs and **ARC** for durable outputs.

For the **label-fraction sweep only** (40 shards), see also [canfar_batch_sweeps.md](canfar_batch_sweeps.md).

## Components

| Piece | Role |
|-------|------|
| Laptop | `python scripts/canfar_launch_from_plan.py` (or `canfar_launch_sweep.py` for shards-only) |
| [`scripts/canfar_headless_job.sh`](../scripts/canfar_headless_job.sh) | Container entry: `vcp` staging + `CANFAR_JOB_KIND` dispatch |
| [`scripts/canfar_launch_from_plan.py`](../scripts/canfar_launch_from_plan.py) | Reads YAML/JSON plan, submits waves, optional inter-wave wait |
| [`scripts/canfar/canfar_work_plan.example.yaml`](../scripts/canfar/canfar_work_plan.example.yaml) | Example waves + jobs |
| **VOS** | Authoritative copies of large inputs |
| **`ARC_RUN_ROOT`** | One run directory shared by phase jobs that must complete before `aggregate` |
| [`scripts/canfar_vcp_prepare.py`](../scripts/canfar_vcp_prepare.py) | One-shot: mirror **`data/`** → **`vos:sfabbro/torchregress/data/`** (see below) |
| [`scripts/canfar_vcp_push_local.sh`](../scripts/canfar_vcp_push_local.sh) | Same as `uv run python scripts/canfar_vcp_prepare.py` |

## Laptop → VOS: one command

From the repo root, with [`vos`](https://pypi.org/project/vos/) on your `PATH` (`uv sync --extra vos` or `uv pip install vos`):

```bash
uv run python scripts/canfar_vcp_prepare.py
```

Defaults: **`./data/`** → **`vos:sfabbro/torchregress/data/`**; prunes local junk under `data/` (`__pycache__`, `.DS_Store`, TabReD `preprocessing/tmp`); runs **`vls`** on the VOS base (if it fails, run **`cadc-get-cert -u YOUR_CADC_USER`** then retry); **`vmkdir -p`**; then one recursive **`vcp -v`**. Re-run after interruption — **`vcp`** resumes.

Useful flags: **`--dry-run`** (print only), **`--no-clean-local`**, **`--vos-base vos:OTHER/torchregress`**, **`--write-vcp-specs FILE`** (emit `VCP_SPECS` rows for headless), **`--quiet`**.

## Phase vocabulary (`--only-phases`)

Must match `NEURIPS_PHASE_KEYS` in `run_neurips_sage_reg_full.py`:

`year_direct`, `multiseed`, `openml_diamonds_multiseed`, `year_labeled_sweep`, `multiseed_year_nl2048`, `catboost`, `tabred`, `synthetic`, `backbone`, `ablations`, `shifts`, `image_rebuttal`, `aggregate`.

Use **`aggregate` alone in the final wave** after all writers have finished; aggregation reads the whole `run_root` tree.

## VOS layout (must mirror `data/`)

Headless jobs use **`VOS_BASE`** (e.g. `vos:sfabbro/torchregress`) and expect inputs under the **same paths as in the clone**, e.g. `data/paper/openml_year.csv` → on VOS `vos:sfabbro/torchregress/data/paper/openml_year.csv`. The prepare script copies **all of `data/`** in one recursive `vcp` to `…/data/`, so keep only experiment inputs there (see table).

### Canonical inputs (deduplicate large files)

To avoid keeping **multi-gigabyte** copies under `docs/research/` (and to mirror one tree on VOS), place optional overrides under **`data/neurips_inputs/`** (gitignored like the rest of `data/`). The NeurIPS driver resolves defaults in this order:

| Role | Preferred (`data/neurips_inputs/`) | Legacy fallback |
|------|-------------------------------------|-------------------|
| Tuning CSV | `supervised_gap_tuning_v3_sweep.csv` | `docs/research/.../supervised_gap_tuning_v3/sweep.csv` |
| Higgs parquet | `FAIR_Universe_HiggsML_data.parquet` | `docs/research/.../higgs_public/extracted/...` |
| Diamonds parquet | `openml_large_tabular_diamonds.parquet` | `data/paper/openml_large_tabular_diamonds.parquet` |

After uploads, you can delete legacy **local** duplicates under `docs/research/` if the canonical copies live under `data/` and on VOS.

### Higgs parquet missing locally (`[skip] missing higgs`)

The FAIR Universe public Higgs ML dump is **not** in git (multi‑GB). If it was removed from `docs/research/.../higgs_public/` (for example after a cleanup of `sage_reg_results`), **re-download** from the FAIR Universe release on Zenodo ([`10.5281/zenodo.15131565`](https://doi.org/10.5281/zenodo.15131565)) or from your own backup, then place the extracted parquet at:

`data/neurips_inputs/FAIR_Universe_HiggsML_data.parquet`

That path is what `run_neurips_sage_reg_full.py` prefers locally; place the same file under `data/` so it is included in the recursive upload.

The cleanup helper `tools/clean_sage_reg_results_heavy_artifacts.py` **no longer deletes** `FAIR_Universe_HiggsML_data.parquet` under `sage_reg_results` unless you pass **`--delete-fair-universe-higgs-parquet`** explicitly.

**Headless `VCP_SPECS`:** after your `data/` tree is final, `uv run python scripts/canfar_vcp_prepare.py --write-vcp-specs ./vcp_specs_for_headless.txt` writes one `vos_rel|scratch_rel` row per file under `data/`.

| Asset | Default repo path | Purpose |
|--------|-------------------|---------|
| Year CSV | `data/paper/openml_year.csv` | Year track, ablations, catboost, multiseed, label sweep |
| Supervised-gap tuning CSV | `data/neurips_inputs/supervised_gap_tuning_v3_sweep.csv` **or** `docs/research/sage_reg_results/2026-04-10/supervised_gap_tuning_v3/sweep.csv` | Multiseed / nl2048 / diamonds |
| OpenML diamonds | `data/neurips_inputs/openml_large_tabular_diamonds.parquet` **or** `data/paper/openml_large_tabular_diamonds.parquet` | `openml_diamonds_multiseed` |
| Higgs parquet | `data/neurips_inputs/FAIR_Universe_HiggsML_data.parquet` **or** `docs/research/.../extracted/FAIR_Universe_HiggsML_data.parquet` | Higgs arm of multiseed (optional) |
| TabReD layout | `data/tabred/<dataset>/...` (default tasks) | Materialized tensors under each task dir; **`data/tabred/.vendor/`** is the local upstream repo only — it is **not** uploaded to VOS (excluded by `canfar_vcp_prepare.py`) |
| Shifts | none (network fetch in driver) | `shifts` phase writes under `--shifts-out-root` |
| Image rebuttal | Synthetic in-process data in [`image_regression_rebuttal.py`](../examples/benchmarks/image_regression_rebuttal.py) | Optional; no VOS assets |

**Multi-file pulls in the container:** set `VCP_SPECS` to newline-separated `vos_rel|scratch_rel` pairs (see [`scripts/canfar_headless_job.sh`](../scripts/canfar_headless_job.sh)); left-hand side is the path suffix under `VOS_BASE`, same as under the git clone.

**Dotfiles / caches outside the repo:** remove anything you do not want uploaded from `data/` before running the script (or use `--no-clean-local` and curate by hand).

### MkDocs `site/` bloat (local disk)

If `mkdocs build` copied **`docs/research/sage_reg_results/`** into `site/research/`, that directory can grow to **tens of GB** (parquets, zips). The repo **`mkdocs.yml`** now **`exclude_docs`** that tree so future builds stay small. Remove an old bloated build with **`rm -rf site/`** and rebuild.

**`data/tabred/`** can be **very large** (materialized tensors); it is real benchmark input, not doc output. Do not delete **`cooking-time/`**, **`delivery-eta/`**, **`maps-routing/`**, etc., unless you will re-run `tools/fetch_tabred_data.py`. The path **`data/tabred/.vendor/yandex-tabred/preprocessing/tmp`** is **upstream preprocessing scratch** (Kaggle pulls / intermediates); it is **not** used by `tabred_sage_ssl_probe` after `info.json` + `.npy` exist. Reclaim space with:

`uv run python tools/fetch_tabred_data.py --prune-preprocessing-tmp-only --out-dir data/tabred`

Smaller dirs under `data/paper/`, `data/nnc_crps/`, etc. are normal caches — prune only if you know you can regenerate them.

## Work plan schema (version 1)

- **`defaults`**: optional `image`, `cores`, `ram_gb`.
- **`jobs`**: each entry has **`id`**, optional **`wave`**, optional **`needs`** (for auto wave assignment if `wave` omitted everywhere), optional **`shard_count`**, optional **`job_kind`**, optional **`only_phases`**, optional **`env`**.

**Sharding:** `shard_count: N` expands to `N` Skaha sessions with `SHARD_ID` / `SHARD_COUNT` (same contract as `sage_year_label_fraction_sweep.py`).

## Ordering and races

1. **One writer per output path.** Do not run two jobs that write the same `run_root/...` subtree concurrently.
2. **Waves:** group parallel-safe jobs in the same wave; run **`aggregate` in a later wave** after prior waves complete.
3. **`--wait-between-waves` (default on):** the launcher polls `Session.info` until every session in a wave reaches a terminal status before submitting the next wave. Disable with `--no-wait-between-waves` only if you manage ordering yourself.

## Quotas

Use `--max-concurrent 8` (or similar) if `Session.create` hits user limits. Forty label shards plus eight NeurIPS jobs can exceed default quotas; stagger waves in the YAML.

## After label shards

Merge CSVs on ARC (same as [canfar_batch_sweeps.md](canfar_batch_sweeps.md)):

```bash
RUN=/arc/projects/ots/torchregress/runs/<run_id>
uv run python tools/collate_csv_glob.py \
  --glob "$RUN/shards/shard_*.csv" \
  --out "$RUN/year_label_fraction_sweep_merged.csv"
```

Expect **5040** data rows + header for the default outer grid with **40** shards.

## Dependencies

```bash
canfar auth login
uv pip install 'torchregress[canfar]'   # canfar API + PyYAML + vos (vcp / vls / vsync)
# vcp-only laptop (no Skaha submitter): uv pip install 'torchregress[vos]'
```

JSON plans work without PyYAML; `.yaml` plans require the **`canfar`** extra (includes `PyYAML`).
