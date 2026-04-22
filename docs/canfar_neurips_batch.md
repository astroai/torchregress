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
| [`scripts/canfar_vcp_prepare.py`](../scripts/canfar_vcp_prepare.py) | One-shot: mirror **`data/`** → VOS; optional **`--write-vcp-specs`** for headless |
| [`scripts/canfar_vcp_push_local.sh`](../scripts/canfar_vcp_push_local.sh) | Thin wrapper: `uv run python scripts/canfar_vcp_prepare.py` |

## Pre-cluster (local)

Before sweeps on Skaha: (1) **`./scripts/ci_local.sh`** — same gates as GitHub (ruff, black, pytest + coverage, CPU benchmark thresholds). (2) Dry-run anything that shells out: e.g. **`uv run python scripts/canfar_vcp_prepare.py --dry-run --skip-auth-check`**. (3) **`uv run python scripts/canfar_launch_from_plan.py … --dry-run`** if your plan supports it. Then commit and push; pull on ARC and match **`VOS_BASE`** / **`VCP_SPECS`** to what you uploaded.

## Phase vocabulary (`--only-phases`)

Must match `NEURIPS_PHASE_KEYS` in `run_neurips_sage_reg_full.py`:

`year_direct`, `multiseed`, `openml_diamonds_multiseed`, `year_labeled_sweep`, `multiseed_year_nl2048`, `catboost`, `tabred`, `synthetic`, `backbone`, `ablations`, `shifts`, `image_rebuttal`, `aggregate`.

Use **`aggregate` alone in the final wave** after all writers have finished; aggregation reads the whole `run_root` tree.

## VOS upload table (defaults the driver expects)

Use a **full VOSpace URI** for uploads (must start with `vos:`, e.g. `vos:sfabbro/torchregress`). Each `vcp` destination is that URI plus the same **path suffix** as under your git clone (so headless `VCP_SPECS` keys match what you pushed — this is a **mirror of local layout on VOS**, not a “relative VOS” protocol).

`canfar_vcp_prepare.py` rejects `--vos-base` without the `vos:` prefix.

### Canonical inputs (deduplicate large files)

To avoid keeping **multi-gigabyte** copies under `docs/research/` (and to mirror one tree on VOS), place optional overrides under **`data/neurips_inputs/`** (gitignored like the rest of `data/`). The NeurIPS driver resolves defaults in this order:

| Role | Preferred (`data/neurips_inputs/`) | Legacy fallback |
|------|-------------------------------------|-------------------|
| Tuning CSV | `supervised_gap_tuning_v3_sweep.csv` | `docs/research/.../supervised_gap_tuning_v3/sweep.csv` |
| Higgs parquet | `FAIR_Universe_HiggsML_data.parquet` | `docs/research/.../higgs_public/extracted/...` |
| Diamonds parquet | `openml_large_tabular_diamonds.parquet` | `data/paper/openml_large_tabular_diamonds.parquet` |

After **`vcp`** uploads the canonical copy to VOS and you have verified runs, you can delete the legacy **local** duplicates to reclaim disk (keep at least one copy until VOS + ARC are validated).

**Upload + `VCP_SPECS`:**

```bash
uv run python scripts/canfar_vcp_prepare.py                    # mirror ./data/ → vos:sfabbro/torchregress/data/
uv run python scripts/canfar_vcp_prepare.py --dry-run --skip-auth-check
uv run python scripts/canfar_vcp_prepare.py --write-vcp-specs ./vcp_specs_for_headless.txt
```

| Asset | Default repo path | Purpose |
|--------|-------------------|---------|
| Year CSV | `data/paper/openml_year.csv` | Year track, ablations, catboost, multiseed, label sweep |
| Supervised-gap tuning CSV | `data/neurips_inputs/supervised_gap_tuning_v3_sweep.csv` **or** `docs/research/sage_reg_results/2026-04-10/supervised_gap_tuning_v3/sweep.csv` | Multiseed / nl2048 / diamonds |
| OpenML diamonds | `data/neurips_inputs/openml_large_tabular_diamonds.parquet` **or** `data/paper/openml_large_tabular_diamonds.parquet` | `openml_diamonds_multiseed` |
| Higgs parquet | `data/neurips_inputs/FAIR_Universe_HiggsML_data.parquet` **or** `docs/research/.../extracted/FAIR_Universe_HiggsML_data.parquet` | Higgs arm of multiseed (optional) |
| TabReD layout | `data/tabred/<dataset>/...` | Included in the **`data/`** mirror; **`data/tabred/.vendor/`** is excluded from VOS (local upstream clone only) |
| Shifts | none (network fetch in driver) | `shifts` phase writes under `--shifts-out-root` |
| Image rebuttal | Synthetic in-process data in [`image_regression_rebuttal.py`](../examples/benchmarks/image_regression_rebuttal.py) | Optional; no VOS assets |

**Multi-file pulls in the container:** set `VCP_SPECS` to newline-separated `vos_rel|scratch_rel` pairs (see [`scripts/canfar_headless_job.sh`](../scripts/canfar_headless_job.sh)). Keys are **paths under your VOS URI** that mirror the clone; generate with **`uv run python scripts/canfar_vcp_prepare.py --write-vcp-specs FILE`** once `data/` matches what you need.

**Dotfiles / caches outside the repo:** keep only what you want uploaded under **`data/`**; the script prunes small junk locally before `vcp`.

### MkDocs `site/` bloat (local disk)

If `mkdocs build` copied **`docs/research/sage_reg_results/`** into `site/research/`, that directory can grow to **tens of GB** (parquets, zips). The repo **`mkdocs.yml`** now **`exclude_docs`** that tree so future builds stay small. Remove an old bloated build with **`rm -rf site/`** and rebuild.

**`data/tabred/`** can be **very large** (materialized tensors); it is real benchmark input, not doc output. Do not delete unless you intend to re-fetch (e.g. `tools/fetch_tabred_data.py`). Smaller dirs under `data/paper/`, `data/nnc_crps/`, etc. are normal caches — prune only if you know you can regenerate them.

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
uv pip install 'torchregress[canfar]'
```

JSON plans work without PyYAML; `.yaml` plans require the **`canfar`** extra (includes `PyYAML`).
