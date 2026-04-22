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

## Phase vocabulary (`--only-phases`)

Must match `NEURIPS_PHASE_KEYS` in `run_neurips_sage_reg_full.py`:

`year_direct`, `multiseed`, `openml_diamonds_multiseed`, `year_labeled_sweep`, `multiseed_year_nl2048`, `catboost`, `tabred`, `synthetic`, `backbone`, `ablations`, `shifts`, `image_rebuttal`, `aggregate`.

Use **`aggregate` alone in the final wave** after all writers have finished; aggregation reads the whole `run_root` tree.

## VOS upload table (defaults the driver expects)

Stage files under your `VOS_BASE` (e.g. `vos:sfabbro/torchregress`) using **repo-relative** paths below unless you override flags / env in the headless job.

| Asset | Default repo path | Purpose |
|--------|-------------------|---------|
| Year CSV | `data/paper/openml_year.csv` | Year track, ablations, catboost, multiseed, label sweep |
| Supervised-gap tuning CSV | `docs/research/sage_reg_results/2026-04-10/supervised_gap_tuning_v3/sweep.csv` | Multiseed / nl2048 / diamonds (same row selection) |
| OpenML diamonds | `data/paper/openml_large_tabular_diamonds.parquet` | `openml_diamonds_multiseed` |
| Higgs parquet | `docs/research/sage_reg_results/2026-04-09/higgs_public/extracted/FAIR_Universe_HiggsML_data.parquet` | Higgs arm of multiseed (optional) |
| TabReD layout | `data/tabred/<dataset>/...` (three default tasks) | TabReD probe (optional fetch if Kaggle present in image) |
| Shifts | none (network fetch in driver) | `shifts` phase writes under `--shifts-out-root` |
| Image rebuttal | per [`image_regression_rebuttal.py`](../examples/benchmarks/image_regression_rebuttal.py) | Optional; use `--include-image-rebuttal` via `NEURIPS_FLAGS` |

**Multi-file pulls in the container:** set `VCP_SPECS` to newline-separated `vos_rel|scratch_rel` pairs (see headless script header). Then point `NEURIPS_*` paths at the scratch files if defaults inside the git checkout are wrong on ARC.

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
