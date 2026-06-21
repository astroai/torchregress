# CANFAR batch: Year label-fraction sweep

For the **full NeurIPS + overnight** manifest (phase shards, label sweep, aggregate), see [canfar_neurips_batch.md](canfar_neurips_batch.md) and [`scripts/canfar_launch_from_plan.py`](../scripts/canfar_launch_from_plan.py).

Run the expensive [`sage_year_label_fraction_sweep`](../examples/benchmarks/sage_year_label_fraction_sweep.py) grid (**240** outer cells = 10 seeds × 3 shifts × 8 label percents) as **40** Skaha **headless** sessions (**6** cells per shard), using the **CANFAR Python API** from your laptop.

## Layout

| Location | Role |
|----------|------|
| **Laptop** | `python scripts/canfar_launch_sweep.py` (uses `canfar.sessions.Session`) |
| **ARC** `/arc/home/sfabbro/src/torchregress` | Git checkout; headless runs `scripts/canfar_headless_job.sh` |
| **`/scratch/torchregress/...`** | Ephemeral copy of Year CSV + `TMPDIR` |
| **`vos:sfabbro/torchregress/...`** | Public VOSpace staging for inputs |
| **`/arc/projects/ots/torchregress/runs/<run_id>/`** | Durable `shards/shard_XX.csv`, `logs/shard_XX.log` |

Override paths with launcher flags (`--arc-project-root`, `--vos-base`, `--torchregress-repo`, `--arc-run-root`).

## One-time: local data → VOSpace

From your **local** torchregress clone (adjust `vos:` path if yours differs):

```bash
# YearPredictionMSD cache used with --no-download on cluster
vcp ./data/paper/openml_year.csv vos:sfabbro/torchregress/data/paper/openml_year.csv
```

For staging **`data/`** on VOS and optional `VCP_SPECS` generation, use [`scripts/canfar_vcp_prepare.py`](../scripts/canfar_vcp_prepare.py) (see [canfar_neurips_batch.md](canfar_neurips_batch.md)).

If you later run benchmarks that need **Higgs** or **shifts** data on CANFAR, upload those files under the same `vos:sfabbro/torchregress/...` tree and extend [`scripts/canfar_headless_job.sh`](../scripts/canfar_headless_job.sh) with additional `vcp` lines (or set `YEAR_VOS_REL` / add env vars).

## Laptop: auth and optional dependency

```bash
canfar auth login
uv pip install 'torchregress[canfar]'   # canfar + PyYAML + vos (vcp); or: pip install 'canfar>=1.3' 'vos>=3.6'
```

## Single launch command (40 sessions, 8 cores, 32 GB each)

```bash
cd /path/to/torchregress
python scripts/canfar_launch_sweep.py --run-id y2026_0422_a
```

Defaults: **40** shards, **`images.canfar.net/skaha/astroml:latest`**, **`cores=8`**, **`ram=32`** (GB, per CANFAR client docs), `ARC_RUN_ROOT=/arc/projects/ots/torchregress/runs/<run-id>`.

Useful flags:

| Flag | Purpose |
|------|---------|
| `--max-concurrent 10` | Throttle parallel `Session.create` calls if quotas bite |
| `--dry-run` | Print manifest fields without calling the API |
| `--manifest-out ./my_manifest.json` | Manifest path (default `./canfar_sweep_<run-id>_manifest.json`) |
| `--shards 40` | Shard count (must match sweep’s `--shard-count` in the headless script) |

## What each headless job does

[`scripts/canfar_headless_job.sh`](../scripts/canfar_headless_job.sh):

1. Requires env `ARC_RUN_ROOT`, `VOS_BASE`, `SHARD_ID`, `SHARD_COUNT` (set by the launcher).
2. `vcp` from `VOS_BASE/data/paper/openml_year.csv` to `/scratch/.../openml_year.csv` (if `vcp` is installed in the image).
3. Runs `sage_year_label_fraction_sweep.py` with **`--shard-id` / `--shard-count`**, **`--dataloader-num-workers 0`**, **`--batch-size 2048`**, **`--catboost-iterations 0`** (override via env in the script if needed).
4. Writes **`$ARC_RUN_ROOT/shards/shard_XX.csv`** and **`shard_XX_summary.json`**, tee stdout/stderr to **`$ARC_RUN_ROOT/logs/shard_XX.log`**.

Install **VOSpace client tools** in the image or ARC environment if `vcp` is missing (`cadc-vos` / OpenCADC documentation).

## Monitor and cancel

From the CANFAR Python client (same machine as launch):

```python
from canfar.sessions import Session
s = Session()
running = s.fetch(kind="headless", status="Running")
print(running)
```

Or use the Science Portal / `canfar` CLI if you prefer interactive checks.

Destroy completed test sessions (example from CANFAR docs):

```python
s.destroy_with(prefix="torchregress-y2026_0422_a-", kind="headless", status="Succeeded")
```

## Merge shard CSVs on ARC (after all jobs succeed)

Expected row count: **40 × 6 × 21 = 5040** data rows (plus header), matching a full unsharded sweep (21 benchmark rows per outer cell).

```bash
RUN=/arc/projects/ots/torchregress/runs/y2026_0422_a
uv run python tools/collate_csv_glob.py \
  --glob "$RUN/shards/shard_*.csv" \
  --out "$RUN/year_label_fraction_sweep_merged.csv"
```

Inspect:

```bash
wc -l "$RUN/year_label_fraction_sweep_merged.csv"
```

## Quotas and stability

- **40 concurrent** jobs may exceed per-user limits; use **`--max-concurrent`** (e.g. 8–16) and rely on the queue.
- Headless logs are retained for a limited time after completion; durable logs are the copies under **`ARC_RUN_ROOT/logs/`**.
- If `vcp` or VOS auth fails inside the pod, fix **CADC certificates** / `~/.config/cadc/` on ARC or use the platform’s documented VOS proxy pattern.

## Sharding reference

Cells are ordered like the nested loops in `run_sweep`: for each **seed**, each **shift_mode**, each **label_pct**, increment a flat index `global_idx` starting at 0. A shard keeps cells where **`global_idx % shard_count == shard_id`**. With **10 / 3 / 8** defaults and **`shard_count=40`**, each shard runs **6** cells.
