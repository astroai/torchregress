# SAGE-Reg reproducibility map

This file links manuscript tables/figures to scripts, library entrypoints, and dated result trees. Paths are **repository-relative** (use your clone root as `REPO_ROOT`).

## Library entrypoints

| Component | Location |
|-----------|----------|
| SAGE-Reg loss, agreement, weights | `torchregress/semi_supervised.py` (`SAGERegLoss`, `SAGERegAgreement`, `disagreement_to_weight`, `SelfAgreementTrainer`, …) |
| Predictive container | `torchregress/prediction.py` (`PredictiveBatch`) |

## Planned main artifacts

| Artifact role | Typical path / pattern | Generator |
|---------------|------------------------|-----------|
| Synthetic / confidence-trap summaries | `docs/research/sage_reg_results/<date>/…` (CSVs, JSON, PNG from benchmark scripts) | `examples/benchmarks/self_agreement_synthetic.py` |
| Backbone comparison (Gaussian / quantile / bar) | dated under `docs/research/sage_reg_results/` | `examples/benchmarks/self_agreement_backbone_comparison.py` |
| Year real-data benchmark | per-run CSV + figures + `*_summary.json` | `examples/benchmarks/self_agreement_realdata_year.py` |
| Higgs OOD-style benchmark | per-run outputs under `docs/research/sage_reg_results/` | `examples/benchmarks/self_agreement_higgs_ood.py` |
| Supervised-gap tuning sweep | `…/supervised_gap_tuning_v3/sweep.csv` (example) | `examples/benchmarks/self_agreement_supervised_gap_tuning.py` |
| Fixed-config confirm | `…/supervised_gap_confirm_v5/` (example) | `examples/benchmarks/self_agreement_supervised_gap_confirm.py` |
| Multi-seed aggregation | e.g. `docs/research/sage_reg_results/2026-04-11/supervised_gap_multiseed_year_32ep/multiseed_summary.json` | `examples/benchmarks/self_agreement_supervised_gap_multiseed.py` |
| One-shot SAGE paper refresh (manifest + `METRICS.md`) | e.g. `docs/research/sage_reg_results/<date>/neurips_sage_reg_full/` | `scripts/run_neurips_sage_reg_full.py` (`--quick` for smoke budgets) |
| Legacy joint SAGE + SPT bundle + combined metrics | e.g. `docs/research/sage_reg_results/<date>/tabular_paper_bundle/tabular_paper_bundle_report.json` | `scripts/run_tabular_paper_bundle.sh` |
| Merged `n_labeled` sweep table | `…/tabular_runs/year_labeled_sweep_collated.json` | `tools/collate_sage_year_labeled_sweep.py` (after `run_tabular_research_experiments.sh` labeled sweep) |
| Semi-supervised demo summary | path passed to `semi_supervised_regression_comparison` | `examples/semi_supervised_regression_comparison.py` |

Long-run **competitiveness** experiment plan (external baselines, extra datasets): `docs/research/paper_strong_experiment_suite.md`.

**Preferred long-run entrypoint:** `scripts/run_neurips_sage_reg_full.py` (Year cache materialization, tabular phases, labeled sweep + collate, multiseed, optional CatBoost/TabReD/Higgs, synthetic/backbone/ablations, aggregation). Leaderboard paste template: `docs/research/higgs_external_scores.template.json`.

### One command (paper-scale SAGE)

```bash
uv run python scripts/run_neurips_sage_reg_full.py
```

Development / CI-sized budgets:

```bash
uv run python scripts/run_neurips_sage_reg_full.py --quick
```

**Shifts placeholder** (writes `README.txt` under `data/shifts/<dataset>/`) runs **by default**; use `--skip-shifts` or `--shifts-out-root` for a custom location. Photo-z is not part of this runner.

OpenML Year cache defaults to `data/paper/openml_year.csv` (created on first run unless `--no-year-download`). Example commands below use `docs/research/sage_reg_results/.../openml_year.csv` as a **local** cache path; that CSV is **not** in git (too large). Committed trees under `docs/research/sage_reg_results/` are **metrics** (`*.json`, `*.csv` summaries, `*.md`, `*.png`) — see `docs/research/sage_reg_results/README.md`. External SSL baselines now include **Mean Teacher** (`MeanTeacher`) and a **Pi-model style consistency** row (`PiModelConsistency`) in Year/Higgs benchmark scripts.

Joint SPT+SAGE pass: `./scripts/run_neurips_paper_bundle.sh` (forwards args to both scripts; **no** `--run-root`).

Optional image rebuttal pack (synthetic, lightweight): `--include-image-rebuttal` on `scripts/run_neurips_sage_reg_full.py` or direct script `examples/benchmarks/image_regression_rebuttal.py`.

Keep **stable filenames** once cited in `main.tex`. Large raw files (parquet, zip) are often local-only; see “What to commit” below.

## Example scripts (quick reference)

| Role | Script |
|------|--------|
| Demo / tutorial comparison | `examples/semi_supervised_regression_comparison.py` |
| Synthetic self-agreement | `examples/benchmarks/self_agreement_synthetic.py` |
| Backbone comparison | `examples/benchmarks/self_agreement_backbone_comparison.py` |
| Year benchmark | `examples/benchmarks/self_agreement_realdata_year.py` |
| Higgs benchmark | `examples/benchmarks/self_agreement_higgs_ood.py` |
| Tuning | `examples/benchmarks/self_agreement_supervised_gap_tuning.py` |
| Confirm | `examples/benchmarks/self_agreement_supervised_gap_confirm.py` |
| Multi-seed | `examples/benchmarks/self_agreement_supervised_gap_multiseed.py` |

## Example: multi-seed Year (32 epochs)

```bash
uv run python examples/benchmarks/self_agreement_supervised_gap_multiseed.py \
  --tuning-csv docs/research/sage_reg_results/2026-04-10/supervised_gap_tuning_v3/sweep.csv \
  --year-cache-path docs/research/sage_reg_results/2026-04-10/openml_year.csv \
  --skip-higgs \
  --out-dir docs/research/sage_reg_results/2026-04-11/supervised_gap_multiseed_year_32ep \
  --seeds 260410 260411 260412 \
  --year-teacher-epochs 32 \
  --year-student-epochs 32
```

For Higgs-only runs: `--skip-year`, `--higgs-dataset-path …`, and matching `--higgs-*-epochs` flags.

### Extra-large OpenML tabular (e.g. diamonds 42225)

`examples/benchmarks/self_agreement_realdata_year.py` can fetch OpenML regression tables via **`--openml-data-id`** (e.g. **42225** diamonds, or **42731** house sales; confirm on OpenML) and cap memory with **`--max-dataset-rows`**, optionally writing **`--cache-path`** for reuse.

For multi-seed supervised-gap confirmation, prefer passing a **materialized table path** to
`self_agreement_supervised_gap_multiseed.py` via **`--year-dataset-path`** (works for `.csv` / `.parquet`),
optionally with **`--year-benchmark-label`** so CSV/JSON rows are not mis-tagged as the Year MSD task.
Materialize pinned bytes once with **`tools/materialize_openml_large_tabular.py`**, then reuse the file.

The one-shot paper runner `scripts/run_neurips_sage_reg_full.py` includes an **OpenML diamonds** phase; because the
diamonds table is small (53,940 rows in the default materialization), it **proportionally shrinks**
`(n_unlabeled, n_test)` relative to the Year MSD protocol so `n_labeled+n_unlabeled+n_test` fits the table.

### Higgs parquet

The FAIR Universe Higgs uncertainty challenge publishes the tabular dump (see [Higgs Uncertainty Challenge](https://fair-universe.lbl.gov/Higgs-Uncertainty-Challenge.html)). The file is **not** vendored in git by default (large); you download or copy it locally and pass the absolute or repo-relative path to `--higgs-dataset-path`.

If you have already staged it under this repository (common layout from local runs):

| Asset | Typical path (from repo root) |
|-------|-------------------------------|
| Extracted parquet | `docs/research/sage_reg_results/2026-04-09/higgs_public/extracted/FAIR_Universe_HiggsML_data.parquet` |
| Original zip (before extract) | `docs/research/sage_reg_results/2026-04-09/higgs_public/raw/FAIR_Universe_HiggsML_data.zip` |

**Do not** use the literal string `/path/to/FAIR_Universe_HiggsML_data.parquet` from old examples—that is only a placeholder.

Example multi-seed command using the in-repo staging path (adjust the date folder if yours differs):

```bash
uv run python examples/benchmarks/self_agreement_supervised_gap_multiseed.py \
  --tuning-csv docs/research/sage_reg_results/2026-04-10/supervised_gap_tuning_v3/sweep.csv \
  --skip-year \
  --higgs-dataset-path docs/research/sage_reg_results/2026-04-09/higgs_public/extracted/FAIR_Universe_HiggsML_data.parquet \
  --higgs-split-scale-factor 10 \
  --higgs-parquet-max-sample-rows 6000000 \
  --out-dir docs/research/sage_reg_results/2026-04-11/supervised_gap_multiseed_higgs_10x \
  --higgs-teacher-epochs 32 \
  --higgs-student-epochs 32
```

### Higgs: larger splits (e.g. 10×) and parquet safety

The public FAIR-style Higgs dump is **~220M rows** of tabular simulation: **PRI** (reconstructed primary) and **DER** (derived) blocks, plus **weights** and **labels** / **detailed_labels**. Confirm/multiseed already use `target_column=labels`, `ood_score_column=PRI_met`, and drop `weights` and `detailed_labels` to avoid obvious leakage into features.

- **10× split sizes** (multiply train / unlabeled / test pools after tuning JSON resolution):  
  `--higgs-split-scale-factor 10` on `self_agreement_supervised_gap_confirm.py` and `self_agreement_supervised_gap_multiseed.py`.
- **Parquet reservoir**: loads use **Polars** `scan_parquet` (lazy) with a seeded random row-index join, then streaming collect—no full materialization of 100M+ row dumps. Raise `--higgs-parquet-max-sample-rows` when the split budget grows. The loader refuses **full-file** reads above `parquet_full_read_row_limit` (override with `--higgs-parquet-full-read-row-limit` only on large-memory machines).
- **Direct benchmark** (single run): `examples/benchmarks/self_agreement_higgs_ood.py` exposes `--scale-split-factor`, `--parquet-max-sample-rows`, and related flags.

## Related internal docs

- Claim boundary and stage status: `papers/neurips_sage_reg/status.md`
- Experiment staging: `docs/research/sage_reg_experiment_plan.md`
- Extended outline / notes: `docs/research/sage_reg_paper_outline.md`
- Research index: `docs/research/README.md`

## What to commit

Prefer **small** JSON/CSV summaries, `selected_configs.json`, and plots for fixed runs. Large downloads (parquet, zip) are often local-only or LFS—see `.gitignore` and team policy.

## Sister track

SPT-Reg artifact map: [papers/neurips_spt_reg/reproducibility.md](../neurips_spt_reg/reproducibility.md).
