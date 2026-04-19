# SAGE competitiveness — wave 2 (paper budget + statistics)

This note captures a **pre-registered style** plan for stronger, less optimization-noise-sensitive evidence under the repo’s existing **supervised-gap** framing (not a universal “beats all SSL regression” claim).

## Optimization hygiene (Year + gap tuning)

- **Cosine LR decay** is supported end-to-end for the Year track:
  - `SelfAgreementTrainer.fit(..., lr_schedule="cosine", lr_min=...)`
  - `examples/benchmarks/self_agreement_realdata_year.py` applies the same schedule to the supervised teacher, confidence-weighted student, Mean Teacher, Pi-model, and SAGE student phases.
- **CLI**
  - Year direct: `--lr`, `--lr-schedule {constant,cosine}`, `--lr-min`.
  - Supervised-gap tuning: `--year-lr`, `--year-lr-schedule`, `--year-lr-min`, plus narrowed grids via `--tau-values`, `--year-n-labeled`, etc.
- **Confirm / multiseed**: `self_agreement_supervised_gap_confirm.py` reads `year_lr`, `year_lr_schedule`, `year_lr_min` from the tuning run’s sidecar JSON (`sweep.json` next to the CSV, or `<csv-stem>.json`), so multiseed confirmation matches the sweep’s optimizer settings.

## Statistical reporting (multiseed)

- `tools/paper_report_common.summarize_multiseed` adds **`gap_bootstrap_95`**: a **nonparametric bootstrap over seeds** (resample seeds with replacement, recompute the mean gap each draw, take 2.5/97.5 percentiles). This complements the existing per-seed table and aggregate mean/std.
- Interpretation: uncertainty is **across rerolls of the semi-supervised protocol at fixed budgets**, not a frequentist CI on infinite population performance.

## Driver script (cache + paper splits + manifest)

Reusable entry point (defaults match NeurIPS **low-label Year** track: `nl=2048`, `nu=131072`, `nt=32768`, `32/32` epochs, cosine LR, offline cache under `data/paper/openml_year.csv`):

```bash
./scripts/wave2_year_gap_experiments.sh cache        # materialize Year CSV if missing
./scripts/wave2_year_gap_experiments.sh tune-medium  # 12-config gap sweep → gap_tune_medium/
./scripts/wave2_year_gap_experiments.sh multiseed    # optional; set WAVE2_OUT_DIR to same day as tune
# Full six seeds (unset WAVE2_SEEDS):
./scripts/wave2_year_gap_experiments.sh multiseed \
  docs/research/sage_reg_results/2026-04-17/wave2_paper_year/gap_tune_medium/sweep.csv \
  docs/research/sage_reg_results/2026-04-17/wave2_paper_year/multiseed_medium
./scripts/wave2_year_gap_experiments.sh aggregate   # refresh neurips sage_paper_report + METRICS.md
```

Latest **paper-split** tuning output (this repo run): `docs/research/sage_reg_results/2026-04-17/wave2_paper_year/gap_tune_medium/` and `manifest.json`. A **2-seed** multiseed smoke completed under `.../multiseed_medium/` (`WAVE2_SEEDS="260410 260411"`); re-run without `WAVE2_SEEDS` for all six seeds (~order of tens of minutes with full unlabeled pool).

## Suggested commands (templates)

Year-only, paper-ish budget (adjust paths to your machine):

```bash
python examples/benchmarks/self_agreement_supervised_gap_tuning.py \
  --skip-higgs \
  --year-n-labeled 2048 --year-n-unlabeled 8192 --year-n-test 8192 \
  --year-teacher-epochs 32 --year-student-epochs 32 \
  --year-lr-schedule cosine --year-lr-min 1e-5 \
  --out-dir docs/research/sage_reg_results/2026-04-17/wave2_year_gap_tune
```

Multiseed confirmation on the resulting `sweep.csv` (after picking best rows / copying `sweep.json` beside it):

```bash
python examples/benchmarks/self_agreement_supervised_gap_multiseed.py \
  --tuning-csv docs/research/sage_reg_results/2026-04-17/wave2_year_gap_tune/sweep.csv \
  --out-dir docs/research/sage_reg_results/2026-04-17/wave2_year_multiseed \
  --seeds 260410 260411 260412 260413 260414
```

Re-aggregate the NeurIPS digest so bootstrap CIs appear under `sage_multiseed`:

```bash
python tools/aggregate_sage_paper_report.py \
  --run-root docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full
```

## Primary readouts (unchanged semantics)

- **Year**: supervised-gap on **test NLL** (lower is better); secondary **Cov90** / calibration diagnostics in the benchmark CSVs.
- **Higgs public** (when enabled): **OOD NLL** gap vs supervised — stress test for trust weighting under covariate shift, not a clean i.i.d. regression benchmark.

## Completed run (2026-04-17)

Executed locally with **cosine LR**, **32/32** teacher–student epochs, **`n_labeled=2048`, `n_unlabeled=8192`, `n_test=8192`**, a **narrow 8-point** hyperparameter grid (two `tau`, two `unlabeled_noise`, `hard_weight_threshold` in `{none, 0.85}`), and **`pseudo_weight_values=[0.8]`** to avoid doubling the confidence-cache phase.

| Artifact | Path |
|----------|------|
| Gap tuning CSV / figure / config JSON | `docs/research/sage_reg_results/2026-04-17/wave2_year_gap_tune/sweep.{csv,png,json}` |
| Multiseed (6 seeds: 260410–260415), Year only | `docs/research/sage_reg_results/2026-04-17/wave2_year_multiseed/multiseed_summary.{json,csv}` |

**Multiseed aggregate (Year, NLL gap vs SupervisedOnly):** mean SAGE−Supervised ≈ **+1.05** (std ≈ 0.79); confidence-weighted baseline mean gap ≈ **+3.54**. So on this **small-label / smaller-pool** slice, SAGE did not close the gap to the supervised teacher under the chosen grid; the run is still useful as a **negative / stress** datapoint and for bootstrap reporting.

**Bootstrap on mean gap (2000 resamples of seeds, seed 42):** SAGE−Sup 95% interval ≈ **[0.45, 1.68]**; confidence−Sup ≈ **[2.87, 4.34]** (see `tools/paper_report_common.summarize_multiseed` on `multiseed_summary.json`).

The main NeurIPS bundle digest was regenerated so existing `sage/multiseed` rows gain **`gap_bootstrap_95`** in `sage_paper_report.json`:

```bash
python tools/aggregate_sage_paper_report.py \
  --run-root docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full \
  --write-markdown
```
