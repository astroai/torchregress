# SAGE-Reg status

Last updated: 2026-04-17

Canonical location for paper-track status. Internal notes also live under `docs/research/`.

**Shared roadmap (SAGE + SPT empirical priorities):** [docs/research/joint_empirical_priorities.md](../../docs/research/joint_empirical_priorities.md)

**Manuscript:** `papers/neurips_sage_reg/main.tex`. Build: `./papers/compile_tex.sh neurips_sage_reg` (uses vendored `papers/neurips_tex/neurips_2026.sty`).

## Executive summary

SAGE-Reg is a coherent paper prototype on `main` with:

- a narrow core API in `torchregress/semi_supervised.py`
- end-to-end support for Gaussian, quantile, and bar / binned-PDF predictors
- synthetic benchmarks that validate the main safety claim
- two **primary** real-data tracks:
  - `year` as the first large IID tabular regression benchmark
  - FAIR Universe Higgs public data as an OOD-oriented stress benchmark
- optional **extra IID-style tabular evidence** (not a third “headline” benchmark): OpenML diamonds (42225) multiseed + a **TabReD quick SSL probe bundle** (targets normalized inside the TabReD harness; use for breadth / sanity, not raw-number comparison to Year/Higgs)

The current evidence is:

- **strong** for "safer than confidence-weighted pseudo-labeling"
- **strong** for the synthetic confidence-trap story
- **good** for OOD-style downweighting on Higgs; **large-scale Higgs** (10× splits, 10M reservoir, 32 ep) shows **stable SAGE OOD NLL vs high-variance supervised OOD NLL** across three seeds (see table below)—use careful wording (variance, protocol).
- **not yet sufficient** for a blanket "beats `SupervisedOnly`" claim on **Year** under **tuned-hyper multiseed** (mean NLL can favor supervised)
- **new (2026-04-11):** a **single-seed labeled-budget sweep** on OpenML Year shows a **clear regime** — SAGE **wins NLL** at **`n_labeled` 2048–4096** (fixed `n_unlabeled=131072`), **loses** at **8192+** with default `YearRealDataConfig` training hypers; see collated `year_labeled_sweep_collated.json` under `docs/research/sage_reg_results/2026-04-11/tabular_runs/`. This supports an **honest "label-scarce IID"** claim rather than a universal NLL win.
- **new (2026-04-17 quick pack):** `scripts/run_neurips_sage_reg_full.py --quick --skip-tabred` completed under `docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full/` and now includes a separate `PiModelConsistency` baseline in Year/Higgs rows. Treat this run as protocol/sanity evidence (1-epoch quick budget), not final paper numbers.
- **new (2026-04-17 full pack refresh):** the same run root now also includes **`nl=1024` in the labeled sweep collate**, **CatBoost Year+Higgs ceilings** (requires `uv pip install catboost`), a **second OpenML tabular multiseed track** on **diamonds (42225)** under `openml_diamonds/` (split sizes are **scaled to the fixed-row table** by `scripts/run_neurips_sage_reg_full.py`), and an updated `sage_paper_report.json` / `METRICS.md` digest.
- **new (2026-04-17 TabReD appendix bundle):** `examples/benchmarks/tabred_sage_ssl_probe.py --quick` was run post-hoc into `docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full/tabred/` so the aggregated digest includes `tabred_bundle` (the original one-shot manifest had TabReD skipped because the runner was invoked with `--skip-tabred` during the long orchestration pass).

**Long-run competitiveness plan** (GBM ceilings, Mean Teacher, multi-seed × `n_labeled`, SPT DA baselines): [docs/research/paper_strong_experiment_suite.md](../../docs/research/paper_strong_experiment_suite.md).

## What is implemented

### Core method

Implemented in `torchregress/semi_supervised.py`:

- multi-view stochastic predictions
- shared predictive representation through `PredictiveBatch`
- consensus predictive law construction
- disagreement computation
- disagreement-to-weight mapping (optional **batch-relative** modes: z-score centering/scale, top-k trust gating)
- supervised + weighted distributional agreement training

### Backbone coverage

- Gaussian: analytic Gaussian cross-entropy / NLL to the consensus Gaussian
- quantile: density cross-entropy on the shared support grid
- bar / binned PDF: PMF cross-entropy on student bins

### Real-data refinement knobs

- feature masking / dropout perturbations
- feature-value mixing perturbations
- `weight_power` for tempered disagreement weighting
- optional `hard_weight_threshold`
- fixed-config tuning and confirm runners
- multi-seed confirmation runner

## Current experimental status

### Stage status

From `docs/research/sage_reg_experiment_plan.md`:

- Stage 1–4b: done
- Stage 5: **active** (close the real-data gap to `SupervisedOnly`)

### Latest fixed-config real-data results

Best `v3` confirm settings are recorded in:

- `docs/research/sage_reg_results/2026-04-10/supervised_gap_confirm_v5/selected_configs.json`

Single-seed confirms (see historical note in prior commits / CSVs in `docs/research/sage_reg_results/`).

### Multi-seed confirmation

First 3-seed fixed-config confirmation:

- `docs/research/sage_reg_results/2026-04-10/supervised_gap_multiseed_v1/multiseed_summary.csv`
- `docs/research/sage_reg_results/2026-04-10/supervised_gap_multiseed_v1/multiseed_summary.json`

### Tabular paper bundle + nl=2048 tuned multiseed (2026-04-11)

- **Bundle script:** `scripts/run_tabular_paper_bundle.sh` — writes `docs/research/sage_reg_results/2026-04-11/tabular_paper_bundle/` (SAGE Year direct with unlabeled fractions **0.25 / 0.5 / 1.0**, multiseed Year+Higgs at `nl=4096`, `nu=131072`, SPT **`full`** on **real** OpenML Year cache, plus `tabular_paper_bundle_report.json` / `METRICS.md`).
- **Year direct (`ufrac=1.0`, bundle):** SAGE NLL **2.003** vs supervised **2.135** (same run as `sage/year_direct/summary.json`) — **better than supervised** under **default** `YearRealDataConfig` hypers (not the tuning-CSV row).
- **Labeled sweep + collate:** `run_sage_year_labeled_sweep` → `tools/collate_sage_year_labeled_sweep.py` → **`year_labeled_sweep_collated.json`**: NLL gap (SAGE − sup) ≈ **−0.45** at `nl=2048`, **−0.13** at **4096**, **+0.10** at **8192**, **+0.14** at **16384**, **+0.12** at **32768** (single seed **260408**).
- **`tabular_runs/multiseed_year_nl2048`:** tuned hyperparameters from `supervised_gap_tuning_v3/sweep.csv`, **three seeds**, Year only: **mean** NLL gap **−0.157**, **median** **+0.030**, **std** **0.287** — **high seed variance** even in the favorable `nl=2048` regime.

### Higher-budget multiseed: Year, 32 epochs, three seeds (2026-04-11)

Output directory:

- `docs/research/sage_reg_results/2026-04-11/supervised_gap_multiseed_year_32ep/`

Fixed config from `supervised_gap_tuning_v3/sweep.csv` (same row as prior confirms): teacher/student **32** epochs, seeds `260410`, `260411`, `260412`, `year` only.

**NLL gap** (`SAGEObjective − SupervisedObjective`; negative means SAGE better):

| Seed   | Supervised NLL | SAGE NLL | Gap      |
|--------|----------------|----------|----------|
| 260410 | 2.196          | 1.976    | **−0.220** |
| 260411 | 2.125          | 2.263    | +0.138   |
| 260412 | 1.820          | 2.040    | +0.220   |

**Aggregate** (from `multiseed_summary.json`): `SAGEMinusSupervisedMean` ≈ **+0.046**, `Std` ≈ **0.191**. So the mean still slightly favors supervised on NLL, with one strong winning seed and two losses; this is **not** a stable multi-seed win yet.

**Cov90** (extra metric): SAGE remains closer to nominal than confidence-weighted pseudo-labeling on this table; confidence-weighted NLL stays catastrophically worse (~+1.8 vs supervised on average).

**Interpretation:** 32 epochs materially changes absolute NLL vs the older ~1.3 supervised reference from shorter-epoch tuning rows (different training budget / evaluation alignment). For the paper, report this three-seed table as the current **Year** evidence and keep the claim boundary: safety vs confidence weighting is solid; **beats supervised-only on mean NLL** is still unproven at three seeds.

A separate **1-epoch pipeline smoke** remains under `…/supervised_gap_multiseed_year_pipeline_verify/` (not comparable to the rows above).

### Large-scale Higgs: 10× splits, 10M parquet reservoir, 32 epochs (2026-04-11)

Output directory:

- `docs/research/sage_reg_results/2026-04-11/supervised_gap_multiseed_higgs_10x/`

Protocol: fixed row from `supervised_gap_tuning_v3/sweep.csv` (same hyperparameters as prior Higgs confirms), **split scale ×10** (`n_train=40960`, unlabeled/test pools scaled accordingly), **`parquet_max_sample_rows=10_000_000`**, teacher/student **32** epochs, seeds `260410`, `260411`, `260412`, Higgs only.

**OOD NLL** (`NLL_OOD`; primary objective in tuning row) and **Cov90_OOD**:

| Seed   | Supervised NLL_OOD | SAGE NLL_OOD | Conf. pseudo NLL_OOD | Supervised Cov90_OOD | SAGE Cov90_OOD |
|--------|-------------------|-------------|----------------------|----------------------|----------------|
| 260410 | 109.8             | 6.91        | 501.3                | 0.586                | 0.510          |
| 260411 | 53.1              | 6.54        | 570.8                | 0.578                | 0.510          |
| 260412 | 15.8              | 6.42        | 544.5                | 0.566                | 0.507          |

**Aggregate** (`multiseed_summary.csv`): `SAGEMinusSupervisedMean` ≈ **−52.9** (NLL_OOD), **Std** ≈ **38.4**. Confidence-weighted pseudo-labeling remains catastrophically worse on NLL_OOD (**mean gap ≈ +479**).

**Interpretation:**

- **SAGE-Reg OOD NLL is stable** (~6.4–6.9) across seeds at this scale; **supervised-only OOD NLL is extremely seed-sensitive** (15–110) while **ID** metrics stay in a reasonable band (e.g. supervised `NLL_ID` ~2.0, `RMSE_ID` ~0.86 on seed 260410). That pattern is consistent with **heteroscedastic Gaussian NLL on near-binary `labels`** plus **different 10M-row subsamples per seed**: occasional **severe OOD miscalibration** (e.g. Cov90_OOD ≈ 0.59 when NLL_OOD explodes) rather than a single obvious code bug.
- For the paper: this run supports a **scale / robustness** story—**semi-supervised agreement training damps pathological OOD density scores** compared to a plain supervised Gaussian head under shift—**provided** you report supervised **variance** (table above + std) and avoid claiming a single representative supervised number.
- **Science / benchmark caveat:** treating Higgs **binary `labels` as Gaussian regression** is contestable; reviewers may ask for Bernoulli/Brier or a continuous physics target. Treat current metrics as **protocol-faithful** to the existing benchmark, not as a claim about optimal Higgs modeling.

**Follow-ups:** (1) more seeds or **median** summaries for supervised OOD NLL; (2) optional **re-tune at 10×** (hyperparameters were selected on smaller splits); (3) log **NLL_ID** / **RMSE_OOD** in `multiseed_summary` for quicker diagnosis; (4) ablation: slightly **lower LR or weight decay** for supervised-only at large \(n\) if collapse is optimization-related.

## What we can claim now

Supported: safer unlabeled signal than scalar confidence weighting; narrow implementation across predictive families; representation sensitivity; reduced damage vs confidence-weighted pseudo-labeling on real tracks.

Supported with qualifiers: on **OpenML Year**, **single-seed** evidence that SAGE **improves NLL vs supervised in a label-scarce regime** (`n_labeled` ≲ 4096 with fixed large unlabeled pool) while **supervised wins at larger `n_labeled`** — see collated sweep; **multiseed at `n_labeled=2048`** still shows **high variance** (mean gap negative, median near zero).

Not yet supported: **tuned-hyper** multiseed **mean** superiority on Year at **`n_labeled=4096`** without seed cherry-picking; universal “beats supervised on IID tabular NLL.”

**Update (Higgs large run):** On **FAIR Higgs at 10× scale** with a 10M-row reservoir, SAGE achieves **much lower and more stable OOD NLL** than supervised across three seeds, but supervised **variance is high**—frame as **shift-stress / calibration stability**, not a blanket “beats supervised everywhere” claim without the Year line also holding.

## Next steps

1. Execute **[paper_strong_experiment_suite.md](../../docs/research/paper_strong_experiment_suite.md)** Tier A: **multiseed × `n_labeled`**, **GBM labeled-only ceiling**, **paper-budget re-tune**, then **second IID OpenML** dataset.
2. **Year narrative:** lead with **label-scarce regime** (sweep + multiseed at `nl∈{2048,4096}`) and **protocol consistency** (tuned row vs default hypers); avoid a single “beats supervised” sentence without those qualifiers.
3. **Higgs:** optional **re-sweep at 10×**; **5+ seeds** / **medians** for supervised OOD NLL; **binary-target caveat** in `main.tex`.
4. Keep **external SSL parity rows** (Mean Teacher + Pi-model consistency) in all Year/Higgs confirms at matched budget.

## Stop / go

**Go** for full paper draft when a clean real benchmark matches or beats supervised-only under multi-seed confirmation and the story is stable.

**Stop** and iterate on method when `year` stays above supervised-only after budgeted confirms and gains are only OOD-proxy.
# SAGE-Reg status (redirect)

**Canonical status for the NeurIPS track:** [papers/neurips_sage_reg/status.md](../../papers/neurips_sage_reg/status.md)

**Reproducibility (commands and paths):** [papers/neurips_sage_reg/reproducibility.md](../../papers/neurips_sage_reg/reproducibility.md)

Supporting materials remain here under `docs/research/` (experiment plan, outlines, dated result trees).
