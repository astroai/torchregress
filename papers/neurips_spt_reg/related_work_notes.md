# SPT-Reg — related work and experimental comparison map

This note ties **literature families**, **factorized method stages**, and **JSON rows** (`year_competing_methods_<profile>.json`, synthetic / tabular siblings). It backs `main.tex` Table `tab:spt_baseline_map`.

## 0. Post-2023 shift, conformal, and adaptation (external SOTA)

| Work | Venue / year | Core idea | Relation to SPT-Reg |
|------|----------------|-----------|----------------------|
| Gibbs & Candès | JMLR 25(162), 2024 | Online conformal under **arbitrary** distribution shifts (DtACI / regret view) | We use **split** conformal as an explicit stage after transport; their line is the right modern anchor for “shift-tolerant conformal” discussion. |
| Kim et al. | NeurIPS 2024 | TTA strengthens accuracy and agreement-on-the-line | Positions **test-time adaptation** as an active 2024 research area; complements our `TargetRefitSmallGaussian` and subspace rows. |
| (Foundational) Tibshirani et al. | NeurIPS 2019 | Weighted conformal under covariate shift | Still the standard citation for **likelihood-ratio / weighted** conformal; we cite it for static tilt; pair with Gibbs & Candès for online drift. |

Other 2023--2025 lines (doubly robust weighted CP, multi-source CP, label-free set adaptation) are surveyed in the ML literature but **not** reimplemented row-for-row here—see `paper_strong_experiment_suite.md` for roadmap.

## 1. Literature ↔ positioning (not always 1:1 with a row)

| Theme | References | Relation to SPT-Reg |
|-------|------------|---------------------|
| Dataset / covariate shift | Quiñonero-Candela et al.; Huang et al. (KMM) | Classic **input** reweighting / matching; SPT focuses on **output-law** transport with a fixed source predictor. |
| Invariant / adversarial DA | Ganin et al. (DANN) | **Not reimplemented** as a benchmark row; positioning only. |
| Conformal prediction | Angelopoulos & Bates | **Rows:** `RawSplitConformalGaussian`; `WeightedSplitConformalGaussian` (density-ratio weights). |
| Conformal under shift | Tibshirani et al.; Gibbs & Candès (2024) | Weighted / online shift-tolerant CP; our stack separates **transport** vs. **conformal** stages. |
| TTA | Kim et al. (2024); subspace TTA for regression (CV/robotics lines) | **Rows:** `SignificantSubspaceGaussian`, `TargetRefitSmallGaussian`. |
| PPI | Chernozhukov et al. | **Metrics** where enabled. |
| Deep ensembles | Lakshminarayanan et al. | **Synthetic / family** stress via MDN rows. |

## 2. Year-track Gaussian rows ↔ isolated factor

| `Method` | Isolated experimental factor | Typical literature bucket |
|----------|------------------------------|---------------------------|
| `SourceGaussian` | No adaptation | Shift-agnostic deployment |
| `FeatureStatNormGaussian` | Input normalization only | Moment matching / simple DA prep |
| `SignificantSubspaceGaussian` | Subspace alignment only | Subspace / TTA-adjacent alignment |
| `PriorTransportGaussian` | Stage A prior transport only | Output / prior correction |
| `RawSplitConformalGaussian` | Conformal on source law | Distribution-free validity |
| `WeightedSplitConformalGaussian` | Weighted split conformal (logistic $p_{\mathrm{tgt}}/p_{\mathrm{src}}$) | Tibshirani et al.–style adapter (benchmark row) |
| `SPTTransportGaussian` | Transport + inflation, **no** conformal | Adaptation path in isolation |
| `SPTRegGaussian` | Full pipeline + conformal | Proposed method |
| `TargetRefitSmallGaussian` | Small-budget target refit | TTA / ridge competitor |

## 3. Year-track BinnedPDF rows

| `Method` | Contrast |
|----------|----------|
| `SourceBinnedPDF` | Ordered-bin head, no transport. |
| `SPTRegBinnedPDF` | Same head + full SPT-Reg stack. |

## 4. Synthetic-only extensions

- **MDN family:** `SourceMDN`, `RawConformalMDN`, `SPTTransportMDN`, `SPTRegMDN`.
- **Oracle refit:** `TargetRefitOracleGaussian`.

## 5. Automation

- Stage-A clip sweep: `tools/sweep_spt_year_stage_a.py`.
- Full run + aggregation: `scripts/run_neurips_spt_reg_full.py` → `tools/aggregate_spt_paper_report.py`.

## 6. Known comparison gaps (for reviewers / roadmap)

- **DANN / CORAL / MMD** retraining baselines on Year.
- **Full online conformal** (Gibbs & Candès-style) coupled to our transport stack.
- **Shifts-Project** regression split: `tools/fetch_shifts_dataset.py` is a placeholder only.

## 7. Analyzing a finished `run_neurips_spt_reg_full` tree

```bash
uv run python tools/analyze_neurips_spt_reg_run.py \
  --run-root reports/neurips_spt_reg/runs/<date>/neurips_spt_reg_full
```

Compare **NLL / CRPS / Cov90 / Width90** for `RawSplitConformalGaussian` vs. `WeightedSplitConformalGaussian` vs. `SPTRegGaussian` on your disk.
