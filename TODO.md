# torchregress TODO

Last updated: 2026-02-27

This TODO now tracks execution status against the adoption-audit plan and the
atomic tranche roadmap.

## Completed (Shipped)

### Core feature tranches
- [x] Prediction-powered inference core (`torchregress.inference`)
  - `ppi_mean_ci`, `ppi_quantile_ci`, `ppi_ols_ci`, diagnostics utilities.
- [x] Ordinal regression core
  - `OrdinalCrossEntropyLoss`, `CumulativeLinkLoss`, `CORALLoss`, ordinal metrics/utilities.
- [x] Censored regression core
  - `CensoredGaussianNLLLoss`, `CensoredQuantileLoss`, `AFTLoss`, censored metrics.
- [x] Long-tail / propensity core
  - `PropensityWeightedLoss`, propensity utilities, tail-focused metrics.
- [x] Constraints + calibration transforms
  - constrained heads + post-hoc calibrators (`VarianceTemperatureScaler`, `IsotonicMeanCalibrator`, `PITCalibrator`).
- [x] Density conformal + uncertain ground-truth core
  - `DensityConformal`, `PrevalenceAdjustedCP`, `MonteCarloConformal`
  - `NoisyTargetGaussianNLL`, `ConsistencyRegLoss`, `PseudoLabelNLL`.
- [x] Causal inference regression core (`torchregress.causal`)
  - `dr_ate`, `dr_cate`, overlap diagnostics, DR examples.

### Adoption/audit closeout
- [x] Audit v1 closeout artifacts regenerated and committed.
- [x] Method catalog/task matrix/comparative evidence generation pipeline stabilized.
- [x] Native leverage matrix formalized and test-backed.
- [x] Example summary governance (profile comparison + threshold checks) stabilized.
- [x] Docs/API/extras drift checks in place and passing.
- [x] CI-style gates green locally (`pytest`, `ruff`, `mypy`, `mkdocs`).

## P0 (Current highest leverage)

### 1) Real-data benchmark depth for adoption claims
- [x] Add at least one additional real dataset each for:
  - OOD/selective prediction
  - uncertain ground-truth
  - censored regression
  - ordinal regression
  - causal effect estimation
- [x] Keep synthetic tracks for fast CI; run heavier real-data tracks manually/scheduled.
  - Real-data comparison tracks now exist for ordinal, censored, uncertain-GT, and causal DR.
  - Synthetic tracks remain in place as CI-friendly baselines.

### 2) Photo-z benchmark parity hardening (RAIL + NNC-CRPS track)
- [x] Finalize manifest-backed dataset parity checks for paper-comparable runs.
- [x] Ensure baseline ingestion covers core RAIL set (`flexzboost`, `pzflow`, `delight`, `bpz`; `lephare` optional).
- [x] Publish one consolidated comparison report with runtime + quality metrics.

### 3) Native-vs-custom closure (remaining API surfaces)
- [x] Extend parity checks where overlap exists with `torchmetrics`/`torch.nn`.
- [x] Confirm no accidental reinvention remains in high-traffic metrics/loss wrappers.
- [x] Document intentional divergences with explicit tests.
  - Added class-weight + `ignore_index` parity checks for classification loss wrappers.
  - Added broader reduction-mode parity checks for core point-loss wrappers.
  - Point-metric wrappers now take native torchmetrics fast-paths for baseline mean reductions.

## P1 (Near-term)

### 4) Comparative evidence quality uplift
- [x] Broaden model-family comparisons for imbalance/long-tail tasks (beyond reweighting-only comparisons).
- [x] Expand conformal+ensemble/BNN/SWAG combinations in calibrated interval benchmarks.
- [x] Add stronger failure-mode notes per example for user decision support.
  - Propensity/long-tail comparison now includes GaussianNLL and quantile-family baselines with interval diagnostics.
  - OOD/selective comparisons now report split-conformal interval metrics across DeepEnsemble, MC dropout, SWAG, and BNN.
  - Example docs and summary notes now include explicit failure-mode guidance.

### 5) Performance governance
- [ ] Add stable microbenchmark baselines for newly added cores (ordinal/censored/uncertain_gt/causal/ppi).
- [ ] Add regression thresholds for these paths where signal/noise ratio is acceptable.

### 6) API consistency sweep (final polish)
- [ ] Re-check naming/signature consistency across old/new modules.
- [ ] Tighten public contract tests for any newly exposed helper APIs.

## P2 (Strategic)

### 7) Real-data-first recommendation guide
- [ ] Publish task-to-method guidance grounded in measured evidence (not capability-only claims).
- [ ] Split “works in synthetic”, “works in real proxy”, “decision-grade real-data” explicitly.

### 8) Scheduled benchmark automation
- [ ] Add lightweight scheduled jobs to refresh governed artifacts.
- [ ] Keep PR CI fast; perform heavy data runs outside default PR path.

## Not planned right now
- [ ] New major algorithm families without evidence-backed benchmark need.
- [ ] Feature expansion that increases API surface before real-data evidence catches up.

## Next execution order
1. Performance governance (P1.5)
2. API consistency sweep (P1.6)
3. Real-data-first recommendation guide (P2.7)
4. Scheduled benchmark automation (P2.8)
