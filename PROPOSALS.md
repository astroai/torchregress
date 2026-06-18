# Proposals — Actionable Work Items

**Last updated:** June 18, 2026
**Focus:** Stable API, high correctness, performance, and impact.
**API-breaking changes** (`.fit()`, `.predict()`) deferred to last phase.

---

## Progress Summary

| Category | Completed | In Progress | Planned |
|---|---|---|---|
| Architecture fixes | 3 | 0 | 0 |
| CI smoketests | 6 | 0 | 0 |
| Documentation | 6 | 0 | 0 |

---

## Phase 1 — Architecture Defaults & Correctness ✅

Fixing undersized default parameters in flow-based losses that caused
under-coverage and poor calibration in benchmarks.

### 1.1 NF hidden_features: 32 → 64 ✅
- **File:** `torchregress-harness/suites/tabular/distributional_bins.py`
- **Change:** `hidden_features=[32, 32]` → `[64, 64]`
- **Evidence:** Architecture sweep on diabetes showed 32-unit layers produced
  overconfident densities (Coverage=0.65). 64-unit layers improved to 0.75
  and won california_housing IS.
- **Key insight:** More transforms backfire (T5 worse than T3). Sweet spot:
  `n_transforms=3, hidden_features=[64, 64]`.
- **Cross-repo impact:** Also fixed in `torchregress-harness/suites/tabular/multivariate_intervals.py`.

### 1.2 SLS hidden_dim: 32 → 64, n_transforms: 2 → 4 ✅
- **File:** `torchregress-harness/suites/tabular/multivariate_intervals.py`
- **Change:** `hidden_dim=32` → `64`, `n_transforms=2` → `4` (matching SLSLoss defaults)
- **Evidence:** Sweep on synthetic_multivariate showed JointCoverage improves
  from 0.88→0.92 (+4.5%), IS from 1.24→1.21.
- **warmup_steps=50** kept (sweep showed no gain above 50 on these datasets;
  library default is 500).

---

## Phase 2 — CI Smoketests ✅

Fast, deterministic tests that catch regressions before benchmarks run.

### 2.1 NF architecture parameter-count check ✅
- **File:** `torchregress-harness/tools/_validate.py` — `validate_nf_architecture()`
- **Approach:** Instantiates NF via DistributionalBinsSuite and MultivariateIntervalsSuite
  at epochs=0 (no training). Checks each flow has >20k params.
- **Detects:** `hidden_features` regressed from `[64,64]` → `[32,32]` (~25k→~9k params).
- **Runtime:** <1s, deterministic.

### 2.2 SLS coverage threshold check ✅
- **File:** `torchregress-harness/tools/_validate.py` — `validate_sls_coverage()`
- **Approach:** Trains SLS on synthetic_multivariate at 1 epoch (200 smoke samples,
  seed=42). Asserts coverage ≥ 0.90.
- **Detects:** Coverage guarantee violations (SLS's core promise).
- **Limitation:** May not catch `hidden_dim`/`n_transforms` regressions since
  undersized flows produce broader densities → wider intervals → higher coverage.
  Architecture regressions need param-count check (see 4.1).
- **Runtime:** ~3s.

---

## Phase 3 — Documentation of Defaults

Documenting recommended minimums so users and developers don't accidentally
use undersized architectures.

### 3.1 SLSLoss docstring expansion ✅
- **File:** `torchregress/src/torchregress/losses/sls.py`
- **Added:** Full Parameters section with recommended minimums for `hidden_dim=64`,
  `n_transforms=4`, `warmup_steps=500`. Sweep evidence in docstring.
- **Notes section:** Explains alternating forward passes and UnionFrontier dynamics.

### 3.2 SLS Architecture Guidelines in CONTRIBUTING.md ✅
- **File:** `torchregress/CONTRIBUTING.md`
- **Added:** Table of recommended minimums with degradation effects.
  Footnote on warmup_steps for small datasets.
  Cross-references harness benchmarks.

### 3.3 NormalizingFlowLoss / create_flow_model docstring expansion ✅
- **File:** `torchregress/src/torchregress/losses/nflows.py`
- **Done:** Documented `create_flow_model` defaults (`n_transforms=5`, `hidden_features=[64,64]`)
  as defaults, plus `NormalizingFlowLoss` class docstring with Architecture Guidance
  table, scalar/multivariate tradeoff note, updated Examples using `create_flow_model`,
  and `.. math::` LaTeX formulation.  Sweep evidence: 32-unit layers → overconfident
  densities; 64-unit layers are the sweet spot.

### 3.5 Negative test: SLS architecture regression catch ✅
- **File:** `torchregress-harness/tools/_validate.py` — `validate_sls_architecture_negative()`
- **Done:** Meta-test that instantiates SLSLoss with old [32,2] config and asserts
  params < 25k (6,767), confirming the CI check correctly catches the regression.
  Shared `_sls_param_count()` helper eliminates duplication with the positive test.

### 3.4 NF Architecture Guidelines in CONTRIBUTING.md ✅
- **File:** `torchregress/CONTRIBUTING.md`
- **Done:** Added table mirroring SLS guidelines: `hidden_features`, `n_transforms`,
  `flow_type`.  Footnotes on n_transforms=3 for tabular data.
  Cross-references harness benchmark evidence.

---

## Phase 4 — Additional CI Coverage

Filling gaps in CI smoketest coverage.

### 4.1 SLS architecture parameter-count check ✅
- **File:** `torchregress-harness/tools/_validate.py` — `validate_sls_architecture()`
- **Done:** Instantiates SLSLoss directly (d=3, context_dim=20, hidden_dim=64,
  n_transforms=4) and checks params >25k.  Correct config: 31,570 params.
  Old config (hidden_dim=32, n_transforms=2): ~18k params (catches the downgrade).
  Integrated into `run_all.py --validate`.
- **Complements:** `validate_sls_coverage()` (coverage guarantee check).

---

## Phase 5 — Future: Stable API Extensions (Low Priority)

Work that can be done without breaking existing APIs.

### 5.1 GaussianNLL architecture guidelines ✅
- **File:** `torchregress/CONTRIBUTING.md`
- **Done:** Added "Gaussian Negative Log-Likelihood Architecture Guidelines"
  section with table of recommended output format, `log_variance`, `fixed_variance`,
  and `min_variance` settings.  Includes guidance on when to choose GaussianNLL
  vs MultivariateGaussian, LowRankGaussian, or GaussianCRPS.  Notes that
  GaussianNLL is the best all-rounder across all benchmark suites.

### 5.2 GaussianNLL coverage CI smoketest ✅
- **File:** `torchregress-harness/tools/_validate.py` — `validate_gaussiannll_coverage()`
- **Done:** Trains GaussianNLL on synthetic_multivariate at 1 epoch (200 smoke
  samples, seed=42) and asserts coverage ≥ 0.90.  Hits 0.989 in 3.3s.
  Integrated into `run_all.py --validate`.
- **Detects:** z_score changes, output dimensionality bugs, loss computation
  breakage.  GaussianNLL has no tunable internal architecture params —
  coverage is the right signal for this loss.

### 5.3 MultivariateGaussian coverage CI smoketest ✅
- **File:** `torchregress-harness/tools/_validate.py` — `validate_multivariate_gaussian_coverage()`
- **Done:** Trains MultivariateGaussian on synthetic_multivariate at 1 epoch (200 smoke
  samples, seed=42) and asserts coverage ≥ 0.90.  Complements the GaussianNLL
  (diagonal) check; covers the full-covariance case.  Integrated into `run_all.py --validate`.
- **Detects:** Cholesky/eigh fallback breakage, jitter regressions, loss computation bugs.

### 5.4 Cross-dataset CI parity ✅
- **Done:** Integrated harness `--validate` into:
  - `scripts/ci_local.sh` (pre-push hook) — runs `run_all.py --validate` if harness is present.
  - `.github/workflows/ci.yml` — clones and validates `torchregress-harness` after benchmark sweep.

### 5.5 MultivariateGaussian negative test ✅
- **File:** `torchregress-harness/tools/_validate.py` — `validate_multivariate_gaussian_coverage_negative()`
- **Done:** Meta-test that constructs MultivariateGaussianLoss with `jitter=0, eps=0`
  and a singular (all-ones) covariance matrix; asserts the loss produces `NaN`/`Inf`.
  Proves that the jitter/eps safety nets are essential and that removing either
  would break the loss instantly — which the coverage smoketest (5.3) would catch.
  Follows the same pattern as the NF and SLS negative tests.

---

## Phase 6 — Future: API Changes (Last Phase)

API-breaking changes deferred until stable foundation is complete.

### 6.1 `.fit()` / `.predict()` interface standardization
- Unify fit/predict signatures across all loss types.
- Consider sklearn-compatible API.

### 6.2 Deprecation of constructor-only flow creation
- Consider making `create_flow_model` the primary API (currently just a helper).
- Simplify the flow + loss construction pattern.

---

## How to Use This Document

1. **Pick a task** from the highest incomplete phase.
2. **Check the evidence** — every architecture change references benchmark data.
3. **Update this document** when a task is completed (✅) or new evidence emerges.
4. **Cross-reference** with `torchregress-harness/PARITY_REPORT.md` for benchmark
   results that motivate changes.
5. **API changes wait** — do not propose `.fit()`/`.predict()` changes until
   Phases 1-5 are complete.
