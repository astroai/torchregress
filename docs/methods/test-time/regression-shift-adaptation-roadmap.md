# Regression under shift: roadmap

Status: planning note  
Scope: reusable regression methods and test-time utilities  
Primary repo: `torchregress`  
Downstream consumer: `torchz`

## 1. Motivation

Many real regression deployments operate under distribution shift:

- **covariate shift**: the feature distribution changes between source and target,
- **selection bias / missing labels**: labeled source data are not representative,
- **target / label shift**: the marginal target distribution changes,
- **imbalance**: rare but scientifically important target regions are poorly covered,
- **calibration drift**: predictive uncertainty degrades even when point metrics remain acceptable.

`torchregress` should provide the **reusable method layer** for these settings: losses, reweighting, adaptation utilities, calibration, conformal wrappers, and evaluation tools. Application repos such as `torchz` should consume these components rather than reimplement them.

This document defines the roadmap for making regression-under-shift a coherent product surface in `torchregress`.

---

## 2. Design principles

### 2.1 Keep `torchregress` model-agnostic
`torchregress` should not own domain-specific architectures or dataset preparation logic. It should expose components that can operate on:

- point predictors,
- Gaussian heads,
- quantile heads,
- binned-PDF heads,
- sample-based predictive objects,
- frozen-feature backbones with lightweight adaptation.

### 2.2 Prefer composable primitives over monolithic pipelines
The library should expose small, testable pieces that can be assembled:

- weighting / reweighting,
- representation alignment,
- label-shift correction,
- lightweight head adaptation,
- conformal recalibration,
- drift-aware diagnostics.

### 2.3 Separate three goals
These are related but not identical:

1. **improve predictive accuracy under shift**
2. **restore calibration / coverage under shift**
3. **estimate or diagnose the shift itself**

APIs and examples should avoid conflating them.

### 2.4 Benchmark before recommending
Methods for imbalance or adaptation should not become default recommendations without clear wins on controlled synthetic drift and at least one real-data benchmark.

---

## 3. Current position

`torchregress` already contains important building blocks:

- imbalanced-regression losses such as `LDSLoss`, `FocalRLoss`, `DensityWeightedLoss`, and `PropensityWeightedLoss`,
- test-time label-shift correction,
- regression-oriented subspace alignment,
- OT/conformal reweighting tools,
- Bayesian linear heads on frozen features,
- conformal and calibration layers.

This is a strong base, but the current surface is still a **collection of parts** more than a unified regression-under-shift framework.

Main gaps:

1. no single canonical user guide for regression shift adaptation,
2. incomplete coverage of deep imbalanced regression beyond LDS/Focal-R,
3. limited benchmark standardization for shift severity and adaptation policies,
4. weak decision workflow for choosing among weighting, adaptation, and recalibration,
5. limited integration examples for dense predictive distributions under shift.

---

## 4. Product goal

Build a first-class `regression under shift` surface in `torchregress` with:

- clear APIs,
- evidence-backed defaults,
- strong examples,
- synthetic and real-data benchmarks,
- compatibility with tabular, image, and frozen-feature workflows.

Success means a downstream repo can do:

```python
pred = model.predictive_batch(x_target)
pred = shift_adapter.adapt(pred, features=z_target)
pred = calibrator.transform(pred)
intervals = conformal.predict_interval(pred)
```

without inventing new glue for each project.

---

## 5. Workstreams

### 5.1 Workstream A — unify the user-facing adaptation API

#### Goal
Expose a coherent adaptation layer rather than a bag of separate modules.

#### Deliverables
- A lightweight high-level wrapper, e.g. `RegressionShiftAdapter` or `ShiftPipeline`
- Shared protocol for:
  - `fit_source(...)`
  - `adapt_target(...)`
  - `transform_predictive_batch(...)`
  - `diagnostics(...)`
- Explicit support for:
  - point / Gaussian / quantile / binned-PDF predictions
  - optional features / embeddings
  - optional source labels
  - unlabeled target batches

This should be orchestration only, not a giant all-in-one method.

### 5.2 Workstream B — finish the imbalanced-regression surface

#### Goal
Upgrade imbalance support from available methods to a more complete and benchmarked module family.

#### Priority additions
1. **FDS (Feature Distribution Smoothing)**
   - the main missing method-family piece relative to the DIR literature,
   - ideally implemented in a backbone-agnostic way:
     - collect features,
     - estimate smoothed feature statistics by label bins,
     - expose a reusable feature-correction utility or training-time hook.

2. Better support for **tail-aware evaluation**
   - dense vs tail slices,
   - per-bin CRPS / NLL / RMSE,
   - tail conditional coverage,
   - catastrophic outlier rate by target density bin.

3. Safer guidance around imbalance
   - emphasize that tail weighting may improve sparse regions while harming calibration,
   - make post-hoc recalibration easy and visible.

#### Non-goal
Do not turn `torchregress` into a benchmark-only reproduction repo for every paper variant.

### 5.3 Workstream C — strengthen covariate-shift adaptation

#### Goal
Make covariate shift a first-class supported scenario.

#### Priority additions
1. **Density-ratio / importance-weight estimation utilities**
   - source-vs-target classifier weighting,
   - simple kernel / nearest-neighbor baselines,
   - ESS diagnostics and clipping helpers.

2. **Frozen-feature head adaptation**
   - build around current Bayesian linear heads,
   - support:
     - weighted refits,
     - recursive target updates,
     - source-target mixing policies,
     - forgetting schedules.

3. **Feature-stat / subspace adaptation**
   - mature the current subspace aligner into a documented, benchmarked default path,
   - add guidance on when to use:
     - simple feature normalization,
     - subspace alignment,
     - weighted conformal only,
     - no adaptation.

#### Nice-to-have
- domain classifier utilities for latent-space reweighting,
- shift score objects returned alongside adapted predictions.

### 5.4 Workstream D — make label-shift support regression-native

#### Goal
Treat continuous-target label shift as a real regression problem, not only a binned workaround.

#### Priority additions
1. Better documentation for current posterior label-shift correction.
2. Canonical utilities for:
   - choosing bin edges,
   - measuring stability to bin count,
   - monitoring prior-ratio degeneracy,
   - mapping corrected class/bin posteriors back to continuous summaries.
3. Experimental support for richer predictive families:
   - quantile-to-bin adapters,
   - support-grid density adapters,
   - sample-based approximations.

#### Important caveat
Continuous label-shift correction is inherently approximate unless the predictive representation supports it well. The docs should say this plainly.

### 5.5 Workstream E — calibration and conformal under shift

#### Goal
Make post-adaptation recalibration easy and well-supported.

#### Priority additions
1. Weighted conformal workflows with explicit shift-aware examples.
2. Coverage diagnostics stratified by:
   - source vs target,
   - density bin,
   - confidence slice,
   - OOD score slice.
3. Simple recipes:
   - adapt only,
   - recalibrate only,
   - adapt then recalibrate,
   - adapt then conformalize.

Under serious shift, improving point RMSE is not enough. Coverage and distribution quality must remain first-class reporting targets.

### 5.6 Workstream F — benchmark program

#### Goal
Move from method availability to decision-grade evidence.

#### Synthetic benchmarks
Include controlled axes:

- covariate shift severity,
- target prior shift,
- long-tail imbalance,
- selection bias,
- noisy features,
- calibration drift.

#### Real-data benchmark expectations
Use at least one downstream benchmark consumer such as `torchz`, but keep the benchmark harness reusable.

#### Standard reports
Every benchmark should report:

- RMSE / MAE,
- CRPS / NLL,
- calibration error / PIT diagnostics,
- coverage at multiple nominal levels,
- tail-slice performance,
- effective sample size after weighting,
- runtime / memory overhead.

---

## 6. Recommended implementation phases

### Phase 1 — consolidate and document
Target: short-term

- Add this roadmap.
- Add a user guide: when to use weighting vs adaptation vs recalibration.
- Add a benchmarked example for:
  - label shift,
  - covariate shift,
  - imbalance + recalibration.
- Expose a small unifying adaptation wrapper.

#### Exit criteria
A user can follow one doc and reproduce three canonical shift workflows end-to-end.

### Phase 2 — complete the imbalance story
Target: medium-term

- Implement FDS.
- Add tail-aware metrics and benchmark summaries.
- Publish comparison docs:
  - baseline,
  - density weighting,
  - LDS,
  - Focal-R,
  - FDS,
  - post-hoc calibration / conformal after each.

#### Exit criteria
There is a benchmark-backed recommendation table for rare-target regression.

### Phase 3 — mature adaptation
Target: medium-term

- Add density-ratio estimation helpers.
- Strengthen head adaptation interfaces.
- Add richer predictive-family adapters for label shift.
- Improve weighted conformal examples under drift.

#### Exit criteria
`torchregress` can support a clean source→target adaptation workflow for frozen-feature predictors and probabilistic heads.

### Phase 4 — evidence-backed defaults
Target: longer-term

- promote only methods that win consistently,
- downgrade methods that are available but fragile,
- tighten docs around default, advanced, and experimental.

#### Exit criteria
Method-selection docs reflect benchmark evidence rather than method novelty.

---

## 7. Proposed file/API additions

### Docs
- Keep this roadmap under `docs/methods/test-time/` while it summarizes implemented test-time APIs.
- Keep future shift-adaptation designs under `docs/research/plans/` until the corresponding code and runnable examples exist.
- Add public example pages under `docs/examples/` only when they are backed by runnable `examples/` scripts.

### Code
Possible additions:

- `torchregress.test_time.pipeline`
- `torchregress.test_time.importance_weighting`
- `torchregress.losses.imbalanced.FeatureDistributionSmoother`
- `torchregress.metrics.shift`
- `torchregress.benchmarks.shift`

---

## 8. Recommended defaults

For now, the practical defaults should remain conservative.

### If the problem is mostly imbalance
Start with:
- a strong probabilistic baseline,
- tail-slice evaluation,
- optional `DensityWeightedLoss`,
- recalibration / conformal validation.

### If the problem is mostly covariate shift
Start with:
- frozen features,
- source-target weighting,
- lightweight head adaptation,
- shift-aware conformal recalibration.

### If the problem is mostly label-shift-like
Start with:
- posterior correction on a discretized predictive representation,
- sensitivity analysis to bins,
- calibration checks after correction.

---

## 9. Explicit non-goals

This roadmap does **not** aim for:

- domain-specific models,
- dataset ingestion pipelines,
- image backbone fine-tuning recipes,
- one-off paper reproductions without reusable abstractions.

Those belong in downstream repos such as `torchz`.

---

## 10. Definition of success

This roadmap succeeds if `torchregress` becomes the obvious place to get:

- reusable regression adaptation methods,
- shift-aware uncertainty tools,
- benchmark-backed imbalance methods,
- clean interfaces consumed by downstream scientific libraries.

In that state, `torchz` and similar repos can focus on science tasks and model architectures, while relying on `torchregress` for the regression-under-shift method layer.
