# Plan to Implement Shift-Aware Optimal-Transport Conformal and Adaptive-Prior Uncertainty Methods in `torchregress`

## Introduction

This document proposes a concrete implementation plan for adding two recent distribution-shift methods to `torchregress`:

1. **Non-exchangeable conformal prediction with optimal transport**, which studies coverage degradation under shift through score-space optimal transport and uses unlabeled target data to learn calibration weights that reduce coverage loss
2. **Adaptive-prior Bayesian uncertainty under distribution shift (VIDS-style)**, which conditions the parameter prior on training and test covariates, approximates the resulting posterior with amortized variational inference, and uses synthetic bootstrap environments to train robustness to plausible covariate shifts

`torchregress` is already organised as a task-first regression and uncertainty library with reusable prediction containers, test-time tooling, conformal methods, and peer support for Bayesian, ensemble, and calibration families rather than a single modelling ideology  . It also already provides a normalised `PredictiveBatch` interface and model-agnostic test-time protocols that are a natural substrate for shift-aware uncertainty tooling  . Finally, the repo enforces strict public export and signature contracts, so any new abstraction must arrive with explicit API and tests from day one .

The two methods should **not** be treated identically.

* The **optimal-transport conformal method** is a strong fit for `torchregress` and should be implemented as a reusable experimental module with a clear path to becoming a first-class method.
* The **adaptive-prior VIDS-style method** is much heavier and should enter as a research-grade experimental module first, with a narrower frozen-feature implementation before any end-to-end version.

---

## Executive summary

### What should be built

* A **shift-aware conformal workstream** under `test_time` plus conformal wrappers
* A **shift-aware adaptive-prior Bayesian workstream** under `algorithms` or a tightly scoped experimental submodule

### What should be merged first

1. **OT-based non-exchangeable conformal**, classification-first
2. **Simplified VIDS-style frozen-feature variant**, not the full paper implementation immediately

### What should not happen

* Do not make either method a top-level default recommendation before benchmark evidence
* Do not implement the full VIDS training stack as the first iteration
* Do not add public API without corresponding snapshot-contract tests

---

## Guiding design principles

## 1. Keep methods modular

New methods should compose with the existing stack

`application model -> PredictiveBatch -> test_time modules -> calibration / conformal / monitoring`

rather than own the entire modelling workflow unless absolutely necessary.

## 2. Separate mature and research-grade abstractions

* OT conformal can plausibly become a stable reusable module
* VIDS-style adaptive-prior inference should start as **experimental**

## 3. Prefer frozen-feature interfaces when possible

This is especially important for the VIDS-style method, which otherwise becomes a full model zoo rather than a reusable uncertainty method.

## 4. Use `PredictiveBatch` everywhere possible

Any new method that emits predictive uncertainty should either produce or easily convert to `PredictiveBatch` so that downstream calibration and evaluation remain unified .

## 5. Preserve repo API discipline

Any public symbol added to `torchregress.test_time`, `torchregress.algorithms`, or other surfaced modules must be accompanied by export and signature tests in the existing public-API contract suite .

---

# Workstream A — Non-exchangeable Conformal Prediction with Optimal Transport

## A.1 Objective

Implement a reusable module for **coverage-gap diagnosis and mitigation under distribution shift** using score-space optimal transport and unlabeled target covariates, following the core ideas of the OT conformal paper .

## A.2 Why this belongs in `torchregress`

This method is a very strong architectural fit:

* it is a **test-time** uncertainty correction layer
* it does not require retraining the base predictor
* it works on top of calibration scores and target-time unlabeled inputs
* it is naturally compatible with conformal prediction
* it addresses **coverage under shift**, which is exactly where `torchregress` already positions conformal as a useful tool

## A.3 Placement in the repo

### Recommended internal placement

* `torchregress/test_time/ot_conformal.py`

### Recommended exports

From `torchregress.test_time.__init__`:

* `OptimalTransportCoverageGap`
* `OTShiftReweighter`
* `WeightedSplitConformalAdapter`

### Optional later placement

If the abstraction stabilises, add thin convenience wrappers under the conformal-facing surface, but keep the computational core in `test_time`.

---

## A.4 Proposed v1 scope

Start with:

* **classification-first**
* **unlabeled target batch setting**
* **free-form calibration weights**
* **weighted-CDF objective first**
* **weighted split conformal inference**
* **coverage-gap diagnostics**

Leave for later:

* full regression version
* Wasserstein objective as default
* label-shift-specialised variants
* sophisticated parametric weighting networks
* online target-stream adaptation

---

## A.5 Proposed public API

```python id="84027"
reweighter = OTShiftReweighter(
    score_mode="classification",
    objective="weighted_cdf",
    weight_parameterization="free",
    entropy_penalty=1e-3,
)

reweighter.fit(
    calibration_scores=cal_scores,
    target_unlabeled_scores=target_scores,
)

adapter = WeightedSplitConformalAdapter(alpha=0.1)
adapter.calibrate(
    calibration_scores=cal_scores,
    calibration_weights=reweighter.weights_,
)

pred_sets = adapter.predict_from_test_scores(test_candidate_scores)
```

### Auxiliary diagnostic interface

```python id="68487"
diag = OptimalTransportCoverageGap().estimate(
    calibration_scores=cal_scores,
    target_score_summary=target_scores,
)
```

### Minimum outputs

* learned calibration weights
* estimated score-distribution discrepancy
* estimated upper-bound proxy
* effective sample size of weighted calibration
* calibrated threshold / quantile

---

## A.6 Implementation phases

## Phase A0 — Design and mathematical scoping

Decide and write down:

* exact score conventions for classification
* how auxiliary target score distributions are represented
* whether the reweighter sees:

  * all class scores
  * min/max score summaries
  * uniform-pseudo-label score summaries
* the optimisation objective
* regularisation on weights
* weighted quantile definition

### Exit criteria

* equations fixed
* shape conventions fixed
* numerical solver chosen
* objective and diagnostics specified

---

## Phase A1 — Core utilities

Implement:

* score summarisation from classification outputs
* weighted empirical CDF utilities
* weighted quantile utilities
* effective sample size
* entropy / KL / smoothness penalties on weights
* coverage-gap proxy computation

### Internal helpers

* `_weighted_ecdf`
* `_weighted_quantile`
* `_normalize_weights`
* `_effective_sample_size`
* `_score_summary_from_logits_or_probs`

---

## Phase A2 — Reweighting module

Implement `OTShiftReweighter` with:

* `fit(calibration_scores, target_unlabeled_scores)`
* `weights_`
* `objective_value_`
* `diagnostics_`

First support:

* free scalar weights over calibration points
* simplex-constrained optimisation
* optional entropy regularisation

Later support:

* feature-conditioned weight network
* minibatch fitting
* temperature or sparsity controls

---

## Phase A3 — Conformal wrapper

Implement `WeightedSplitConformalAdapter` with:

* `calibrate(calibration_scores, calibration_weights)`
* `predict_from_test_scores(candidate_scores)`
* `threshold_`
* `coverage_diagnostics(...)`

This wrapper should remain small and composable rather than attempt to own the full classifier pipeline.

---

## Phase A4 — Integration with `PredictiveBatch`

For classification-style outputs, add helper conversion utilities that take model outputs and produce:

* candidate-label scores
* conformity / nonconformity scores
* set predictions
* size diagnostics

Where reasonable, expose outputs in `PredictiveBatch.extra` so that downstream evaluation tooling can inspect:

* threshold
* calibrated set size
* weighted ESS
* shift score

---

## A.7 Tests

## Unit tests

* weighted ECDF matches hand-computed result
* weighted quantile matches expected threshold on toy data
* weights always normalise and remain non-negative
* no-shift case returns near-uniform weights
* identical calibration and target score distributions yield negligible discrepancy
* stronger score shift increases estimated discrepancy

## Behaviour tests

* weighted split conformal reduces to ordinary split conformal under uniform weights
* calibration-set permutation does not change result
* entropy regularisation stabilises extreme weights
* effective sample size drops appropriately for concentrated weights

## Synthetic coverage tests

Construct toy classification settings with:

* no shift
* covariate shift only
* label shift only
* combined shift

Check:

* baseline split conformal undercovers under shift
* OT-reweighted conformal improves coverage
* set sizes do not explode unreasonably relative to alternatives

## Public API tests

Update the snapshot contract suite for any new public exports and signatures under `test_time` .

---

## A.8 Benchmarks

## Benchmark A — toy score-shift classification

Purpose:

* verify mechanism under controlled shift

Metrics:

* marginal coverage
* mean set size
* 90th percentile set size
* ESS of weighted calibration
* estimated gap proxy

Compare against:

* ordinary split conformal
* naive importance weighting if available
* uniform-weight weighted conformal sanity baseline

## Benchmark B — regression-as-classification

Purpose:

* verify compatibility with `torchregress` regression style

Use a discretised target head or binned predictive density and compare:

* marginal coverage
* interval width
* CRPS / NLL of base predictor
* conformal interval efficiency

## Benchmark C — shift severity sweep

Vary shift continuously and report:

* coverage degradation curve
* coverage recovery after OT reweighting
* set-size inflation curve

---

## A.9 Documentation

Add:

* `docs/test_time/ot_shift_conformal.md`
* one example notebook or script
* one benchmark report page
* method-catalog entry with cautious maturity label

The documentation should explain clearly:

* what the method guarantees and what it does not
* that it targets **coverage under shift**, not general density estimation
* why unlabeled target data help
* when the method is likely to fail

---

## A.10 Merge criteria

Promote beyond experimental only if:

* coverage improves materially under synthetic shift
* no-shift degradation is negligible
* implementation is stable and numerically well behaved
* regression-as-classification use is demonstrated
* docs and examples are decision-grade

---

# Workstream B — Adaptive-Prior Bayesian Uncertainty under Distribution Shift (VIDS-style)

## B.1 Objective

Implement an **adaptive-prior Bayesian uncertainty module** inspired by the VIDS paper, where uncertainty responds to how a test covariate sits relative to training covariates, using a posterior approximation conditioned on both train and test covariate context .

## B.2 Why this is harder

This method is much heavier than OT conformal:

* it changes the prior, not just post-hoc calibration
* it requires amortized variational inference
* it needs synthetic shifted environments
* it implicitly owns part of the training pipeline
* it is much less naturally model-agnostic

Therefore this should enter as a **research-track experimental module**, not as a stable first-class abstraction.

---

## B.3 Recommended implementation strategy

Do **not** start with the paper’s full end-to-end system.

Start with a staged narrowing:

### Stage 1

Frozen-feature VIDS-style model

### Stage 2

Optional end-to-end encoder version

### Stage 3

Only if warranted, richer adaptive-prior families and full environment training

This keeps the first implementation coherent and benchmarkable.

---

## B.4 Placement in the repo

### Recommended internal placement

* `torchregress/algorithms/adaptive_prior_vi.py`

This is preferable to `test_time` because the method owns training and inference, not just target-time adjustment. It also fits better than `ensemble`, because it is not fundamentally an ensemble abstraction.

### Recommended exports

From `torchregress.algorithms`:

* `AdaptivePriorGuide`
* `SyntheticEnvironmentSampler`
* `VIDSRegressor`

Keep the exported surface minimal.

---

## B.5 Proposed v1 scope

Implement a **frozen-feature regression version** with:

* external feature extractor or user-supplied features
* summary of training covariate distribution in feature space
* test-point-conditioned prior parameters
* amortized variational posterior for last-layer parameters
* Gaussian predictive output
* synthetic environment generation from bootstrap subsets
* calibration and shift benchmarks

Do not start with:

* full end-to-end deep feature learning
* classification-first multiclass architecture
* high-dimensional weight posterior over entire deep network
* energy-based prior over all network parameters
* elaborate environment curriculum

---

## B.6 Proposed public API

```python id="20814"
model = VIDSRegressor(
    in_features=d,
    hidden_dim=128,
    posterior_family="diagonal_gaussian",
    prior_family="adaptive_diagonal_gaussian",
    feature_mode="frozen",
)

model.fit(
    x_train_features=phi_train,
    y_train=y_train,
    n_environments=32,
    bootstrap_fraction=0.3,
)

pred = model.predict_distribution(phi_test)
# returns PredictiveBatch with mean/std and diagnostics
```

### Optional lower-level components

```python id="19527"
sampler = SyntheticEnvironmentSampler(
    bootstrap_fraction=0.3,
    n_environments=32,
)

guide = AdaptivePriorGuide(
    in_features=d,
    context_summary="set_mean_var",
    posterior_family="diagonal_gaussian",
)
```

---

## B.7 Internal architecture for v1

## Training inputs

* frozen features for train covariates
* responses
* optionally validation split

## Context representation

Need a permutation-invariant summary of the training covariates or environment covariates.

Start simple:

* feature mean
* feature variance
* maybe top-k principal directions

Then combine with a test feature vector to predict prior parameters.

## Prior network

A small MLP that maps:

* training-context summary
* test feature vector

to prior parameters for a last-layer weight posterior.

## Posterior guide

A variational family conditioned on:

* training context
* test feature
* observed training responses

For v1, keep it diagonal Gaussian and last-layer only.

## Predictive head

Gaussian regression head with:

* predictive mean
* epistemic uncertainty from posterior sampling
* optional aleatoric head later

---

## B.8 Synthetic environment generation

The paper motivates bootstrap-based synthetic environments as a way to simulate plausible covariate shifts from the original dataset alone .

In `torchregress`, v1 should implement only a restrained version:

* draw bootstrap subsamples
* optionally stratify by target range or feature clusters
* compute environment-specific training-context summaries
* train the guide to perform well across these environments

The sampler should be a reusable utility, but initially internal to the method.

---

## B.9 Training objective

Use a conservative first objective:

* predictive NLL on environment-held-out points
* KL term between posterior guide and adaptive prior
* variance regularisation to avoid posterior collapse
* optional environment-robustness penalty across synthetic shifts

Keep the objective small enough to debug.

Do **not** reproduce every nuance of the paper in v1.

---

## B.10 Tests

## Unit tests

* adaptive prior parameters change when test features move away from training summary
* train-set permutation does not change set-summary features
* posterior samples have expected shapes
* predictive variance remains positive
* `PredictiveBatch` output is valid

## Synthetic behaviour tests

Construct a regression problem where one covariate dimension is poorly supported in training but appears at test time. Then test that:

* deterministic Gaussian head is overconfident
* adaptive-prior model increases predictive variance in unsupported regions
* uncertainty does not inflate excessively in in-distribution regions

## Environment-sampler tests

* bootstrap sampler produces correctly sized environments
* environment summaries vary across draws
* repeated seeds reproduce identical environments

## Public API tests

Add export and signature snapshots if any symbols are made public under `algorithms` .

---

## B.11 Benchmarks

## Benchmark A — structured synthetic covariate-support gap

Purpose:

* verify the central claimed mechanism

Build a problem where training support is narrow in one covariate direction and test support expands into unseen regions.

Compare:

* Gaussian deterministic head
* deep ensemble
* SWAG
* Bayesian last-layer / BLR baseline
* VIDS-style adaptive prior model

Metrics:

* RMSE
* NLL
* CRPS
* PIT / calibration
* uncertainty-distance relation to training support

## Benchmark B — low-shot covariate shift

Train with one support pattern and evaluate under a shifted covariate regime. Check whether the adaptive prior actually improves predictive uncertainty rather than merely inflating all intervals.

## Benchmark C — ablation study

Ablate:

* adaptive prior vs fixed prior
* synthetic environments on/off
* frozen features vs learned features if Stage 2 is attempted
* diagonal posterior vs simpler deterministic baseline

These ablations are mandatory. Without them, the method will be too hard to interpret.

---

## B.12 Documentation

Add:

* `docs/algorithms/adaptive_prior_vi.md`
* synthetic demo
* benchmark page
* explicit warning that the method is experimental and heavier than standard UQ methods

The docs must explain:

* what problem the method is trying to solve
* why fixed priors may under-react to covariate shift
* what the adaptive prior conditions on
* why this is **not** a generic BNN replacement
* when to prefer simpler methods such as BLR, SWAG, ensembles, or conformal wrappers

---

## B.13 Merge criteria

This workstream should remain experimental unless it shows at least one real win versus simpler baselines:

* materially better NLL or CRPS under covariate shift
* better calibration under shift without severe in-distribution penalty
* uncertainty that tracks support mismatch in a way deterministic and simpler Bayesian baselines do not

If not, it should remain a research prototype.

---

# Shared infrastructure tasks

## 1. Prediction-container interoperability

Both workstreams should either emit or convert to `PredictiveBatch` so they compose with existing calibration and test-time tooling .

## 2. Test-time protocol compatibility

Where possible, make new methods interoperate with existing `SupportsPredictiveBatch` and `AdaptationBatch` conventions .

## 3. Public API discipline

All new public symbols must update the repo’s export and signature snapshot tests .

## 4. Documentation consistency

Update:

* method-selection matrix
* examples index
* comparative evidence docs
* any generated catalog snapshots if the methods become public

---

# Recommended implementation order

## Milestone 1 — OT conformal MVP

Deliver:

* score summarisation utilities
* OTShiftReweighter
* WeightedSplitConformalAdapter
* toy synthetic benchmark
* docs and examples
* public API tests

## Milestone 2 — OT conformal hardening

Deliver:

* regression-as-classification support
* stronger diagnostics
* optional Wasserstein objective
* benchmark sweep across shift severity

## Milestone 3 — VIDS-style frozen-feature prototype

Deliver:

* SyntheticEnvironmentSampler
* AdaptivePriorGuide
* VIDSRegressor
* synthetic covariate-support benchmark
* docs
* public API tests if exported

## Milestone 4 — VIDS go/no-go decision

Promote only if the frozen-feature prototype demonstrates a real gain. Otherwise keep private or experimental and do not widen the public API.

---

# Resourcing and risk

## Low-risk / high-value component

**OT conformal** is the lower-risk, higher-value addition.

Reasons:

* clean modular interface
* post-hoc usage
* unlabeled target support
* direct compatibility with the current `test_time` philosophy
* easy to benchmark against existing conformal baselines

## High-risk / research-heavy component

**VIDS-style adaptive prior** is the higher-risk component.

Reasons:

* larger architectural footprint
* more training complexity
* higher chance of benchmark sensitivity
* more difficult to keep library-general

That means the two workstreams should not be budgeted equally.

---

# Final recommendation

Implement **both**, but with explicitly different maturity targets.

### OT conformal

Implement as a **serious experimental feature with a path to first-class support**. It fits `torchregress` well and addresses a real gap in conformal validity under shift .

### VIDS-style adaptive-prior uncertainty

Implement only as an **experimental research-track module**, starting with a frozen-feature version. It is conceptually interesting, but too heavy to justify immediate broad public exposure without stronger benchmark evidence and simpler ablations .

In practical terms, the right strategy is:

* build the OT conformal method first
* stabilise and benchmark it
* then build a narrow VIDS-style prototype
* only widen the public surface of the second method if it clearly beats BLR, ensembles, SWAG, or simpler shift-aware baselines

Turn this into a repo-ready markdown file with proposed class signatures, acceptance checklists, and a phased issue breakdown.
