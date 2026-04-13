# Plan to Implement Bayesian Learning Rule Abstractions and Tests for `torchregress`

## Introduction

This document proposes a concrete implementation plan for **Bayesian Learning Rule (BLR)** abstractions in `torchregress`, together with the required tests, documentation, examples, and benchmark gates needed to make the feature maintainable and useful.

In this plan, **BLR** is treated as a **posterior-updating Bayesian regression head on top of fixed learned features**, rather than as a new end-to-end Bayesian training framework. That framing matches the current `torchregress` design: the library is already organised as a task-first regression and uncertainty toolkit, with reusable prediction containers and architecture-agnostic test-time tooling rather than ownership of full application models  .

This proposal therefore aims to add a **fast, exact, low-maintenance Bayesian adaptation layer** that complements existing `BayesianNeuralNetwork`, `HeteroscedasticBNN`, `SWAG`, and ensemble methods, instead of duplicating them  .

---

## 1. Motivation and fit within `torchregress`

`torchregress` already exposes:

* task-first method guidance rather than method-family ideology 
* reusable test-time components sitting between an application model and downstream calibration / conformal / monitoring 
* a normalised prediction container, `PredictiveBatch`, intended to standardise outputs across tools 
* explicit test-time interfaces such as `SupportsPredictiveBatch` and `AdaptationBatch` for model-agnostic tooling 

BLR fits naturally into this architecture because it is most useful as:

* a **last-layer Bayesian adapter**
* a **fast few-shot or online update mechanism**
* a **low-compute uncertainty-aware head**
* a **test-time adaptation primitive**
* a **bridge between deterministic backbones and calibrated probabilistic outputs**

The proposed implementation should therefore live beside existing test-time tools, not as a replacement for existing Bayesian or ensemble families.

---

## 2. Goal

Add a new BLR component family that allows users to:

1. fit a Bayesian linear head on top of arbitrary feature representations
2. update that posterior exactly in batch or recursively
3. produce predictive means and variances compatible with `PredictiveBatch`
4. use the head as a lightweight adaptation layer under distribution shift
5. benchmark it fairly against ridge heads, ensembles, `SWAG`, and existing BNN tools

---

## 3. Non-goals for the first release

The first release should **not** attempt to cover:

* full-network Bayesian training
* variational layer replacements throughout arbitrary networks
* nonlinear kernels
* general probabilistic programming surfaces
* classification-first BLR interfaces
* highly expressive heteroscedastic neural likelihoods
* full-covariance multi-output Bayesian regression

Those are all valid future directions, but including them at the start would dilute the design and slow down delivery.

---

## 4. Scope of the first implementation

### 4.1 Core deliverables

The first implementation should include two public abstractions:

* `BayesianLinearHead`
* `RecursiveBayesianHead`

### 4.2 Supported setting in v1

The supported setting should be:

* regression only
* fixed features as input
* Gaussian likelihood
* conjugate Gaussian prior on weights
* scalar target and diagonal multi-target support
* exact posterior updates
* predictive mean and variance output
* optional `PredictiveBatch` conversion

### 4.3 Supported usage modes

Two modes should be supported:

* **batch posterior fitting**
* **streaming / sequential posterior updating**

This gives immediate value for both static datasets and test-time adaptation workflows.

---

## 5. Proposed module placement

### Recommendation

Implement BLR under:

* `torchregress/test_time/bayes.py`

and export symbols from:

* `torchregress/test_time/__init__.py`

### Why this placement is the right one

This is the cleanest fit because the current `test_time` package is already defined as **reusable test-time adaptation utilities without owning model architectures** . It also already defines the model-agnostic interfaces needed to integrate a posterior-updating head cleanly with external predictors and feature extractors .

Placing BLR under `ensemble` would be less precise, because BLR is not primarily an ensemble method and should not be framed as another heavy epistemic-UQ family alongside `DeepEnsemble`, `SWAG`, or variational BNNs.

---

## 6. Proposed public API

## 6.1 `BayesianLinearHead`

Purpose: exact batch Bayesian linear regression on top of fixed features.

```python id="44516"
head = BayesianLinearHead(
    in_features=d,
    out_features=1,
    fit_intercept=True,
    prior_mean=0.0,
    prior_precision=1.0,
    noise_variance=1.0,
    jitter=1e-6,
)

head.fit(phi_train, y_train)

pred = head.predict(phi_test, return_std=True)
# pred["mean"], pred["variance"], pred["std"]
```

### Required methods

* `fit(features, y, sample_weight=None)`
* `predict(features, return_std=False, include_noise=True)`
* `predictive_batch(features, include_noise=True)`
* `sample_weights(n_samples)`
* `reset_posterior()`

### Required public attributes / properties

* `posterior_mean`
* `posterior_covariance`
* `posterior_precision`
* `is_fitted`

---

## 6.2 `RecursiveBayesianHead`

Purpose: exact or algebraically equivalent recursive updates for sequential adaptation.

```python id="34616"
head = RecursiveBayesianHead(
    in_features=d,
    out_features=1,
    fit_intercept=True,
    prior_mean=0.0,
    prior_precision=1.0,
    noise_variance=1.0,
    forgetting_factor=1.0,
    jitter=1e-6,
)

head.partial_fit(phi_batch, y_batch)
pred = head.predict(phi_test, return_std=True)
```

### Required methods

* `partial_fit(features, y, sample_weight=None)`
* `predict(features, return_std=False, include_noise=True)`
* `predictive_batch(features, include_noise=True)`
* `reset_posterior()`

### Optional but useful methods

* `state_dict()`
* `load_state_dict()`
* `clone()`

---

## 6.3 `PredictiveBatch` integration

`torchregress` already provides a normalised `PredictiveBatch` abstraction for reusable tooling . BLR should integrate with it directly.

For a Gaussian BLR head:

* `mean` should populate `PredictiveBatch.mean`
* predictive standard deviation should populate `PredictiveBatch.std`
* optionally `point` may equal `mean`
* `extra` should include posterior diagnostics such as:

  * `epistemic_variance`
  * `aleatoric_variance`
  * `posterior_trace`
  * `n_observations_seen`

This makes BLR immediately interoperable with downstream calibration or conformal layers.

---

## 7. Mathematical specification

## 7.1 Model

For a feature vector (\phi(x) \in \mathbb{R}^d),

[
y = \phi(x)^\top w + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma^2)
]

Prior over weights:

[
w \sim \mathcal{N}(m_0, S_0)
]

Posterior after observing ((\Phi, y)):

[
S_N^{-1} = S_0^{-1} + \sigma^{-2}\Phi^\top \Phi
]

[
m_N = S_N\left(S_0^{-1}m_0 + \sigma^{-2}\Phi^\top y\right)
]

Predictive mean for a new feature vector (\phi_*):

[
\mu_* = \phi_*^\top m_N
]

Predictive variance:

[
\mathrm{Var}(y_* \mid \phi_*, \mathcal D) = \phi_*^\top S_N \phi_* + \sigma^2
]

## 7.2 Recursive update

The recursive implementation should use a mathematically equivalent sequential update rule based on rank-one or block updates. A hard requirement is that recursive and batch fits agree numerically within tolerance on the same data ordering.

## 7.3 Multi-target v1 policy

For (y \in \mathbb{R}^k), v1 should support **independent-output BLR heads**. That means one posterior per output dimension, sharing input features but not cross-target covariance.

This is much simpler, easier to test, and sufficient for the first release.

---

## 8. Architecture design choices

## 8.1 Feature ownership

BLR should **not** own feature extraction. The head accepts feature matrices directly.

This keeps the abstraction composable:

* tabular model backbone -> BLR head
* neural embedding model -> BLR head
* representation extractor from application repo -> BLR head

## 8.2 Numerical representation

Internally maintain posterior state using one of:

* covariance form
* precision form
* Cholesky factorisation where appropriate

Recommended approach:

* use precision and Cholesky solves for stability in batch mode
* use covariance or inverse-precision recursive updates for streaming mode
* add configurable `jitter`

## 8.3 Observation noise

In v1, support:

* fixed scalar noise variance
* optional per-output scalar noise variance

Do not start with learned heteroscedastic noise.

## 8.4 Intercept handling

Provide `fit_intercept=True` and implement it via feature augmentation rather than a separate special-case parameter.

## 8.5 Device and dtype behavior

BLR should work on:

* CPU
* CUDA tensors when available

and behave predictably across:

* `float32`
* `float64`

Tests should confirm acceptable numeric agreement.

---

## 9. Implementation work plan

## Phase 0 — design note and acceptance criteria

### Deliverables

* short design document
* API signatures
* exact mathematical specification
* list of non-goals
* benchmark success criteria
* decision on module placement and exports

### Exit criteria

* API frozen for v1
* naming settled
* acceptance tests enumerated

---

## Phase 1 — core batch BLR implementation

### Work items

1. create `torchregress/test_time/bayes.py`
2. implement `BayesianLinearHead`
3. implement posterior fit logic
4. implement Gaussian predictive mean / variance
5. implement `predictive_batch`
6. add docstrings and shape validation
7. add dtype / device handling
8. add small internal helper routines:

   * prior initialisation
   * design-matrix augmentation for intercept
   * Cholesky-safe solve
   * positive-variance clamps

### Exit criteria

* deterministic batch fit works
* posterior state accessible
* predictive variance correct
* `PredictiveBatch` output supported

---

## Phase 2 — recursive BLR implementation

### Work items

1. implement `RecursiveBayesianHead`
2. add `partial_fit`
3. implement sequential posterior updates
4. add forgetting-factor option
5. store update count and adaptation diagnostics
6. ensure algebraic agreement with batch BLR

### Exit criteria

* recursive updates stable
* batch and recursive fits agree on toy data
* online adaptation example runs end to end

---

## Phase 3 — public API integration

`torchregress` maintains strict public export and signature snapshot tests for public modules , so API integration must be explicit.

### Work items

1. export new classes from `torchregress.test_time.__init__`
2. update `__all__`
3. update `tests/test_public_api_contracts.py`
4. add signature snapshots for:

   * `BayesianLinearHead.fit`
   * `BayesianLinearHead.predict`
   * `RecursiveBayesianHead.partial_fit`
   * `predictive_batch`
5. decide whether to expose a minimal top-level import path or keep BLR only under `test_time`

### Recommendation

Keep BLR public under `torchregress.test_time` first. Do not lift it to the package top level in v1.

---

## Phase 4 — documentation

### Required docs

1. `docs/test_time/bayesian_heads.md`
2. one API reference page
3. one narrative example page
4. one benchmark results page or section

### Documentation content

The docs should cover:

* what BLR is
* when to use it
* when not to use it
* how it differs from:

  * `BayesianNeuralNetwork`
  * `HeteroscedasticBNN`
  * `SWAG`
  * `DeepEnsemble`
* how to connect it to external feature extractors
* how to convert outputs to `PredictiveBatch`
* how to wrap with conformal prediction afterward

This is especially important because `torchregress` already positions conformal and probabilistic methods as complementary rather than interchangeable .

---

## Phase 5 — examples

### Example 1: synthetic exactness demo

File suggestion:

* `examples/test_time_bayesian_linear_head.py`

Content:

* generate synthetic linear data
* fit BLR
* compare posterior mean to ground truth
* visualise predictive intervals
* compare batch vs recursive fit

### Example 2: frozen-feature adaptation demo

File suggestion:

* `examples/test_time_bayesian_head_shift.py`

Content:

* train or simulate a deterministic feature extractor
* induce mild target shift
* compare:

  * fixed deterministic linear head
  * re-fit ridge head
  * BLR head
* evaluate uncertainty calibration

### Example 3: integration with `PredictiveBatch`

File suggestion:

* `examples/test_time_blr_predictive_batch.py`

Content:

* show BLR head output
* wrap into `PredictiveBatch`
* run downstream calibration or interval diagnostics

---

## 10. Test plan

## 10.1 Unit tests

Create:

* `tests/test_test_time_bayes.py`

### Required unit tests

#### Posterior correctness

* posterior mean equals closed-form reference on tiny synthetic data
* posterior covariance equals analytic solution
* prior-only prediction behaves correctly with zero observations

#### Prediction correctness

* predictive mean matches manual calculation
* predictive variance is non-negative
* predictive variance decreases as data accumulate in well-conditioned settings
* `include_noise=True` and `include_noise=False` behave as intended

#### Shape handling

* scalar output works
* multi-target independent outputs work
* batch dimension checks are enforced
* intercept mode behaves correctly

#### Numeric stability

* collinear features do not crash when jitter is set
* `float32` and `float64` produce similar results within tolerance
* large and small prior precisions behave sensibly

#### Recursive consistency

* batch fit and recursive fit agree
* single-sample repeated updates equal block update
* forgetting factor changes posterior as expected

#### Device tests

* CPU behavior correct
* CUDA behavior correct when CUDA is available

#### `PredictiveBatch` interop

* `predictive_batch()` returns valid structure
* mean and std fields are populated correctly
* extra diagnostics are present

---

## 10.2 Public API contract tests

Because the repo enforces public export and signature snapshots , the following must be added to `tests/test_public_api_contracts.py`:

* exports in `test_time.__all__`
* signature snapshot strings
* parameter-order contracts where appropriate

This prevents API drift and keeps BLR aligned with existing package discipline.

---

## 10.3 Property-style tests

Add randomized tests for:

* invariance of batch fit to permutation of rows
* equivalence of repeated scalar updates and grouped updates
* positivity of predictive variance
* monotonic behavior of posterior confidence with more data in simple settings

---

## 10.4 Regression tests

Once benchmarks are running, add small regression tests that guard against:

* silent changes in predictive variance formula
* accidental omission of observation noise in total variance
* recursive-update bugs under forgetting factor
* incorrect output dtype promotion

---

## 11. Benchmark plan

The feature should not be considered “strong” or “recommended” until it demonstrates a real advantage on at least one meaningful axis.

## 11.1 Benchmark A — exactness and numerics

Purpose: verify mathematical implementation.

Tasks:

* synthetic linear Gaussian data
* known prior and noise
* compare posterior and predictive moments to closed form

Metrics:

* posterior mean error
* posterior covariance Frobenius error
* predictive mean / variance error
* runtime for batch and recursive modes

---

## 11.2 Benchmark B — practical low-shot adaptation

Purpose: test BLR in the regime where it is most likely to matter.

Setup:

* deterministic feature extractor
* small target-domain adaptation set
* mild covariate or response shift

Compare against:

* ordinary least squares head
* ridge regression head
* frozen Gaussian deterministic head
* `DeepEnsemble`
* `SWAG`
* `BayesianNeuralNetwork`

These are already represented in the library’s method framing and documentation  .

Metrics:

* RMSE
* NLL
* CRPS
* calibration score / PIT diagnostics
* interval coverage
* selective risk if applicable
* wall-clock adaptation time
* parameter count and memory

Success criterion:

BLR should either:

* improve calibration or NLL in low-shot adaptation, or
* deliver comparable predictive quality at materially lower adaptation cost

---

## 11.3 Benchmark C — online adaptation under drift

Purpose: validate the recursive interface.

Setup:

* data stream with gradual drift
* periodic evaluation after each chunk

Compare:

* static head
* periodically re-fit batch head
* recursive BLR with and without forgetting

Metrics:

* rolling RMSE
* rolling NLL / CRPS
* calibration drift
* adaptation latency

Success criterion:

Recursive BLR should outperform or match static alternatives under drift while remaining much cheaper than repeated full refits.

---

## 12. Documentation and method-catalog integration

The library currently treats Bayesian, SWAG, and ensemble methods as peer options in a task-first catalog rather than presenting one family as the default ideology . BLR should be integrated in the same style.

### Recommended catalog framing

Do not frame BLR as “the new Bayesian answer.”

Frame it as:

* **Bayesian posterior-updating head**
* strong for:

  * low-shot adaptation
  * online updates
  * cheap uncertainty-aware last-layer inference
* weaker for:

  * deeply nonlinear epistemic uncertainty
  * end-to-end posterior inference
  * multimodal targets
  * expressive heteroscedastic modeling

### Catalog status recommendation

Initial maturity label:

* `Available` or `Experimental`

Promote only after benchmarks and examples are in place.

---

## 13. Risks and mitigation

## Risk 1: feature overlap with existing BNN and SWAG methods

### Mitigation

Keep BLR narrowly defined as a last-layer posterior updater and document the distinction clearly.

---

## Risk 2: poor calibration in misspecified nonlinear settings

### Mitigation

Be explicit that BLR is not a substitute for richer predictive backbones. Encourage conformal wrapping or stronger base models where appropriate.

---

## Risk 3: numerical instability with ill-conditioned features

### Mitigation

Use Cholesky-based solves, jitter, and rigorous conditioning tests.

---

## Risk 4: API sprawl

### Mitigation

Keep v1 to two classes and a small number of methods. Avoid adding noise-estimation and robust-likelihood features prematurely.

---

## Risk 5: method-catalog clutter without evidence

### Mitigation

Require benchmark wins or efficiency advantages before giving BLR stronger recommendation status.

---

## 14. Acceptance criteria for merge

A production-quality first merge should require all of the following:

1. `BayesianLinearHead` implemented and documented
2. `RecursiveBayesianHead` implemented and documented
3. `PredictiveBatch` integration working
4. unit tests passing
5. public API contract tests updated
6. at least one synthetic exactness example
7. at least one practical adaptation benchmark
8. docs page explaining BLR vs BNN / SWAG / ensembles
9. benchmark results showing either:

   * better low-shot adaptation quality, or
   * comparable quality at materially lower cost

Without that evidence, the feature should remain experimental.

---

## 15. Recommended milestone schedule

## Milestone 1 — core implementation

* batch BLR
* recursive BLR
* unit tests
* export integration

## Milestone 2 — examples and docs

* exactness example
* adaptation example
* narrative docs
* API docs

## Milestone 3 — benchmarking and maturity decision

* low-shot adaptation benchmark
* online drift benchmark
* method-catalog entry
* recommendation level decided

---

## 16. Future extensions after v1

Only after the basic abstraction proves useful:

1. empirical-Bayes noise estimation
2. low-rank multi-output covariance
3. robust Student-(t) observation model
4. conjugate Bayesian updates with sample weights and decay
5. local or mixture-of-BLR heads in representation space
6. integration with conformal wrappers for guaranteed coverage
7. application-specific adapters for tabular and scientific regression pipelines

That order keeps the implementation disciplined.

---

## 17. Final recommendation

Implement BLR in `torchregress` as a **small, exact, posterior-updating regression-head family under `torchregress.test_time`**, not as a new full-network Bayesian subsystem.

That approach is well aligned with the current package structure:

* `torchregress` already exposes lightweight, lazy-loaded modular subpackages including `test_time` and `prediction` 
* `test_time` is already designed for reusable adaptation utilities 
* `PredictiveBatch` already provides the right prediction abstraction for downstream interoperability 
* public API contracts are already enforced and should be extended for BLR from the start 

This plan keeps the feature focused, testable, and valuable without inflating the library’s conceptual surface area.

I can next turn this into a repo-ready markdown file with exact class signatures, method docstrings, and a checklist format.
