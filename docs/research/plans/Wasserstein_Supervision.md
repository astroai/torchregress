# Plan to Implement Stable 2-Wasserstein Supervision and Self-Supervised Covariance Estimation in `torchregress`

## Introduction

This document proposes a concrete implementation plan for adding the main ideas from *Towards Self-Supervised Covariance Estimation in Deep Heteroscedastic Regression* to `torchregress` .

The paper targets a central problem in deep heteroscedastic regression: learning a sample-dependent covariance when direct covariance labels are unavailable. It argues that standard negative log-likelihood training can be unstable because covariance learning is entangled with residual errors, and proposes two complementary ingredients :

1. a **stable 2-Wasserstein-based supervision objective** for Gaussian mean-and-covariance prediction, using an upper bound that avoids eigendecomposition for non-commutative covariance matrices
2. a **self-supervised pseudo-labeling scheme** that estimates local target covariance from neighbourhood structure in input space, enabling covariance supervision even without ground-truth covariance labels

This is a strong fit for `torchregress`. The library already positions heteroscedastic Gaussian regression, uncertainty estimation, and multivariate probabilistic regression as core use cases, and explicitly treats Gaussian, ensemble, conformal, and related methods as peer tools in a task-first framework  . The proposed additions therefore extend an existing strength of the package rather than opening an unrelated branch.

The key recommendation of this plan is:

* implement the **2-Wasserstein bound loss** as a first-class loss in `torchregress.losses`
* implement the **covariance pseudo-labeler** as an experimental algorithm/util
* expose the **hybrid training recipe** in examples and docs, not as a core API abstraction initially

---

## 1. Goals

The implementation should let users:

1. supervise Gaussian mean and covariance prediction using a stable 2-Wasserstein surrogate
2. generate covariance pseudo-labels from neighbour structure when direct covariance labels are unavailable
3. train multivariate heteroscedastic regressors more stably than plain Gaussian NLL in regimes where covariance learning is difficult
4. mix supervised, self-supervised, and standard likelihood objectives in staged or hybrid schedules
5. benchmark covariance-learning quality against existing `torchregress` Gaussian and heteroscedastic baselines

---

## 2. Non-goals for the first release

The first release should **not** attempt to:

* fully reproduce every experiment in the paper
* introduce a monolithic trainer abstraction for all self-supervised heteroscedastic workflows
* make pseudo-label self-supervision the default recommendation for all tasks
* guarantee that neighbourhood pseudo-labels are statistically correct in all modalities
* replace Gaussian NLL or CRPS as standard baselines
* implement large modality-specific pipelines such as pose-specific image augment logic in the core package

Those can come later if the basic abstractions prove broadly useful.

---

## 3. Why this belongs in `torchregress`

This paper is directly aligned with the library’s current scope.

`torchregress` already emphasizes:

* heteroscedastic Gaussian regression as a primary workflow
* decomposition-aware uncertainty methods and multivariate Gaussian losses as mainstream options
* modular building blocks rather than one-off end-to-end application code

The paper’s main technical contribution is exactly the kind of reusable abstraction that libraries benefit from: a stable Gaussian-distribution loss for mean + covariance learning. Its strongest empirical claim is also relevant to library design: the 2-Wasserstein bound is much cheaper than heavier covariance-learning methods while remaining accurate on synthetic and UCI benchmarks .

The pseudo-label component is a weaker, more heuristic contribution, but still useful as an experimental add-on.

---

## 4. What should be implemented

## 4.1 Core feature to implement as stable public API

### `GaussianWassersteinBoundLoss`

This is the central addition.

It should implement the paper’s stable surrogate for the Gaussian 2-Wasserstein distance:
[
|\mu_1-\mu_2|_2^2 + |\Sigma_1^{1/2}-\Sigma_2^{1/2}|_F^2
]
used as an upper bound in the non-commutative covariance case .

This is the piece most worth turning into a reusable loss.

---

## 4.2 Experimental feature to implement behind a lighter recommendation

### `NeighborhoodCovariancePseudoLabeler`

This should implement the paper’s heuristic covariance pseudo-labeling algorithm:

* compute neighbourhoods in input space
* weight neighbours using a Mahalanobis-distance-based soft weighting
* estimate a local target covariance from neighbour targets
* use that covariance as pseudo-supervision for covariance prediction

This is useful, but should begin as experimental because its success depends strongly on geometry, representation quality, and modality.

---

## 4.3 Example-only recipe

### Hybrid schedule

The paper’s pose experiments suggest the 2-W objective is particularly useful as a stabilising pretraining or early-stage supervision mechanism, followed by likelihood-based fine-tuning .

That means `torchregress` should provide:

* an example of **2-W pretraining + Gaussian NLL fine-tuning**
* optionally a callback or tiny utility for staged objective schedules

But this should remain an example recipe at first, not a central public abstraction.

---

## 5. Recommended repo placement

## 5.1 Loss implementation

Place the core loss in:

* `torchregress/losses/gaussian_wasserstein.py`

Export from:

* `torchregress/losses/__init__.py`

Recommended public names:

* `GaussianWassersteinBoundLoss`
* optionally `gaussian_wasserstein_bound_loss`

---

## 5.2 Pseudo-label implementation

Place the pseudo-labeler in:

* `torchregress/algorithms/covariance_pseudo_labels.py`

Recommended public names:

* `NeighborhoodCovariancePseudoLabeler`
* `mahalanobis_covariance_pseudo_labels`

Why `algorithms` rather than `losses`:

* it is not itself a loss
* it is a data- or representation-driven heuristic estimator
* it is best treated as a reusable preprocessing / supervision utility

---

## 5.3 Optional helper utilities

If needed:

* `torchregress/utils/covariance.py`
* `torchregress/utils/neighbors.py`

Possible helpers:

* SPD matrix checks
* Cholesky reconstruction helpers
* covariance square-root conversion
* batched Mahalanobis computations
* neighbour weighting utilities

---

## 6. Proposed public API

## 6.1 Core loss

```python id="26781"
loss_fn = GaussianWassersteinBoundLoss(
    covariance_parameterization="cholesky",
    reduction="mean",
    mean_weight=1.0,
    covariance_weight=1.0,
    eps=1e-6,
)

loss = loss_fn(
    pred_mean=mu_hat,
    pred_scale_tril=L_hat,
    target_mean=y,
    target_covariance=cov_target,
)
```

### Required supported parameterisations

At minimum:

* `"covariance"`: full covariance matrices
* `"cholesky"`: lower-triangular Cholesky factor
* `"sqrt"`: symmetric or generic covariance square-root representation
* `"diagonal"`: diagonal covariance or variance vector

The `"cholesky"` path should likely be the default.

---

## 6.2 Pseudo-labeler

```python id="26782"
labeler = NeighborhoodCovariancePseudoLabeler(
    n_neighbors=32,
    metric="mahalanobis",
    weighting="softmax",
    regularization=1e-5,
)

cov_pseudo = labeler.fit_predict(X_train, Y_train)
```

Optional richer interface:

```python id="26783"
cov_pseudo = labeler.predict_for_query(
    x_query=x_batch,
    X_reference=X_train,
    Y_reference=Y_train,
)
```

This second interface is useful when pseudo-labels must be recomputed on embeddings rather than raw inputs.

---

## 6.3 Hybrid training example

```python id="26784"
for epoch in range(pretrain_epochs):
    cov_pseudo = labeler.fit_predict(features, y)
    loss = wasserstein_loss(mu_hat, L_hat, y, cov_pseudo)

for epoch in range(finetune_epochs):
    loss = gaussian_nll_loss((mu_hat, log_var_or_cov), y)
```

This should appear in examples and docs, not necessarily as a public class.

---

## 7. Mathematical scope

## 7.1 First-release mathematical target

The v1 implementation should cover multivariate Gaussian regression with:

* target mean vector (y)
* predicted mean (\hat\mu(x))
* predicted covariance (\hat\Sigma(x))
* target covariance label or pseudo-label (\tilde\Sigma(x))

The primary loss is:

[
\mathcal L_{\text{W2-bound}} =
|\hat\mu - y|_2^2 +
|\hat\Sigma^{1/2} - \tilde\Sigma^{1/2}|_F^2
]

or the equivalent batched form consistent with the chosen covariance parameterisation.

This should be documented explicitly as a **stable upper-bound surrogate objective**, not the exact Gaussian 2-W loss in the general non-commutative case .

---

## 7.2 Pseudo-label semantics

The pseudo-labeler should estimate a local covariance by:

1. finding nearest neighbours of (x_i)
2. computing neighbour weights from Mahalanobis distance
3. estimating a weighted local mean
4. estimating a weighted local covariance of neighbour targets

The implementation should support shrinkage or diagonal jitter to ensure SPD outputs.

---

## 8. Implementation phases

## Phase 0 — design note and API freeze

Decide:

* public names
* supported parameterisations
* whether target covariance inputs are covariance, Cholesky, or square-root by default
* behaviour for diagonal outputs
* SPD enforcement rules
* reduction semantics
* pseudo-labeler interface and whether it supports raw inputs only or arbitrary embeddings

### Exit criteria

* equations fixed
* shape conventions fixed
* parameterisation strategy fixed
* acceptance tests enumerated

---

## Phase 1 — core Wasserstein-bound loss

Implement:

* `GaussianWassersteinBoundLoss`
* batched support for:

  * mean vectors
  * diagonal covariance
  * full covariance via Cholesky
* conversion helpers
* reduction modes
* SPD-safe numerics

### Required behaviours

* no eigendecomposition in the main training path
* stable gradients
* support for multivariate targets
* compatibility with existing training loops that currently use Gaussian losses

### Exit criteria

* numerically stable on toy data
* diagonal and full-covariance modes work
* shape errors and invalid SPD inputs are handled cleanly

---

## Phase 2 — pseudo-labeler implementation

**Repo status (v1):** implemented as ``torchregress.algorithms.NeighborhoodCovariancePseudoLabeler`` and ``mahalanobis_covariance_pseudo_labels`` (Mahalanobis or Euclidean neighbour metric, softmax weights, SPD projection). Phase 3+ (scheduled hybrid examples, dedicated algorithm doc page, broader benchmarks) remain open.

Implement:

* `NeighborhoodCovariancePseudoLabeler`
* Mahalanobis-distance neighbour search
* softmax weighting
* weighted local mean / covariance
* shrinkage and jitter for stability

### Optional v1 simplifications

If needed, allow:

* Euclidean distance fallback
* global covariance estimate of (X) for Mahalanobis metric
* batchwise approximate nearest neighbours later

### Exit criteria

* pseudo-labels are SPD
* toy neighbourhood examples behave sensibly
* works for both univariate and multivariate targets

---

## Phase 3 — examples and recipes

Add examples:

1. synthetic univariate heteroscedastic regression
2. synthetic multivariate heteroscedastic regression
3. UCI-style tabular demonstration
4. hybrid schedule example

The examples should compare:

* Gaussian NLL
* `GaussianWassersteinBoundLoss`
* Gaussian NLL after 2-W pretraining
* pseudo-label vs oracle covariance label when oracle is available synthetically

### Exit criteria

* examples run end to end
* there is at least one notebook or script showing the intended practical pattern

---

## Phase 4 — docs and method catalog integration

Add docs:

* `docs/losses/gaussian_wasserstein.md`
* `docs/algorithms/covariance_pseudo_labels.md`
* Future runnable example page, promoted to `docs/examples/` only after a matching `examples/` script exists.

Update method-selection docs if the method proves good enough.

Initially frame it as:

* **experimental but promising**
* especially useful when covariance labels or pseudo-labels exist
* not a drop-in replacement for all Gaussian objectives

---

## 9. Tests

## 9.1 Unit tests for the loss

Add a new test file, e.g.:

* `tests/test_gaussian_wasserstein_loss.py`

### Required tests

* zero loss when predicted mean and covariance match targets
* diagonal mode matches hand-computed result
* full-covariance mode behaves correctly on small 2D examples
* batched reduction modes are correct
* loss is non-negative
* gradients are finite for well-formed SPD inputs
* invalid covariance shapes raise clear errors
* near-singular target covariances are stabilised by jitter

---

## 9.2 Unit tests for the pseudo-labeler

Add:

* `tests/test_covariance_pseudo_labels.py`

### Required tests

* returns SPD or PSD + jitter-stabilised covariance
* nearest-neighbour weighting sums to one
* identical local targets produce near-zero covariance
* local target spread increases pseudo-label covariance
* diagonal and full covariance outputs have expected shapes
* reproducibility with fixed seeds where relevant

---

## 9.3 Behaviour tests on synthetic data

Construct toy problems where the true covariance is known.

Check that:

* the Wasserstein-bound loss recovers covariance better than plain NLL in the target setup used for the test
* pseudo-label supervision helps covariance prediction relative to a no-supervision baseline
* the hybrid schedule outperforms or matches pure NLL in difficult optimisation settings

---

## 9.4 Public API contract tests

Because `torchregress` uses strict public export and signature snapshots, any new public symbols in `losses` or `algorithms` must be added to the existing contract tests .

Update:

* exports
* signatures
* parameter-order expectations where appropriate

This is mandatory if the symbols are public.

---

## 10. Benchmark plan

## Benchmark A — bivariate synthetic convergence

Reproduce a simplified version of the paper’s diagnostic setup:

* target and predicted Gaussians with mismatched mean/covariance
* compare convergence of:

  * Gaussian NLL
  * KL-style supervision if available
  * Wasserstein-bound supervision

Metrics:

* mean error
* covariance Frobenius error
* exact Gaussian KL if oracle available
* exact Gaussian 2-W if feasible offline for evaluation
* optimisation stability

Purpose:

* verify the central claim that the 2-W bound is more stable for covariance learning than residual-sensitive alternatives

---

## Benchmark B — synthetic multivariate heteroscedastic regression

Use multivariate synthetic data with known input-dependent covariance.

Compare:

* Gaussian NLL
* diagonal Gaussian baseline
* existing multivariate Gaussian loss path
* Wasserstein-bound loss with oracle covariance labels
* Wasserstein-bound loss with pseudo-labels
* hybrid 2-W pretrain + NLL fine-tune

Metrics:

* MSE
* NLL
* covariance error
* correlation recovery
* runtime
* memory

Purpose:

* test both accuracy and computational tradeoff

The paper’s tables on pages 8–9 suggest this is exactly where the method is strongest: good covariance quality without the heavy compute of TIC-like methods .

---

## Benchmark C — UCI-style tabular regression

If a lightweight tabular benchmark is already available or easy to add, compare the same set of methods on tabular data.

Metrics:

* MSE
* NLL
* covariance quality where proxy metrics exist
* compute time
* memory

Purpose:

* determine whether the improvement generalises beyond synthetic settings

---

## Benchmark D — ablation study

Ablate:

* Wasserstein bound loss vs Gaussian NLL
* pseudo-label supervision on/off
* pseudo-label metric: Mahalanobis vs Euclidean
* neighbour count (k)
* shrinkage strength
* hybrid schedule vs single-objective training

This is especially important because the pseudo-label mechanism is heuristic.

---

## 11. Documentation guidance

The docs should be explicit about what each component is for.

## 11.1 For the loss

State:

* this is a stable Gaussian supervision loss for mean + covariance
* it is especially useful when covariance labels or pseudo-labels exist
* it is an upper-bound surrogate in the general non-commutative case
* it does not replace calibration or conformal methods
* it does not by itself solve epistemic uncertainty

## 11.2 For the pseudo-labeler

State:

* this is heuristic self-supervision
* success depends on neighbourhood quality
* better embeddings may help
* raw-input neighbourhoods may fail on highly structured modalities
* users should benchmark against simpler baselines

## 11.3 For the hybrid recipe

State:

* use the 2-W objective to stabilise covariance learning early
* then fine-tune with Gaussian NLL or another proper scoring-rule objective
* this is a practical recipe, not a theorem

---

## 12. Risks and mitigation

## Risk 1: pseudo-labels are poor in raw input space

This is the biggest practical risk.

### Mitigation

Allow pseudo-labeling on arbitrary user-provided features or embeddings, not only raw inputs.

---

## Risk 2: users overinterpret the Wasserstein-bound loss as exact Gaussian 2-W

### Mitigation

Name and document it explicitly as a bound or surrogate.

---

## Risk 3: covariance parameterisation instability

### Mitigation

Prefer Cholesky or square-root interfaces, add SPD checks, shrinkage, and jitter.

---

## Risk 4: the loss improves covariance metrics but harms predictive NLL

### Mitigation

Make hybrid training a first-class example and benchmark it explicitly.

---

## Risk 5: API sprawl

### Mitigation

Keep the public surface minimal at first:

* one core loss
* one pseudo-labeler
* examples for schedules

---

## 13. Acceptance criteria for merge

The implementation should be considered successful if:

1. `GaussianWassersteinBoundLoss` is implemented, documented, and tested
2. it supports diagonal and multivariate full-covariance use
3. `NeighborhoodCovariancePseudoLabeler` is implemented and clearly marked experimental
4. examples show at least one synthetic setting where the method materially helps covariance learning
5. public API contract tests are updated
6. docs explain when to use the loss and when to avoid the pseudo-labeler
7. a benchmark demonstrates one of the following:

   * better covariance learning than plain Gaussian NLL at similar compute
   * comparable performance to heavier methods at much lower compute
   * improved hybrid training behaviour

---

## 14. Recommended release staging

## Stage 1

Ship:

* `GaussianWassersteinBoundLoss`
* tests
* synthetic example
* docs page

This should be the first merge.

## Stage 2

Ship:

* `NeighborhoodCovariancePseudoLabeler`
* synthetic self-supervised example
* ablation tests
* docs page marked experimental

## Stage 3

Ship:

* hybrid schedule example
* benchmark report
* optional method-catalog mention

Only after this stage should the method be surfaced prominently in recommendations.

---

## 15. Final recommendation

Implement this paper in `torchregress`, but do it in layers.

### Strong yes

* **`GaussianWassersteinBoundLoss`** as a public loss

### Conditional yes

* **`NeighborhoodCovariancePseudoLabeler`** as an experimental algorithm

### Example only

* **hybrid 2-W pretraining + Gaussian NLL fine-tuning**

This matches both the paper’s strongest contribution and the current architecture of `torchregress`: it adds a reusable, stable covariance-learning loss to a library that already prioritises heteroscedastic Gaussian regression and uncertainty-aware regression workflows   .
