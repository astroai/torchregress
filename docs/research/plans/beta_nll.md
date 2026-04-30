# Plan to Implement Five High-Impact Modern Regression Meefthods in `torchregress`

## Introduction

This document proposes a concrete implementation roadmap for five high-impact regression methods, all from roughly 2022 onward, that are strong candidates for addition to `torchregress`.

The selection is based on four criteria:

1. **scientific impact**
2. **fit with the current `torchregress` scope**
3. **engineering leverage**
4. **evidence of adoption**, including public code and strong venues when available

This is not a pure citation ranking. It is a practical product-and-research ranking for `torchregress`.

The current package already covers a broad base of regression losses and uncertainty methods, including Gaussian NLL/CRPS, multivariate and low-rank Gaussian losses, ensembles, Bayesian neural network variants, SWAG, MDNs, flows, and multiple conformal families  . The plan therefore focuses on methods that meaningfully extend that surface rather than duplicate it.

The five recommended additions are:

1. **β-NLL** for robust heteroscedastic training, from ICLR 2022, with official public code
2. **TIC-TAC** for improved covariance estimation in deep heteroscedastic regression, from ICML 2024, with official public code
3. **Effective Bayesian Heteroscedastic Regression** using natural heteroscedastic heads and Laplace-style posterior approximation, from NeurIPS 2023, with official public code
4. **Stable 2-Wasserstein bound supervision plus self-supervised covariance pseudo-labeling**, from ICLR 2025
5. **Faithful Heteroscedastic Regression**, an important modern baseline in the same family, also directly discussed as a relevant comparison in the 2025 covariance paper

---

## Portfolio view

| Method                              | Priority | Public maturity target | Main value                                            |
| ----------------------------------- | -------: | ---------------------- | ----------------------------------------------------- |
| β-NLL                               |        1 | Stable public API      | Cheap, high-impact heteroscedastic baseline           |
| 2-Wasserstein bound                 |        2 | Stable public API      | Strong new covariance-supervision loss                |
| TIC-TAC                             |        3 | Experimental first     | Structured covariance learning beyond plain NLL       |
| Natural heteroscedastic Laplace     |        4 | Experimental first     | Strong Bayesian heteroscedastic path                  |
| Faithful heteroscedastic regression |        5 | Stable or semi-stable  | Important modern baseline and lightweight alternative |

---

## Guiding principles

## 1. Add methods as composable building blocks

Where possible, new methods should appear as:

* losses
* heads
* wrappers
* metrics
* small training utilities

not as one large monolithic trainer.

## 2. Keep benchmark discipline

Every new method should be compared against:

* existing Gaussian NLL baselines
* deep ensembles / SWAG where relevant
* simpler heteroscedastic baselines already in the package
* calibration and interval metrics, not only RMSE

## 3. Separate “stable” from “experimental”

Some additions are obvious general-purpose primitives. Others are promising but heavier or less universally reliable.

## 4. Prefer low-friction public APIs

A user should be able to adopt most methods by changing one loss, one head, or one wrapper, rather than rewriting their whole stack.

---

# Workstream 1 — β-NLL

## Why this method matters

The official repository for *On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks* presents β-NLL as a mitigation for unstable or poor heteroscedastic likelihood training and provides a compact reference implementation of the loss . This is probably the single highest-ROI missing method because it is both influential and trivial to integrate.

## Goal

Add a robust heteroscedastic regression loss that rescales each sample’s NLL contribution using the predicted variance raised to a configurable exponent β.

## Recommended placement

* `torchregress/losses/beta_nll.py`

Public exports:

* `BetaNLLLoss`
* `beta_nll_loss`

## Proposed API

```python
loss_fn = BetaNLLLoss(beta=0.5, reduction="mean", variance_mode="diagonal")
loss = loss_fn(pred_mean, pred_variance, target)
```

Optional extension:

* multivariate diagonal variance
* full covariance later only if clearly justified

## Implementation scope

### v1

* scalar and diagonal-Gaussian β-NLL
* detach-based weighting, consistent with the reference implementation
* reduction modes
* numerical epsilon handling

### v2

* support for multivariate structured covariance if benchmarking shows real value

## Tests

* β = 0 reduces to Gaussian NLL-like weighting behavior
* loss is finite for positive variances
* larger variance changes sample weighting as expected
* gradients remain finite
* matches the reference formula on toy inputs

## Benchmarks

Compare against:

* Gaussian NLL
* MSE with post-hoc variance head if applicable
* Faithful
* 2-Wasserstein-bound loss on matching tasks

Metrics:

* RMSE
* NLL
* convergence stability
* sensitivity to learning rate

## Merge target

This should be a **stable public addition** early. It is cheap and immediately useful.

---

# Workstream 2 — Stable 2-Wasserstein Bound Supervision

## Why this method matters

The ICLR 2025 paper studies self-supervised covariance estimation in deep heteroscedastic regression and proposes a stable upper-bound surrogate for Gaussian 2-Wasserstein supervision that avoids eigendecomposition in the non-commutative covariance case . This is one of the strongest genuinely new loss ideas among recent heteroscedastic regression papers.

## Goal

Add a stable distribution-level loss for supervising mean and covariance jointly.

## Recommended placement

* `torchregress/losses/gaussian_wasserstein.py`

Public exports:

* `GaussianWassersteinBoundLoss`
* `gaussian_wasserstein_bound_loss`

## Proposed API

```python
loss_fn = GaussianWassersteinBoundLoss(
    covariance_parameterization="cholesky",
    mean_weight=1.0,
    covariance_weight=1.0,
    reduction="mean",
)
loss = loss_fn(pred_mean, pred_scale_tril, target_mean, target_covariance)
```

## Implementation scope

### v1

* diagonal covariance
* full covariance through Cholesky or square-root parameterization
* no eigendecomposition in the training path
* batched multivariate support

### v2

* exact commutative Wasserstein mode when available
* mixed supervised / hybrid schedule helpers

## Tests

* zero loss when mean and covariance match targets
* diagonal and full-covariance modes behave correctly
* gradients are finite for SPD inputs
* output remains non-negative
* jitter stabilizes near-singular targets

## Benchmarks

Compare against:

* Gaussian NLL
* β-NLL
* TIC-TAC
* Faithful

Metrics:

* covariance Frobenius error
* NLL
* MSE
* runtime
* memory
* calibration metrics downstream

## Merge target

This should also be a **stable public addition**. It is a core loss-level primitive, not just a paper-specific trick.

---

# Workstream 3 — TIC-TAC

## Why this method matters

The official TIC-TAC repository describes the ICML 2024 method as a framework for improved covariance estimation using a Taylor-Induced Covariance parameterization and Task Agnostic Correlations metric . This is one of the most serious recent proposals for structured covariance learning in deep heteroscedastic regression.

## Goal

Add structured covariance-learning support that goes beyond plain Gaussian NLL heads.

## Recommended placement

Split the implementation:

* `torchregress/algorithms/tictac.py`
* `torchregress/metrics/tac.py`

Optional helpers:

* `torchregress/utils/covariance.py`

Public exports:

* `TaylorInducedCovarianceHead`
* `TICTACRegressor` or `TICHead`
* `TaskAgnosticCorrelations`
* `task_agnostic_correlations`

## Proposed API

```python
head = TaylorInducedCovarianceHead(base_model=backbone, target_dim=d)
pred = head(x)
```

and

```python
metric = TaskAgnosticCorrelations()
score = metric(pred_covariance, target_covariance)
```

## Implementation scope

### v1

* core TIC covariance construction
* metric implementation for evaluation
* example integration with an existing mean network

### v2

* more faithful reproduction of the paper’s setup
* optional application-specific variants

## Tests

* covariance outputs are SPD or PSD + stabilized
* TIC output shapes match target dimension
* TAC metric behaves sensibly on known covariance matrices
* covariance scales correctly under simple affine transformations

## Benchmarks

Compare against:

* Gaussian NLL
* β-NLL
* 2-Wasserstein-bound supervision
* diagonal Gaussian baselines

Metrics:

* covariance error
* TAC
* NLL
* memory
* runtime

## Merge target

This should begin as **experimental**. It is important, but heavier and more specialized than β-NLL or the Wasserstein-bound loss.

---

# Workstream 4 — Effective Bayesian Heteroscedastic Regression with Natural Heads and Laplace

## Why this method matters

The NeurIPS 2023 repository explicitly describes a Bayesian heteroscedastic regression method built around natural parameterization and Laplace approximation, extending Laplace-style machinery to heteroscedastic Gaussian likelihoods . This is the strongest missing Bayesian heteroscedastic path.

## Goal

Add a principled Bayesian heteroscedastic option that is stronger than generic BNN wrappers and more directly suited to regression with sample-dependent variance.

## Recommended placement

* `torchregress/algorithms/heteroscedastic_laplace.py`

Optional public exports:

* `NaturalHeteroscedasticHead`
* `NaturalReparamHead`
* `HeteroscedasticLaplaceRegressor`

This could also later justify a small `bayes` submodule, but not initially.

## Proposed API

```python
head = NaturalHeteroscedasticHead(in_features=h, out_features=d)
model = HeteroscedasticLaplaceRegressor(base_model=net, head=head)
model.fit(train_loader)
pred = model.predict_distribution(x_test)
```

## Implementation scope

### v1

* natural-parameter head
* reparameterization wrapper from mean/variance outputs
* predictive distribution helper
* no full Laplace machinery if that is too heavy initially; head support alone is already useful

### v2

* actual Laplace posterior approximation
* posterior predictive sampling
* uncertainty decomposition helpers

## Tests

* natural head outputs valid parameterization
* mean/variance to natural reparameterization is numerically consistent
* predictive variance stays positive
* posterior approximation wrappers preserve shapes and output semantics

## Benchmarks

Compare against:

* Gaussian NLL
* β-NLL
* SWAG
* current BNN path
* deep ensembles
* 2-W pretraining + NLL fine-tuning if relevant

Metrics:

* NLL
* RMSE
* epistemic calibration
* OOD / shift behavior where possible

## Merge target

This should start **experimental**. It is important, but the engineering footprint is larger than a new loss.

---

# Workstream 5 — Faithful Heteroscedastic Regression

## Why this method matters

The 2025 covariance-supervision paper repeatedly uses Faithful heteroscedastic regression as a major comparison point and discusses its tradeoffs relative to newer covariance-learning methods . Even if it is not the most glamorous addition, it is an important baseline that users now expect to see.

## Goal

Add a modern lightweight heteroscedastic training alternative that decouples mean fitting and variance fitting more cleanly than vanilla NLL.

## Recommended placement

* `torchregress/losses/faithful.py`

Public exports:

* `FaithfulHeteroscedasticLoss`
* optionally `faithful_heteroscedastic_loss`

## Proposed API

```python
loss_fn = FaithfulHeteroscedasticLoss(reduction="mean")
loss = loss_fn(pred_mean, pred_variance, target)
```

If the method requires a two-term objective or specific stop-gradient pattern, keep that internal to the loss.

## Implementation scope

### v1

* basic faithful objective
* diagonal variance first
* clear docs on when it helps and when it does not

### v2

* structured covariance variant only if justified

## Tests

* finite gradients
* sensible behavior under residual changes
* stable variance predictions on toy problems
* comparison against Gaussian NLL on a simple pathology example

## Benchmarks

Compare against:

* Gaussian NLL
* β-NLL
* 2-Wasserstein bound
* TIC-TAC

Metrics:

* convergence stability
* MSE
* NLL
* covariance quality where possible

## Merge target

This can be **stable or semi-stable** relatively early because it is small and important as a benchmark baseline.

---

# Shared benchmark suite

## Objective

Build one modern heteroscedastic regression benchmark harness so all five methods can be compared in a consistent way.

## Suggested benchmark tiers

### Tier A — toy pathologies

* classic heteroscedastic instability examples
* synthetic bivariate Gaussian mismatch problems
* varying-amplitude sinusoidal regression

### Tier B — multivariate synthetic regression

* known input-dependent covariance
* diagonal and full-covariance scenarios
* increasing target dimensionality

### Tier C — tabular real datasets

* UCI-style regression
* compute-time and memory tracking
* calibration metrics

### Tier D — application-specific examples

Only after the core methods work:

* pose/keypoint tasks
* image-conditioned structured covariance
* astronomy-specific multivariate regression if relevant

## Common metrics

Use:

* RMSE / MAE
* Gaussian NLL
* CRPS when applicable
* PIT / calibration diagnostics
* interval coverage after conformalization if added downstream
* covariance Frobenius error
* correlation recovery metrics
* runtime
* memory

---

# Shared documentation plan

Add a new guide page such as:

* `docs/guide/modern_heteroscedastic_regression.md` after the guide is promoted from research notes

It should explain:

* why vanilla Gaussian NLL can be problematic
* when to use β-NLL
* when to use Wasserstein-bound supervision
* when TIC-TAC is worth the compute
* when Faithful is a good lightweight baseline
* when the Bayesian heteroscedastic path is justified

Each method should also have:

* one dedicated docs page
* one minimal example
* one benchmark snippet

---

# Recommended implementation order

## Milestone 1

* `BetaNLLLoss`
* tests
* docs
* minimal examples

## Milestone 2

* `GaussianWassersteinBoundLoss`
* tests
* synthetic covariance-supervision benchmark
* docs

## Milestone 3

* `FaithfulHeteroscedasticLoss`
* shared benchmark harness updated
* docs

## Milestone 4

* experimental `TICHead` / `TaskAgnosticCorrelations`
* structured covariance benchmarks
* docs

## Milestone 5

* experimental natural heteroscedastic head + Laplace path
* Bayesian uncertainty benchmarks
* docs

---

# Final recommendation

If the goal is to make `torchregress` materially stronger over the next development cycle, the best path is:

1. **β-NLL first**
2. **2-Wasserstein bound loss second**
3. **Faithful third**
4. **TIC-TAC fourth**
5. **Natural heteroscedastic Laplace fifth**

That ordering balances impact, fit, and engineering cost.

The first three are immediate wins.
The fourth adds serious structured covariance capability.
The fifth gives the package a stronger modern Bayesian heteroscedastic story.
