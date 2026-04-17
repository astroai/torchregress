# torchregress: research note on the 5 most impactful regression methods to implement since 2022

## 1. Scope and objective

This note identifies the five post-2022 regression method families that are most worth implementing in **torchregress**.

The goal is **not** to list the five most cited papers. The goal is to identify methods that are:

* scientifically credible,
* useful across many regression tasks,
* mature enough to justify a library abstraction,
* reasonably well adopted in code or downstream work,
* and sufficiently distinct from one another to justify separate support.

The intended outcome is a concrete implementation roadmap for a PyTorch regression library.

---

## 2. Selection criteria

A method was prioritised if it scores well on most of the following axes:

### 2.1 Research legitimacy

* publication in a strong venue,
* clear methodological contribution,
* evidence that the method is being used or discussed beyond the original paper.

### 2.2 Practicality for a reusable library

* can be exposed as a clean loss, wrapper, head, or trainer abstraction,
* does not require a highly specialised architecture or domain-specific pipeline,
* integrates naturally with standard PyTorch training loops.

### 2.3 Distinct user value

* improves one of the key regression pain points:

  * predictive accuracy,
  * heteroscedastic uncertainty,
  * epistemic uncertainty,
  * target imbalance,
  * calibrated intervals with guarantees.

### 2.4 Ecosystem signal

* active public implementations,
* GitHub usage or stars,
* adoption in benchmark papers, uncertainty libraries, or downstream packages.

---

## 3. Executive recommendation

The five method families that torchregress should implement first are:

1. **Conformal regression wrappers**, especially modern variants such as **CQR** and **UACQR**
2. **β-NLL** for heteroscedastic Gaussian regression
3. **Balanced MSE / BMC** for imbalanced regression
4. **Faithful heteroscedastic regression**
5. **Packed ensembles** for efficient epistemic uncertainty

These five together cover the most important modern needs of regression users:

* accurate point prediction,
* predictive intervals with finite-sample coverage guarantees,
* calibrated aleatoric uncertainty,
* improved behaviour under skewed target distributions,
* and computationally efficient epistemic uncertainty.

---

## 4. Recommended methods

## 4.1 Conformal regression wrappers

### Why it matters

Conformal prediction is the strongest addition from a reliability point of view. It does not depend on a specific neural architecture. Instead, it wraps a trained regressor and returns prediction intervals with finite-sample coverage guarantees under exchangeability assumptions.

This makes it ideal for a library such as torchregress because it is:

* broadly useful,
* model-agnostic,
* easy to compose with many base regressors,
* and directly aligned with honest uncertainty quantification.

### What to implement

Implement a common wrapper:

* `SplitConformalRegressor`
* `CQRRegressor` for conformalised quantile regression
* `UACQRRegressor` for uncertainty-aware adaptive conformal quantile regression

### Core abstractions

#### Base requirements

A conformal wrapper should accept one of:

* a point regressor,
* a quantile regressor,
* a mean-variance regressor.

#### Core API sketch

```python
model = MLPRegressor(...)
conf = SplitConformalRegressor(model, score="abs")
conf.fit(train_loader, calib_loader)
pred = conf.predict(x)
lo, hi = conf.predict_interval(x, alpha=0.1)
```

For quantile-based conformalisation:

```python
model = QuantileRegressor(..., quantiles=[0.05, 0.95])
conf = CQRRegressor(model)
conf.fit(train_loader, calib_loader)
lo, hi = conf.predict_interval(x, alpha=0.1)
```

### Why it belongs in torchregress

This is not just a paper implementation. It is a library-level capability.

A good regression package in 2026 should offer:

* point predictions,
* uncertainty estimates,
* and valid predictive intervals.

Conformal support is the cleanest route to the last item.

### Design notes

* keep the conformity score modular,
* separate training and calibration phases,
* support multi-output regression if feasible,
* expose conditional coverage diagnostics even though only marginal coverage is guaranteed.

---

## 4.2 β-NLL for heteroscedastic Gaussian regression

### Why it matters

Many users train a network to predict both a mean and a variance and optimise Gaussian negative log-likelihood. In practice, standard NLL can produce poor optimisation behaviour, especially when the variance head starts to absorb errors in undesirable ways.

β-NLL is an elegant fix. It modifies the regression objective by reweighting the contribution of each sample using the predicted variance. This often yields better optimisation and better uncertainty behaviour with almost no additional implementation complexity.

### What to implement

A simple loss module:

* `BetaGaussianNLLLoss(beta=0.5)`

Possibly also:

* `BetaStudentTLoss(...)` later, if the library expands beyond Gaussian heads.

### Core API sketch

```python
head = GaussianHead(in_dim, out_dim)
loss_fn = BetaGaussianNLLLoss(beta=0.5)
mean, logvar = model(x)
loss = loss_fn(mean, logvar, y)
```

### Why it belongs in torchregress

This is one of the highest value-per-line additions:

* small implementation,
* strong modern research motivation,
* broadly applicable,
* immediate benefit for users already doing mean-variance regression.

### Design notes

* expose both `var` and `logvar` interfaces,
* support reduction modes,
* allow optional floor or clipping on variance,
* ensure numerical stability for mixed precision.

---

## 4.3 Balanced MSE / BMC for imbalanced regression

### Why it matters

Regression target distributions are often highly skewed. Standard MSE overemphasises dense regions of the target space and neglects tails. This is a major issue in scientific regression, long-tail estimation, and many vision tasks.

Balanced MSE and related balanced objective formulations address this directly. They are among the strongest post-2022 additions for users whose regression targets are imbalanced or heavy-tailed in frequency.

### What to implement

At minimum:

* `BalancedMSELoss`
* `BMCLoss`

Potential extensions:

* multidimensional balanced losses,
* adaptive binning strategies,
* class-frequency estimation utilities for continuous labels.

### Core API sketch

```python
loss_fn = BalancedMSELoss(bin_edges=edges)
loss = loss_fn(pred, target)
```

or

```python
loss_fn = BMCLoss(noise_sigma=init_sigma)
loss = loss_fn(pred, target)
```

### Why it belongs in torchregress

This fills a gap that many generic regression libraries ignore.

Users usually know about:

* MSE,
* MAE,
* Huber.

But they often do not have good default tools for imbalanced continuous targets. Balanced losses are therefore a strong differentiator for torchregress.

### Design notes

* keep the implementation general, not vision-specific,
* provide utilities to inspect target imbalance,
* support scalar and vector targets,
* benchmark explicitly on long-tail regression datasets.

---

## 4.4 Faithful heteroscedastic regression

### Why it matters

A persistent problem in heteroscedastic regression is that the variance head can influence the mean estimate in ways that degrade point prediction quality. Faithful heteroscedastic regression proposes a training strategy that decouples these roles more carefully so that uncertainty estimation does not corrupt the mean.

This is especially relevant in scientific settings where:

* the mean estimate matters a lot,
* uncertainty is required,
* and a naïve heteroscedastic objective can create misleading confidence structure.

### What to implement

This should probably not be a new model class. It is better expressed as a trainer recipe or objective option:

* `FaithfulGaussianTrainer`
* or `FaithfulHeteroscedasticLoss`

### Core API sketch

```python
trainer = FaithfulGaussianTrainer(model, optimizer, ...)
trainer.fit(train_loader)
```

or

```python
loss_fn = FaithfulGaussianLoss(...)
loss = loss_fn(mean, logvar, target)
```

### Why it belongs in torchregress

It is more specialised than β-NLL, but it addresses a real failure mode of uncertainty-aware regression. For a library that aims to be serious about uncertainty, this deserves first-class support.

### Design notes

* document clearly when to prefer this over ordinary Gaussian NLL,
* compare directly against NLL and β-NLL,
* include calibration metrics, not just RMSE.

---

## 4.5 Packed ensembles for epistemic uncertainty

### Why it matters

Deep ensembles remain one of the strongest baselines for epistemic uncertainty, but they are expensive. Packed ensembles provide a more efficient approximation by sharing computation while preserving much of the diversity benefit.

For regression users, this matters whenever:

* train/test shift is present,
* OOD detection matters,
* or interval width should reflect model uncertainty rather than just observation noise.

### What to implement

A wrapper such as:

* `PackedEnsembleRegressor`

This should work over arbitrary base models that emit:

* point predictions,
* quantiles,
* or mean-variance outputs.

### Core API sketch

```python
base = MLPRegressor(...)
model = PackedEnsembleRegressor(base, ensemble_size=4, alpha=2)
out = model(x)
mean = out.mean
std_epistemic = out.std_epistemic
```

### Why it belongs in torchregress

This gives torchregress a credible epistemic-uncertainty story without forcing users to train fully independent ensembles every time.

### Design notes

* design the output object carefully,
* separate aleatoric and epistemic aggregation when the base model predicts variance,
* expose member-wise predictions for diagnostics,
* benchmark cost versus true deep ensembles.

---

## 5. Why these five, and not others

Several important methods were considered but are lower priority for initial inclusion.

## 5.1 Deep Evidential Regression

This remains influential, but it is not post-2022, and its uncertainty semantics are debated. It is still worth supporting later under an experimental namespace.

## 5.2 Generic transformer regressors

Architecture matters, but torchregress should first focus on reusable regression-specific abstractions rather than committing to one backbone family.

## 5.3 Diffusion or flow-based full predictive distributions

These are interesting, but they are currently too heavy and too architecture-dependent for an initial general-purpose regression library.

## 5.4 Distributionally robust optimisation in general

Very useful, but often not regression-specific enough to justify prioritisation over the five methods above.

## 5.5 Large foundation-model-style regressors

These may become important, especially for multimodal or tabular foundation settings, but they are not yet the cleanest first addition to a focused PyTorch regression package.

---

## 6. Proposed package structure

A sensible structure would be:

```text
torchregress/
  losses/
    mse.py
    huber.py
    gaussian_nll.py
    beta_nll.py
    balanced_mse.py
    bmc.py
    faithful.py
  heads/
    deterministic.py
    gaussian.py
    quantile.py
  conformal/
    split.py
    cqr.py
    uacqr.py
    scores.py
    diagnostics.py
  ensembles/
    packed.py
    utils.py
  metrics/
    regression.py
    calibration.py
    interval.py
  models/
    mlp.py
    resnet.py
    ft_transformer.py
  trainers/
    standard.py
    heteroscedastic.py
    faithful.py
  benchmarks/
    tabular/
    vision/
    synthetic/
```

---

## 7. Minimal core abstractions

## 7.1 Prediction heads

Support three universal output head types:

### Deterministic head

Outputs:

* `y_hat`

### Gaussian head

Outputs:

* `mean`
* `logvar`

### Quantile head

Outputs:

* `q_low`, `q_mid`, `q_high`, or arbitrary quantile set

These three cover nearly all of the recommended methods.

---

## 7.2 Output containers

Use typed output containers so downstream wrappers remain clean:

```python
RegressionOutput(pred)
GaussianRegressionOutput(mean, logvar)
QuantileRegressionOutput(quantiles, values)
EnsembleRegressionOutput(member_preds, aggregated)
```

This will make conformal wrappers and metrics much easier to maintain.

---

## 7.3 Metrics to ship with the library

A strong regression library should not stop at RMSE.

Recommended built-in metrics:

### Point prediction

* MSE
* RMSE
* MAE
* median absolute error
* R²

### Distributional / uncertainty

* Gaussian NLL
* CRPS if predictive distributions are supported later
* calibration error for variance models
* interval coverage
* average interval length
* conditional coverage diagnostics by bins of target or covariates

### Imbalanced regression

* tail RMSE
* per-bin RMSE
* balanced MAE / balanced RMSE summaries

---

## 8. Implementation priority

## Phase 1: immediate, high-value

### 1. β-NLL

Reason:

* extremely easy to add,
* immediately useful,
* minimal dependency on other design choices.

### 2. Gaussian and quantile heads

Reason:

* needed by several later methods.

### 3. Split conformal and CQR

Reason:

* large practical impact,
* strong user-facing value,
* clean abstraction.

## Phase 2: strong differentiators

### 4. Balanced MSE / BMC

Reason:

* unusual and valuable for many scientific and long-tail regression tasks.

### 5. Faithful heteroscedastic training

Reason:

* completes the uncertainty-aware regression story.

## Phase 3: more advanced uncertainty tooling

### 6. Packed ensembles

Reason:

* more engineering work,
* but high value for epistemic uncertainty.

### 7. UACQR

Reason:

* excellent modern conformal extension,
* best added once the basic conformal API is stable.

---

## 9. Recommended benchmark suite

The benchmark suite should reflect the reasons these methods exist.

## 9.1 Standard tabular regression

Examples:

* UCI-style datasets
* OpenML tabular regression tasks

Purpose:

* measure basic predictive quality,
* test whether advanced methods degrade or improve ordinary regression.

## 9.2 Heteroscedastic synthetic benchmarks

Construct datasets with input-dependent noise.

Purpose:

* compare Gaussian NLL, β-NLL, and faithful heteroscedastic training,
* evaluate variance calibration and interval quality.

## 9.3 Long-tail target benchmarks

Use or construct datasets where the target distribution is strongly imbalanced.

Purpose:

* measure benefit of Balanced MSE / BMC,
* inspect tail-region accuracy.

## 9.4 Shifted or OOD benchmarks

Create train/test covariate shift.

Purpose:

* assess epistemic uncertainty quality for packed ensembles,
* check whether interval width reflects uncertainty under distribution shift.

## 9.5 Interval benchmark protocol

For conformal methods, always report:

* empirical coverage,
* average interval length,
* stratified coverage,
* failure modes under distribution shift.

---

## 10. Concrete recommendation for torchregress v0.1

If the first public release must stay focused, the best v0.1 content is:

### Must-have

* deterministic regression heads
* Gaussian regression head
* quantile regression head
* MSE / MAE / Huber / Gaussian NLL
* `BetaGaussianNLLLoss`
* `SplitConformalRegressor`
* `CQRRegressor`
* interval and calibration metrics

### Strongly recommended

* `BalancedMSELoss`
* `BMCLoss`

### v0.2 candidates

* `FaithfulGaussianTrainer`
* `PackedEnsembleRegressor`
* `UACQRRegressor`

This gives users a coherent story from day one:

* ordinary regression,
* probabilistic regression,
* quantile regression,
* conformal intervals,
* and improved robustness to heteroscedasticity and target imbalance.

---

## 11. Final ranking

### Rank 1 — Conformal regression wrappers

Best overall addition for reliability, generality, and user-facing value.

### Rank 2 — β-NLL

Best small-footprint upgrade to modern heteroscedastic regression.

### Rank 3 — Balanced MSE / BMC

Best addition for long-tail and imbalanced regression.

### Rank 4 — Faithful heteroscedastic regression

Best principled refinement for mean-variance modelling.

### Rank 5 — Packed ensembles

Best efficient epistemic uncertainty addition.

---

## 12. Bottom line

If torchregress wants to be a serious modern regression library rather than just a collection of standard losses, it should prioritise:

* **valid predictive intervals** via conformal wrappers,
* **better heteroscedastic training** via β-NLL and faithful objectives,
* **tail-aware learning** via balanced losses,
* **efficient epistemic uncertainty** via packed ensembles.

That set is modern, defensible, useful, and substantially more distinctive than shipping only MSE, Huber, and a few model backbones.
