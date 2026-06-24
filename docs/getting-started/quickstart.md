# Quick Start

This page assembles the canonical workflows for the four most common
regression regimes in torchregress. Each workflow is **self-contained**:
copy a block, run it, and validate against the listed evaluation metrics
before committing to a method.

!!! info "Design intent"
    Every loss in this page is interchangeable in a standard PyTorch
    training loop. The differences are *semantic* (what the loss
    optimises) and *contractual* (what the model must output), not
    procedural. Replace `loss_fn` and the head's final layer; the rest
    of the loop is unchanged.

---

## 1. Point regression with mask and sample-weight support

When you need a point prediction and your data has **missing targets** or
**per-sample weights**, use the weighted wrappers instead of `torch.nn.MSELoss`.
The wrappers also support `reduction="none"` for per-sample loss returns.

```python
import torch
import torch.nn as nn
from torchregress.losses import WeightedMSELoss, WeightedHuberLoss
from torchregress.metrics import mae, r2_score, rmse

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))

# Huber is a robust default for mild outliers; delta=1.0 transitions
# between quadratic and linear at |r| = 1.
loss_fn = WeightedHuberLoss(delta=1.0)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for x, y, mask, w in dataloader:
    pred = model(x)
    loss = loss_fn(pred, y, mask=mask, weights=w)
    loss.backward(); optimizer.step(); optimizer.zero_grad()

# Evaluation: report RMSE + MAE + R² as secondary summaries
with torch.no_grad():
    y_pred = model(x_test)
    print(f"RMSE: {rmse(y_pred, y_test):.4f}  "
          f"MAE: {mae(y_pred, y_test):.4f}  "
          f"R²: {r2_score(y_pred, y_test):.4f}")
```

**When not to use this:** if your data is heteroscedastic (noise varies
with $x$) or you need uncertainty intervals, use §2 instead.

→ API: [`WeightedHuberLoss`](../api/losses.md#weightedhuberloss), [`WeightedMSELoss`](../api/losses.md#weightedmseloss); metrics [`rmse`](../api/metrics.md#rmse), [`mae`](../api/metrics.md#mae), [`r2_score`](../api/metrics.md#r2_score).

---

## 2. Heteroscedastic Gaussian regression

The model outputs a **mean** and a **log-variance**; the loss is the
Gaussian negative log-likelihood, a *proper scoring rule* that admits
gradient-based optimisation. This is the canonical
[aleatoric uncertainty](concepts.md)
head.

```python
from torchregress.losses import GaussianNLLLoss
from torchregress.metrics import crps_gaussian, gaussian_nll

# Head: [mean, log_var] per example
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))

loss_fn = GaussianNLLLoss()

for x, y in dataloader:
    out = model(x)                      # [B, 2]
    loss = loss_fn(out, y)              # -log p(y | μ(x), σ²(x))
    loss.backward(); optimizer.step(); optimizer.zero_grad()

# Inference + density-aware evaluation
with torch.no_grad():
    out = model(x_test)
    mu, logvar = out[:, 0], out[:, 1]
    std = torch.exp(0.5 * logvar)
    lower, upper = mu - 1.96 * std, mu + 1.96 * std
    crps = crps_gaussian(mu, std, y_test)        # proper scoring rule
    gnll = gaussian_nll(mu, logvar, y_test)     # NLL on test set
```

**Caveat:** Gaussian NLL can suffer from variance-collapse early in
training (the model predicts $\sigma \to 0$ and exploits the NLL's density
peak). The [`BetaNLLLoss`](../api/losses.md#betanllloss) variant detaches the
variance term to mitigate this. Always inspect a
[reliability diagram](../methods/calibration.md) before trusting the
intervals.

**For full epistemic decomposition**, wrap this head in an ensemble or
add a posterior approximation (see §5).

→ API: [`GaussianNLLLoss`](../api/losses.md#gaussiannllloss); evaluation: [`crps_gaussian`](../api/metrics.md#crps_gaussian), [`gaussian_nll`](../api/metrics.md#gaussian_nll).

---

## 3. Robust regression under contamination

When the data contains outliers or heavy-tailed noise, the $L_2$ penalty
in MSE / Gaussian NLL gives unbounded influence to a single bad
example. Use an M-estimator with a bounded influence function.

```python
from torchregress.losses import (
    WeightedHuberLoss, CauchyLoss, TukeyBiweightLoss, AdaptiveRobustLoss
)

# Mild outliers (5–10%): Huber is a safe default
loss_fn = WeightedHuberLoss(delta=1.0)

# Moderate outliers (10–25%): Cauchy is more aggressive
loss_fn = CauchyLoss(c=1.0)

# Severe contamination (>25%): Tukey rejects outliers entirely
loss_fn = TukeyBiweightLoss(c=4.685)

# Unknown noise regime: learn the shape parameter
loss_fn = AdaptiveRobustLoss()

# IMPORTANT: when the loss has learnable parameters (AdaptiveRobust,
# Barron), pass them to the optimizer:
optimizer = torch.optim.Adam([
    {"params": model.parameters(), "lr": 1e-3},
    {"params": loss_fn.parameters(), "lr": 1e-2},
])
```

The choice of $\rho(r)$ is not a hyperparameter to tune blindly —
inspect the [empirical influence function](../methods/visualization.md) and
the [uncertainty vs. error diagnostic](../methods/visualization.md) to
validate that outliers are actually being down-weighted.

---

## 4. Conformal prediction for guaranteed coverage

Apply [conformal prediction](../methods/conformal/index.md) on top of any
pre-trained model to obtain prediction intervals with **distribution-free
coverage guarantees** under exchangeability.

```python
from torchregress.losses import SplitConformal, CQR

# Split conformal: residuals |y - ŷ| on a held-out calibration set
cp = SplitConformal(alpha=0.1)                       # target 90% coverage
cp.calibrate(y_pred_cal, y_cal)
lower, upper = cp.predict_interval(y_pred_test)

# Conformalized Quantile Regression (CQR): adapt to heteroscedastic noise
cqr = CQR(alpha=0.1)
cqr.calibrate(quantile_preds_cal, y_cal)             # quantile predictions
lower, upper = cqr.predict_interval(quantile_preds_test)
```

**Important:** conformal prediction provides *coverage*, not density
estimation. The intervals can be wide if the underlying model is poor
or the data has shifted; this is a feature, not a bug. If you also
need aleatoric/epistemic decomposition, use a probabilistic model
(§2) **and** conformal calibration.

---

## 5. Epistemic uncertainty via ensembles

For decomposable uncertainty, train an **ensemble of likelihood heads**
and decompose the predictive variance via the Law of Total Variance:

$$
\underbrace{\operatorname{Var}_{\text{total}}[y \mid x]}_{\text{predictive spread}}
= \underbrace{\frac{1}{M}\sum_{m=1}^M \sigma_m^2(x)}_{\text{aleatoric}}
+ \underbrace{\frac{1}{M}\sum_{m=1}^M (\mu_m(x) - \bar\mu(x))^2}_{\text{epistemic}}
$$

```python
from torchregress.ensemble import DeepEnsemble
from torchregress.metrics import uncertainty_decomposition

# A heteroscedastic head that outputs [mean, log_var]
def heteroscedastic_head():
    return nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))

ens = DeepEnsemble(model_factory=heteroscedastic_head,
                   n_members=5, seed=0)
ens.fit(train_loader, optimizer_factory=torch.optim.Adam, lr=1e-3)

# Per-member predictions
mu_per_member = torch.stack([m(x_test) for m in ens.models], dim=0)  # [M, B]
var_per_member = ...  # from each member's (mean, log_var) head

# Decomposition
decomp = uncertainty_decomposition(mu_per_member, var_per_member)
# decomp["epistemic_variance"], decomp["aleatoric_variance"], decomp["total_variance"]
```

**Tradeoff:** training and inference costs scale linearly with ensemble
size. For a single-pass alternative, see
[`EvidentialRegressionLoss`](../losses/advanced.md); for a cheaper posterior
approximation, see [`SWAG`](../methods/ensemble/index.md) or
[`IVON`](../methods/algorithms/ivon.md).

---

## 6. Distribution shift at test time

When the target distribution at deployment differs from training
(population shift, instrument drift, non-stationary process), use
**test-time adaptation** primitives.

```python
from torchregress.test_time import (
    BayesianLinearHead, ScoreCDFReweighter, WeightedSplitConformalAdapter
)

# 1. Closed-form Bayesian last layer — adapts to a few labeled or
#    unlabeled examples without backprop
head = BayesianLinearHead(in_features=64, out_features=1, noise_variance=0.1)
head.fit(features_train, y_train)              # one-shot
head.partial_fit(features_stream, y_stream)   # streaming with forgetting

# 2. OT-style conformal reweighting under non-exchangeable shift
rw = ScoreCDFReweighter().fit(cal_scores, target_scores)
adapter = WeightedSplitConformalAdapter(alpha=0.1).calibrate(cal_scores, rw.weights_)
mask = adapter.predict_from_test_scores(test_scores)

# 3. Full pipeline: prior transport + feature alignment + conformal
from torchregress.test_time import (
    ShiftFactoredPredictiveTransport, ShiftFactoredTransportConfig
)
transport = ShiftFactoredPredictiveTransport(ShiftFactoredTransportConfig())
transport.fit_source(source_pb, source_targets=src_y, source_inputs=src_x)
adapted_pb = transport.adapt_unlabeled_target(target_pb, target_inputs=tgt_x)
```

---

## 7. Causal inference (ATE / CATE)

For treatment-effect estimation under confounding, use **doubly-robust
estimators** with cross-fitting and overlap diagnostics.

```python
from torchregress.causal import dr_ate, causal_overlap_report

ate = dr_ate(x, t, y,
             outcome_model=lambda: GradientBoostingRegressor(n_estimators=50),
             propensity_model=lambda: GradientBoostingClassifier(n_estimators=50),
             folds=2, alpha=0.05, seed=42)
print(f"ATE = {ate['estimate']:.3f}  CI = [{ate['ci_lower']:.3f}, {ate['ci_upper']:.3f}]")

# Always inspect overlap diagnostics — a low ESS in either arm means
# the estimate is dominated by a few influential points
print(f"min ESS: {ate['diagnostics']['min_group_ess']:.1f}  "
      f"overlap rate: {ate['diagnostics']['overlap_rate']:.3f}")
```

---

## Decision aid: which workflow is right for you?

| Your situation | Start at | Why |
|:---------------|:---------|:----|
| Plain regression, complete data, no uncertainty needed | §1 | Smallest model, fastest to train |
| Noise varies with $x$ or you need predictive intervals | §2 | Heteroscedastic head + proper scoring rule |
| Data has outliers or heavy-tailed noise | §3 | Bounded influence function |
| You need coverage guarantees on intervals | §4 | Distribution-free coverage under exchangeability |
| You need to know *what the model doesn't know* | §5 | Decomposition into aleatoric / epistemic |
| Test data is shifted from training | §6 | Test-time adaptation primitives |
| You need treatment-effect estimates under confounding | §7 | Doubly-robust estimation with diagnostics |

For the full task matrix, see
[Method Selection](../guide/method-selection.md). For derivations of the
scoring rules and decomposition formulas, see
[Mathematical Foundations](../guide/math/index.md).

---

## Next steps

<div class="grid cards" markdown>

-   :material-function: __Mathematical Foundations__
    -   Derivations of NLL, CRPS, interval score, and decomposition formulas
    -   [Mathematical Foundations](../guide/math/index.md)

-   :material-table: __Method Selection__
    -   Task-first guidance with capability matrices
    -   [Method Selection Matrix](../guide/method-selection.md)

-   :material-chart-line: __Diagnostics & Visualization__
    -   Reliability diagrams, PIT histograms, residual plots
    -   [Visualization Methods](../methods/visualization.md)

</div>
