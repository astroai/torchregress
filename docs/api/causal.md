# Causal API

Complete reference for `torchregress.causal`. This package provides
**doubly-robust (DR)** ATE / CATE / policy-value estimators with **cross-fitting**
and overlap diagnostics. Inputs are PyTorch tensors; outcome and propensity
models can be any scikit-learn-style `fit` / `predict` object.

For background, see [Causal inference methods](../methods/causal.md).

---

## Estimators

| Symbol | Description |
|:-------|:------------|
| `dr_ate(x, t, y, *, outcome_model, propensity_model, folds=2, alpha=0.05, seed=42, trim_threshold=0.05, eps=1e-4)` | Cross-fitted doubly-robust ATE with normal-approximation CI and overlap diagnostics. Returns `{"estimate", "se", "ci_lower", "ci_upper", "alpha", "n_samples", "dr_scores", "propensity", "mu1_hat", "mu0_hat", "diagnostics"}`. |
| `dr_cate(x, t, y, *, cate_model, outcome_model, propensity_model, folds=2, alpha=0.05, seed=42, trim_threshold=0.05, eps=1e-4)` | Cross-fitted DR CATE via pseudo-outcome regression. Returns `{"ate_estimate", "ate_se", "ate_ci_lower", "ate_ci_upper", "ate_ci_low", "ate_ci_high", "alpha", "cate_hat", "pseudo_outcome", "propensity", "mu1_hat", "mu0_hat", "diagnostics"}`. |
| `dr_policy_value(x, t, y, *, policy, outcome_model, propensity_model, folds=2, seed=42, eps=1e-4)` | AIPW value estimate for a binary treatment policy. Returns `{"estimate", "se", "n_samples"}`. |

**References:** Robins, Rotnitzky, Zhao (JASA 1994); Chernozhukov et al.,
"Double/debiased machine learning for treatment and structural parameters"
(Econometrics Journal 2018).

```python
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from torchregress.causal import dr_ate, dr_cate, dr_policy_value

x = torch.randn(2000, 5)
t = (torch.rand(2000) > 0.5).float()
y = x.sum(1) + t + 0.1 * torch.randn(2000)

# ATE
ate = dr_ate(x, t, y,
             outcome_model=LinearRegression,
             propensity_model=LogisticRegression,
             folds=2, alpha=0.05, seed=42)
# ate["estimate"], ate["ci_lower"], ate["ci_upper"], ate["diagnostics"]

# CATE
cate = dr_cate(x, t, y,
               cate_model=LinearRegression,
               outcome_model=LinearRegression,
               propensity_model=LogisticRegression,
               folds=2)
# cate["cate_hat"], cate["ate_estimate"], ...

# Policy value
policy = (dr_cate(... if computed, else torch.zeros(2000)) > 0).float()
val = dr_policy_value(x, t, y, policy=policy,
                      outcome_model=LinearRegression,
                      propensity_model=LogisticRegression)
```

---

## Diagnostics

| Symbol | Description |
|:-------|:------------|
| `causal_overlap_report(propensity, treatment, *, trim_threshold=0.05, eps=1e-6)` | Returns `{"n_samples", "n_treated", "n_control", "propensity_min", "propensity_max", "propensity_mean", "overlap_rate", "trim_threshold", "n_trimmed", "treated_ess", "control_ess", "min_group_ess"}`. Inspect before trusting ATE estimates. |

The diagnostics include **effective sample size (ESS)** for each treatment arm
based on IPW weights; very low `treated_ess` or `control_ess` is a strong
signal of overlap failure.

---

## Quick example

```python
import torch
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from torchregress.causal import dr_ate, causal_overlap_report

x = torch.randn(2000, 5)
t = (torch.rand(2000) > 0.5).float()
y = x[:, 0] + t * x[:, 1] + 0.1 * torch.randn(2000)

ate = dr_ate(x, t, y,
             outcome_model=lambda: GradientBoostingRegressor(n_estimators=50, max_depth=3),
             propensity_model=lambda: GradientBoostingClassifier(n_estimators=50, max_depth=3),
             folds=2, alpha=0.05, seed=42)
print(f"ATE = {ate['estimate']:.3f}  CI = [{ate['ci_lower']:.3f}, {ate['ci_upper']:.3f}]")
print(f"Overlap rate: {ate['diagnostics']['overlap_rate']:.3f}, "
      f"min ESS: {ate['diagnostics']['min_group_ess']:.1f}")
```

---

## Next steps

- [Causal inference methods](../methods/causal.md) — background, decision guidance, references
- [Diagnostics visualizations](../api/viz.md)
- [DR diagnostics](../api/viz.md) — reliability plots for residual checks
