# Causal Inference

Causal inference is the study of how one variable (the treatment $T$) affects another (the outcome $Y$). Unlike standard regression which measures **correlation**, causal inference seeks to estimate the **Average Treatment Effect (ATE)** and **Conditional Average Treatment Effect (CATE)** after adjusting for confounding variables $X$.

---

## The Fundamental Problem

In observational data, we only ever see one outcome for each individual: either the treated outcome $Y(1)$ or the control outcome $Y(0)$. We never observe the **counterfactual** $Y(1-T)$.

$$Y_i = T_i Y_i(1) + (1-T_i) Y_i(0)$$

Causal inference methods in **torchregress** use **doubly-robust** estimation [1, 4] to overcome this missing data problem.

---

## Why Use Causal Inference?

| Benefit | Description |
|:--------|:------------|
| **Policy Evaluation** | Estimate the impact of a new treatment or policy before implementation. |
| **Bias Correction** | Adjust for confounders that influence both treatment assignment and outcome. |
| **Heterogeneity** | Identify which subgroups benefit most from a treatment (CATE). |
| **Double Robustness** | Consistent estimates if *either* the outcome model or propensity model is correct. |

---

## Core Methods

### 1. Doubly Robust ATE ([`dr_ate`](../api/causal.md#torchregress.causal.dr_ate))

The Doubly Robust (DR) estimator [1] combines an **outcome model** $\hat{\mu}(x, t)$ and a **propensity model** $\hat{e}(x) = P(T=1 \mid x)$ to estimate the average treatment effect across the entire population.

$$\hat{\tau}_{\text{DR}} = \frac{1}{n}\sum_{i=1}^n \left[ \hat{\mu}(x_i, 1) - \hat{\mu}(x_i, 0) + \frac{T_i(Y_i - \hat{\mu}(x_i, 1))}{\hat{e}(x_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}(x_i, 0))}{1 - \hat{e}(x_i)} \right]$$

```python
from torchregress.causal import dr_ate

# Estimate ATE + 95% Confidence Interval
ate, ci = dr_ate(y, t, mu0, mu1, propensity)
```

### 2. Conditional ATE ([`dr_cate`](../api/causal.md#torchregress.causal.dr_cate))

Estimates the treatment effect as a function of the covariates $X$. This is crucial for **personalised medicine** or **targeted marketing**, where we want to know $\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid x]$.

```python
from torchregress.causal import dr_cate

# Per-sample treatment effect estimates
cate = dr_cate(y, t, mu0, mu1, propensity, x)
```

### 3. Policy Evaluation ([`dr_policy_value`](../api/causal.md#torchregress.causal.dr_policy_value))

Estimates the expected outcome if we were to apply a specific treatment policy $\pi(x)$ to the entire population.

```python
# What happens if we treat only if feature 0 > 0.5?
policy = (x[:, 0] > 0.5).float()
value = dr_policy_value(y, t, mu0, mu1, propensity, policy)
```

---

## Method Comparison

| Method | Outcome Model? | Propensity Model? | Robustness |
|:-------|:--------------:|:-----------------:|:-----------|
| **Regression Adjust** | ✅ | ❌ | Sensitive to outcome model |
| **IPW (Weighting)** | ❌ | ✅ | Sensitive to propensity model |
| **Doubly Robust** | ✅ | ✅ | **Robust to either** |

---

## Requirements: The Three Pillars

For causal estimates to be valid, three assumptions must hold:

1. **Unconfoundedness**: No unmeasured variables influence both $T$ and $Y$.
2. **Positivity**: Every individual has a non-zero probability of being treated or not ($0 < \hat{e}(x) < 1$).
3. **Exchangeability**: The treated and control groups are comparable after adjusting for $X$.

!!! warning "Positivity Violations"

    If propensity scores are near 0 or 1, the DR estimator becomes unstable. Always check for overlap using [`causal_overlap_report`](../api/causal.md#torchregress.causal.causal_overlap_report).

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Robins et al. ["Estimation of Regression Coefficients When Some Regressors Are Not Always Observed."](https://www.jstor.org/stable/2290910) *JASA*, 1994. |
| 2 | Athey & Imbens. ["Recursive Partitioning for Heterogeneous Causal Effects."](https://www.pnas.org/doi/10.1073/pnas.1510489113) *PNAS*, 2016. |
| 3 | van der Laan & Rose. *Targeted Learning*. Springer, 2011. |
| 4 | Chernozhukov et al. ["Double/Debiased Machine Learning for Treatment and Structural Parameters."](https://arxiv.org/abs/1608.00060) *Econometrics J.*, 2018. |

---

## Next Steps
- Learn about [Propensity Weighted Losses](../losses/imbalanced.md)
- View the [Causal DR Uplift Comparison](../examples/causal_dr_uplift_comparison.md)
- View the [Causal DR Comparison (Real Covariates)](../examples/causal_dr_realdata_comparison.md)
- Explore [Inference Methods Detail](inference.md)
