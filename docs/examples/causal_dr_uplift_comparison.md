# Doubly Robust Causal Inference Comparison

This guide demonstrates how to perform doubly robust estimation of Average Treatment Effects (ATE) and Conditional Average Treatment Effects (CATE) under confounded treatment assignment.

| # | Reference |
|:-:|:----------|
| 1 | Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). [**Estimation of Regression Coefficients when Some Regressors Are Not Always Observed**](https://www.jstor.org/stable/2290910). *Journal of the American Statistical Association*. |
| 2 | Chernozhukov, V., et al. (2018). [**Double/debiased machine learning for treatment and structural parameters**](https://arxiv.org/abs/1608.00060). *The Econometrics Journal*. |

---

## Mathematical Formulations

Causal inference aims to estimate the causal effect of a binary treatment $T_i \in \{0, 1\}$ on an outcome $Y_i \in \mathbb{R}$, adjusting for confounding covariates $X_i \in \mathbb{R}^D$.

### Propensity and Outcome Models
We define:
1.  **Propensity Score** $e(X_i)$: The probability of treatment assignment:
    $$e(X_i) = P(T_i = 1 \mid X_i)$$
2.  **Outcome Models** $\mu_1(X_i), \mu_0(X_i)$: The expected potential outcomes:
    $$\mu_1(X_i) = \mathbb{E}[Y_i \mid X_i, T_i = 1], \quad \mu_0(X_i) = \mathbb{E}[Y_i \mid X_i, T_i = 0]$$

### Doubly Robust ATE Estimator

The Doubly Robust (DR) ATE estimator combines propensity weighting and outcome regression:

$$\hat{\tau}_{\text{DR}} = \frac{1}{N} \sum_{i=1}^N \left[ \left( \mu_1(X_i) + \frac{T_i (Y_i - \mu_1(X_i))}{e(X_i)} \right) - \left( \mu_0(X_i) + \frac{(1 - T_i) (Y_i - \mu_0(X_i))}{1 - e(X_i)} \right) \right]$$

This estimator is **doubly robust** because it is unbiased if *either* the propensity model $e(X)$ *or* the outcome models $\mu_1(X), \mu_0(X)$ are correctly specified (but not necessarily both).

### Doubly Robust CATE Estimator

For heterogeneous treatment effects conditional on covariates $X$, we compute DR pseudo-outcomes $\tilde{Y}_i$:

$$\tilde{Y}_i = \mu_1(X_i) - \mu_0(X_i) + \frac{T_i (Y_i - \mu_1(X_i))}{e(X_i)} - \frac{(1 - T_i) (Y_i - \mu_0(X_i))}{1 - e(X_i)}$$

We then regress the pseudo-outcomes $\tilde{Y}_i$ on the covariates $X_i$ using a target regression model $\tau(X)$ to estimate CATE:

$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$$

### Overlap Diagnostics

Propensity score division by $e(X_i)$ or $1 - e(X_i)$ can lead to high variance if propensity scores are near 0 or 1 (positivity violations).
1.  **Overlap Rate**: The fraction of samples whose propensity scores lie within a trimmed region (e.g. $[0.05, 0.95]$).
2.  **Effective Sample Size (ESS)**:
    Measures the numerical stability of the propensity weights $w_i$:
    $$\text{ESS} = \frac{\left( \sum_i w_i \right)^2}{\sum_i w_i^2}$$
    A small ESS relative to $N$ indicates high variance due to extreme weights.

---

## Task-First Context

*   **When to Use**: Use Doubly Robust causal models when estimating the impact of an action or treatment (e.g. ad uplift, clinical intervention, pricing change) from observational data with confounding variables.
*   **Comparison Notes**: Cross-fitting (splitting the data into $K$ folds to fit nuisance models and estimate pseudo-outcomes) is enabled by default in `dr_ate` and `dr_cate` to prevent overfitting bias.

---

## Code Example

Below is the complete, self-contained code comparing naive treatment differences against Doubly Robust ATE and CATE estimates on synthetic uplift data.

```python
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from torchregress.causal import causal_overlap_report, dr_ate, dr_cate

def main() -> None:
    # Set seed
    torch.manual_seed(260227)
    n_samples, n_features = 1200, 6

    # Generate synthetic confounded causal dataset
    x = torch.randn(n_samples, n_features)
    base = 0.5 * x[:, 0] - 0.4 * x[:, 1] + 0.3 * x[:, 2] ** 2
    tau = 0.4 + 0.3 * torch.tanh(x[:, 0]) # True heterogeneous treatment effect (CATE)

    # Propensity scores (assignment probability)
    p = torch.sigmoid(0.9 * x[:, 0] - 0.7 * x[:, 1] + 0.25 * x[:, 2]).clamp(0.03, 0.97)
    t = torch.bernoulli(p) # Treatment assignment

    y0 = base + 0.25 * torch.randn(n_samples) # Control outcome
    y = y0 + t * tau # Observed outcome

    true_ate = float(tau.mean().item())

    # 1. Naive Difference in Means (confounded)
    treated = t > 0.5
    control = ~treated
    naive_ate = float(y[treated].mean().item() - y[control].mean().item())

    # 2. Doubly Robust ATE
    ate_result = dr_ate(
        x, t, y,
        outcome_model=LinearRegression,
        propensity_model=LogisticRegression(max_iter=1000),
        folds=3,
        alpha=0.05,
        seed=260227,
        trim_threshold=0.05,
    )

    # 3. Doubly Robust CATE
    cate_result = dr_cate(
        x, t, y,
        cate_model=LinearRegression,
        outcome_model=LinearRegression,
        propensity_model=LogisticRegression(max_iter=1000),
        folds=3,
        alpha=0.05,
        seed=260227,
        trim_threshold=0.05,
    )

    print(f"True ATE: {true_ate:.5f}")
    print(f"Naive ATE: {naive_ate:.5f} (Error: {abs(naive_ate - true_ate):.5f})")
    print(f"DR-ATE Estimate: {ate_result['estimate']:.5f} (Error: {abs(ate_result['estimate'] - true_ate):.5f})")
    print(f"  DR-ATE 95% CI: [{ate_result['ci_low']:.5f}, {ate_result['ci_high']:.5f}]")
    print(f"DR-CATE ATE Estimate: {cate_result['ate_estimate']:.5f} (Error: {abs(cate_result['ate_estimate'] - true_ate):.5f})")
    print(f"  Overlap Rate: {ate_result['diagnostics']['overlap_rate']:.4f}")
    print(f"  Min Group ESS: {ate_result['diagnostics']['min_group_ess']:.2f}")

if __name__ == "__main__":
    main()
```
