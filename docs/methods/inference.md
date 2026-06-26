# Prediction-Powered Inference (PPI)

> ← [Semi-Supervised](semi_supervised.md) | [Visualization](visualization.md) →

PPI is a statistical framework for performing **valid inference** using a combination of a small set of gold-standard labels and a large set of machine learning predictions.  It produces confidence intervals that are (1) **narrower** than using gold labels alone and (2) **statistically valid** even if the ML model is arbitrarily biased.

!!! abstract "Why this matters"
    In many domains — healthcare, remote sensing, social science, NLP evaluation — obtaining accurate labels is expensive, but cheap ML predictions are abundant.  PPI lets you exploit this asymmetry for rigorous statistical analysis.

---

## Background and Motivation

### The Problem

Suppose you want to estimate a population parameter $\theta$ (a mean, a quantile, or a regression coefficient).  You have two data sources:

| Source | Size | Quality | Cost |
|:-------|:----:|:--------|:-----|
| **Gold-standard labels** $Y_i$ | $n$ (small) | Accurate | Expensive (expert annotation, spectroscopy, laboratory assay) |
| **ML predictions** $\hat{f}(X_i)$ | $N \gg n$ (large) | Potentially biased | Cheap (automated model) |

Three naive approaches all have critical flaws:

| Approach | Problem |
|:---------|:--------|
| Use only gold labels | Valid but **imprecise** — CIs are wide because $n$ is small |
| Use only predictions | **Invalid** — systematic model bias contaminates the estimate |
| Mix them ad hoc | No coverage guarantee — unclear how to combine |

### The PPI Solution

PPI uses the large prediction set for **precision** and the small gold set for **bias correction**.  The **rectified estimator** is:

$$\boxed{\;\hat\theta_{\text{PPI}} = \underbrace{\hat\theta_N(\hat{f})}_{\substack{\text{prediction}\\\text{on all } N}} \;-\; \underbrace{\bigl(\hat\theta_n(\hat{f}) - \hat\theta_n(Y)\bigr)}_{\substack{\text{bias correction}\\\text{on labelled }n}}\;}$$

The first term uses all $N$ samples for precision.  The bias correction (middle two terms) cancels the model's systematic error, estimated on the labelled subset.

!!! success "Validity guarantee"
    The resulting CIs are valid **regardless of model quality**.  If the model is accurate ($R^2 \approx 1$), CIs are much tighter.  If the model is poor ($R^2 \approx 0$), PPI gracefully degrades to gold-label-only inference, never making things worse (PPI++).

---

## Core Assumptions

PPI's validity rests on a small number of assumptions:

| Assumption | Description | Diagnostic |
|:-----------|:------------|:-----------|
| **Exchangeability** | Labelled and unlabelled samples drawn from the **same population** (MCAR) | Compare covariate distributions |
| **External model** | The prediction model $\hat{f}$ must be trained on **data separate from** the internal dataset | Prevents overfitting bias |
| **Full covariates** | All features needed by $\hat{f}$ are observed for **every unit** (labelled and unlabelled) | Check for missingness |

!!! warning "When PPI breaks"
    If labelling is **not random** (e.g., only easy-to-classify examples get labels), the bias correction is invalid.  This is the Missing Not At Random (MNAR) setting; extensions exist but require additional modelling \[5\].

---

## Available Estimators

| Function | Estimand | Typical Application |
|:---------|:---------|:-------------------|
| `ppi_mean_ci` | Population mean $\mathbb{E}[Y]$ | Average treatment effect, prevalence estimation |
| `ppi_calibrated_mean_ci` | Same, with affine post-hoc calibration of the score | Mis-scaled but informative scores (OLS on labeled pairs) |
| `ppi_quantile_ci` | Population quantile $Q_\tau$ | Median income, tail risk |
| `ppi_ols_ci` | OLS coefficients $\beta$ | Regression with proxy labels |
| `ppi_diagnostics` | — | Prediction quality assessment |

### Population Mean

```python
from torchregress.inference import PPIConfig, ppi_mean_ci

ci = ppi_mean_ci(
    y_labeled=y_gold,
    pred_labeled=f_hat_gold,
    pred_unlabeled=f_hat_all,
    config=PPIConfig(alpha=0.05),
)
print(f"95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
```

### Population Quantile

```python
from torchregress.inference import PPIConfig, ppi_quantile_ci

ci = ppi_quantile_ci(
    y_labeled=y_gold,
    pred_labeled=f_hat_gold,
    pred_unlabeled=f_hat_all,
    q=0.5,  # median
    config=PPIConfig(alpha=0.05),
)
```

### OLS Coefficients

```python
from torchregress.inference import PPIConfig, ppi_ols_ci

ci = ppi_ols_ci(
    x_labeled=x_gold, y_labeled=y_gold,
    x_unlabeled=x_all, pred_labeled=f_hat_gold, pred_unlabeled=f_hat_all,
    config=PPIConfig(alpha=0.05),
)
```

### Calibrated PPI for the mean

Chen et al. (*Calibeating Prediction-Powered Inference*, [arXiv:2604.21260](https://arxiv.org/abs/2604.21260)) study **post-hoc calibration** of a fixed scalar score $m(X)$ on the labeled sample before semisupervised mean estimation.  Torchregress implements their **linearly calibrated** mean (Section 3.3): fit an affine map by ordinary least squares on labeled pairs,

$$
m_n^\star(x) := \hat a + \hat b\, m(x), \qquad (\hat a, \hat b) \in \arg\min_{a,b \in \mathbb{R}} \;\sum_{i=1}^n \{Y_i - a - b\, m(X_i)\}^2.
$$

Then apply the **same rectified mean** as `ppi_mean_ci`, but with $m_n^\star$ in place of $m$:

$$
\hat\psi = \underbrace{\frac{1}{N}\sum_{j=1}^N m_n^\star(\tilde X_j)}_{\text{unlabeled plug-in}} + \underbrace{\frac{1}{n}\sum_{i=1}^n \bigl(Y_i - m_n^\star(X_i)\bigr)}_{\text{labeled residual}}.
$$

Bootstrap percentile intervals **refit** $(\hat a, \hat b)$ on each labeled bootstrap replicate (and draw independent unlabeled bootstrap rows), which accounts for calibration uncertainty.  The paper also analyzes isotonic calibration theoretically; we omit it here in favor of the lightweight affine map (pure PyTorch), which they relate to prognostic-score style adjustment and to PPI++ at first order.

```python
from torchregress.inference import PPIConfig, ppi_calibrated_mean_ci

ci_cal = ppi_calibrated_mean_ci(
    y_labeled=y_gold,
    pred_labeled=f_hat_gold,
    pred_unlabeled=f_hat_all,
    config=PPIConfig(alpha=0.05, n_boot=2000, seed=0),
)
```

!!! tip "When this helps"
    Use calibrated PPI when the score tracks $Y$ but has the **wrong slope or intercept** (common for models trained on another population or loss).  If the score is already conditionally unbiased, gains over `ppi_mean_ci` may be modest.

---

## Inference vs prediction (and “better on all metrics”)

Earlier guidance distinguished two kinds of follow-up work:

| Track | What it meant | Role |
|:------|:--------------|:-----|
| **(b) Code** | Wire **existing** pieces together (e.g. `SplitConformal` + PPI) in a runnable workflow | Shows a **defensible pipeline**, not a single magic estimator |
| **(c) Docs** | Explain **which metric** each method optimizes and where tradeoffs live | Sets expectations so “better everywhere” is not promised |

**There is no single procedure that simultaneously maximizes every metric** (interval width vs coverage vs bias vs sharpness vs conditional validity).  What you *can* do is **separate estimands** and use the right tool per question:

| Goal | Typical tool in torchregress | Contract (informal) |
|:-----|:----------------------------|:--------------------|
| **Uncertainty for a summary** (e.g. $\mathbb{E}[Y]$, OLS $\beta$) | `ppi_mean_ci`, `ppi_calibrated_mean_ci`, `ppi_ols_ci`, … | Uses labeled + unlabeled; **bias-aware** rectification |
| **Finite-sample predictive bands** for individual $Y$ | `torchregress.losses.SplitConformal`, CQR, … | **Exchangeability / split** assumptions; calibrate scores on a **held-out** labeled fold |

**Conformalizing** in the strict sense means building **nonconformity scores** on labeled data and extrapolating intervals to new predictions — it does **not** replace PPI’s job for $\mathbb{E}[Y]$, but it **complements** it when you also care about **per-unit** coverage.

### Recommended composition

1. **Split the labeled data** (at least two folds): one fold fits **affine calibration** (or any post-hoc map), another supports **PPI** bias correction together with unlabeled scores, and optionally a third (or reuse a fold carefully) feeds **split conformal** calibration.
2. Use **`ppi_calibrated_mean_ci`** when you are comfortable refitting $(\hat a,\hat b)$ on every labeled bootstrap replicate (often good variance when $n$ is not tiny).
3. Use **`SplitConformal`** on **residuals** $|Y - \hat Y|$ where $\hat Y$ is the **same** post-calibrated score you will deploy at test time, calibrated only on data **not** used to cherry-pick that map if you need clean marginal guarantees.

A minimal end-to-end sketch is in [`examples/ppi_mean_plus_split_conformal.py`](https://github.com/sfabbro/torchregress/blob/main/examples/ppi_mean_plus_split_conformal.py).

---

## Efficiency and When to Expect Gains

The precision gain from PPI depends on the **predictive quality** of the model:

$$\text{Variance ratio} \approx \frac{1}{1 - R^2}$$

| Model quality | $R^2$ | Effective sample size multiplier | CI width reduction |
|:-------------|:-----:|:-------------------------------:|:-----------------:|
| Excellent | 0.95 | $\sim 20\times$ | $\sim 78\%$ |
| Good | 0.80 | $\sim 5\times$ | $\sim 55\%$ |
| Moderate | 0.50 | $\sim 2\times$ | $\sim 30\%$ |
| Poor | 0.10 | $\sim 1.1\times$ | $\sim 5\%$ |
| Useless | 0.00 | $1\times$ (no gain) | $0\%$ |

!!! tip "Rule of thumb"
    PPI is most valuable when $R^2 > 0.5$ and $N/n > 10$.  Use `ppi_diagnostics` to assess before committing.

---

## Complete Example

```python
import torch
from torchregress.inference import ppi_mean_ci, ppi_diagnostics

# Setting: estimating mean patient recovery time
# Gold labels: expert-assessed recovery (expensive, n = 150)
# ML predictions: automated scoring (cheap, N = 10,000)
torch.manual_seed(42)

# Gold-standard labels
y_gold = 14.0 + 3.0 * torch.randn(150)  # days

# ML predictions on gold set (correlated but biased)
f_hat_gold = y_gold + 1.5 + 0.8 * torch.randn(150)  # bias = +1.5 days

# ML predictions on full population
f_hat_all = 15.5 + 3.0 * torch.randn(10_000)

# --- Gold-only CI (wide) ---
gold_mean = y_gold.mean()
gold_se = y_gold.std() / (150 ** 0.5)
print(f"Gold-only 95% CI: [{gold_mean - 1.96*gold_se:.2f}, {gold_mean + 1.96*gold_se:.2f}]")
print(f"  Width: {2 * 1.96 * gold_se:.2f} days")

# --- PPI CI (narrower) ---
from torchregress.inference import PPIConfig
ci = ppi_mean_ci(y_gold, f_hat_gold, f_hat_all, config=PPIConfig(alpha=0.05))
print(f"PPI      95% CI: [{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}]")
print(f"  Width: {ci['ci_upper'] - ci['ci_lower']:.2f} days")

# Diagnostics
diag = ppi_diagnostics(y_gold, f_hat_gold, f_hat_all)
```

---

## Comparison with Related Approaches

| Method | Uses Unlabelled | Valid CIs | Corrects Bias | Assumptions |
|:-------|:--------------:|:---------:|:-------------:|:------------|
| **Gold-only** | ❌ | ✅ | N/A | None |
| **Prediction-only** | ✅ | ❌ | ❌ | Model is correct |
| **Semi-supervised** | ✅ | ❌ (heuristic) | Partially | Model class |
| **PPI** | ✅ | ✅ | ✅ | Exchangeability |
| **PPI++** | ✅ | ✅ | ✅ | Same + tuned $\lambda$ |
| **Calibrated PPI (mean)** | ✅ | ✅ (bootstrap) | ✅ | Post-hoc $f \circ m$ on labeled data \[6\] |

---

## Strengths and Limitations

!!! success "Strengths"
    - **Model-agnostic** — valid even when $\hat{f}$ is arbitrarily wrong
    - **Never hurts** (PPI++) — asymptotically never increases variance over gold-only
    - **Modular** — works for means, quantiles, regression, classification
    - **Broad applicability** — biomedicine, remote sensing, social science, NLP

!!! warning "Limitations"
    - Requires **MCAR labelling** — if labels are obtained non-randomly, extensions are needed
    - No gain when predictions are **uninformative** ($R^2 \approx 0$)
    - Model must be trained on **external data** (not the same dataset)
    - The proliferation of PPI variants (PPI, PPI++, Cross-PPI, Cross-PPBoot) creates a usability gap \[5\]

---

## References

| # | Reference |
|:-:|:----------|
| 1 | A. Angelopoulos et al. ["Prediction-Powered Inference."](https://www.science.org/doi/10.1126/science.adi6029) *Science*, 382(6671):669–674, **2023**. |
| 2 | A. Angelopoulos, S. Bates, T. Zrnic. ["PPI++: Fast Valid Inference via Tuned Cross-Prediction."](https://arxiv.org/abs/2311.01453) *arXiv:2311.01453*, **2023**. |
| 3 | T. Zrnic, E. Candès. ["Cross-Prediction-Powered Inference."](https://arxiv.org/abs/2402.04351) *PNAS*, **2024**. |
| 4 | J. Gronsbell et al. "Efficient and Robust Semi-Supervised Estimation of ATE." *JASA*, **2024**. |
| 5 | J. Miao et al. ["Demystifying Prediction Powered Inference."](https://arxiv.org/abs/2601.20819) *arXiv:2601.20819*, **2025**. |
| 6 | Y. Chen et al. ["Calibeating Prediction-Powered Inference."](https://arxiv.org/abs/2604.21260) *arXiv:2604.21260*, **2026**. |

---

## Next Steps

- [Calibration Metrics](../metrics/calibration.md) — evaluate your model's predictive quality for PPI
- [Causal Inference](causal.md) — doubly-robust estimation (related methodology)
