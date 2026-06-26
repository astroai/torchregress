# Conformal Prediction

Conformal Prediction (CP) is a modern framework for uncertainty quantification that provides **mathematically guaranteed** prediction intervals for any regression model.

!!! abstract "The CP Guarantee"

    For a chosen miscoverage level $\alpha \in (0, 1)$, a conformal predictor constructs an interval $\hat{C}(x)$ such that:
    $$\boxed{\,P\!\bigl(Y_{n+1} \in \hat{C}(X_{n+1})\bigr) \;\geq\; 1 - \alpha\,}$$
    This guarantee holds for **any** model, **any** distribution, and **any** sample size $n$, provided the data is **exchangeable**.

---

## Why Use Conformal Prediction?

Traditional uncertainty estimates (e.g., from Gaussian NLL) are often **miscalibrated**. They rely on strong parametric assumptions (like normality) that rarely hold in practice. Conformal prediction fixes this post-hoc.

### Key Advantages

- **Distribution-Free**: No assumptions about the shape of the noise or the model's error distribution.
- **Model-Agnostic**: Works with any neural network, tree-based model, or ensemble.
- **Finite-Sample Validity**: The coverage guarantee is exact, even for small calibration sets.
- **No Retraining**: Apply CP to your existing pre-trained models.

---

## Core Workflow

The conformal workflow consists of three distinct phases:

### 1. Training (Standard)

Train your regression model on a training set $\mathcal{D}_{\text{train}}$ as you normally would.

### 2. Calibration (Offline)

Evaluate the model on a **held-out** calibration set $\mathcal{D}_{\text{cal}}$ to compute "non-conformity scores" $s_i$. These scores measure how "unusual" an observation is relative to the model's prediction.

```python
from torchregress.losses import SplitConformal

# Target 90% coverage (alpha = 0.1)
cp = SplitConformal(alpha=0.1)
cp.calibrate(y_pred_cal, y_cal)
```

### 3. Prediction (Inference)

At test time, the conformal predictor uses the calibrated scores to produce an interval $\hat{C}(x)$ around the model's point prediction.

```python
lower, upper = cp.predict_interval(y_pred_test)
```

---

## Method Selection Matrix

| Method | Best For | Adaptive Width? | API Reference |
|:-------|:---------|:---------------:|:--------------|
| **`SplitConformal`** | Baseline, homoscedastic noise | ❌ | [SplitConformal](../../api/conformal.md) |
| **`CQR`** \[2\] | Heteroscedasticity, skewed noise | ✅ | [CQR](../../api/conformal.md) |
| **`UACQR`** | Same as CQR + width-normalized scores | ✅ | [UACQR](../../api/conformal.md) |
| **`DensityConformal`** | Imbalanced or long-tail data | ✅ | [DensityConformal](../../api/conformal.md) |
| **`MonteCarloConformal`** | Ensembles, Bayesian models | ✅ | [MonteCarloConformal](../../api/conformal.md) |
| **`LocalConformal`** | Local feature-space coverage (LVD) | ✅ | [LocalConformal](../../api/conformal.md) |
| **`LocalConformalMAD`** | Local feature-space + MAD scaling | ✅ | [LocalConformalMAD](../../api/conformal.md) |
| **`CVPlus`** \[5\] | Cross-validation ensembles | ❌ | [CVPlus](../../api/conformal.md) |
| **`EnsembleBatchCP`** \[6\] | Bootstrap / bagging ensembles (EnbPI) | ❌ | [EnsembleBatchCP](../../api/conformal.md) |
| **`CTI`** \[3\] | Multimodal, complex distributions | ✅ | [CTI](../../api/conformal.md) |

---

## Mathematical Deep Dive

### Non-Conformity Scores

The heart of CP is the **non-conformity score** $s(x, y)$, which quantifies how poorly a model prediction $\hat{y}$ fits a true value $y$. Common scores include:

- **Absolute Residual**: $s_i = |y_i - \hat{y}_i|$ (used in `SplitConformal`)
- **Quantile Score**: $s_i = \max(\hat{q}_{\text{lo}} - y_i, y_i - \hat{q}_{\text{hi}})$ (used in `CQR`)

### The Conformal Quantile

Given $n$ calibration scores $\{s_1, \dots, s_n\}$, we compute the $(1-\alpha)(1+1/n)$ quantile:

$$\hat{q} = \text{Quantile}\left(\{s_i\}_{i=1}^n, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$

The prediction interval is then:

$$\hat{C}(x) = \{ y : s(x, y) \leq \hat{q} \}$$

### Marginal vs. Conditional Coverage

Standard CP guarantees **marginal coverage** on average over all possible test points:
$$P\bigl(Y_{n+1} \in \hat{C}(X_{n+1})\bigr) \geq 1 - \alpha$$

To achieve **conditional coverage** (validity within local regions of the feature space, i.e., $P(Y \in \hat{C}(x) \mid X=x) \geq 1-\alpha$), **torchregress** supports:

1. **Mondrian Conformal Prediction**: Calibrate independently for discrete subgroups or categories.
2. **Normalized/Adaptive Scores**: Divide residuals by a difficulty estimate $\hat{\sigma}(x)$ (e.g., in `UACQR` or `MonteCarloConformal`), adapting interval width locally.
3. **Locally Valid Conformal Prediction**: Apply kernel-weighted local density estimators in the embedding space (e.g., in `LocalConformal`).

### The Exchangeability Assumption & OOD Limits

The mathematical guarantees of conformal prediction rely fundamentally on **exchangeability**.

!!! info "Mathematical Definition of Exchangeability"
    A sequence of random variables $Z_1, \dots, Z_{N}$ (where $Z_i = (X_i, Y_i)$) is exchangeable if their joint probability distribution is invariant under any permutation $\sigma$ of the indices:
    $$P(Z_1, Z_2, \dots, Z_N) = P(Z_{\sigma(1)}, Z_{\sigma(2)}, \dots, Z_{\sigma(N)})$$
    This is a weaker assumption than the independent and identically distributed (i.i.d.) assumption, as it allows for certain forms of global correlation (e.g., drawing without replacement) but excludes temporal ordering or systematic drifts.

#### Out-of-Distribution (OOD) Failures
If test samples violate exchangeability (e.g., due to concept drift, temporal autocorrelation, or shifted covariates), the empirical quantiles calculated on $\mathcal{D}_{\text{cal}}$ will not reflect the test distribution. This leads to **under-coverage** (actual coverage drops below $1 - \alpha$).

#### Mitigating Covariate Shift
When the feature distribution shifts ($P_{\text{test}}(X) \neq P_{\text{cal}}(X)$) but the label distribution condition holds ($P_{\text{test}}(Y \mid X) = P_{\text{cal}}(Y \mid X)$), **torchregress** allows passing importance weights $w(x) = \frac{p_{\text{test}}(x)}{p_{\text{cal}}(x)}$.
Instead of a uniform quantile, the threshold $\hat{q}$ is computed by solving the weighted quantile equation:
$$\sum_{i=1}^n \tilde{p}_i \mathbb{I}(s_i \leq s) \geq 1 - \alpha$$
where the normalized weights are:
$$\tilde{p}_i = \frac{w(X_i)}{\sum_{j=1}^n w(X_j) + w(X_{n+1})}, \quad \tilde{p}_{n+1} = \frac{w(X_{n+1})}{\sum_{j=1}^n w(X_j) + w(X_{n+1})}$$

### Coordinate-wise vs. Joint Multi-Target Conformal

For multi-output regression ($\mathbf{Y} \in \mathbb{R}^D$), calibrating each coordinate independently at level $1-\alpha$ (coordinate-wise conformal prediction) does not guarantee joint coverage of the vector.

* **Coordinate-wise Coverage**: Guarantees $P(Y_{d, n+1} \in \hat{C}_d(X_{n+1})) \geq 1 - \alpha$ for each dimension $d$ individually.
* **Joint Coverage (Bonferroni Bound)**: To guarantee that the entire true vector $\mathbf{Y}_{n+1}$ falls inside the hyperrectangle $\prod_{d=1}^D \hat{C}_d(X_{n+1})$ with probability at least $1 - \alpha$, we must apply the Bonferroni correction, calibrating each individual dimension at level $1 - \alpha / D$:
  $$P\bigl(\mathbf{Y}_{n+1} \in \prod_{d=1}^D \hat{C}_d(X_{n+1})\bigr) \geq 1 - \sum_{d=1}^D \frac{\alpha}{D} = 1 - \alpha$$
* **Joint Conformal Regions (Mahalanobis / Ellipsoidal)**: Rather than coordinate-wise hyperrectangles (which grow excessively wide under Bonferroni correction for large $D$), joint approaches map vector errors into a single scalar non-conformity score (e.g., Mahalanobis distance $s_i^2 = (\mathbf{y}_i - \hat{\mathbf{y}}_i)^\top \mathbf{\Sigma}^{-1} (\mathbf{y}_i - \hat{\mathbf{y}}_i)$). This yields an ellipsoidal joint confidence region that is significantly tighter when dimensions are highly correlated. (Note: `MultiTargetConformal` in **torchregress** performs coordinate-wise calibration; if joint coverage is needed, adjust the input `alpha` using the Bonferroni correction $\alpha/D$).


---

## Best Practices

!!! tip "Calibration Set Size"

    While CP works for any $n$, a larger calibration set (e.g., $n > 500$) results in **more stable** and **tighter** intervals. If your calibration set is too small, $\hat{q}$ will be conservative (too large).

!!! warning "Data Leakage"

    The calibration set **must not** be used during model training. If the model has already "seen" the calibration data, its residuals will be artificially small, leading to under-coverage at test time.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Vovk et al. *Algorithmic Learning in a Random World*. Springer, 2005. |
| 2 | Romano et al. ["Conformalized Quantile Regression."](https://arxiv.org/abs/1905.03222) *NeurIPS*, 2019. |
| 3 | Luo & Zhou. ["Conformal Thresholded Intervals for Efficient Regression."](https://arxiv.org/abs/2407.14495) *AAAI*, 2025. |
| 4 | Angelopoulos & Bates. ["A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification."](https://arxiv.org/abs/2107.07511) *Technical Report*, 2021. |
| 5 | Barber et al. ["Predictive inference with the jackknife+."](https://arxiv.org/abs/1905.02928) *The Annals of Statistics*, 2021. |
| 6 | Xu & Xie. ["Conformal prediction interval for dynamic time-series."](https://arxiv.org/abs/2010.14144) *ICML*, 2021. |

---

## Next steps

Continue with conformal prediction:

- [Predictors and Scores](predictors.md) — per-class API reference with parameter tables for SplitConformal, CQR, UACQR, and all variants
- [Distributional Conformal](distributional.md) — CDF/density-based methods (CTI, DistributionalConformal)
- [Test-Time Shift-Aware Conformal](../test-time/ot-shift-conformal.md) — maintain coverage when the test distribution shifts
- [Conformal Regression Example](../../examples/conformal_regression_example.md) — runnable comparison of all conformal predictors
