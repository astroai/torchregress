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
| **`SplitConformal`** | Baseline, homoscedastic noise | ❌ | [`SplitConformal`](../../api/losses.md#torchregress.losses.conformal.SplitConformal) |
| **`CQR`** [2] | Heteroscedasticity, skewed noise | ✅ | [`CQR`](../../api/losses.md#torchregress.losses.conformal.CQR) |
| **`UACQR`** | Same as CQR + width-normalized scores | ✅ | [`UACQR`](../../api/losses.md#torchregress.losses.conformal.UACQR) |
| **`DensityConformal`** | Imbalanced or long-tail data | ✅ | [`DensityConformal`](../../api/losses.md#torchregress.losses.conformal.DensityConformal) |
| **`MonteCarloConformal`** | Ensembles, Bayesian models | ✅ | [`MonteCarloConformal`](../../api/losses.md#torchregress.losses.conformal.MonteCarloConformal) |
| **`CTI`** [3] | Multimodal, complex distributions | ✅ | [`CTI`](../../api/losses.md#torchregress.losses.conformal.CTI) |

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

Standard CP guarantees **marginal coverage**: $P(\text{coverage}) \geq 1-\alpha$ on average over all possible test points. To achieve **conditional coverage** (coverage for specific regions of feature space), **torchregress** supports:

1. **Mondrian Groups**: Calibrate separately for different categories.
2. **Normalised Scores**: Scale $s_i$ by a difficulty estimate $\hat{\sigma}(x)$.

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
| 3 | Sadinle et al. ["Least Ambiguous Set-Valued Classifiers with Bounded Error Levels."](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1395341) *JASA*, 2019. |
| 4 | Angelopoulos & Bates. ["A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification."](https://arxiv.org/abs/2107.07511) *Technical Report*, 2021. |

---

## Next Steps
- Explore [Predictors and Scores](predictors.md)
- Learn about [Distributional Conformal Prediction](distributional.md)
- View the [Conformal Regression Example](../../examples/conformal_regression_example.md)
