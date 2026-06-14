# Distributional Conformal Prediction

These methods leverage **full predictive distributions** — CDF, density, or class-probability models — to produce tighter, more adaptive prediction intervals than point-prediction methods.

!!! success "Best of both worlds"
    Parametric models (GaussianNLL, MDN, normalizing flows) provide **shape information** about the predictive distribution.  Distributional conformal adds a **finite-sample coverage guarantee** on top — no matter how wrong the parametric assumptions are.

---

## DistributionalConformal

Conformal prediction via the **Probability Integral Transform** (PIT).

!!! abstract "Summary"
    **Score:**  $\;s_i = \bigl\lvert 2\,F(y_i \mid x_i) - 1\bigr\rvert$
    **Interval:**  $\;\bigl[F^{-1}(\alpha_{\mathrm{lo}} \mid x),\; F^{-1}(\alpha_{\mathrm{hi}} \mid x)\bigr]$
    **Requires:**  CDF values $F(y \mid x)$ and an inverse-CDF (ICDF) function

### Mathematical Details

If the predictive CDF $F(y \mid x)$ is correctly specified, the PIT value $U = F(Y \mid X)$ is uniformly distributed on $[0, 1]$.  The PIT nonconformity score measures **how far** $U$ is from uniform:

$$s = \bigl\lvert 2 U - 1\bigr\rvert \;\in\; [0, 1]$$

After computing the conformal quantile $\hat{q}$, the adjusted quantile levels are:

$$\alpha_{\mathrm{lo}} = \frac{1 - \hat{q}}{2}, \qquad \alpha_{\mathrm{hi}} = \frac{1 + \hat{q}}{2}$$

and the prediction interval is $\bigl[F^{-1}(\alpha_{\mathrm{lo}} \mid x),\; F^{-1}(\alpha_{\mathrm{hi}} \mid x)\bigr]$.

### Usage

```python
import torch
from torchregress.losses import DistributionalConformal

# Stage 1: Compute CDF values on calibration set
F_cal = model.cdf(x_cal, y_cal)  # shape: (n_cal,)

# Stage 2: Calibrate
dcp = DistributionalConformal(alpha=0.1)
dcp.calibrate(F_cal, y_cal)

# Stage 3: Predict intervals via inverse CDF
def icdf_fn(quantile_levels, x):
    """quantile_levels: (2,) → [α_lo, α_hi]; x: (n, d) → (n, 2)"""
    return model.icdf(x, quantile_levels)

lower, upper = dcp.predict_intervals_from_cdf(icdf_fn, x_test)
```

### Integration with torchregress Models

=== "GaussianNLLLoss"

    ```python
    from torch.distributions import Normal
    from torchregress.losses import DistributionalConformal

    # After training:
    with torch.no_grad():
        mean, logvar = model(x_cal).chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        dist = Normal(mean.squeeze(), std.squeeze())
        F_cal = dist.cdf(y_cal.squeeze())

    dcp = DistributionalConformal(alpha=0.1)
    dcp.calibrate(F_cal, y_cal)

    # ICDF function for test-time intervals
    def icdf_fn(levels, x_batch):
        mean, logvar = model(x_batch).chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        d = Normal(mean.squeeze(-1), std.squeeze(-1))
        return torch.stack([d.icdf(levels[0]), d.icdf(levels[1])], dim=-1)

    lower, upper = dcp.predict_intervals_from_cdf(icdf_fn, x_test)
    ```

=== "Mixture Density Network"

    ```python
    # MDN outputs: weights π_k, means μ_k, stds σ_k
    # CDF = Σ_k π_k Φ((y - μ_k) / σ_k)
    def mdn_cdf(x, y):
        pi, mu, sigma = model.predict_components(x)
        components = Normal(mu, sigma)
        return (pi * components.cdf(y.unsqueeze(-1))).sum(-1)

    F_cal = mdn_cdf(x_cal, y_cal)
    dcp = DistributionalConformal(alpha=0.1)
    dcp.calibrate(F_cal, y_cal)
    ```

=== "Normalizing Flow"

    ```python
    # Flow models provide log_prob; CDF via numerical integration
    # or use the flow's built-in CDF if available
    F_cal = flow_model.cdf(x_cal, y_cal)
    dcp = DistributionalConformal(alpha=0.1)
    dcp.calibrate(F_cal, y_cal)
    ```

!!! tip "When to use"
    When your model produces **full distributions** (CDF/ICDF).  DistributionalConformal gives **tighter intervals** than SplitConformal because it respects the shape of the predictive distribution — wider where the model predicts high variance, narrower where it's confident.

---

## CTI

**Conformal Thresholded Intervals** — produces the **smallest possible** prediction sets by using negative log-density as the nonconformity score.

!!! abstract "Summary"
    **Score:**  $\;s_i = -\log p(y_i \mid x_i)$
    **Interval:**  Density level set $\;\{y : -\log p(y\mid x) \leq \hat{q}\}$
    **Requires:**  A log-density function $\log p(y \mid x)$

### Mathematical Details

The prediction set at density threshold $e^{-t}$ is:

$$C_t(x) = \bigl\{y : -\log p(y \mid x) \leq t\bigr\}$$

By the **Neyman–Pearson lemma**, density level sets are the **smallest** prediction sets for a given coverage level.  CTI conformally calibrates the threshold $t = \hat{q}$ to ensure finite-sample coverage:

$$P\bigl(Y_{n+1} \in C_{\hat{q}}(X_{n+1})\bigr) \;\geq\; 1 - \alpha$$

!!! success "Advantages over CQR"
    - Produces **asymmetric** intervals that follow the density shape
    - Naturally handles **multimodal** distributions (disjoint prediction sets)
    - **Optimal** in the sense of shortest total interval length

### Usage

```python
import torch
from torchregress.losses import CTI

# Compute log-density on calibration set
log_density_cal = model.log_prob(x_cal, y_cal)  # (n_cal,)

# Calibrate
cti = CTI(alpha=0.1, grid_size=1000)
cti.calibrate(log_density_cal, y_cal)

# Predict using density evaluation on a grid
def density_fn(y_grid, x_single):
    """y_grid: (grid_size,), x_single: (d,) → log p(y|x): (grid_size,)"""
    return model.log_prob_grid(x_single, y_grid)

lower, upper = cti.predict_intervals_from_density(
    density_fn, x_test, y_min=-5.0, y_max=5.0
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `grid_size` | `int` | 500 | Resolution of the grid for level-set computation. |

!!! tip "When to use"
    When you have a **density model** (normalizing flow, MDN, KDE) and want the **tightest possible intervals**.  Especially powerful for **multimodal** target distributions.

!!! warning "Computational Cost & Dimensionality Limit"
    Because `CTI` and `R2CConformal` rely on evaluating density or probabilities over a discretized grid of the target space, they are computationally limited to **low-dimensional** (typically 1D or 2D) targets. For high-dimensional regression, the grid resolution required scales exponentially ($\mathcal{O}(\text{grid\_size}^D)$), making level-set search computationally intractable.

---

## R2CConformal

**Regression-to-Classification Conformal Prediction.**  Discretises the target space into bins, treats regression as softmax classification, and applies the **Adaptive Prediction Sets** (APS) algorithm.

!!! abstract "Summary"
    **Score:**  Cumulative probability of bins ranked above the true bin
    **Interval:**  Union of high-probability bins
    **Requires:**  Softmax probabilities over target bins

### Usage

```python
import torch
from torchregress.losses import R2CConformal

# Define target bins
bin_edges = torch.linspace(-5.0, 5.0, 101)  # 100 bins

# Model outputs softmax probabilities over bins
probs_cal = model(x_cal)  # (n_cal, 100)

# Calibrate
r2c = R2CConformal(alpha=0.1, bin_edges=bin_edges)
r2c.calibrate(probs_cal, y_cal)

# Predict
lower, upper = r2c.predict_interval(model(x_test))
```

!!! tip "When to use"
    When your target distribution is **highly multimodal** or skewed and you want conformal sets that can be **disjoint** (multiple disconnected intervals covering separate modes).

!!! quote "Reference"
    R. Izbicki, R. Shimizu, R. Stern. "Flexible distribution-free conditional predictive bands using density estimators." *AISTATS*, **2020**.

---

## Method Summary

| Method | Model Output | Handles Multimodal? | Interval Optimality |
|:-------|:------------|:-------------------:|:-------------------:|
| `DistributionalConformal` | CDF / ICDF | No (single contiguous interval) | Near-optimal |
| `CTI` | Log-density | ✅ Disjoint level sets | ✅ **Optimal** (Neyman–Pearson) |
| `R2CConformal` | Softmax bins | ✅ Disjoint bin unions | Near-optimal |
