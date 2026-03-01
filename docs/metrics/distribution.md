# Distributional Metrics

Distributional metrics evaluate **probabilistic forecasts** — how well does the predicted probability distribution $F$ match the true (but unknown) data-generating process $G$? Unlike point metrics, these assess both **calibration** (reliability) and **sharpness** (precision).

---

## Proper Scoring Rules

A scoring rule $S(F, y)$ is **proper** if its expected value is minimised when the predicted distribution $F$ is equal to the true distribution $G$ [1].

$$\mathbb{E}_{y \sim G} [S(G, y)] \leq \mathbb{E}_{y \sim G} [S(F, y)]$$

In **torchregress**, we prioritise proper scoring rules for evaluating all probabilistic models.

---

## Continuous Ranked Probability Score (CRPS)

The CRPS is the most widely used proper scoring rule for univariate regression. It can be viewed as the integral of the pinball loss over all possible quantiles $\tau \in [0, 1]$.

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} [F(z) - \mathbf{1}_{z \geq y}]^2 dz$$

### Properties

- **Units**: Same as the target variable $y$.
- **Point Mass**: Reduces to Mean Absolute Error (MAE) if $F$ is a point mass.
- **Duality**: Simultaneously rewards **calibration** (is the truth within the predicted range?) and **sharpness** (is the predicted range narrow?).

### Implementation

```python
from torchregress.metrics import crps_gaussian, energy_score

# For Gaussian models
loss = crps_gaussian(mu, sigma, y_true)

# For non-parametric models (e.g., Ensembles, BNNs) using samples
loss = energy_score(y_samples, y_true)
```

→ See [Mathematical Foundations](../math/index.md) for the Gaussian closed-form derivation. API Reference: [`crps_gaussian`](../api/metrics.md#torchregress.metrics.distribution.crps_gaussian).

---

## Multivariate: Energy Score

The **Energy Score (ES)** [2] is the multivariate generalisation of CRPS to $\mathbb{R}^d$. It evaluates the joint distribution of multiple targets, capturing correlations that univariate CRPS misses.

$$\text{ES}(F, y) = \mathbb{E}_{Y \sim F} \|Y - y\|^\beta - \frac{1}{2} \mathbb{E}_{Y, Y' \sim F} \|Y - Y'\|^\beta$$

where $\beta \in (0, 2)$ (default is $\beta=1$).

### Implementation

```python
from torchregress.metrics import energy_score

# y_samples: [num_samples, batch_size, num_targets]
score = energy_score(y_samples, y_true)
```

API Reference: [`energy_score`](../api/metrics.md#torchregress.metrics.distribution.energy_score).

---

## Calibration: Probability Integral Transform (PIT)

A model is **perfectly calibrated** if its predictive CDF $F(y \mid x)$, when evaluated at the true value $y$, is uniformly distributed on $[0, 1]$ [3].

$$U = F(Y \mid X) \sim \text{Uniform}(0, 1)$$

### Diagnosing Miscalibration

- **U-Shaped**: The model is **overconfident** (true values fall in the tails too often).
- **Hump-Shaped**: The model is **underconfident** (true values fall in the center too often).
- **Skewed**: The model has a consistent bias (predicting too high or too low).

### Implementation

To visualise calibration, use the **PIT Histogram** diagnostic from the visualization module:

```python
from torchregress.viz import plot_pit_histogram

# Generate a PIT histogram to visualize calibration
plot_pit_histogram(y_pred_dist, y_true, bins=20)
```

API Reference: [`plot_pit_histogram`](../api/viz.md#torchregress.viz.diagnostic.plot_pit_histogram).

---

## Summary Matrix

| Metric | Best For | Proper? | API Reference |
|:-------|:---------|:-------:|:--------------|
| **NLL** | Parametric models | ✅ | [`gaussian_nll`](../api/metrics.md#torchregress.metrics.distribution.gaussian_nll) |
| **CRPS** | Univariate UQ | ✅ | [`crps_gaussian`](../api/metrics.md#torchregress.metrics.distribution.crps_gaussian) |
| **Energy Score** | Multivariate UQ | ✅ | [`energy_score`](../api/metrics.md#torchregress.metrics.distribution.energy_score) |
| **PIT** | Calibration check | — | [`plot_pit_histogram`](../api/viz.md#torchregress.viz.diagnostic.plot_pit_histogram) |

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Gneiting & Raftery. ["Strictly Proper Scoring Rules, Prediction, and Estimation."](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) *JASA*, 2007. |
| 2 | Gneiting & Katzfuss. ["Probabilistic Forecasting."](https://www.annualreviews.org/doi/abs/10.1146/annurev-statistics-062713-085831) *Annual Review of Statistics*, 2014. |
| 3 | Dawid, A. P. ["Statistical Theory: The Prequential Approach."](https://www.jstor.org/stable/2345714) *JRSS A*, 1984. |

---

## Next Steps
- Learn about [Calibration Metrics](calibration.md)
- View the [Distributional Conformal Tutorial](../conformal/distributional.md)
- Explore [Normalizing Flow Examples](../examples/normalizing_flows_multitarget.md)
