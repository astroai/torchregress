# Distributional Metrics

> ← [Interval Metrics](interval.md) | [Calibration Metrics](calibration.md) →

Distributional metrics evaluate **probabilistic forecasts** — how well does the predicted probability distribution $F$ match the true (but unknown) data-generating process $G$? Unlike point metrics, these assess both **calibration** (reliability) and **sharpness** (precision).

---

## Proper Scoring Rules

A scoring rule $S(F, y)$ is **proper** if its expected value is minimised when the predicted distribution $F$ is equal to the true distribution $G$ (Ref. 1).

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

# For Gaussian models — note: argument order is (mean, y_true, std)
loss = crps_gaussian(mu, y_true, sigma)

# For non-parametric models (e.g., Ensembles, BNNs) using samples
loss = energy_score(y_samples, y_true)
```

→ See [Mathematical Foundations](../guide/math/index.md) for the Gaussian closed-form derivation. API Reference: [crps_gaussian](../api/metrics.md).

### Estimators and conventions

| Forecast type | Function | Estimator | Reference implementation |
|:--|:--|:--|:--|
| Gaussian $\mathcal N(\mu, \sigma^2)$ | `crps_gaussian` | closed form | `scoringrules.crps_normal`, `properscoring.crps_gaussian` |
| $M$ samples $x_1, \dots, x_M$ | `crps_from_samples` | fair: $\frac1M\sum_i \lvert x_i - y\rvert - \frac{1}{2M(M-1)}\sum_{i \ne j}\lvert x_i - x_j\rvert$ | `scoringrules.crps_ensemble(estimator="fair")` |
| $K$ quantiles $q_{\tau_k}$ | `continuous_ranked_probability_score` | $2\sum_k w_k\,\text{QS}_{\tau_k}$, trapezoid $w_k = (\tau_{k+1} - \tau_{k-1})/2$, $\tau_0 = 0$, $\tau_{K+1} = 1$ | `scoringrules.quantile_score` with these weights |

The fair estimator is unbiased for the CRPS of the distribution the members
are drawn from. The biased $1/M^2$ version (`properscoring.crps_ensemble`,
`scoringrules` `estimator="nrg"`) favours small ensembles. One sample is a
Dirac forecast, so the CRPS is the absolute error. The quantile form only
approximates the integral: it has no information beyond the outermost levels,
so use the sample or closed-form estimator when you have samples or a
distribution.

`vario_score` / `VarioScore` (Zamo & Naveau) is positively oriented,
$\nu_\rho = \tfrac12\mathbb E\lvert X - X'\rvert^\rho - \mathbb E\lvert X - y\rvert^\rho$
(equal to $-\text{CRPS}$ at $\rho = 1$), so `VarioScore.higher_is_better` is
`True`. `variogram_score` sums over pairs $i < j$, which is half of
`scoringrules.variogram_score`. All reference conventions are pinned in
`tests/metrics/test_reference_parity.py`.

---

## Multivariate: Energy Score

The **Energy Score (ES)** (Ref. 2) is the multivariate generalisation of CRPS to $\mathbb{R}^d$. It evaluates the joint distribution of multiple targets, capturing correlations that univariate CRPS misses.

$$\text{ES}(F, y) = \mathbb{E}_{Y \sim F} \|Y - y\|^\beta - \frac{1}{2} \mathbb{E}_{Y, Y' \sim F} \|Y - Y'\|^\beta$$

where $\beta \in (0, 2)$ (default is $\beta=1$).

### Implementation

```python
from torchregress.metrics import energy_score

# y_samples: [num_samples, batch_size, num_targets]
score = energy_score(y_samples, y_true)
```

API Reference: [energy_score](../api/metrics.md).

---

## Calibration: Probability Integral Transform (PIT)

A model is **perfectly calibrated** if its predictive CDF $F(y \mid x)$, when evaluated at the true value $y$, is uniformly distributed on $[0, 1]$ (Ref. 3).

$$U = F(Y \mid X) \sim \text{Uniform}(0, 1)$$

### Diagnosing Miscalibration

- **U-Shaped**: The model is **overconfident** (true values fall in the tails too often).
- **Hump-Shaped**: The model is **underconfident** (true values fall in the center too often).
- **Skewed**: The model has a consistent bias (predicting too high or too low).

### Implementation

To visualise calibration, use the **PIT Histogram** diagnostic from the visualization module:

```python
from torchregress.viz import plot_pit_histogram

# Generate a PIT histogram to visualize calibration for Gaussian predictions
plot_pit_histogram(y_pred, y_pred_std, y_true, n_bins=20)
```

API Reference: [plot_pit_histogram](../api/viz.md).

---

## Highest Posterior Density (HPD) Coverage

For a 1D predictive density $p(y \mid x)$ on a grid, the HPD level of an observation $y$ is the mass of the smallest region that contains $y$ and consists of the highest-density points:

$$\ell(y) = \int_{\{y' \,:\, p(y') \ge p(y)\}} p(y')\, dy'$$

For a calibrated predictive density, $\ell(y)$ is uniform on $[0, 1]$. `highest_posterior_density_level` returns $\ell$ for each target, and `highest_posterior_density_coverage` returns the fraction of targets with $\ell(y) \le \alpha$.

!!! warning "`alpha` is the HPD mass, not the miscoverage"
    In `highest_posterior_density_coverage(support, density, y_true, alpha=0.1)`, `alpha` is the **nominal probability mass of the HPD region**. Everywhere else in torchregress (conformal classes, `prediction_interval_coverage`, ...) `alpha` is the **miscoverage** rate. For calibrated predictions the returned coverage is approximately `alpha`: `alpha=0.9` returns about 0.90 (the coverage of a 90% region), while the default `alpha=0.1` returns about 0.10 (the coverage of a 10% region), not 0.90. Pass `alpha=0.9` to evaluate a 90% HPD region.

```python
import torch
from torchregress.metrics import highest_posterior_density_coverage

support = torch.linspace(-6.0, 6.0, 1201)
mu = torch.randn(2000)  # predicted means
density = torch.exp(-0.5 * (support[None, :] - mu[:, None]) ** 2)  # unit-variance Gaussians (unnormalised)
y_true = mu + torch.randn(2000)  # calibrated: targets drawn from the predictions

cov_90 = highest_posterior_density_coverage(support, density, y_true, alpha=0.9)  # approx. 0.90
cov_10 = highest_posterior_density_coverage(support, density, y_true)  # alpha=0.1 default: approx. 0.10
```

---

## Unified Metrics Report

For comprehensive evaluation, use the `distribution_metrics_report` helper. It consolidates NLL, CRPS, Energy Score, PIT uniformity, and coverage into a single dictionary.

```python
from torchregress.metrics import distribution_metrics_report

# dist: torch.distributions.Distribution
# y_true: Ground truth tensor
results = distribution_metrics_report(dist=dist, y_true=y_true)

print(f"CRPS: {results['crps']:.4f}")
print(f"PIT KS: {results['pit_ks']:.4f}")
print(f"90% Coverage: {results['coverage_90']:.2%}")
```

This is the recommended way to evaluate complex probabilistic models, as it provides a multi-faceted view of model performance.

`results["crps"]` uses the best representation available. A Normal `dist`
gets the closed form. `samples` (or samples drawn from a non-Normal `dist`)
get the fair ensemble estimator. Only `y_pred_quantiles` use the quantile
approximation. Before 0.3 the report always reduced samples to 7 quantiles,
which underestimated the CRPS by 3-4%.

!!! info "PIT from quantile forecasts"
    A quantile forecast says nothing about the CDF beyond its outermost levels
    $\tau_1$ and $\tau_K$. A target below $q_{\tau_1}$ therefore gets the
    randomised PIT $U\tau_1$, and a target above $q_{\tau_K}$ gets
    $\tau_K + U(1 - \tau_K)$, with $U \sim \text{Uniform}(0, 1)$. For a
    calibrated forecast this has exactly the distribution of the true PIT in
    each tail. Mapping such targets to exactly 0 or 1 (as before 0.3) put
    point masses at the ends and made `pit_ks` / `pit_chi2` reject even
    perfect forecasts. The draw uses a fixed seed by default; pass
    `generator=torch.Generator().manual_seed(...)` to change it.

---

## Summary Matrix

| Metric | Best For | Proper? | API Reference |
|:-------|:---------|:-------:|:--------------|
| **NLL** | Parametric models | ✅ | [gaussian_nll](../api/metrics.md) |
| **CRPS** | Univariate uncertainty | ✅ | [crps_gaussian](../api/metrics.md) |
| **Energy Score** | Multivariate uncertainty | ✅ | [energy_score](../api/metrics.md) |
| **PIT** | Calibration check | — | [plot_pit_histogram](../api/viz.md) |

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Gneiting & Raftery. ["Strictly Proper Scoring Rules, Prediction, and Estimation."](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) *JASA*, 2007. |
| 2 | Gneiting & Katzfuss. ["Probabilistic Forecasting."](https://www.annualreviews.org/doi/abs/10.1146/annurev-statistics-062713-085831) *Annual Review of Statistics*, 2014. |
| 3 | Dawid, A. P. ["Statistical Theory: The Prequential Approach."](https://www.jstor.org/stable/2345714) *JRSS A*, 1984. |

---

## Limitations

1. **CRPS sample requirements**: `crps_from_samples` requires $S \ge 50$ MC samples per test point for reliable estimates. Using fewer samples introduces Monte Carlo noise that can obscure model comparisons. For Gaussian models, use the analytic `crps_gaussian` instead — it is exact and faster.
2. **Energy Score is $\mathcal{O}(S^2)$**: Computing the pairwise-expectation term requires evaluating $\|Y - Y'\|^\beta$ for all $S(S-1)/2$ sample pairs. For $S > 500$, this becomes a runtime bottleneck.
3. **PIT is necessary but not sufficient**: A uniform PIT histogram indicates marginal calibration but does not guarantee conditional calibration $F(y \mid X = x)$. A model can pass the PIT test while being poorly calibrated for specific subpopulations.
4. **NLL sensitivity**: NLL is dominated by low-probability events. A single point with $p(y \mid x) \approx 0$ produces an arbitrarily large NLL contribution. CRPS is more robust to these tail events.

## Recommendations

- **Default suite**: Report CRPS (interpretable, robust) + NLL (sensitive to tail calibration) + PIT (calibration diagnostic) for a complete picture.
- **For Gaussian models**: Always use `crps_gaussian` (analytic, exact) over `crps_from_samples`.
- **For multivariate targets**: Use `energy_score` ($\beta=1$) as the multivariate CRPS analogue.
- **PIT bin count**: Use 20–50 bins for the PIT histogram. Too few bins hide miscalibration patterns; too many create noise.
- **[distribution_metrics_report](../api/metrics.md)** consolidates CRPS, NLL, PIT, and coverage into a single call.

## Next Steps
- Learn about [Calibration Metrics](calibration.md)
- View the [Distributional Conformal Tutorial](../methods/conformal/distributional.md)
- Explore [Normalizing Flow Examples](../examples/normalizing_flows_multitarget.md)
