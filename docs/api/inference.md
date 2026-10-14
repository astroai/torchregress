# Inference API

Complete reference for `torchregress.inference`. This package implements
**Prediction-Powered Inference (PPI)** — confidence-interval estimators that
combine a small trusted-labeled set with a large model-predicted unlabeled set
to improve statistical efficiency while preserving frequentist coverage.

For background, see [PPI + conformal](../guide/method-selection.md) and the
[inference example](../examples/index.md).

---

## Configuration

| Symbol | Description |
|:-------|:------------|
| `PPIConfig` | Frozen dataclass with `alpha` (target error rate), `method` (always `"bootstrap"`), `n_boot` (default `2000`), `seed`. Bootstrap means use O(N) memory (chunked); for samples above 100,000 points the bootstrap mean of that component is drawn from its CLT approximation `N(mean, var/N)`. All PPI functions keep the floating dtype of their inputs (float64 stays float64). |

---

## Mean CI (rectified)

| Symbol | Description |
|:-------|:------------|
| `ppi_mean_ci(y_labeled, pred_labeled, pred_unlabeled, *, config=None)` | PPI CI for a population mean. Estimator: `E[Y] ≈ mean(pred_unlabeled) + mean(y_labeled − pred_labeled)`. Returns `{"estimate", "se", "ci_lower", "ci_upper", "alpha", "n_labeled", "n_unlabeled", "bootstrap_samples"}`. |
| `ppi_pp_mean_ci` | `(y_labeled, pred_labeled, pred_unlabeled, *, lambdas=None, cross_fits=0, alpha=0.05)` — PPI++ confidence interval for a population mean: selects the power-tuning parameter λ minimizing first-order PPI variance (optionally cross-fitted). |

**Reference:** Angelopoulos, Bates, Fannjiang, Jordan, Zrnic,
"Prediction-Powered Inference" (Science 2023).

```python
from torchregress.inference import ppi_mean_ci, PPIConfig

cfg = PPIConfig(alpha=0.1, n_boot=2000, seed=42)
res = ppi_mean_ci(y_labeled, pred_labeled, pred_unlabeled, config=cfg)
# res["estimate"], res["ci_lower"], res["ci_upper"]
```

---

## Linearly-calibrated mean CI

| Symbol | Description |
|:-------|:------------|
| `ppi_calibrated_mean_ci` | `(y_labeled, pred_labeled, pred_unlabeled, *, config=None)` — Like `ppi_mean_ci`, but fits an affine map `m⋆(x) = â + b̂ m(x)` by OLS on labeled pairs, then refits `(â, b̂)` on every bootstrap resample. Returns the same dict as `ppi_mean_ci` but with `method="ppi_calibrated_mean_ci"`. |

**Reference:** Chen et al., "Linearly Calibrated Prediction-Powered Inference" (arXiv 2026).

```python
from torchregress.inference import ppi_calibrated_mean_ci
res = ppi_calibrated_mean_ci(y_labeled, pred_labeled, pred_unlabeled,
                              config=PPIConfig(alpha=0.1, n_boot=2000, seed=42))
```

---

## Quantile CI

| Symbol | Description |
|:-------|:------------|
| `ppi_quantile_ci` | `(y_labeled, pred_labeled, pred_unlabeled, *, q, config=None)` — PPI CI for a target quantile (Angelopoulos et al., 2023). Inverts the rectified CDF `F(θ) = mean_u 1{f_u ≤ θ} + mean_l(1{y ≤ θ} − 1{f_l ≤ θ})`: estimate `inf{θ : F(θ) ≥ q}`, CI `{θ : |F(θ) − q| ≤ z·sqrt(F_u(1−F_u)/N + Var_l(rectifier)/n)}` on a grid of observed values. Analytic (CLT) interval: `n_boot`/`seed` are unused, `bootstrap_samples` is 0 and `se` is the normal-equivalent half-width. |

---

## OLS coefficient CI

| Symbol | Description |
|:-------|:------------|
| `ppi_ols_ci` | `(x_labeled, y_labeled, x_unlabeled, pred_labeled, pred_unlabeled, *, add_intercept=True, config=None)` — PPI CI for linear-regression coefficients. `β̂ = β̂_unlabeled + β̂_(labeled residual)`. Bootstrap refits both regressions. Returns `{"coef", "se", "ci_lower", "ci_upper", "alpha", "n_labeled", "n_unlabeled", "bootstrap_samples"}`. Default `n_boot=1000`. |

---

## Diagnostics

| Symbol | Description |
|:-------|:------------|
| `ppi_diagnostics` | `(y_labeled, pred_labeled, pred_unlabeled)` — Returns `{"n_labeled", "n_unlabeled", "prediction_label_correlation", "residual_rmse_labeled", "residual_mean_labeled", "prediction_mean_shift_unlabeled_vs_labeled", "prediction_range_overlap_ratio"}`. Use to assess PPI validity before trusting a CI. |

---

## Orthogonal (double/debiased ML) inference

Cross-fitted estimation of a low-dimensional target `theta` in the partially linear
model `y = theta x + eta(z) + eps` (Chernozhukov et al., 2018); the nuisances
`E[y|z]` and `E[x|z]` are ridge regressions on a feature basis of `z`, fitted out of fold
on one shared split.

| Symbol | Description |
|:-------|:------------|
| `orthogonal_partially_linear(y, x, z, *, folds=5, ridge=1e-6, nuisance_degree=3, nuisance_features=None, confidence=0.95, seed=0)` | Returns an `OrthogonalEstimate` (`theta`, influence-function `sigma`, `ci_low`, `ci_high`, `n`, `folds`, `nuisance_r2_x`, `nuisance_r2_y`, `cross_fitted`). `ridge` is a float penalty (default `1e-6`, a jitter for the polynomial basis) or `"gcv"` / `"loo"`: the penalty is then chosen per nuisance and training fold by generalized cross-validation or exact leave-one-out error (closed form from one SVD, unpenalised intercept). `nuisance_features` maps `z` to a custom basis. |
| `random_fourier_features(z, *, n_features=256, bandwidth="median", standardize=True, polynomial_degree=3, seed=0)` | Random Fourier features of a Gaussian kernel (Rahimi and Recht, 2007) for `nuisance_features`. `bandwidth="median"` is the median pairwise distance of the standardised `z` on a seeded subsample; a float fixes it. `polynomial_degree` appends per-covariate powers of the standardised `z` (`0` for the bare features). |
| `median_heuristic_bandwidth(z, *, n_subsample=1000, seed=0)` | The median-heuristic kernel bandwidth used above. |
| `naive_linear_estimate(y, x)` | OLS slope ignoring `z`: the biased baseline. |

```python
from torchregress.inference import orthogonal_partially_linear, random_fourier_features

estimate = orthogonal_partially_linear(
    y, x, z,
    nuisance_features=lambda z: random_fourier_features(z, seed=0),
    ridge="gcv",
)
print(estimate.theta, estimate.ci_low, estimate.ci_high)
```

!!! tip "Choosing the nuisance basis"
    Monte Carlo over 200 replications, n = 500, 95% intervals (harness
    `orthogonal_inference` designs):

    | Design | Nuisance | Bias | Coverage | SE ratio |
    |:--|:--|--:|--:|--:|
    | CCDDHNR-2018 (p = 20, theta = 0.5) | RFF, unit bandwidth, `ridge=1e-2` (old recipe) | +0.008 | 0.785 | 0.65 |
    | | RFF, median bandwidth, `ridge="gcv"`, `polynomial_degree=1` | +0.004 | 0.955 | 1.02 |
    | | cubic polynomial (default basis) | +0.005 | 0.930 | 0.91 |
    | cubic (p = 5, theta = 1) | RFF, unit bandwidth, `ridge=1e-2` (old recipe) | +0.060 | 0.820 | 0.73 |
    | | RFF, median bandwidth, `ridge="gcv"`, `polynomial_degree=1` | +0.016 | 0.900 | 0.84 |
    | | the same with `polynomial_degree=3` (default) | -0.002 | 0.945 | 0.93 |
    | | cubic polynomial (default basis) | -0.001 | 0.945 | 0.92 |

    Random features alone cannot reproduce a cubic trend, which is why
    `polynomial_degree` defaults to 3 (cubic per-covariate terms). Across five
    designs (CCDDHNR-2018, linear, cubic, smooth interactions, 1-D periodic;
    300 replications) its 95% coverage stays within 0.90-0.95, whereas
    `polynomial_degree=1` falls to 0.78 on cubic confounding; on the smooth
    sigmoid nuisances of CCDDHNR degree 1 is slightly better (0.94 vs 0.92).
    The previous recipe (`W ~ N(0, 1/K)` on raw `z`, no polynomial part,
    `ridge=1e-2`) is `random_fourier_features(z, bandwidth=K**0.5,
    standardize=False, polynomial_degree=0)` with `ridge=1e-2`.

**Reference:** Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, Robins,
"Double/debiased machine learning for treatment and structural parameters" (Econometrics
Journal, 2018).

---

## Quick example

```python
import torch
from torchregress.inference import ppi_mean_ci, ppi_quantile_ci, ppi_ols_ci, PPIConfig

# Toy data
y_l = torch.randn(40)
p_l = y_l + 0.1 * torch.randn(40)
p_u = torch.randn(2000) + 0.05    # model on unlabeled
x_l = torch.randn(40, 3)
x_u = torch.randn(2000, 3)

# Mean CI
cfg = PPIConfig(alpha=0.1, n_boot=2000, seed=42)
mean_ci = ppi_mean_ci(y_l, p_l, p_u, config=cfg)

# Quantile CI
q_ci = ppi_quantile_ci(y_l, p_l, p_u, q=0.9, config=cfg)

# OLS coefficient CI
ols_ci = ppi_ols_ci(x_l, y_l, x_u, p_l, p_u, add_intercept=True, config=cfg)
# ols_ci["coef"] -> list of length 4 (intercept + 3 coefs)
```

---

## Next steps

- [Uncertainty decomposition](../guide/uncertainty-decomposition.md)
- [PPI examples](../examples/index.md) — coverage and efficiency in practice
- [Method selection guide](../guide/method-selection.md) — when PPI vs other UQ approaches
