# Algorithms API

Complete reference for `torchregress.algorithms`. For background and decision
guidance, see [Algorithms overview](../methods/index.md) and the
per-method pages under Methods.

---

## Robust regression

### Iteratively Reweighted Least Squares (IRLS)

| Symbol | Description |
|:-------|:------------|
| `IRLSConfig` | Dataclass with `base_loss`, `max_iter`, `tol`, `delta`, `weight_fn`, `weight_params`, `variance_type`, `epsilon`, `return_all_predictions`, `batch_size`. |
| `iteratively_reweighted_least_squares(model, x, y_true, …)` | Run IRLS using `huber` / `tukey` / `power` weighting, with `predicted` / `fixed` / `robust` variance estimation. Returns `(y_pred, loss_history, final_precision[, all_predictions])`. |
| `huber_weights(scaled_residuals, delta)` | Huber weighting function. |
| `tukey_weights(scaled_residuals, c)` | Tukey biweight; zero weight for `\|r\| > c`. |
| `power_weights(scaled_residuals, a, b)` | DAOPHOT-style power weighting. |
| `estimate_variance(residuals, y_pred, …)` | Estimate variance from predicted / fixed / robust sources. |
| `extract_mean_and_residuals(y_pred, y_true)` | Normalize heterogeneous model outputs. |

**Reference:** Beaton & Tukey, "The Fitting of Power Series…" (Technometrics, 1974).

```python
from torchregress.algorithms import iteratively_reweighted_least_squares, IRLSConfig

cfg = IRLSConfig(base_loss="gaussian", max_iter=10, weight_fn="huber",
                 delta=1.0, variance_type="robust")
y_pred, losses, precision = iteratively_reweighted_least_squares(
    model, x, y_true, config=cfg
)
```

---

## Error-in-Variables (EIV) algorithms

| Symbol | Description |
|:-------|:------------|
| `SIMEX` | Simulation-Extrapolation: add simulated noise `λ * Σ_u`, refit, extrapolate to `λ = -1`. Args: `model_factory`, `train_func`, `sigma_u`, `lambdas`, `n_simulations`, `extrapolation_order`. |
| `RegressionCalibration` | Classical RC: computes `E[X | W] = μ_w + (Σ_x (Σ_x + Σ_u)⁻¹) (W − μ_w)` with PSD projection. `fit_transform` for one-shot use. `posterior(...)` returns posterior mean and covariance (supports per-sample `σ_u`). |
| `LatentNN` | Latent-input neural regressor: jointly optimizes network params and per-sample latent clean inputs with a Gaussian quadratic penalty. |
| `ErrorAwareFeatureEncoder` | Feature encoder with explicit input-noise awareness. |
| `NoiseAwareRegressor` | Wrapper for noise-aware prediction heads. |
| `NeighborhoodCovarianceConfig` | Configuration for neighborhood-based covariance pseudo-labels. |
| `NeighborhoodCovariancePseudoLabeler` | Generate pseudo-covariance targets from feature neighborhoods. |
| `mahalanobis_covariance_pseudo_labels` | Functional form of `NeighborhoodCovariancePseudoLabeler`. |

**Reference:** Carroll, Ruppert, Stefanski, Crainiceanu,
*Measurement Error in Nonlinear Models* (2nd ed., 2006).

```python
from torchregress.algorithms import SIMEX, RegressionCalibration

# SIMEX
simex = SIMEX(model_factory=lambda: MyModel(), train_func=train_one,
              sigma_u=0.1, lambdas=[0.5, 1.0, 1.5, 2.0], n_simulations=10,
              extrapolation_order=2)
simex.fit(X_train, y_train)
y_pred_corrected = simex.predict(X_test)

# Regression calibration
rc = RegressionCalibration(sigma_u=0.1)
X_clean = rc.fit_transform(X_noisy)
post_mean, post_cov = rc.posterior(X_noisy)
```

---

## IVON (Bayesian learning rule)

| Symbol | Description |
|:-------|:------------|
| `IVON` | Variational Online Newton optimizer. Fits `q(θ) = N(μ, Σ)` over parameters using natural-gradient updates. Use inside `optimizer.sampled_params(train=True)` to draw MC samples. Supports `hess_approx ∈ {"price", "gradsq"}`, distributed `sync`, bias correction, LR rescaling. |

**Reference:** Khan & Rue, "The Bayesian Learning Rule" (JMLR 2023); Shen et al.,
"Variational Learning is Effective for Large Deep Networks" (ICML 2024).

```python
from torchregress.algorithms import IVON
optimizer = IVON(model.parameters(), lr=1e-2, ess=1000, weight_decay=1e-4)

for x, y in loader:
    with optimizer.sampled_params(train=True):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
    optimizer.step()
```

---

## TIC-TAC covariance parameterisation

| Symbol | Description |
|:-------|:------------|
| `TaylorInducedCovarianceHead` | Wraps a backbone `nn.Module` and predicts a covariance matrix derived from the Jacobian `J Jᵀ` and Hessian `H_ij = Tr(H_i H_j)` of the mean prediction w.r.t. inputs, with learnable scaling `k1, k2, k3` (input-dependent or global) plus a jitter term. |

**Reference:** Shukla et al., "TIC-TAC: A Framework For Improved Covariance
Estimation In Deep Heteroscedastic Regression" (ICML 2024).

```python
from torchregress.algorithms import TaylorInducedCovarianceHead
head = TaylorInducedCovarianceHead(backbone, target_dim=2, input_dim=10)
mean, cov = head(x)   # cov: [B, 2, 2]
```

---

## Heteroscedastic Laplace (last-layer)

| Symbol | Description |
|:-------|:------------|
| `NaturalHeteroscedasticHead` | Gaussian natural-parameterisation head `(η₁ = μ/σ², η₂ = -1/(2σ²))` with `exp` or `softplus` link. |
| `NaturalReparamHead` | Re-usable reparameterisation helper mapping `(f1, f2)` to `(mean, log_var)`. |
| `HeteroscedasticLaplaceRegressor` | Last-layer Laplace approximation over a heteroscedastic head. `fit(loader, lr, epochs)` then `predict_distribution(x, n_samples)` returns a `PredictiveBatch` with `(mean, std, samples, epistemic_variance, aleatoric_variance)`. |

**Reference:** Immer et al., "Effective Bayesian Heteroscedastic Regression" (NeurIPS 2023).

---

## VIDS (variational inference under distribution shift)

| Symbol | Description |
|:-------|:------------|
| `SyntheticEnvironmentSampler` | Bootstrap-based synthetic environment generator (`bootstrap_fraction`, `n_environments`). |
| `AdaptivePriorGuide` | Amortised variational posterior over parameters, conditioned on `(context_X, context_Y)`. |
| `AdaptivePriorNetwork` | Adaptive prior network that conditions on `(context_X, x_query)`. |
| `VIDSRegressor` | Variational regressor under covariate shift. `fit(x_train, y_train, n_environments, …)`; `predict_distribution(x_test, n_samples)` returns decomposed uncertainty in a `PredictiveBatch`. |

**Reference:** Slavutsky et al., "Quantifying Uncertainty in the Presence of
Distribution Shifts" (NeurIPS 2025).

---

## Quick example

```python
# Robust + noise-aware pipeline
from torchregress.algorithms import (
    iteratively_reweighted_least_squares, RegressionCalibration, LatentNN
)

# Correct noisy features
rc = RegressionCalibration(sigma_u=0.05)
X_clean = rc.fit_transform(X_noisy)

# Train an error-in-variables model with latent inputs
lnn = LatentNN(model_factory=lambda: MyModel(),
               sigma_x=0.05, sigma_y=0.1, epochs=500)
lnn.fit(X_noisy, y)
y_pred = lnn.predict(X_test)
```

## Next steps

- [Algorithms overview](../methods/index.md)
- [EIV losses](../losses/index.md#error-in-variables-losses)
- [Censored regression](../losses/censored.md)
- [Robust losses](../losses/robust.md)
