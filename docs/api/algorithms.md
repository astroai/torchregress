# Algorithms API

Complete reference for `torchregress.algorithms`. For background, decision
guidance, and detailed usage walkthroughs, see [Algorithms overview](../methods/index.md).

---

## IRLS — robust regression

→ Method guide: [IRLS](../methods/algorithms/irls.md). Losses: [Robust losses](../losses/robust.md).

| Symbol | Description |
|:-------|:------------|
| `IRLSConfig(base_loss, max_iter, tol, delta, weight_fn, weight_params, variance_type, epsilon, return_all_predictions, batch_size)` | Configuration dataclass. |
| `iteratively_reweighted_least_squares(model, x, y_true, *, config)` | Run IRLS. Returns `(y_pred, loss_history, final_precision[, all_predictions])`. |
| `huber_weights(scaled_residuals, delta)` | Huber weighting function. |
| `tukey_weights(scaled_residuals, c)` | Tukey biweight; zero weight for `\|r\| > c`. |
| `power_weights(scaled_residuals, a, b)` | Power weighting. |
| `estimate_variance(residuals, y_pred, …)` | Variance from predicted / fixed / robust sources. |
| `extract_mean_and_residuals(y_pred, y_true)` | Normalize heterogeneous outputs. |

---

## EIV algorithms

→ Method guides: [SIMEX](../methods/algorithms/simex.md), [RC](../methods/algorithms/rc.md), [LatentNN](../methods/algorithms/latentnn.md), [Error-aware](../methods/algorithms/error_aware.md). Losses: [EIV losses](../losses/eiv.md).

| Symbol | Description |
|:-------|:------------|
| `SIMEX(model_factory, train_func, sigma_u, lambdas, n_simulations, extrapolation_order)` | Simulation-Extrapolation. |
| `PredictionSIMEX(...)` | SIMEX variant optimised for prediction (shrinkage on extrapolated coefs). |
| `RegressionCalibration(sigma_u)` | Classical RC with PSD projection. `fit_transform`, `posterior(...)` for per-sample cov. |
| `LatentNN(model_factory, sigma_x, sigma_y, epochs, …)` | Joint latent-input + network optimization. |
| `ErrorAwareFeatureEncoder(…)` | Encoder with input-noise awareness. |
| `NoiseAwareRegressor(…)` | Noise-aware prediction head wrapper. |
| `NeighborhoodCovarianceConfig(…)` | Config for neighborhood-based cov targets. |
| `NeighborhoodCovariancePseudoLabeler(…)` | Pseudo-covariance targets from neighborhoods. |
| `mahalanobis_covariance_pseudo_labels(…)` | Functional form of the above. |

---

## IVON

→ Method guide: [IVON](../methods/algorithms/ivon.md).

| Symbol | Description |
|:-------|:------------|
| `IVON(params, lr, ess, weight_decay, …)` | Variational Online Newton. Fits `q(θ) = N(μ, Σ)`. Use `optimizer.sampled_params(train=True)` for MC samples. Supports `hess_approx ∈ {"price", "gradsq"}`, distributed `sync`, bias correction. |

---

## TIC-TAC

→ Method guide: [TIC-TAC](../methods/algorithms/tictac.md).

| Symbol | Description |
|:-------|:------------|
| `TaylorInducedCovarianceHead(backbone, target_dim, input_dim)` | Jacobian+Hessian-derived covariance head with learnable scaling `k1, k2, k3`. Returns `(mean, cov)` where `cov: [B, d, d]`. |

---

## Heteroscedastic Laplace

→ Method guide: [Heteroscedastic Laplace](../methods/algorithms/heteroscedastic_laplace.md).

| Symbol | Description |
|:-------|:------------|
| `NaturalHeteroscedasticHead` | Gaussian natural-parameterisation head `(η₁=μ/σ², η₂=-1/(2σ²))`. |
| `NaturalReparamHead` | Helper: maps `(f1, f2)` to `(mean, log_var)`. |
| `HeteroscedasticLaplaceRegressor` | Last-layer Laplace. `fit(loader, …)` → `predict_distribution(x, n_samples)` → `PredictiveBatch(mean, std, samples, epistemic_variance, aleatoric_variance)`. |

---

## VIDS

→ Method guide: [Adaptive prior VI](../methods/algorithms/adaptive_prior_vi.md).

| Symbol | Description |
|:-------|:------------|
| `SyntheticEnvironmentSampler(bootstrap_fraction, n_environments)` | Bootstrap-based synthetic env generator. |
| `AdaptivePriorGuide(…)` | Amortised variational posterior given `(context_X, context_Y)`. |
| `VIDSRegressor(…)` | Variational regressor under covariate shift. `fit(…)` → `predict_distribution(x, n_samples)`. |

---

## Warmup MC

| Symbol | Description |
|:-------|:------------|
| `WarmupMCTrainer(model, n_warmup, n_samples)` | Runs warmup epochs then collects MC predictive samples for Bayesian model averaging. |

---

## Next steps

- [Algorithms overview](../methods/index.md) — decision guidance & method comparisons
- [EIV losses](../losses/index.md#error-in-variables-losses)
- [Robust losses](../losses/robust.md)
- [Censored regression](../losses/censored.md)
