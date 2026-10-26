# Algorithms API

Complete reference for `torchregress.algorithms`. For background, decision
guidance, and detailed usage walkthroughs, see [Algorithms overview](../methods/index.md).

---

## IRLS — robust regression

→ Method guide: [IRLS](../methods/algorithms/irls.md). Losses: [Robust losses](../losses/robust.md).

| Symbol | Description |
|:-------|:------------|
| `IRLSProblem` | Protocol requiring `residuals(state)` and `weighted_fit(state, precision)`; optional covariance, whitening, and state-step callbacks are discovered structurally. |
| `IRLSCallbacks` | Frozen functional adapter for the generic problem protocol. |
| `IRLSFitConfig` | Robust weights/scale, base precision, masks, continuation stages and objective/weight/parameter convergence tolerances. |
| `IRLSContinuationStage` | One weight family/tuning stage in a continuation schedule. |
| `iteratively_reweighted_fit` | `(problem, initial_state, config)` — solver-agnostic IRLS that refits after every weight update and returns `IRLSFitResult`. |
| `IRLSFitResult`, `IRLSIteration` | Frozen result and history diagnostics with effective sample size and finite/termination status. |
| `IRLSBreakdownError` | Exception a caller-owned solver raises to return a diagnostic `"breakdown"` result. |
| `robust_weights` | Huber, Pseudo-Huber, Tukey, Cauchy, Barron-style or custom IRLS multipliers. |
| `reweight_fixed_predictions` | Robust reweighting of a fixed model. Does not refit. |
| `IRLSConfig` | Configuration for fixed-prediction compatibility reweighting. |
| `huber_weights(scaled_residuals, delta)` | Huber weighting function. |
| `tukey_weights(scaled_residuals, c)` | Tukey biweight; zero weight for `\|r\| > c`. |
| `power_weights(scaled_residuals, a, b)` | Power weighting. |
| `estimate_variance(residuals, y_pred, …)` | Variance from predicted / fixed / robust sources. |
| `extract_mean_and_residuals(y_pred, y_true)` | Normalize heterogeneous outputs. |

---

## EIV algorithms

→ Method guides: [SIMEX](../methods/algorithms/simex.md), [RC](../methods/algorithms/rc.md), [LatentNN](../methods/algorithms/latentnn.md), [Forward generative inversion](../methods/algorithms/forward_inversion.md), [Error-aware](../methods/algorithms/error_aware.md). Losses: [EIV losses](../losses/eiv.md).

| Symbol | Description |
|:-------|:------------|
| `SIMEX` | `(model_factory, train_func, sigma_u, lambdas, n_simulations, extrapolation_order)` — Simulation-Extrapolation. |
| `PredictionSIMEX(...)` | SIMEX variant optimised for prediction (shrinkage on extrapolated coefs). |
| `RegressionCalibration` | `(sigma_u)` — Classical RC with PSD projection. `fit_transform`, `posterior(...)` for per-sample cov. |
| `LatentNN` | `(model_factory, sigma_x, sigma_y, epochs, …)` — Joint latent-input + network optimization. |
| `ForwardInversionRegressor` | `(decoder=None, *, n_labels, n_observables, hidden_dims=(128,128), sigma_x, learn_noise=True, likelihood="gaussian"|"student_t"|"wls", degrees_of_freedom=5.0, epochs, batch_size, lr, ...)` — Generative `y -> x` surrogate (inverse regression / auto-decoder); `fit(X, y, train_decoder=True)`, `predict(X, *, method="gauss_newton"|"lbfgs", sigma_x=None, mask=None, return_std=False, return_info=False)`, `predict_with_covariance(...)`. `predict(return_info=True)` also returns per-object stationarity / Fisher-condition diagnostics for the non-convex test-time inversion. |
| `build_decoder_mlp(n_labels, n_observables, hidden_dims=(128,128), activation="silu")` | Default MLP decoder for `ForwardInversionRegressor`. |
| `ErrorAwareFeatureEncoder` | Encoder with input-noise awareness. |
| `NoiseAwareRegressor` | Noise-aware prediction head wrapper. |
| `NeighborhoodCovarianceConfig` | Config for neighborhood-based cov targets. |
| `NeighborhoodCovariancePseudoLabeler` | Pseudo-covariance targets from neighborhoods. |
| `mahalanobis_covariance_pseudo_labels(…)` | Functional form of the above. |

---

## Group DRO — worst-regime training

→ Method guide: [Continuous Group DRO](../methods/algorithms/group_dro.md).

| Symbol | Description |
|:-------|:------------|
| `ContinuousGroupDRO` | `(group_edges, *, eta=0.01, base_loss="mse", min_group_weight=0.0, update_every=1, reduction="mean")` — exponentiated-gradient group weights over continuous target bins. `forward(y_pred, target, mask=None, weights=None)` updates `q` in training mode; `group_weights()`, `group_losses(...)`, `fit_group_edges(targets)`, `reset()`. `eval()` freezes the adversarial weights. |
| `GroupDROTrainer` | `(*, model_factory, group_edges, base_loss="mse", eta=0.01, epochs=100, lr=1e-3, batch_size=64, weight_decay=0.0, seed=None)` — end-to-end `fit(X, y)` / `predict(X)`; exposes `group_weights_`, `group_losses_`, `history_`. |

---

## Variational imbalanced regression (VIR)

→ Method guide: [Variational Imbalanced Regression](../methods/algorithms/vir.md).

| Symbol | Description |
|:-------|:------------|
| `VIRRegressor` | `(encoder, *, feature_dim=None, latent_dim=None, n_bins=20, kernel_sigma=1.0, smoothing_shrinkage=5.0, prior=(0.0,1.0,2.0,0.1), reg_weight=0.01, max_importance_weight=10.0, kl_weight=1e-3, epochs=100, batch_size=64, lr=1e-3, weight_decay=0.0, inference_smoothing="two_pass", n_refine=1, seed=None, device=None, verbose=False)` — encoder + probabilistic label-similarity latent smoothing + NIG evidential head; `fit(X, y)`, `predict(X, return_std=True)` returns NIG mean and predictive uncertainty, and `predict_interval(X, confidence=0.9)` returns lower/upper bounds. Single-output targets. |
| `VIRLoss` | `(*, prior=(0.0,1.0,2.0,0.1), reg_weight=0.01, n_bins=20, kernel_sigma=1.0, min_nu=2.0, min_beta=1e-6, max_importance_weight=10.0, reduction="mean")` — reweighted NIG evidential loss; `fit(train_targets)` estimates the smoothed label density. Consumes the raw 3-unit NIG head output. |

---

## IVON

→ Method guide: [IVON](../methods/algorithms/ivon.md).

| Symbol | Description |
|:-------|:------------|
| `IVON` | `(params, lr, ess, weight_decay, …)` — Variational Online Newton. Fits `q(θ) = N(μ, Σ)`. Use `optimizer.sampled_params(train=True)` for MC samples. Supports `hess_approx ∈ {"price", "gradsq"}`, distributed `sync`, bias correction. |

---

## TIC-TAC

→ Method guide: [TIC-TAC](../methods/algorithms/tictac.md).

| Symbol | Description |
|:-------|:------------|
| `TaylorInducedCovarianceHead` | `(backbone, target_dim, input_dim)` — Jacobian+Hessian-derived covariance head with learnable scaling `k1, k2, k3`. Returns `(mean, cov)` where `cov: [B, d, d]`. |

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
| `AdaptivePriorGuide` | Amortised variational posterior given `(context_X, context_Y)`. |
| `VIDSRegressor` | Variational regressor under covariate shift. `fit(…)` → `predict_distribution(x, n_samples)`. |
| `sample_synthetic_environments` | Functional helper that bootstraps synthetic training environments for VIDS. |

---

## Warmup MC

| Symbol | Description |
|:-------|:------------|
| `WarmupMCTrainer` | `(model, n_warmup, n_samples)` — Runs warmup epochs then collects MC predictive samples for Bayesian model averaging. |

---

## Next steps

- [Algorithms overview](../methods/index.md) — decision guidance & method comparisons
- [EIV losses](../losses/index.md#error-in-variables-losses)
- [Robust losses](../losses/robust.md)
- [Censored regression](../losses/censored.md)
