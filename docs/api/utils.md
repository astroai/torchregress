# Utilities API

Complete reference for `torchregress.utils`. For background, see
[Method selection](../guide/method-selection.md).

---

## Predictive containers (`prediction`)

Top-level utilities for representing predictive distributions in a normalized
container used across test-time tooling, calibration, and conformal code.

| Symbol | Description |
|:-------|:------------|
| `PredictiveBatch` | Frozen dataclass carrying `point` / `mean` / `std` / `quantiles` (+ `quantile_levels`) / `bar_logits` (+ `bin_edges`) / `samples` / `support` (+ `density`) / `extra`. `.with_density(n_support=200, range_margin=0.05)` auto-resolves `support` and `density` from `bar_logits`, `quantiles`, or `samples`. |
| `quantiles_to_density_grid` | `(quantiles, quantile_levels, *, n_support=200, range_margin=0.05)` — Convert quantile predictions to a regular density grid on a per-row support. Crossing quantiles are repaired by sorting each row (monotone rearrangement). **The grid is the predictive law truncated to `[q_0, q_K]` and renormalised** — see the note below. |
| `bars_to_density_grid` | `(bar_logits, bin_edges, *, n_support=200, range_margin=0.05)` — Convert piecewise-constant bar distributions to a regular density grid. |
| `samples_to_density_grid` | `(samples, *, n_support=200, range_margin=0.05)` — Convert scalar predictive samples to a regular density grid. |

!!! note "Truncation semantics of `quantiles_to_density_grid`"
    The tail mass outside the outermost quantiles (`τ_0` below `q_0` and `1 − τ_K`
    above `q_K`) is not represented: the density is zero on the `range_margin`
    margins and the grid is renormalised to integrate to one. The grid therefore
    encodes the predictive law *truncated to* `[q_0, q_K]` and renormalised, so its
    CDF at knot `k` is `(τ_k − τ_0) / (τ_K − τ_0)` rather than `τ_k`, and densities
    are inflated by `1 / (τ_K − τ_0)`. For levels `(0.05, 0.5, 0.75)` the grid median
    sits at the `(0.5 − 0.05) / 0.7 ≈ 64`-th percentile of the grid. Use wide outer
    levels (for example 0.01 and 0.99) when the tails matter. This behaviour is
    unchanged pending a maintainer decision;
    `PredictiveBatch.with_density` and the test-time transport code consume this grid
    as the predictive density.

```python
import numpy as np
from torchregress.prediction import PredictiveBatch, samples_to_density_grid

batch = PredictiveBatch(mean=mu, std=sigma, samples=samples, extra={"family": "gaussian"})
density_batch = batch.with_density(n_support=256)
```

---

## Density basis (`utils.bspline`)

| Symbol | Description |
|:-------|:------------|
| `BSplineDensityBasis` | `(breakpoints, degree=3)` — Unit-integral (M-spline) B-spline basis on a clamped knot vector. `evaluate`, `density`, `bin_integrals(edges, *, dtype=None, device=None)` (returns a fresh tensor in the dtype/device of `edges`; `float64` for sequences), `cdf`, `cumulative_moment`, `absolute_deviation`, `basis_means`; constructors `from_uniform`, `from_quantiles`. Guide: [Grid & basis densities](../losses/density_basis.md). |

---

## Augmentations (`utils.augment`)

| Symbol | Description |
|:-------|:------------|
| `Augmentation` | `(probability=0.5)` — Base class for input augmentations. |
| `Adversarial` | `(model, loss_fn, epsilon=0.05, steps=3, alpha=None, probability=0.5, random_start=False)` — FGSM/PGD-style adversarial augmentation. |
| `EnsemblePerturbationAugmenter` | `(n_samples=20, perturb_method="gaussian", sigma=0.1, feature_wise=True, device=None)` — Input perturbation designed to expose ensemble disagreement. |

---

## Gaussian output helpers (`utils.gaussian_output`)

| Symbol | Description |
|:-------|:------------|
| `split_mean_log_variance` | `(y_pred, *, split_dim=-1, mean_only_log_var="error")` — Split a `(…, 2D)` output (or tuple / dict) into `(mean, log_var)`. |
| `variance_from_logvar` | `(log_var, *, min_logvar=-8.0, max_logvar=6.0, eps=1e-8)` — Numerically-stable `exp(log_var)` with clipping. |

---

## Ordinal utilities (`utils.ordinal`)

| Symbol | Description |
|:-------|:------------|
| `cumulative_probs_to_pmf` | `(cumulative_probs, eps=1e-8)` — Convert cumulative probabilities to a class PMF. |
| `cumulative_logits_to_pmf` | `(logits)` — Convert cumulative logits to a class PMF. |
| `ordinal_predict` | `(y_pred, *, encoding="cumulative_logits", threshold=0.5, strategy="argmax", num_classes=None, return_pmf=False)` — Decode ordinal outputs into class-index predictions. |
| `CORALHead` | `(in_features, num_classes)` — Shared-weight ordinal output head with monotonic bias constraints; maps features to cumulative logits. |

---

## Propensity utilities (`utils.propensity`)

| Symbol | Description |
|:-------|:------------|
| `ipw_weights` | `(propensity, observed=None, *, clip_min=0.01, clip_max=0.99, normalize=True)` — Inverse-propensity weights `1 / e` (or `t / e + (1 − t) / (1 − e)` when `observed` is given) with clipping and optional mean-one normalisation; the dtype follows `float_dtype(propensity)`. |

---

## PyTorch compatibility (`utils.pytorch_compat`)

| Symbol | Description |
|:-------|:------------|
| `set_all_seeds` | `(seed)` — Seed Python + NumPy + PyTorch (CPU + CUDA). |

---

## Reduction (`utils.reduction`)

| Symbol | Description |
|:-------|:------------|
| `_safe_denominator` | Replace a non-positive denominator with one so empty reductions yield `0`, not `NaN`. |

---

## Semi-supervised (`utils.semisupervised`)

| Symbol | Description |
|:-------|:------------|
| `generate_pseudo_labels` | `(prediction, *, log_variance=None, confidence=None, confidence_threshold=0.0, max_std=None, min_confidence=0.0)` — Pseudo-labels, confidence weights, and accept-mask tensors. |
| `update_ema_teacher_` | `(teacher, student, *, momentum=0.99)` — In-place EMA teacher update. |

---

## Tensor ops (`utils.tensor_ops`)

| Symbol | Description |
|:-------|:------------|
| `float_dtype` | `(*tensors)` — Floating dtype that boolean / integer intermediates should adopt: the promoted dtype of the floating tensors among `tensors` (float64 inputs keep float64), else `torch.get_default_dtype()`. Never hard-codes float32. |
| `masked_reduction` | `(tensor, mask, reduction="mean")` — `mean` / `sum` / `max` / `min` / `none` ignoring `mask == False`; NaN at masked positions never leaks. |
| `masked_mean` | `(tensor, mask, dim=None, keepdim=False)` — Mean over the `True` entries of `mask`; masked NaN / Inf are ignored. |
| `masked_sum` | `(tensor, mask, dim=None, keepdim=False)` — Sum over the `True` entries of `mask`; masked NaN / Inf are ignored. |

---

## Target transforms (`utils.transform`)

| Symbol | Description |
|:-------|:------------|
| `TargetTransform` | Base class. `forward(y)` (also `__call__`), `inverse(z)`. |
| `IdentityTransform` | `f(y) = y`. |
| `LogTransform` | `(eps=1e-6)` — `f(y) = log(y + eps)`. |
| `BoxCoxTransform` | `(lam=0.0, eps=1e-6)` — `f(y) = ((y + eps)^λ − 1) / λ` for `λ ≠ 0` (`log` at `λ = 0`), evaluated with `expm1` so small `λ` does not cancel. |
| `SqrtTransform` | `(eps=1e-6)` — `f(y) = sqrt(y + eps)`. |
| `YeoJohnsonTransform` | `(lam=1.0)` — Yeo-Johnson signed-target transform (`expm1` / `log1p` forms). |
| `make_target_transform` | `(name, **kwargs)` — Factory: name → transform instance. |

---

## Validation (`utils.validation`)

NaN is rejected by `validate_positive` (`ValueError`). The other validators
(`validate_range`, `validate_quantile`, ...) are internal; see
[Internal helpers](#internal-helpers-not-re-exported).

| Symbol | Description |
|:-------|:------------|
| `validate_positive` | `(value, param_name, allow_zero=False)` — Assert `value > 0` (or `≥ 0`); NaN is rejected. |

---

## Internal helpers (not re-exported)

These helpers are internal plumbing. They are **not** re-exported from
`torchregress.utils` and are not part of the public API. Import them from their
submodule if needed, for example `from torchregress.utils.tensor_ops import convert_to_tensor`.
Some have a public equivalent elsewhere: `parse_heteroscedastic_output` in `torchregress.ensemble`,
`low_rank_output_dim` and `split_low_rank_gaussian_output` in `torchregress.losses`.

| Symbol | Submodule | Description |
|:-------|:----------|:------------|
| `normal_cdf` | `utils.distributions` | `(z)` — Standard-normal CDF `0.5 * erfc(-z / √2)`; keeps full relative precision in the lower tail. |
| `parse_heteroscedastic_output` | `utils.gaussian_output` | `(output)` — Accepts tuple / dict / concatenated-tensor layouts; returns `(mean, log_var)`. A 2-D `[batch, 2 * n_outputs]` tensor is split along dim 1; higher-rank tensors are split along the last dim, like `split_mean_log_variance`. |
| `low_rank_output_dim` | `utils.gaussian_output` | `(n_features, rank)` — Total output dim of a low-rank Gaussian head `(mean, cov_factor, cov_diag)`. |
| `split_low_rank_gaussian_output` | `utils.gaussian_output` | `(y_pred, n_features, rank)` — Split a low-rank output into `(mean, cov_factor, cov_diag)`. |
| `labels_to_levels` | `utils.ordinal` | `(target, num_classes)` — Convert class-index labels to cumulative binary levels `1[y > k]`; the dtype follows `float_dtype(target)`. |
| `class_probs_to_levels` | `utils.ordinal` | `(target_probs, *, class_dim=-1, eps=1e-8)` — Convert per-class PMF targets to cumulative levels `P(y > k)`. |
| `normalize_class_probs` | `utils.ordinal` | `(target_probs, *, class_dim=-1, eps=1e-8)` — Normalise non-negative class probabilities along `class_dim`. |
| `get_device` | `utils.pytorch_compat` | `(device_str=None)` — Resolve the requested device, falling back to the best available one. |
| `quantile_loss` | `utils.quantile` | `(y_pred, y_true, quantile)` — Functional pinball loss for a single quantile. |
| `multi_quantile_loss` | `utils.quantile` | `(y_pred, y_true, quantiles, quantile_weights=None)` — Functional pinball loss averaged over multiple quantiles. |
| `apply_mask` | `utils.tensor_ops` | `(tensor, mask)` — Zero out entries where `mask` is `False` (`torch.where`, so NaN / Inf at masked positions is dropped). |
| `convert_to_tensor` | `utils.tensor_ops` | `(x, *, dtype=None, device=None)` — Cast list / ndarray / tensor to `torch.Tensor`. |
| `ensure_batch_dim` | `utils.tensor_ops` | `(x)` — Add a leading batch dim if absent. |
| `prepare_cross_covariance` | `utils.tensor_ops` | `(cov_xy, n_dims_x, n_dims_y, device, dtype=None)` — Validate / build a `[Dy, Dx]` cross-covariance. |
| `prepare_model_input_for_gradients` | `utils.tensor_ops` | `(x)` — Enable autograd on inputs for Jacobian / Hessian computations. |
| `compute_model_gradients` | `utils.tensor_ops` | `(y_pred, x, n_features_y, create_graph=None)` — Per-sample Jacobian of predictions with respect to inputs. |
| `calculate_gaussian_nll` | `utils.tensor_ops` | `(residuals, var, eps=1e-8)` — Gaussian NLL per sample. `var` may match `residuals` (elementwise diagonal), be a per-sample variance `[B]` that is broadcast over the feature axis, or be a full covariance `[B, D, D]`. |
| `calculate_propagated_variance` | `utils.tensor_ops` | `(grad, sigma_x, sigma_y=None, sigma_xy=None)` — Propagate input uncertainty through the Jacobian: `J Σₓ Jᵀ (+ Σ_y + cross terms)`. |
| `check_tensor` | `utils.validation` | `(tensor, name="tensor", max_elements=200_000_000)` — Validate a tensor (type, size, NaN / Inf) and raise an informative error. |
| `validate_metric_inputs` | `utils.validation` | `(y_pred, y_true)` — Standard point-metric input checks. |
| `validate_quantile` | `utils.validation` | `(q)` — Assert `0 ≤ q ≤ 1` and return a tensor; NaN is rejected. |
| `validate_range` | `utils.validation` | `(value, min_value, max_value, param_name)` — Assert `min_value ≤ value ≤ max_value`; NaN is rejected. |
| `validate_reduction` | `utils.validation` | `(reduction, valid_reductions=None)` — Assert `mean` / `sum` / `none`. |
| `validate_sample_weight` | `utils.validation` | `(sample_weight, batch_size)` — Validate and flatten per-sample metric weights. |
| `validate_weights` | `utils.validation` | `(weights, batch_size, allow_none=True, *, flatten=False)` — Validate per-sample loss weights. |
| `validate_url` | `utils.security` | `(url, allowed_schemes=("http", "https"))` — URL scheme allowlist for downloads (used by example loaders). |
| `subsample_rows` | `utils.numpy_stats` | `(X, max_rows, *, random_state)` — Deterministic row subsampling to at most `max_rows` rows. |
| `winsorize` | `utils.numpy_stats` | `(X, clip_quantile)` — Clip values along axis 0 to the lower / upper quantiles. |

---

## OpenML relaxed (`utils.openml_relaxed`)

Offline-friendly OpenML ARFF loading that skips the stale-MD5 check. Numeric,
boolean, and numeric-coded nominal (for example `{0,1}`) columns are kept as
`float32`; rows with any non-finite value are dropped.

| Symbol | Description |
|:-------|:------------|
| `fetch_openml_regression_frame_skip_checksum` | `(*, data_id, target_column="target", download_timeout=600.0, json_timeout=120.0)` — Load `(X, y, tag)` float32 arrays for an OpenML dataset id without MD5 verification. |
| `fetch_openml_regression_with_sklearn_fallback` | `(*, data_id, name, version, target_column)` — Resolve a dataset by id or by `name` / `version` and load `(X, y, tag)`; kept under its historical name, it always uses the relaxed ARFF parser. |

---

## Method catalog (`method_catalog`)

Scriptable method metadata behind the [Method Catalog](../reports/method_catalog_generated.md)
and the [method-selection guide](../guide/method-selection.md); available as
`tr.method_catalog`.

| Symbol | Description |
|:-------|:------------|
| `MethodMetadata` | Frozen dataclass: `name`, `family`, `public_path`, `task_tags`, `maturity`, capability flags (`multimodal`, `multi_target`, `non_gaussian`, `epistemic`, `aleatoric`, `decomposition`, `calibration`, `ood_support`, `imbalance`, `noisy_features_eiv`; each `"yes"` / `"partial"` / `"no"`), and `notes`. |
| `list_methods` | `(*, family=None, task_tag=None, capability_filters=None, maturity=None)` — Cataloged methods as dict rows; capability filters are exact matches on `yes` / `partial` / `no`. |
| `get_method_metadata` | `(name)` — Metadata dict for one cataloged method. |
| `list_task_recommendations` | `()` — Per-task recommended starting point, strong alternatives, and notes. |
| `list_decision_workflow_steps` | `()` — Ordered decision-workflow questions with primary recommendations and caveats. |
| `list_comparative_evidence_rows` | `()` — Comparative-evidence rows (examples, comparison grade, fairness controls, metric coverage, gaps) per task. |

---

## Health check (`health`)

| Symbol | Description |
|:-------|:------------|
| `check_health` | `()` — Print versions and device, then run an import check, a tensor-op check, and a one-step training smoke test; exits with status 1 on failure. Import with `from torchregress.health import check_health`. |

---

## Quick example

```python
import torch
from torchregress.utils import (
    BoxCoxTransform, ipw_weights, masked_mean, validate_positive,
)

# Masked mean ignoring missing targets (NaN at masked positions does not leak)
y = torch.tensor([1.0, float("nan"), 3.0, 4.0])
y_pred = torch.tensor([1.5, 0.0, 2.0, 4.5])
mask = ~torch.isnan(y)
loss = masked_mean((y - y_pred) ** 2, mask)

# Box-Cox transform for positive targets
bxcx = BoxCoxTransform(lam=0.5)
y_pos = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_t = bxcx.forward(y_pos)     # transform
y_back = bxcx.inverse(y_t)    # invert

# IPW weights for causal reweighting
e = torch.tensor([0.7, 0.2, 0.6, 0.4, 0.9])
w = ipw_weights(e, clip_min=0.1, clip_max=0.9, normalize=True)

validate_positive(0.5, "scale")
```

## Next steps

- [Losses API](losses.md)
- [Metrics API](metrics.md)
- [Calibration API](calibration.md)
