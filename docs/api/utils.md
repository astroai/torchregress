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
| `quantiles_to_density_grid(quantiles, quantile_levels, *, n_support=200, range_margin=0.05)` | Convert monotone quantile predictions to a regular density grid on a per-row support. |
| `bars_to_density_grid(bar_logits, bin_edges, *, n_support=200, range_margin=0.05)` | Convert piecewise-constant bar distributions to a regular density grid. |
| `samples_to_density_grid(samples, *, n_support=200, range_margin=0.05)` | Convert scalar predictive samples to a regular density grid. |

```python
import numpy as np
from torchregress.prediction import PredictiveBatch, samples_to_density_grid

batch = PredictiveBatch(mean=mu, std=sigma, samples=samples, extra={"family": "gaussian"})
density_batch = batch.with_density(n_support=256)
```

---

## Augmentations (`utils.augment`)

| Symbol | Description |
|:-------|:------------|
| `Augmentation` | Base class for input augmentations. |
| `GaussianNoise(std)` | Adds `N(0, std²)` noise. |
| `Adversarial(model, loss_fn, epsilon, steps, alpha, probability, random_start)` | FGSM/PGD-style adversarial augmentation. |
| `MixUp(alpha)` | Mixup augmentation for regression. |
| `FeatureMask(p, mask_value=0.0)` | Per-feature masking augmentation. |
| `EnsemblePerturbationAugmenter` | Input perturbation designed to expose ensemble disagreement. |

---

## Distributions (`utils.distributions`)

| Symbol | Description |
|:-------|:------------|
| `normal_cdf(x)` | Standard-normal CDF implemented as a PyTorch op. |

---

## Gaussian output helpers (`utils.gaussian_output`)

| Symbol | Description |
|:-------|:------------|
| `split_mean_log_variance(out)` | Split `(B, 2D)` output into `(mean, log_var)`. |
| `variance_from_logvar(log_var, min_logvar=-8.0, max_logvar=6.0, eps=1e-8)` | Numerically-stable `exp(log_var)` with clipping. |
| `parse_heteroscedastic_output(out)` | Accepts tuple / dict / concatenated-tensor layouts; returns `(mean, log_var)`. |
| `low_rank_output_dim(target_dim, cov_rank)` | Total output dim of a low-rank Gaussian head `(mean, cov_factor, cov_diag)`. |
| `split_low_rank_gaussian_output(out, cov_rank, target_dim)` | Split a low-rank output into `(mean, cov_factor, cov_diag)`. |

---

## Label utilities (`utils.labels`)

| Symbol | Description |
|:-------|:------------|
| `encode_onehot(y, num_classes)` | One-hot encode integer class labels. |
| `decode_onehot(probs)` | Argmax decoding with optional threshold. |
| `label_smoothing(probs, alpha)` | Smooth a one-hot / hard label by `α`. |
| `soft_to_hard_labels(probs, dim=-1)` | Convert soft probabilities to hard labels. |
| `combine_binary_average(probs_a, probs_b)` | Average of two binary probability vectors. |
| `combine_binary_weighted_average(probs_a, probs_b, weight_a)` | Weighted average of two binary probability vectors. |

---

## Ordinal utilities (`utils.ordinal`)

| Symbol | Description |
|:-------|:------------|
| `labels_to_levels(y, n_levels)` | Convert level indices to a `\[0,1\]` level target. |
| `class_probs_to_levels(probs, support)` | Expected level from class probabilities. |
| `cumulative_probs_to_pmf(cum_probs)` | Convert cumulative probabilities to PMF. |
| `cumulative_logits_to_pmf(cum_logits)` | Convert cumulative logits to PMF. |
| `normalize_class_probs(probs, dim=-1, eps=1e-8)` | Normalise probabilities to a valid simplex. |
| `ordinal_predict(probs, support)` | Expected ordinal value with optional support. |

---

## Propensity utilities (`utils.propensity`)

| Symbol | Description |
|:-------|:------------|
| `ipw_weights(t, propensity, clip=0.05, normalize=True)` | IPW weights `w = t / e + (1 − t) / (1 − e)` with trimming and normalisation. |

---

## PyTorch compatibility (`utils.pytorch_compat`)

| Symbol | Description |
|:-------|:------------|
| `convert_reduction_type(reduction)` | Validate and standardise a reduction string. |
| `convert_to_pytorch_loss(name)` | Map a name like `"mse"` to `nn.MSELoss` etc. |
| `extract_output_size(out)` | Best-effort shape inference for a model output. |
| `set_seed(seed)` | Seed Python + NumPy + PyTorch (CPU + CUDA). |
| `set_all_seeds(seed)` | Same as `set_seed`. |
| `get_device(prefer_cuda=True)` | Pick the best available device. |

---

## Quantile utilities (`utils.quantile`)

| Symbol | Description |
|:-------|:------------|
| `quantile_loss(y_pred, y, tau)` | Functional pinball loss for a single quantile. |
| `multi_quantile_loss(y_pred, y, quantiles)` | Functional pinball for multiple quantiles. |

---

## Reduction (`utils.reduction`)

| Symbol | Description |
|:-------|:------------|
| `reduce_per_sample(losses, mask=None, weights=None, reduction="mean")` | Apply mask / weights / reduction consistently to a per-sample loss tensor. |

---

## Scaling / hardware (`utils.scaling`)

| Symbol | Description |
|:-------|:------------|
| `GradientAccumulation(steps)` | Iterate effective batches over smaller `DataLoader` steps. |
| `StandardScaler` | Fit / transform a `StandardScaler`-style normaliser. |
| `compile_model(model, **kwargs)` | One-liner for `torch.compile`. |

---

## Semi-supervised (`utils.semisupervised`)

| Symbol | Description |
|:-------|:------------|
| `generate_pseudo_labels(model, x_unlabeled, threshold=…)` | Confidence-thresholded pseudo labels. |
| `update_ema_teacher_(student, teacher, decay)` | In-place EMA teacher update. |

---

## Tensor ops (`utils.tensor_ops`)

| Symbol | Description |
|:-------|:------------|
| `apply_mask(x, mask, value=0.0)` | Element-wise masking utility. |
| `convert_to_tensor(x, dtype=None, device=None)` | Cast list / ndarray / tensor to `torch.Tensor`. |
| `ensure_batch_dim(x)` | Add a leading batch dim if absent. |
| `masked_reduction(x, mask, reduction="mean")` | Reduction ignoring `mask == False`. |
| `masked_mean(x, mask, dim=None)` | Mean ignoring masked values. |
| `masked_sum(x, mask, dim=None)` | Sum ignoring masked values. |
| `prepare_param(x, target_shape)` | Broadcast / tile a parameter tensor. |
| `prepare_sigma(sigma, target_dim)` | Build a positive σ from scalar / 1D / 2D input. |
| `prepare_covariance(cov, dim, eps=1e-6)` | Build a PSD `[D, D]` covariance. |
| `prepare_cross_covariance(cov, dim)` | Build a `[D, D]` cross-covariance. |
| `prepare_model_input_for_gradients(x, requires_grad=True)` | Enable autograd on inputs for Jacobian / Hessian. |
| `batched_linalg_solve(A, B)` | Solve `A X = B` in batches with a `pinv` fallback. |
| `standardize(x, dim=0, eps=1e-5)` | Standardise to mean 0, std 1. |
| `unstandardize(x, mean, std)` | Inverse of `standardize`. |
| `compute_model_gradients(model, x, y, loss_fn, create_graph=False)` | Per-sample gradients. |
| `calculate_gaussian_nll(mean, log_var, target, reduction="mean")` | Diagonal Gaussian NLL. |
| `calculate_propagated_variance(jvp, cov)` | `cov_out = J @ cov_in @ Jᵀ`. |

---

## Target transforms (`utils.transform`)

| Symbol | Description |
|:-------|:------------|
| `TargetTransform` | Base class. `forward(y)`, `inverse(z)`. |
| `IdentityTransform` | `f(y) = y`. |
| `LogTransform` | `f(y) = log(y)`, defined for `y > 0`. |
| `BoxCoxTransform(lmbda)` | `f(y) = (yᵏ − 1) / k` for `k ≠ 0`. |
| `SqrtTransform` | `f(y) = sqrt(y)`. |
| `YeoJohnsonTransform(lmbda)` | `Yeo-Johnson` signed-target transform. |
| `log_transform`, `log_inverse` | Functional form. |
| `boxcox_transform`, `boxcox_inverse` | Functional form. |
| `sqrt_transform`, `sqrt_inverse` | Functional form. |
| `yeojohnson_transform`, `yeojohnson_inverse` | Functional form. |
| `make_target_transform(name, **kwargs)` | Factory: name → transform instance. |

---

## Validation (`utils.validation`)

| Symbol | Description |
|:-------|:------------|
| `check_tensor(x, name="x")` | Validate tensor, raise informative error. |
| `validate_batch_consistency(*tensors)` | Assert all tensors share batch dim. |
| `validate_integer(x, name)` | Assert `x` is an integer. |
| `validate_metric_inputs(y_pred, y)` | Standard point-metric input checks. |
| `validate_positive(x, name)` | Assert `x > 0`. |
| `validate_quantile(tau, name)` | Assert `0 < tau < 1`. |
| `validate_range(x, low, high, name)` | Assert `low ≤ x ≤ high`. |
| `validate_reduction(reduction)` | Assert `mean` / `sum` / `none`. |
| `validate_same_device(*tensors)` | Assert tensors are on the same device. |
| `validate_sample_weight(w, y)` | Validate a sample-weights tensor. |
| `validate_shape(x, shape, name)` | Assert tensor has expected shape. |
| `validate_weights(w, y)` | Validate per-sample weights. |

---

## Security (`utils.security`)

| Symbol | Description |
|:-------|:------------|
| `validate_url(url, allowed_schemes=("https",))` | URL allowlist for downloads (used by example loaders). |

---

## NumPy stats (`utils.numpy_stats`)

| Symbol | Description |
|:-------|:------------|
| `subsample_rows(X, n=None, random_state=0)` | Deterministic row subsampling. |
| `winsorize(X, quantile=None)` | Winsorise extreme values. |

---

## OpenML relaxed (`utils.openml_relaxed`)

| Symbol | Description |
|:-------|:------------|
| `fetch_openml_relaxed(...)` | `sklearn.datasets.fetch_openml` with safer defaults and robust caching for offline / unstable OpenML access. |

---

## Quick example

```python
import torch
from torchregress.utils import (
    convert_to_tensor, masked_mean, split_mean_log_variance,
    BoxCoxTransform, ipw_weights, validate_positive,
)

# Masked mean ignoring missing targets
y = torch.tensor([1.0, float("nan"), 3.0, 4.0])
mask = ~torch.isnan(y)
loss = masked_mean((y - y_pred) ** 2, mask)

# Box-Cox transform for positive targets
bxcx = BoxCoxTransform(lmbda=0.5)
y_t = bxcx.forward(y)         # transform
y_back = bxcx.inverse(y_t)    # invert

# IPW weights for causal reweighting
t = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
e = torch.tensor([0.7, 0.2, 0.6, 0.4, 0.9])
w = ipw_weights(t, e, clip=0.1, normalize=True)
```

## Next steps

- [Losses API](losses.md)
- [Metrics API](metrics.md)
- [Calibration API](calibration.md)


## Scaling & Hardware
