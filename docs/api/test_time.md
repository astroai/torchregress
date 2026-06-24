# Test-Time Adaptation API

Complete reference for `torchregress.test_time`. This package provides
**model-agnostic** test-time adaptation and uncertainty utilities for
regression: Bayesian linear heads, label-shift correction, OT-based conformal
reweighting, feature alignment, EMA ensembling, and confidence-based sample
selection. Every primitive operates on `torch.Tensor` or `numpy.ndarray`
inputs and returns a [`PredictiveBatch`](utils.md) or
self-describing dataclass (see the
[Predictive containers](utils.md) section in the Utilities API).

For background, see [Bayesian Linear Regression](../methods/test-time/bayesian-linear-regression.md),
[Optimal Transport Conformal](../methods/test-time/ot-shift-conformal.md), and
[Shift Adaptation Roadmap](../methods/test-time/regression-shift-adaptation-roadmap.md). The
shared `PredictiveBatch` container is documented in the
[Utilities API — Predictive containers](utils.md) section.

---

## Adaptation interfaces (`test_time.base`)

| Symbol | Description |
|:-------|:------------|
| `AdaptationBatch` | Frozen dataclass for unlabeled target-time inputs — `x`, `predictions: PredictiveBatch`, `representations`, `sigma_x`. |
| `SupportsPredictiveBatch` | `Protocol` for models exposing `predict_distribution(X, **kwargs) -> PredictiveBatch`. |
| `SupportsRepresentation` | `Protocol` for models exposing `representation_dict(x) -> dict[str, Tensor]` (feature access). |
| `SupportsAdaptationParameters` | `Protocol` for models exposing `adaptation_parameter_groups() -> dict[str, list[Parameter]]` (TTA-targeted params). |
| `flatten_adaptation_parameters(groups)` | Flatten a `dict[name, Iterable[Parameter]]` into a deduplicated list. |

```python
from torchregress.test_time import (
    AdaptationBatch, SupportsPredictiveBatch, flatten_adaptation_parameters
)
assert isinstance(my_model, SupportsPredictiveBatch)
batch = AdaptationBatch(x=x_test, predictions=predictive_batch)
flat = flatten_adaptation_parameters(my_model.adaptation_parameter_groups())
```

---

## Bayesian linear heads (`test_time.bayes`)

Conjugate Gaussian linear regression on **fixed features** (closed-form
posterior, online updates).

| Symbol | Description |
|:-------|:------------|
| `BayesianLinearHead(in_features, out_features=1, *, fit_intercept=True, prior_mean=0.0, prior_precision=1.0, noise_variance=1.0, jitter=1e-6)` | Closed-form conjugate Gaussian linear regression. Posterior precision `Λ = Λ₀ + σ⁻² Φᵀ W Φ`, canonical `h = h₀ + σ⁻² Φᵀ (Wy)`. `fit(features, y, sample_weight=None)`, `reset_posterior()`, `predict(features, return_std=False, include_noise=True)`, `sample_weights(n_samples, generator=None)`. |
| `BayesianLinearHead.posterior_mean` | Posterior mean `[out_features, d_eff]`. |
| `BayesianLinearHead.posterior_covariance` | Posterior covariance `[d_eff, d_eff]` (Cholesky-solved). |
| `BayesianLinearHead.predictive_batch(features, *, include_noise=True)` | Returns a `PredictiveBatch` with `point`/`mean`/`std` and `extra` containing `epistemic_variance`, `aleatoric_variance`, `posterior_trace`, `n_observations_seen`. |
| `RecursiveBayesianHead(...)` | Adds `forgetting_factor ∈ (0, 1]` and `partial_fit(features, y, sample_weight=None)` for streaming updates; `fit` performs a full reset + one-shot update. |

**Reference:** Bishop, *Pattern Recognition and Machine Learning* (2006),
§3.3 (conjugate Bayesian linear regression).

```python
import torch
from torchregress.test_time import BayesianLinearHead

phi = torch.randn(64, 5)        # fixed features
y   = torch.randn(64, 1)
head = BayesianLinearHead(in_features=5, out_features=1, noise_variance=0.1)
head.fit(phi, y)
batch = head.predictive_batch(phi[:8])   # mean / std / epistemic / aleatoric
samples = head.sample_weights(n_samples=16)   # \[16, 1, 6\]  (with intercept)
```

---

## Label-shift correction (`test_time.label_shift`)

EM-based target-prior estimation and Gaussian predictive adjustment under
**marginal label shift** $p_{\text{target}}(y) \neq p_{\text{source}}(y)$.

| Symbol | Description |
|:-------|:------------|
| `LabelShiftEMConfig` | Dataclass: `max_iter=100`, `tol=1e-6`, `eps=1e-8`. |
| `LabelShiftEstimate` | Frozen result: `source_prior`, `target_prior`, `iterations`, `converged`. |
| `apply_label_shift_correction(probabilities, *, source_prior, target_prior, eps=1e-8)` | Posterior correction under prior ratio $p_{\text{tgt}}/p_{\text{src}}$. |
| `estimate_target_prior_em(probabilities, *, source_prior=None, sample_weights=None, sample_size=None, random_state=0, config=None)` | EM target-prior estimate (Lipton–Wang–Smola 2018). |
| `PosteriorLabelShiftAdapter(*, source_prior=None, sample_size=None, random_state=0, config=None)` | Reusable adapter; `.estimate(probs)`, `.transform(probs, target_prior=...)`, `.fit_transform(probs)`. |
| `GaussianLabelShiftConfig` | Dataclass: `n_bins=32`, `estimation_rows`, `top_fraction=0.5`, `reference_size=2048`, `seed=0`, `eps=1e-8`. |
| `gaussian_bin_edges_from_targets(targets, n_bins)` | Quantile-based bin edges. |
| `gaussian_bin_probabilities(mean, std, bin_edges, *, eps=1e-8)` | Discretize Gaussian predictions to bin probabilities. |
| `gaussian_moments_from_binned_probabilities(probabilities, bin_edges, *, eps=1e-8)` | Reconstruct Gaussian `(mean, std)` from bin probs. |
| `correct_gaussian_predictions_for_label_shift(*, mean, std, source_targets, features=None, config=None)` | End-to-end: discretize → EM estimate → backproject → corrected `(mean, std, metadata)`. |

**Reference:** Lipton, Wang, Smola, "Detecting and Correcting for Label Shift
with Black Box Predictors" (ICML 2018).

```python
import numpy as np
from torchregress.test_time import correct_gaussian_predictions_for_label_shift

mean, std, meta = correct_gaussian_predictions_for_label_shift(
    mean=test_mean, std=test_std, source_targets=train_y,
    features=test_x,                           # optional, enables local consistency weights
)
print(meta["estimate_converged"], meta["selected_rows"])
```

---

## Optimal-transport conformal reweighting (`test_time.ot_conformal`)

Score-CDF matching reweighting for **classification-style nonconformity
scores** under non-exchangeable target shift. v1 is a lightweight surrogate
of OT reweighting (no external OT solver).

| Symbol | Description |
|:-------|:------------|
| `OptimalTransportCoverageGap(n_grid=129)` | Diagnostics: `l2_cdf_gap`, `ks_max_abs`, `n_calibration`, `n_target` between uniform-weight calibration and target ECDFs. |
| `ScoreCDFReweighter(*, score_mode="classification", objective="weighted_cdf", weight_parameterization="free", entropy_penalty=1e-3, n_grid=129, n_steps=200, learning_rate=0.05)` | Learns simplex weights over calibration points by minimising the L₂ gap on a 1-D score grid (Adam). `.fit(calibration_scores, target_unlabeled_scores)`. Exposes `.weights_`, `.objective_value_`, `.diagnostics_["ess_inv_square", "cdf_l2_on_grid"]`. |
| `WeightedSplitConformalAdapter(alpha=0.1)` | Weighted split-conformal threshold using `torchregress.losses.conformal._weighted_quantile`. `.calibrate(calibration_scores, calibration_weights)`, `.predict_from_test_scores(candidate_scores)`, `.coverage_diagnostics(...)`. |
| `weighted_split_classification_predictive_batch(adapter, candidate_scores, *, gap_diagnostics=None, calibration_ess_inv_square=None)` | Build a `PredictiveBatch` with `point`/set-size and `extra` containing `label_inclusion_mask`, `alpha`, `threshold`, optional `shift_gap_diagnostics`. |

```python
import torch
from torchregress.test_time import ScoreCDFReweighter, WeightedSplitConformalAdapter

rw = ScoreCDFReweighter(n_steps=200).fit(cal_scores, tgt_scores)
adapter = WeightedSplitConformalAdapter(alpha=0.1).calibrate(cal_scores, rw.weights_)
mask = adapter.predict_from_test_scores(test_candidate_scores)   # [B, K] bool
```

---

## Shift calibration primitives (`test_time.calibration`)

Backward-compatible re-export of the variance-inflating temperature calibrator
used by `ShiftFactoredPredictiveTransport` and any other shift-aware pipeline.

| Symbol | Description |
|:-------|:------------|
| `RepresentationShiftInflator` | Re-export of `torchregress.calibration.shift.RepresentationShiftInflator`. Scales the per-example predicted std by a feature-shift-driven temperature $T \in [T_{\min}, T_{\max}]$ (configurable base, slope, ceiling, and optional Winsorization quantile). |

```python
from torchregress.test_time import RepresentationShiftInflator

cal = RepresentationShiftInflator(base_temperature=1.0, slope=0.2,
                                    max_temperature=2.0, clip_quantile=0.05)
cal.fit(source_features)                # learns per-source calibration
std_calibrated = cal.calibrate_std(test_std, test_features)
```

---

## Confidence and selection (`test_time.selection`)

| Symbol | Description |
|:-------|:------------|
| `entropy_scores(probabilities, *, eps=1e-8)` | Shannon entropy of normalised probabilities. |
| `confidence_scores(probabilities)` | Max probability per row. |
| `pseudo_label_targets(probabilities)` | `(argmax_labels, max_weights)` for self-training. |
| `select_high_confidence(probabilities, *, min_confidence=None, max_entropy=None, top_fraction=None, min_count=1)` | Composite selector with confidence / entropy / top-k gates. |
| `LocalConsistencyConfig` | Dataclass: `k=5`, `temperature=1.0`, `reference_size`, `max_exact_rows=4096`, `query_chunk_size=2048`, `random_state`, `eps=1e-8`. |
| `local_consistency_weights(features, probabilities, config=None)` | FTAT-style neighbourhood agreement (Bhattacharyya-like inner product of $\sqrt{p \cdot q}$), rescaled to mean 1. |

```python
import numpy as np
from torchregress.test_time import local_consistency_weights, select_high_confidence

w = local_consistency_weights(features, probabilities)         # [B] weights
mask = select_high_confidence(probabilities, top_fraction=0.5, min_count=32)
```

---

## Feature / subspace alignment (`test_time.subspace`)

| Symbol | Description |
|:-------|:------------|
| `SubspaceAlignmentState` | Frozen dataclass: `source_mean`, `target_mean`, `source_scale`, `target_scale`, `components`, `feature_weights`, `rank`. |
| `WeightedSubspaceMomentAligner(*, rank=None, variance_threshold=0.95, target_sample_size=None, random_state=0, clip_quantile=None, max_scale_ratio=10.0, eps=1e-6)` | SSA-style low-rank alignment with **regression-significance weighting**. `.fit(X_source, y_source=None)`, `.transform(X_target)`, `.fit_transform(...)`. |
| `FeatureStatNormalizer(*, target_sample_size=None, random_state=0, clip_quantile=None, max_scale_ratio=10.0, eps=1e-6)` | Low-risk per-feature mean/std alignment (no PCA). |

**Reference:** Fernando, Habrard, Sebban, Tuytelaars, "Unsupervised Visual
Domain Adaptation Using Subspace Alignment" (ICCV 2013).

```python
import numpy as np
from torchregress.test_time import WeightedSubspaceMomentAligner

aligner = WeightedSubspaceMomentAligner(rank=10, variance_threshold=0.95)
aligner.fit(X_train, y_train)
X_test_aligned = aligner.transform(X_test)
```

---

## Dynamic ensembling (`test_time.dynamic`)

| Symbol | Description |
|:-------|:------------|
| `ParameterEMA(decay=0.99)` | Exponential moving average over trainable parameters. `.initialize(model)`, `.update(model)`, `.copy_to(model)`. |

```python
from torchregress.test_time import ParameterEMA
ema = ParameterEMA(decay=0.99)
ema.initialize(model)
for x, y in loader:
    ...
    ema.update(model)
ema.copy_to(model)   # evaluate with EMA weights
```

---

## Shift-factored predictive transport (`test_time.transport`)

End-to-end orchestrator: combines prior transport, feature alignment,
uncertainty inflation, and conformal calibration for predictive batches
under target shift.

| Symbol | Description |
|:-------|:------------|
| `ShiftFactoredTransportConfig` | Comprehensive dataclass: `n_support=256`, `support_margin=0.05`, `alpha=0.1`, `top_fraction=0.5`, `min_selection_count=16`, `local_consistency_k=5`, `prior_estimation_rows`, `prior_transport_strength=0.5`, `prior_ratio_clip=2.0`, `prior_transport_requires_convergence=True`, `prior_transport_min_selected_fraction`, `prior_transport_max_prior_tv`, `random_state=0`, `enable_alignment=True`, `allow_input_alignment_rerun=False`, `enable_uncertainty_inflation=True`, `uncertainty_base_temperature=1.0`, `uncertainty_slope=0.2`, `uncertainty_max_temperature=2.0`, `uncertainty_clip_quantile=0.05`, `gaussian_conformal_uses_native_interval=True`, `eps=1e-8`. |
| `ShiftFactoredTransportState` | Frozen dataclass: `source_support`, `source_prior`, `source_targets`, `source_inputs`, `source_representations`, `last_target_prior`, `conformal_method`, `metadata`. |
| `ShiftFactoredPredictiveTransport(config=None)` | `.fit_source(source_predictions, source_targets, *, source_inputs=None, source_representations=None)`, `.adapt_unlabeled_target(*, target_predictions=None, target_inputs=None, target_representations=None, predictor=None)`, `.calibrate_target(calibration_predictions, calibration_targets, *, method=None)`, `.predict(...)`, `.apply_conformal(batch)`, `.ppi_target_ci(estimand, labeled_targets, labeled_predictions, unlabeled_predictions, *, x_labeled=None, x_unlabeled=None, q=None, alpha=0.1, n_boot=2000, seed=None)`. |

The transport supports PPI (`mean` / `quantile` / `ols` estimands) and
dispatches conformal calibration to `cqr` / `cti` / `interval` / `split`
based on the predictive family.

```python
import numpy as np
from torchregress.test_time import ShiftFactoredPredictiveTransport, ShiftFactoredTransportConfig
from torchregress.prediction import PredictiveBatch

src_pb = PredictiveBatch(mean=src_mean, std=src_std)
config = ShiftFactoredTransportConfig(alpha=0.1, enable_alignment=True)
transport = ShiftFactoredPredictiveTransport(config)
transport.fit_source(src_pb, source_targets=src_y, source_inputs=src_x)
adapted = transport.adapt_unlabeled_target(target_predictions=tgt_pb, target_inputs=tgt_x)
calibrated = transport.calibrate_target(cal_pb, cal_y)  # fits conformal threshold
final = transport.predict(target_predictions=tgt_pb, target_inputs=tgt_x)
```

---

## Quick example

```python
import numpy as np
import torch
from torchregress.test_time import (
    BayesianLinearHead, ScoreCDFReweighter, WeightedSplitConformalAdapter,
    WeightedSubspaceMomentAligner, ParameterEMA,
)

# 1. Closed-form Bayesian linear regression on fixed features
head = BayesianLinearHead(in_features=8, out_features=1, noise_variance=0.1)
head.fit(features, labels)
batch = head.predictive_batch(query_features)   # mean / std / epistemic / aleatoric

# 2. OT-style conformal reweighting under target shift
rw = ScoreCDFReweighter().fit(cal_scores, tgt_scores)
adapter = WeightedSplitConformalAdapter(alpha=0.1).calibrate(cal_scores, rw.weights_)
mask = adapter.predict_from_test_scores(test_scores)

# 3. Feature alignment
aligner = WeightedSubspaceMomentAligner(rank=8)
aligner.fit(X_train, y_train)
X_test_aligned = aligner.transform(X_test)

# 4. EMA over model parameters
ema = ParameterEMA(decay=0.999)
ema.initialize(model)
for x, y in loader: ... ; ema.update(model)
ema.copy_to(model)
```

## Next steps

- [Bayesian Linear Regression](../methods/test-time/bayesian-linear-regression.md)
- [Optimal Transport Conformal](../methods/test-time/ot-shift-conformal.md)
- [Shift Adaptation Roadmap](../methods/test-time/regression-shift-adaptation-roadmap.md)


## Test-Time Adaptation Details

### ScoreCDFReweighter

Learns target weights by minimising the Wasserstein/CDF gap between source and target predictions.

### WeightedSplitConformalAdapter

Weighted split-conformal calibration using per-sample weights (e.g. from `ScoreCDFReweighter`).

### BayesianLinearHead

Conjugate Gaussian Bayesian linear head for closed-form posterior updates and predictive uncertainty at test time.

### ShiftFactoredPredictiveTransport

Orchestrates covariate shift correction, feature alignment, and conformal calibration.
