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
| `flatten_adaptation_parameters` | `(groups)` — Flatten a `dict[name, Iterable[Parameter]]` into a deduplicated list. |

```python
from torchregress.test_time import (
    AdaptationBatch, SupportsPredictiveBatch, flatten_adaptation_parameters
)
assert isinstance(my_model, SupportsPredictiveBatch)
batch = AdaptationBatch(x=x_test, predictions=predictive_batch)
# Any dict of name -> iterable of parameters works
flat = flatten_adaptation_parameters(
    {"head": my_model.head.parameters(), "norm": my_model.norm.parameters()}
)
```

---

## Bayesian linear heads (`test_time.bayes`)

Conjugate Gaussian linear regression on **fixed features** (closed-form
posterior, online updates).

| Symbol | Description |
|:-------|:------------|
| `BayesianLinearHead` | `(in_features, out_features=1, *, fit_intercept=True, prior_mean=0.0, prior_precision=1.0, noise_variance=1.0, jitter=1e-6)` — Closed-form conjugate Gaussian linear regression. Posterior precision `Λ = Λ₀ + σ⁻² Φᵀ W Φ`, canonical `h = h₀ + σ⁻² Φᵀ (Wy)`. `fit(features, y, sample_weight=None)`, `reset_posterior()`, `predict(features, return_std=False, include_noise=True)`, `sample_weights(n_samples, generator=None)`. |
| `BayesianLinearHead.posterior_mean` | Posterior mean `[out_features, d_eff]`. |
| `BayesianLinearHead.posterior_covariance` | Posterior covariance `[d_eff, d_eff]` (Cholesky-solved). |
| `BayesianLinearHead.predictive_batch(features, *, include_noise=True)` | Returns a `PredictiveBatch` with `point`/`mean`/`std` and `extra` containing `epistemic_variance`, `aleatoric_variance`, `posterior_trace`, `n_observations_seen`. |
| `RecursiveBayesianHead` | Adds `forgetting_factor ∈ (0, 1]` and `partial_fit(features, y, sample_weight=None)` for streaming updates; `fit` performs a full reset + one-shot update. |

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
| `LabelShiftEMConfig` | Dataclass: `max_iter=100`, `tol=1e-6` (max per-class prior change), `eps=1e-8`, `loglik_tol=None` (optional: also stop once the mean target log-likelihood gains less than this per iteration). |
| `LabelShiftEstimate` | Frozen result: `source_prior`, `target_prior`, `iterations`, `converged`. |
| `apply_label_shift_correction` | `(probabilities, *, source_prior, target_prior, eps=1e-8)` — Posterior correction under prior ratio $p_{\text{tgt}}/p_{\text{src}}$. |
| `estimate_target_prior_em` | `(probabilities, *, source_prior, sample_weights=None, sample_size=None, random_state=0, config=None)` — EM target-prior estimate (Saerens–Latinne–Decaestecker 2002). Requires calibrated source posteriors (see below); classes with source prior `<= eps` are excluded and get target prior 0. |
| `PosteriorLabelShiftAdapter` | `(*, source_prior, sample_size=None, random_state=0, config=None)` — Reusable EM adapter (`source_prior` is required); `.estimate(probs, sample_weights=None)` returns a `LabelShiftEstimate`, `.transform(probs, target_prior=None)` reweights the posteriors (estimating the target prior first when none is given). |
| `GaussianLabelShiftConfig` | Dataclass: `n_bins=32`, `estimation_rows`, `top_fraction=0.5`, `reference_size=2048`, `seed=0`, `eps=1e-8`. |
| `gaussian_bin_edges_from_targets` | `(targets, n_bins)` — Quantile-based bin edges. |
| `gaussian_bin_probabilities` | `(mean, std, bin_edges, *, eps=1e-8)` — Discretize Gaussian predictions to bin probabilities. |
| `gaussian_moments_from_binned_probabilities` | `(probabilities, bin_edges, *, eps=1e-8)` — Reconstruct Gaussian `(mean, std)` from bin probs. |
| `correct_gaussian_predictions_for_label_shift` | `(*, mean, std, source_targets, features=None, config=None)` — End-to-end: discretize → EM estimate → backproject → corrected `(mean, std, metadata)`. |
| `estimate_target_prior_bbse` | `(probabilities_source, labels_source, probabilities_target, *, cond_threshold=1e8)` — Estimate target priors under label shift with BBSE (Black-Box Shift Estimation): invert the source confusion matrix against the target mean predicted-label distribution. |

EM iterates the Saerens et al. fixed point: with source posteriors $p_{ik} = p_s(y = k \mid x_i)$
and source prior $\pi$,

$$
\hat q^{(t+1)}_k = \frac{1}{n} \sum_{i=1}^{n}
\frac{p_{ik}\, \hat q^{(t)}_k / \pi_k}{\sum_j p_{ij}\, \hat q^{(t)}_j / \pi_j},
$$

which increases the target log-likelihood
$\frac{1}{n}\sum_i \log \sum_k p_{ik}\, q_k / \pi_k$ at every step.

!!! warning "EM needs calibrated posteriors; BBSE does not"
    The EM estimate is consistent only when `probabilities` are **calibrated source
    posteriors** whose source average equals `source_prior`
    ($\mathbb{E}_s[p_s(y = k \mid x)] = \pi_k$) and label shift holds at the level of the
    classes or bins ($p(x \mid y = k)$ shared by source and target). Binned Gaussians
    (`gaussian_bin_probabilities(mean, std, edges)`) qualify only if `N(mean, std)` is a
    calibrated predictive $p_s(y \mid x)$. A prediction that is a noisy copy of the label
    (`pred = y + noise`) with `std` the residual spread gives the *likelihood*
    $p(\text{pred} \mid y)$ instead: no shrinkage toward the prior, over-dispersed bins whose
    source average differs from the source prior, and EM stops at a biased fixed point.
    On such a synthetic Gaussian shift (5 quantile bins) EM reached
    prior TV 0.26 against 0.32 uncorrected, while BBSE, which only inverts a confusion matrix,
    reached 0.06; with calibrated posteriors and a bin-level shift EM is within 0.01 of the
    true prior (20 000 target rows).
    Calibrate the posteriors first, pass the **mean source posterior** as `source_prior`
    (which restores the fixed-point condition above), or use `estimate_target_prior_bbse`.
    On many overlapping bins the maximum-likelihood prior is an ill-posed deconvolution:
    iterating to `tol` fits sampling noise, so set `loglik_tol` (e.g. `1e-4`).

| # | Reference |
|:-:|:----------|
| 1 | Saerens, Latinne, Decaestecker, "Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure", *Neural Computation* 14(1), 2002. |
| 2 | Lipton, Wang, Smola, "Detecting and Correcting for Label Shift with Black Box Predictors" (ICML 2018). |
| 3 | Alexandari, Kundaje, Shrikumar, "Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation" (ICML 2020). |

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

## COSA Output Adaptation (`test_time.cosa`)

COSA-style (Conformal Output Space Adaptation) residual adaptation for point predictions
and uncertainty under delayed label observations.

| Symbol | Description |
|:-------|:------------|
| `DelayedLabelResidualAdapter` | `(base_model, *, ema_beta=0.1, scale_ema_beta=0.1)` — Tracks running EMA of residuals and variance inflation factors to adjust point, mean, std, and quantiles at test time. `.partial_fit(X, y)` updates state (the first update measures residuals against the uncorrected base predictions), `.predict_distribution(X)` returns adapted PredictiveBatch. |

```python
from torchregress.test_time import DelayedLabelResidualAdapter

adapter = DelayedLabelResidualAdapter(base_model, ema_beta=0.2)
# Under streaming protocol:
# 1. Predict first
adapted_batch = adapter.predict_distribution(X_t)
# 2. Observe delayed labels later
adapter.partial_fit(X_old, y_old)
```

---

## Optimal-transport conformal reweighting (`test_time.ot_conformal`)

Score-CDF matching reweighting for **classification-style nonconformity
scores** under non-exchangeable target shift, and weighted split-conformal regression.

| Symbol | Description |
|:-------|:------------|
| `OptimalTransportCoverageGap(n_grid=129)` | Diagnostics: `l2_cdf_gap`, `ks_max_abs` between uniform-weight calibration and target ECDFs. |
| `ScoreCDFReweighter(...)` | Learns simplex weights over calibration points by minimising the L₂ gap on a 1-D score grid. |
| `WeightedSplitConformalAdapter(alpha=0.1)` | Weighted split-conformal threshold for classification-style nonconformity scores. |
| `WeightedConformalRegressionAdapter` | `(alpha=0.1, classifier=None)` — Weighted split-conformal regression using classifier-based density ratio estimation. `.fit_density_ratio(X_cal, X_tgt)`, `.compute_density_ratios(X)`, `.calibrate(y_pred_cal, y_cal, X_cal, X_tgt)`, `.predict_interval(y_pred, X)` (returns infinite intervals when the calibration mass is too small for the requested `alpha`, per Tibshirani et al. 2019). |
| `weighted_split_classification_predictive_batch(...)` | Build a `PredictiveBatch` from a calibrated WeightedSplitConformalAdapter. |

```python
import torch
from torchregress.test_time import WeightedConformalRegressionAdapter

adapter = WeightedConformalRegressionAdapter(alpha=0.1)
# Fit density ratio estimator and calibrate scores
adapter.calibrate(y_pred_cal, y_cal, X_cal, X_tgt)
# Predict intervals under covariate shift
lower, upper = adapter.predict_interval(y_pred_test, X_test)
```

---

## Density-ratio weights & joint TTA (`test_time.shift_weights`, `test_time.joint_tta`)

| Symbol | Description |
|:-------|:------------|
| `DomainClassifierRatioEstimator` | Density-ratio `w(x) = p_target(x) / p_source(x)` estimated via a probabilistic domain classifier. `.fit(X_source, X_target)`, then `.weights(X)` / `.weights_for(X)`. The classifier odds are corrected by `n_source / n_target` before clipping, so the weights are normalised for unequal pool sizes. |
| `estimate_label_shift_weights` | Per-point importance weights under label shift via the BBSE-per-point identity from source/target posterior probabilities. |
| `JointTTAResult` | Frozen outcome of `JointDistributionalTTA.adapt_and_calibrate`: the adapted model, the fitted conformal calibrator, and a diagnostics dict. |
| `JointDistributionalTTA` | Adapt a distributional regressor at test time (feature alignment + filtered pseudo-label rounds), freeze, then recalibrate once with fixed importance weights. `.adapt_and_calibrate(model, X_cal_src, y_cal_src, X_target)`, `.predict_intervals(result, X_test)`. |

---

## Streaming evaluation (`test_time.benchmark`)

| Symbol | Description |
|:-------|:------------|
| `CausalTTAHarness` | Causal streaming evaluation harness for test-time adaptation protocols (predict → observe delayed labels → update). |

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
| `entropy_scores` | `(probabilities, *, eps=1e-8)` — Shannon entropy of normalised probabilities. |
| `confidence_scores` | `(probabilities)` — Max probability per row. |
| `pseudo_label_targets` | `(probabilities)` — `(argmax_labels, max_weights)` for self-training. |
| `select_high_confidence` | `(probabilities, *, min_confidence=None, max_entropy=None, top_fraction=None, min_count=1, scores=None)` — Composite selector with confidence / entropy / top-k gates. `scores` (one value per row, higher is more confident) replaces the maximum probability as the ranking key for `top_fraction` and the `min_count` fallback. |
| `LocalConsistencyConfig` | Dataclass: `k=5`, `temperature=1.0`, `reference_size`, `max_exact_rows=4096`, `query_chunk_size=2048`, `random_state`, `eps=1e-8`. |
| `local_consistency_weights` | `(features, probabilities, config=None)` — FTAT-style neighbourhood agreement (Bhattacharyya-like inner product of $\sqrt{p \cdot q}$), rescaled to mean 1. |

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
| `WeightedSubspaceMomentAligner` | `(*, rank=None, variance_threshold=0.95, target_sample_size=None, random_state=0, clip_quantile=None, max_scale_ratio=10.0, eps=1e-6)` — SSA-style low-rank alignment with **regression-significance weighting**. `.fit(X_source, y_source=None)`, `.transform(X_target)`, `.fit_transform(...)`. |
| `FeatureStatNormalizer` | `(*, target_sample_size=None, random_state=0, clip_quantile=None, max_scale_ratio=10.0, eps=1e-6)` — Low-risk per-feature mean/std alignment (no PCA). |

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
| `ParameterEMA` | `(decay=0.99)` — Exponential moving average over trainable parameters. `.initialize(model)`, `.update(model)`, `.copy_to(model)`. |

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

`fit_source` discretises the source predictive distributions on the support grid
(`n_support` points spanning the source labels plus `support_margin`) and uses their
**mean** as the source prior — the marginal the predictive posteriors integrate to, which
the EM prior estimate in `adapt_unlabeled_target` needs (see the warning above). EM runs on
the `top_fraction` most confident target rows and stops once the mean target
log-likelihood gains less than `1e-4` nats per row per iteration; the estimate is then
shrunk toward the source prior and ratio-clipped (`prior_transport_strength`,
`prior_ratio_clip`). Metadata reports `target_prior_raw` (EM), `target_prior`
(stabilised), `prior_source_target_tv`, `estimate_converged` and `transport_applied`.

!!! info "Confidence selection on fine grids"
    The `top_fraction` most confident rows are ranked on the **unrenormalised in-grid peak
    probability** (the maximum bin probability times the predictive mass inside the support
    grid). A predictive cut by the grid edge has its peak inflated by renormalisation, so
    ranking on the renormalised maximum favoured those edge rows on fine grids and the raw
    EM prior overshot the shift by up to ~0.3 |shift| at 256 bins; with the in-grid peak the
    error is below 0.1 |shift| at 16, 64 and 256 bins. Predictives with a constant spread
    carry no per-row confidence either way: `top_fraction=1.0` is then equivalent and
    uses every row.

The transport supports PPI (`mean` / `quantile` / `ols` estimands) and
dispatches conformal calibration to `cqr` / `cti` / `interval` / `split`
based on the predictive family. The conformal threshold is the exact split-conformal
order statistic `k = ceil((n + 1)(1 - alpha))` of the calibration scores (`+inf`, i.e. an
unbounded interval, when `k > n`). Outputs keep the floating dtype of the predictions
(float64 predictions stay float64).

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
