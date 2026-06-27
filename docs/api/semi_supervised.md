# Semi-supervised API

Complete reference for `torchregress.semi_supervised`. Every exported class, function, and dataclass is listed here. For conceptual background and worked examples, see the [Semi-supervised guide](../methods/semi_supervised.md).

→ **Related:** [Ensemble API](ensemble.md) · [Losses API](losses.md) · [Methods overview](../methods/index.md)

---

## Module overview

The semi-supervised module provides **teacher–student consistency** workflows
for regression. The core loop:

1. A teacher model (EMA or frozen) generates predictive views on unlabeled data.
2. A **consensus predictive batch** is built by aligning the views on a shared
   support grid and averaging their densities.
3. Per-sample **trust weights** are computed from inter-view disagreement,
   teacher uncertainty, or conformal interval width.
4. The student is trained with a weighted distributional pseudo-loss that
   matches its predictions to the consensus.

The module is designed to compose with any `PredictiveBatch`-emitting model —
Gaussian heads, quantile heads, MDN, binning heads, and density/support heads
are all supported through the common density-on-support representation.

Typical call pattern:
```python
trainer = TeacherStudentTrainer(
    optimizer=optimizer,
    supervised_loss_fn=my_supervised_loss,
    predictive_batch_fn=my_predictive_batch_fn,
    augment_fn=my_augmentation,
)
history = trainer.fit(model, labeled_loader, unlabeled_loader, epochs=10)
```

---

## Trainer classes

| Symbol | Description |
|:-------|:------------|
| `TeacherStudentTrainer` | Modular teacher–student consistency trainer for semi-supervised regression. Orchestrates labeled and unlabeled training steps. Custom sample-weighting policies can be injected for continuous heteroscedastic pseudo-labeling, conformal width gating, and target label shift prior correction. Constructor accepts `optimizer`, `supervised_loss_fn`, `predictive_batch_fn`, optional `augment_fn`, `unsupervised_loss_fn`, `sample_weight_fn`, and hyperparameters (`n_views`, `agreement_weight`, `ema_decay`, `n_support`, …). |
| `SelfAgreementTrainer` | Backward-compatible wrapper around `TeacherStudentTrainer` with the SAGE-Reg default weight/loss policies. Used for NeurIPS SAGE-Reg benchmarks. Exposes additional methods: `compute_agreement()`, `unsupervised_loss()`, and includes `mean_disagreement` in training history. |

---

## Composite loss

| Symbol | Description |
|:-------|:------------|
| `SAGERegLoss` | Composite supervised + weighted distributional pseudo-supervision loss. Inherits from `nn.Module`. `forward(supervised_loss, unlabeled_views)` returns `SAGERegOutput`. Internal `agreement()` method builds consensus, computes per-sample KL disagreement, converts to trust weights via `disagreement_to_weight`, and returns a `SAGERegAgreement`. Configurable: `tau` (weight sharpness), `agreement_weight` (pseudo-loss coefficient), `detach_weights`, `weight_power`, `hard_weight_threshold`, `batch_relative_mode`, `batch_trust_top_k`. |

---

## Output dataclasses

| Symbol | Fields | Description |
|:-------|:-------|:------------|
| `SAGERegAgreement` | `loss: Tensor`, `disagreement: Tensor`, `weights: Tensor`, `consensus: PredictiveBatch` | Distributional agreement statistics for one unlabeled batch. |
| `SAGERegOutput` | `total_loss: Tensor`, `supervised_loss: Tensor`, `agreement: SAGERegAgreement` | Composite supervised + self-agreement objective. |

---

## Consensus & agreement functions

These functions form the backbone of the pseudo-supervision pipeline.
They operate on sequences of `PredictiveBatch` views and produce a
single consensus prediction plus per-sample agreement scores.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `build_consensus_predictive_batch` | `(predictive_views, *, n_support=128, range_margin=0.05, gaussian_std_span=4.0, min_scale=1e-4, eps=1e-8) → PredictiveBatch` | Build the consensus predictive law for a set of stochastic views. Projects each view onto a shared support grid, averages densities, and returns mean/std/support/density via `PredictiveBatch`. |
| `predictive_agreement_score` | `(predictive_views, *, ..., reduction="none") → Tensor` | Average pairwise symmetric KL divergence across predictive views. Requires at least two views. Supports `"none"`, `"mean"`, `"sum"` reduction. |
| `perturbation_instability_score` | `(predictive_views, *, ..., reduction="none") → Tensor` | Alias for `predictive_agreement_score`. Compute representation/prediction instability under augmentations. |

---

## Pseudo-loss

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `distributional_pseudo_loss` | `(student_prediction, consensus_prediction, *, sample_weights=None, reduction="mean", n_support=128, …) → Tensor` | Backbone-aware pseudo-supervision loss on a shared predictive representation. Automatically selects the appropriate loss pathway based on the student's predictive head type: Gaussian NLL for `(mean, std)` heads, cross-entropy for binning heads, or general cross-entropy on the common density support. |

$$\\mathcal{L}_{\\text{pseudo}} = -\\int p_{\\text{consensus}}(y) \\log p_{\\text{student}}(y) \\, dy$$

---

## Weighting helpers

These functions convert disagreement, uncertainty, or conformal width into
per-sample trust weights for pseudo-label filtering.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `disagreement_to_weight` | `(disagreement, tau, *, power=1.0, hard_weight_threshold=None, batch_relative_mode=None, batch_trust_top_k=None, eps=1e-8) → Tensor` | Convert disagreement scores into trust weights: `w = exp(-d / τ)^p`. Supports z-score batch-relative normalisation, top-k hard gating, and hard threshold masking. |
| `uncertainty_to_weight` | `(predictive_batch, tau, *, power=1.0, …) → Tensor` | Convert teacher predictive uncertainty (std) into trust weights. Averages std over output dimensions if multivariate, then delegates to `disagreement_to_weight`. |
| `conformal_width_to_weight` | `(lower, upper, tau=None, *, threshold=None) → Tensor` | Compute pseudo-label weights or masks based on conformal prediction interval width. Returns soft weights `exp(-width / τ)` when `tau` is set, or binary mask `(width ≤ threshold)` when `threshold` is set. |

$$w_i = \\exp\\!\\left(-\\frac{d_i}{\\tau}\\right)^p \\quad \\text{with optional hard threshold } w_i \\geq w_{\\min}$$

---

## Next steps

- [Semi-supervised methods guide](../methods/semi_supervised.md) — conceptual walkthrough and usage
- [Teacher–student demo](../examples/semi_supervised_regression_comparison.py)
- [Teacher–student demo](../examples/semi_supervised_regression_comparison.py)
- [Losses API](losses.md) — supervised loss function catalogue
- [Ensemble API](ensemble.md) — ensemble methods that pair well with pseudo-label training
