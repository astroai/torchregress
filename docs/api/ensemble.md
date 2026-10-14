# Ensemble API

Complete reference for `torchregress.ensemble`. Every class, layer, and helper
is listed here. For background and decision guidance, see
[Ensembles overview](../methods/ensemble/index.md).

---

## Base classes

| Symbol | Description |
|:-------|:------------|
| `BaseEnsembleModel` | Foundation class. Each call to `forward(x)` runs every member and returns a stacked tensor `[M, batch, ...]` (a list only for non-tensor member outputs). Provides `predict` (mean + variance) and `predict_full_covariance`, which run members in eval mode and restore their previous modes. `base_seed` seeds member initialisation inside a forked RNG (the global RNG state is unchanged). |
| `EnsembleFitConfig` | Dataclass holding training options: `epochs`, `lr`, `optimizer_cls`, `optimizer_kwargs`, `optimizer_factory`, `verbose`, `device`. Built by `fit()`. |

**`BaseEnsembleModel.fit(...)`** trains each member independently with the
given criterion (no adversarial training term). `optimizer_factory(model) -> Optimizer`
lets you supply custom per-member optimizers (e.g. AdamW + Muon).

---

## Mean-only and heteroscedastic deep ensembles

| Symbol | Output | Predicts |
|:-------|:-------|:---------|
| `BaseEnsembleModel` (deep ensemble) | `dict` | `mean`, `variance` (epistemic variance from member disagreement) |
| `HeteroscedasticEnsembleModel` | `dict` | `mean`, `variance` (total), `epistemic_variance`, `aleatoric_variance` (each member outputs `(μ, log_σ²)`) |
| `HeteroscedasticEnsembleModel.predict_full_covariance` | `dict` | `mean`, `epistemic_covariance`, `aleatoric_covariance`, `total_covariance` |

`HeteroscedasticEnsembleModel` inherits from `BaseEnsembleModel` (same training loop).
Member log-variances are converted with the `GaussianNLLLoss` training bounds
(`log_var` clamped to `[log(1e-6), 30]`), here and in the batch/packed ensembles.

---

## Specialized ensemble models

→ **Guide:** [Ensemble methods](../methods/ensemble/index.md)

| Symbol | Description |
|:-------|:------------|
| `BinnedPDFEnsembleModel` | Members predict bin logits; ensemble averages `softmax` probabilities |
| `RandomPartitionEnsembleModel` | Each member uses its own bin edges; ensemble averages CDFs on shared grid |
| `CumulativeLinkEnsembleModel` | Members predict cumulative-link logits; ensemble averages PMFs |
| `MDNEnsembleModel` | Members predict mixture components; mixture-of-mixtures aggregation |
| `HeteroscedasticBatchEnsembleModel` | Parameter-efficient shared-backbone ensemble with per-member `(μ, log_σ²)` outputs |

```python
from torchregress.ensemble import HeteroscedasticBatchEnsembleModel

model = HeteroscedasticBatchEnsembleModel(
    backbone=backbone, output_size=1, ensemble_size=4  # backbone.feature_dim required
)
out = model(x)        # {"means": [B, M, D], "log_vars": [B, M, D]}
pred = model.predict(x)
# pred["mean"], pred["variance"], pred["epistemic_variance"], pred["aleatoric_variance"]
```

---

## Packed (parameter-efficient) ensembles

| Symbol | Description |
|:-------|:------------|
| `BatchEnsembleLinear` | Rank-1 perturbation linear layer (shared weight + per-member `r`, `s` vectors). |
| `BatchEnsembleMLPBackbone` | Shared-backbone MLP using `BatchEnsembleLinear` throughout (TabM-style). |
| `BatchEnsembleRegressor` | Facade over `HeteroscedasticBatchEnsembleModel` (or mean-only) with `alpha` scaling of fast weights. |
| `BatchEnsembleOutput` | Structured output of `BatchEnsembleRegressor.predict_output`: `mean`, `member_means`, `epistemic_variance`, `aleatoric_variance`, `predictive_variance`, `std_epistemic`. |

```python
bb = BatchEnsembleMLPBackbone(input_size=10, hidden_size=64, ensemble_size=4,
                             hidden_dims=[64, 64])
model = BatchEnsembleRegressor(bb, feature_dim=bb.feature_dim, output_dim=1,
                               ensemble_size=4, alpha=1.0, heteroscedastic=True)
out: BatchEnsembleOutput = model.predict_output(x)
```

---

## MC-Dropout

| Symbol | Description |
|:-------|:------------|
| `MCDropoutWrapper` | Wraps any model with `nn.Dropout` layers; enables dropout at inference time and runs `n_samples` forward passes. Provides `mc_forward`, `predict_with_uncertainty` and `predict_interval`. |
| `enable_dropout` | `(model)` — Set all `nn.Dropout` modules to `train()` mode (for MC-Dropout inference). |

**References:** Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016).

---

## SWAG

| Symbol | Description |
|:-------|:------------|
| `SWAG` | Single-SWAG: collects running first and second moments of weights during SGD; `sample(scale)` draws from the diagonal + low-rank Gaussian posterior. |
| `MultiSWAG` | M independent SWAG models → disagreement between SWAGs = epistemic, intra-SWAG spread = aleatoric. |

**Reference:** Maddox et al., "A Simple Baseline for Bayesian Uncertainty Estimation in Deep Learning" (NeurIPS 2019).

```python
# Typical SWAG workflow
model = MyModel()
swag = SWAG(model, max_num_models=20)
# ... train normally for `warmup` epochs ...
for epoch in range(swag_epochs):
    train_one_epoch(model, optimizer, loader)
    swag.collect_model(model)   # collect snapshot

# Inference: sample N times
preds = []
for _ in range(N):
    swag.sample(scale=0.5)
    with torch.no_grad():
        preds.append(swag(x_test))
mean, std = torch.stack(preds).mean(0), torch.stack(preds).std(0)
```

---

## Bayesian Neural Networks

| Symbol | Description |
|:-------|:------------|
| `VariationalLinear` | Bayesian linear layer with `weight_mu`, `weight_log_sigma`. Uses local reparameterization for efficient gradient estimation. Exposes `kl_divergence()`. |
| `BayesianNeuralNetwork` | MLP of `VariationalLinear` layers with ELBO training. Methods: `mc_forward(x, n)`, `predict_with_uncertainty`, `predict_interval`. |
| `HeteroscedasticBNN` | BNN with `(μ, log_σ²)` outputs. `predict_with_decomposition` returns `(mean, aleatoric_var, epistemic_var)`. |

**Reference:** Blundell et al., "Weight Uncertainty in Neural Networks" (ICML 2015).

---

## Combiners

| Symbol | Description |
|:-------|:------------|
| `StackingEnsemble` | Concatenates base-model predictions and feeds them into a `meta_learner` (e.g. another `nn.Module`). |
| `SoftmaxModelCombiner` | Learns softmax weights over a fixed pool of base models. `predict_with_uncertainty` returns (mean, total_var) via law of total variance. |

All combiners use `_batched_ensemble_forward`, which uses `torch.func.vmap` /
`stack_module_state` for fast batched inference when gradients aren't required.

---

## Laplace & snapshot ensembles

| Symbol | Description |
|:-------|:------------|
| `FullNetworkLaplace` | Diagonal empirical-Fisher Laplace approximation over **all** network parameters for post-hoc uncertainty. |
| `SnapshotEnsemble` | Collects model snapshots at the minima of a cyclic cosine learning-rate schedule; snapshots form a cheap deep ensemble. |

---

## Utilities

| Symbol | Description |
|:-------|:------------|
| `parse_heteroscedastic_output` | Splits a model output into `(mean, log_var)`. Accepts tuple, dict, or concatenated tensor layouts. |

---

## Quick example

```python
import torch.nn as nn
from torchregress.ensemble import BaseEnsembleModel, HeteroscedasticEnsembleModel

# Plain deep ensemble
ensemble = BaseEnsembleModel(MyModel, ensemble_size=5)
ensemble.fit(train_loader, criterion=nn.MSELoss(), epochs=10, lr=1e-3)
pred = ensemble.predict(x_test)  # {"mean", "variance"}

# Heteroscedastic ensemble
hetro_ens = HeteroscedasticEnsembleModel(MyGaussianModel, ensemble_size=5)
hetro_ens.fit(loader, criterion=GaussianNLLLoss(), epochs=10)
pred = hetro_ens.predict(x_test)
# pred["mean"], pred["variance"], pred["epistemic_variance"], pred["aleatoric_variance"]
```

## Next steps

- [Ensemble methods](../methods/ensemble/index.md) — peer-method matrix, decision guidance
- [Uncertainty decomposition](../guide/uncertainty-decomposition.md) — what each ensemble contract returns
