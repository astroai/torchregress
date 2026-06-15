# Ensemble API

Complete reference for `torchregress.ensemble`. Every class, layer, and helper
is listed here. For background and decision guidance, see
[Ensembles overview](../methods/ensemble/index.md).

---

## Base classes

| Symbol | Description |
|:-------|:------------|
| `BaseEnsembleModel` | Foundation class. Each call to `forward(x)` runs every member and stacks outputs. Provides `predict` (mean + variance) and `predict_full_covariance`. |
| `EnsembleFitConfig` | Dataclass holding training options: `epochs`, `lr`, `optimizer_cls`, `optimizer_kwargs`, `optimizer_factory`, `device`, `adversarial_training`, etc. Passed to `fit()`. |

**`BaseEnsembleModel.fit(...)`** trains each member independently. With
`adversarial_training=True`, each member is also trained on an FGSM/PGD-style
adversarial objective. `optimizer_factory(model) -> Optimizer` lets you supply
custom per-member optimizers (e.g. AdamW + Muon).

---

## Mean-only and heteroscedastic deep ensembles

| Symbol | Output | Predicts |
|:-------|:-------|:---------|
| `DeepEnsemble` | `stacked` | Just the per-member means (epistemic variance from disagreement) |
| `HeteroscedasticEnsembleModel` | `dict` | `mean`, `variance` (total), `epistemic_variance`, `aleatoric_variance` (each member outputs `(μ, log_σ²)`) |
| `HeteroscedasticEnsembleModel.predict_full_covariance` | `dict` | `mean`, `epistemic_covariance`, `aleatoric_covariance`, `total_covariance` |

Both inherit from `BaseEnsembleModel`; same training loop and adversarial options.

---

## Specialized ensemble models

| Symbol | Family | Use case |
|:-------|:-------|:---------|
| `BinnedPDFEnsembleModel` | Discrete PDF | Members predict bin logits; ensemble averages `softmax` probabilities |
| `RandomPartitionEnsembleModel` | Discrete PDF (irregular grids) | Each member uses its own bin edges; ensemble projects to a shared evaluation grid and averages CDFs |
| `CumulativeLinkEnsembleModel` | Ordinal | Members predict cumulative-link logits; ensemble averages PMFs |
| `MDNEnsembleModel` | Mixture Density Network | Members predict mixture components; ensemble aggregates as a mixture-of-mixtures (avoids label switching) |
| `HeteroscedasticBatchEnsembleModel` | Batch-ensemble | Parameter-efficient shared-backbone ensemble with per-member `(μ, log_σ²)` outputs |

```python
from torchregress.ensemble import HeteroscedasticBatchEnsembleModel

model = HeteroscedasticBatchEnsembleModel(
    backbone=backbone, input_size=128, output_size=1, ensemble_size=4
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
| `PackedEnsembleRegressor` | Facade over `HeteroscedasticBatchEnsembleModel` (or mean-only) with `alpha` scaling of fast weights. |
| `PackedEnsembleOutput` | Structured output: `mean`, `member_means`, `epistemic_variance`, `aleatoric_variance`, `predictive_variance`, `std_epistemic`. |

```python
bb = BatchEnsembleMLPBackbone(input_size=10, hidden_size=64, ensemble_size=4,
                             hidden_dims=\[64, 64\])
model = PackedEnsembleRegressor(bb, feature_dim=bb.feature_dim, output_dim=1,
                                ensemble_size=4, alpha=1.0, heteroscedastic=True)
out: PackedEnsembleOutput = model.predict_output(x)
```

---

## MC-Dropout

| Symbol | Description |
|:-------|:------------|
| `MCDropoutWrapper` | Wraps any model with `nn.Dropout` layers; enables dropout at inference time and runs `n_samples` forward passes. |
| `MCDropoutModel` | MLP with built-in `Dropout` layers and `predict_with_uncertainty` / `predict_interval` methods. |
| `enable_dropout(model)` | Set all `nn.Dropout` modules to `train()` mode (for MC-Dropout inference). |

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
| `BayesianModelAveraging` | Learns softmax weights over a fixed pool of base models. `predict_with_uncertainty` returns (mean, total_var) via law of total variance. |
| `StackingEnsemble` | Concatenates base-model predictions and feeds them into a `meta_learner` (e.g. another `nn.Module`). |
| `DynamicEnsembleWeighting` | Sliding-window performance-based weight updates: lower recent MSE → higher weight. |

All combiners use `_batched_ensemble_forward`, which uses `torch.func.vmap` /
`stack_module_state` for fast batched inference when gradients aren't required.

---

## Utilities

| Symbol | Description |
|:-------|:------------|
| `parse_heteroscedastic_output` | Splits a model output into `(mean, log_var)`. Accepts tuple, dict, or concatenated tensor layouts. |

---

## Quick example

```python
import torch.nn as nn
from torchregress.ensemble import DeepEnsemble, HeteroscedasticEnsembleModel

# Plain deep ensemble
ensemble = DeepEnsemble(MyModel, ensemble_size=5)
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
- [Uncertainty decomposition](../guide/uncertainty-decomposition.md) — what each ensemble contract actually returns


## Detailed Class References

### DeepEnsemble

Orchestrates Deep Ensembles of $M$ independently trained model members:

$$
\bar{\mu}(x) = \frac{1}{M} \sum_{m=1}^M \mu_m(x)
$$

### BaseEnsembleModel

Foundation base class for all ensemble architectures in **torchregress**:

$$
f_{\text{ensemble}}(x) = \text{stack}\left([f_m(x)]_{m=1}^M\right)
$$

### HeteroscedasticEnsembleModel

Ensemble model where members output both mean $\mu_m(x)$ and log-variance $\log\sigma^2_m(x)$. Evaluates aleatoric and epistemic uncertainty via the Law of Total Variance.

### HeteroscedasticBatchEnsembleModel

Parameter-efficient BatchEnsemble variant where each member outputs a heteroscedastic predictive distribution.

### BinnedPDFEnsembleModel

Ensemble of discrete PDF estimators predicting target bin logits. Averages probability mass functions across members:

$$
p_{\text{ensemble}}(y) = \frac{1}{M}\sum_{m=1}^M \operatorname{softmax}(f_m(x))
$$

### RandomPartitionEnsembleModel

Ensemble of discrete PDF estimators defined on random partition boundaries. Members' cumulative distribution functions (CDFs) are interpolated onto a unified evaluation grid:

$$
F_{\text{ensemble}}(y) = \frac{1}{M}\sum_{m=1}^M F_m(y)
$$

### MDNEnsembleModel

Ensemble of Mixture Density Networks. Member component parameters are merged to form a unified Gaussian Mixture Model:

$$
p(y \mid x) = \frac{1}{M}\sum_{m=1}^M \sum_{k=1}^K \pi_{m, k} \mathcal{N}(y \mid \mu_{m, k}, \sigma_{m, k}^2)
$$

### MCDropoutModel

Point prediction MLP containing dropout layers active during both training and test-time evaluation to sample predictions.

### BatchEnsembleLinear

Rank-1 perturbation linear layer parameterized by shared weights $W$ and per-member vectors $r_m, s_m$:

$$
y = (X \circ r_m) W \circ s_m
$$

### BatchEnsembleMLPBackbone

Multi-layer MLP backbone built entirely using parameter-efficient `BatchEnsembleLinear` layers.
