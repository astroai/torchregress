# Ensemble Methods Reference

Detailed API reference for all ensemble and Bayesian UQ methods.

---

## BaseEnsembleModel

!!! abstract "Summary"
    Shared base for member-based ensembles: builds `ensemble_size` copies of a base module, runs them in parallel, and stacks outputs.

```python
from torchregress.ensemble import BaseEnsembleModel

# Subclasses (DeepEnsemble, HeteroscedasticEnsembleModel, …) wrap this pattern.
```

→ Full API: [`BaseEnsembleModel`](../../api/ensemble.md#torchregress.ensemble.BaseEnsembleModel).

---

## DeepEnsemble

!!! abstract "Summary"
    Train $M$ **independently initialised** copies of a base model.  
    Epistemic uncertainty = variance of member predictions.

```python
from torchregress.ensemble import DeepEnsemble

ensemble = DeepEnsemble(base_model=MyModel, ensemble_size=5)

# Train each member with different random seeds
for member in ensemble.members:
    train_model(member, train_loader)

# Predict
preds = ensemble.forward(x_test)        # list of M tensors
mean = torch.stack(preds).mean(dim=0)    # ensemble mean
epi  = torch.stack(preds).var(dim=0)     # epistemic variance
```

Optional adversarial smoothing during member training:

```python
ensemble.fit(
    train_loader,
    loss_fn,
    epochs=20,
    adversarial_training=True,
    adversarial_epsilon=0.01,
    adversarial_steps=1,
    adversarial_loss_weight=1.0,
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `base_model` | `type` or `nn.Module` | — | Model class or instance to ensemble |
| `ensemble_size` | `int` | 5 | Number of independent members |
| `device` | `str` | `"cpu"` | Target device |

!!! tip "When to use"
    Use when your base model outputs **only point predictions** ($\hat{y}$) and you want epistemic uncertainty via disagreement.  For **aleatoric + epistemic**, use `HeteroscedasticEnsembleModel`.

!!! note "Adversarial training"
    The original deep-ensemble recipe adds an optional adversarial loss term on FGSM-style perturbed inputs. `torchregress` now exposes that directly in `fit(...)`, including multi-step and random-start variants for stronger smoothing.

---

## HeteroscedasticEnsembleModel

!!! abstract "Summary"
    Each member predicts $(\mu, \log\sigma^2)$ — enabling full **aleatoric + epistemic** decomposition.

```python
from torchregress.ensemble import HeteroscedasticEnsembleModel

ensemble = HeteroscedasticEnsembleModel(
    base_model=HeteroModel,  # outputs [mean, log_var]
    ensemble_size=5,
)

# After training:
result = ensemble.predict(x_test)
```

**`predict(x)` output:**

| Key | Shape | Description |
|:----|:------|:------------|
| `"mean"` | $(n, d)$ | $\bar\mu = \frac{1}{M}\sum_m \mu_m$ |
| `"variance"` | $(n, d)$ | Total = aleatoric + epistemic |
| `"aleatoric_variance"` | $(n, d)$ | $\frac{1}{M}\sum_m \sigma_m^2$ |
| `"epistemic_variance"` | $(n, d)$ | $\frac{1}{M}\sum_m (\mu_m - \bar\mu)^2$ |

---

## HeteroscedasticBatchEnsembleModel

!!! abstract "Summary"
    Memory-efficient ensemble via **rank-1 perturbations** of shared weights.  
    Each member applies $W_m = W \circ (r_m\, s_m^\top)$ where $r_m, s_m$ are per-member vectors.

```python
from torchregress.ensemble import HeteroscedasticBatchEnsembleModel

backbone = nn.Sequential(nn.Linear(10, 64), nn.ReLU())
batch_ens = HeteroscedasticBatchEnsembleModel(
    backbone=backbone,
    input_size=64,
    output_size=2,    # mean + log_var
    ensemble_size=4,
)
result = batch_ens.predict(x_test)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `backbone` | `nn.Module` | — | Shared feature extractor |
| `input_size` | `int` | — | Input dim to batch-ensemble head |
| `output_size` | `int` | — | Output dim ($2 \times d$ for mean + logvar) |
| `ensemble_size` | `int` | 4 | Number of virtual members |

!!! tip "When to use"
    When compute or memory budget prohibits $M$ full models.  BatchEnsemble achieves ~80 % of full-ensemble UQ quality at ~20 % extra cost.

!!! quote "Reference"
    Y. Wen, D. Tran, J. Ba. "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning." *ICLR*, **2020**.

---

## BatchEnsembleLinear

!!! abstract "Summary"
    Rank-1 **BatchEnsemble** perturbation of a linear layer: shared weight $W$ plus per-member vectors $r_m, s_m$ so each virtual member uses an effective weight $W \circ (r_m s_m^\top)$ (see Wen et al., 2020).

```python
from torchregress.ensemble import BatchEnsembleLinear

layer = BatchEnsembleLinear(in_features=10, out_features=64, ensemble_size=4)
```

→ API: [`BatchEnsembleLinear`](../../api/ensemble.md#torchregress.ensemble.BatchEnsembleLinear).

---

## BatchEnsembleMLPBackbone

!!! abstract "Summary"
    Shared-backbone MLP that applies **BatchEnsemble** perturbations through the hidden stack (not only the readout). Suited to tabular-style deep models and efficient ensemble backbones.

```python
from torchregress.ensemble import BatchEnsembleMLPBackbone

backbone = BatchEnsembleMLPBackbone(
    input_size=20,
    hidden_size=128,
    ensemble_size=4,
    hidden_dims=(128, 64),
)
features = backbone(x)  # (batch, ensemble_size, feature_dim)
```

→ API: [`BatchEnsembleMLPBackbone`](../../api/ensemble.md#torchregress.ensemble.BatchEnsembleMLPBackbone).

---

## BinnedPDFEnsembleModel

!!! abstract "Summary"
    Average member predictions in **probability space** for discrete PDFs or regression-as-classification heads.

```python
from torchregress.ensemble import BinnedPDFEnsembleModel

ensemble = BinnedPDFEnsembleModel(
    base_model=MyBinLogitModel,
    ensemble_size=5,
    support_values=torch.linspace(0.0, 4.0, 64),
)
result = ensemble.predict(x_test)
# result: {"probabilities", "log_probabilities", "mean", "variance"}
```

!!! tip "When to use"
    Prefer this over averaging logits when each member predicts a **binned predictive PDF**.  The ensemble should average the predictive distribution, not the pre-softmax parameters.

---

## RandomPartitionEnsembleModel

!!! abstract "Summary"
    Each member predicts a softmax PDF on **its own bin edges**. The ensemble averages **CDFs** mapped to a common evaluation grid, then differencing yields a coherent aggregate PDF (avoids averaging incompatible logits).

```python
from torchregress.ensemble import RandomPartitionEnsembleModel

member_edges = [torch.linspace(0.0, 1.0, 33) for _ in range(5)]
ensemble = RandomPartitionEnsembleModel(
    base_model=MyPartitionHead,
    ensemble_size=5,
    member_bin_edges=member_edges,
)
result = ensemble.predict(x_test)
# result: "probabilities", "cdf_at_edges", "bin_edges", "mean", "variance", ...
```

→ API: [`RandomPartitionEnsembleModel`](../../api/ensemble.md#torchregress.ensemble.RandomPartitionEnsembleModel).

---

## CumulativeLinkEnsembleModel

!!! abstract "Summary"
    Average **ordinal CDF / PMF predictions** across members for cumulative-link heads.

```python
from torchregress.ensemble import CumulativeLinkEnsembleModel

ensemble = CumulativeLinkEnsembleModel(
    base_model=MyOrdinalModel,
    ensemble_size=5,
    support_values=torch.arange(64, dtype=torch.float32),
)
result = ensemble.predict(x_test)
```

!!! tip "When to use"
    Use when the model predicts ordered thresholds / cumulative logits and you want ensemble averaging in the implied probability space rather than averaging thresholds directly.

---

## MDNEnsembleModel

!!! abstract "Summary"
    Ensemble an MDN by forming a **mixture of mixtures**, not by naively averaging component parameters.

```python
from torchregress.ensemble import MDNEnsembleModel

ensemble = MDNEnsembleModel(
    base_model=MyMDNModel,
    ensemble_size=5,
    n_components=4,
    n_features=1,
)
result = ensemble.predict(x_test)
# result: {"mixture_weights", "component_means", "component_stds", "mean", "variance"}
```

!!! warning "Do not average MDN weights component-wise"
    Component 1 in member A is not generally the same mode as component 1 in member B. `MDNEnsembleModel` avoids this label-switching problem by concatenating all member components with a `1 / M` weight factor.

---

## SWAG / MultiSWAG

!!! abstract "Summary"
    **S**tochastic **W**eight **A**veraging — **G**aussian.  
    Fits a Gaussian over the SGD **weight trajectory**, then samples weight configurations for MC prediction.

=== "SWAG (single)"

    ```python
    from torchregress.ensemble import SWAG

    swag = SWAG(base_model, rank=20)

    # During training — periodically collect snapshots:
    for epoch in range(epochs):
        train_one_epoch(model, ...)
        if epoch >= swa_start:
            swag.collect_model(model)

    # At inference — sample from weight posterior:
    predictions = swag.sample_predictions(x_test, n_samples=30)
    mean = predictions.mean(0)
    var  = predictions.var(0)
    ```

=== "MultiSWAG"

    ```python
    from torchregress.ensemble import MultiSWAG

    # Multiple independent SWAG models for better diversity
    multi_swag = MultiSWAG(base_model, n_models=3, rank=20)
    ```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `rank` | `int` | 20 | Rank of low-rank weight covariance approximation |
| `n_samples` | `int` | 30 | MC samples at inference |

!!! quote "Reference"
    W. Maddox et al. "A Simple Baseline for Bayesian Deep Learning." *NeurIPS*, **2019**.

---

## MC-Dropout

!!! abstract "Summary"
    Keep dropout **active** during inference and aggregate multiple stochastic forward passes.

=== "MCDropoutModel"

    ```python
    from torchregress.ensemble import MCDropoutModel

    mc_model = MCDropoutModel(base_model, n_samples=20)
    result = mc_model.predict(x_test)
    # result: dict with "mean", "variance", "samples"
    ```

=== "MCDropoutWrapper"

    ```python
    from torchregress.ensemble import MCDropoutWrapper

    mc = MCDropoutWrapper(model, n_samples=20)
    mean, var = mc(x_test)
    ```

=== "enable_dropout utility"

    ```python
    from torchregress.ensemble import enable_dropout

    model.eval()
    enable_dropout(model)  # keeps dropout active in eval mode
    ```

!!! quote "Reference"
    Y. Gal, Z. Ghahramani. "Dropout as a Bayesian Approximation." *ICML*, **2016**.

---

## Bayesian Neural Networks

!!! abstract "Summary"
    Replace `nn.Linear` with `VariationalLinear` — weights are sampled from a learned posterior $q(\mathbf{w}) = \mathcal{N}(\boldsymbol\mu, \boldsymbol\sigma^2)$ during each forward pass.

=== "VariationalLinear"

    ```python
    from torchregress.ensemble import VariationalLinear

    layer = VariationalLinear(in_features=64, out_features=32)
    # Each forward pass samples weights from N(μ, σ²)
    ```

=== "BayesianNeuralNetwork"

    ```python
    from torchregress.ensemble import BayesianNeuralNetwork

    bnn = BayesianNeuralNetwork(model, n_samples=10)
    # Training loss includes KL divergence term:
    # L_total = E_q[NLL] + β · KL(q || p)
    ```

=== "HeteroscedasticBNN"

    ```python
    from torchregress.ensemble import HeteroscedasticBNN

    hbnn = HeteroscedasticBNN(model, n_samples=10)
    result = hbnn.predict(x_test)
    # result: dict with mean, epistemic_variance, aleatoric_variance
    ```

---

## Ensemble Combiners

### BayesianModelAveraging

Weight members by posterior probability from held-out log-likelihood:

```python
from torchregress.ensemble import BayesianModelAveraging

bma = BayesianModelAveraging()
weights = bma.fit(member_predictions_val, y_val)
combined = bma.predict(member_predictions_test)
```

### StackingEnsemble

Learned meta-model combining member outputs:

```python
from torchregress.ensemble import StackingEnsemble

stacker = StackingEnsemble(meta_model=nn.Linear(5, 1))
stacker.fit(member_predictions_val, y_val)
combined = stacker.predict(member_predictions_test)
```

### DynamicEnsembleWeighting

**Input-dependent** weighting — different members trusted more in different input regions:

```python
from torchregress.ensemble import DynamicEnsembleWeighting

dew = DynamicEnsembleWeighting(input_dim=10, n_members=5)
combined = dew(x_test, member_predictions_test)
```

---

## Utility: `parse_heteroscedastic_output`

Splits a model's concatenated `[mean, log_var]` output:

```python
from torchregress.ensemble import parse_heteroscedastic_output

mean, log_var = parse_heteroscedastic_output(model_output)
```
