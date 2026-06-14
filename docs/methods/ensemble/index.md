# Ensemble Methods & Uncertainty Quantification

Ensemble methods are the most robust and widely used techniques for estimating **epistemic uncertainty** (model ignorance) in deep learning. By training multiple models and measuring where they **disagree**, ensembles provide well-calibrated uncertainty estimates without the heavy computational burden of traditional Bayesian methods.

---

## Why Ensembles?

In deep learning, a single model's point prediction $\hat{y}$ is often **overconfident**. Ensembles fix this by averaging predictions and providing a measure of "disagreement" between members.

### Key Benefits

- **Epistemic Uncertainty**: Captures uncertainty due to limited data (disagreement between members).
- **Aleatoric Uncertainty**: When using heteroscedastic members, captures data noise (mean of member variances).
- **Better Generalisation**: Averaging multiple models reduces the variance of the final prediction.
- **OOD Detection**: High disagreement (epistemic variance) is a strong signal for Out-of-Distribution inputs.

---

## Uncertainty Decomposition

When ensemble members predict both a mean $\mu(x; \mathbf{w}_m)$ and a variance $\sigma^2(x; \mathbf{w}_m)$, the total predictive uncertainty can be decomposed into two distinct components using the **Law of Total Variance** [1].

### Mathematical Derivation

Let $\mathbf{w}$ represent the model weights, distributed according to the posterior distribution $p(\mathbf{w} \mid \mathcal{D})$ given the training dataset $\mathcal{D}$. The total predictive variance of the target $Y$ for a new input $x$ is:
$$\text{Var}(Y \mid x, \mathcal{D}) = \mathbb{E}_{\mathbf{w} \mid \mathcal{D}}\left[\text{Var}(Y \mid x, \mathbf{w})\right] + \text{Var}_{\mathbf{w} \mid \mathcal{D}}\left(\mathbb{E}[Y \mid x, \mathbf{w}]\right)$$

Replacing the expectations with empirical averages over $M$ ensemble members (or posterior samples) $\mathbf{w}_1, \dots, \mathbf{w}_M$, we obtain:
$$\boxed{\; \sigma_{\text{total}}^2(x) \;=\; \underbrace{\frac{1}{M}\sum_{m=1}^{M}\sigma^2(x; \mathbf{w}_m)}_{\text{Aleatoric (Expected Data Noise)}} \;+\; \underbrace{\frac{1}{M}\sum_{m=1}^{M}\bigl(\mu(x; \mathbf{w}_m) - \bar\mu(x)\bigr)^2}_{\text{Epistemic (Model Disagreement)}} \;}$$

where $\bar\mu(x) = \frac{1}{M}\sum_{m=1}^{M} \mu(x; \mathbf{w}_m)$ is the ensemble mean prediction.

* **Aleatoric Uncertainty**: Represents irreducible data noise (e.g., measurement error, stochastic physics). Because it is a property of the data-generating process, it **does not** shrink as the training dataset size $N \to \infty$.
* **Epistemic Uncertainty**: Represents model parameters/structure ignorance. It **shrinks** to zero in regions covered by training data as $N \to \infty$, but remains high in out-of-distribution (OOD) or data-sparse regions.

### Bayesian Sampling & Variational Limitations

While SWAG and BNNs offer a principled Bayesian approach to approximate $p(\mathbf{w} \mid \mathcal{D})$, they carry significant practical and theoretical limitations:

#### 1. Stochastic Weight Averaging Gaussian (SWAG) Limits
* **Local Mode Bias**: SWAG fits a Gaussian distribution $\mathcal{N}(\boldsymbol\theta_{\text{SWA}}, \mathbf{\Sigma}_{\text{SWAG}})$ over the SGD weight trajectory. Since neural network loss landscapes are highly non-convex with many symmetric basins, SWAG only models a single local mode. It cannot capture disjoint global modes (unlike `MultiSWAG`, which runs multiple independent SWAG chains).
* **Trajectory Dependency**: The quality of the covariance matrix $\mathbf{\Sigma}_{\text{SWAG}}$ depends heavily on the SGD learning rate schedule during the collection phase. A learning rate that is too small fails to explore the local basin boundary, underestimating epistemic uncertainty.

#### 2. Variational BNN (VI) Limits
* **Mean-Field Approximation**: Standard variational inference assumes a fully factorized posterior (e.g., diagonal covariance where weight variables are independent). This ignores strong weight correlations, causing BNNs to systematically **underestimate** epistemic uncertainty and output overconfident predictions.
* **Prior/KL Sensitivity**: Optimizing the Evidence Lower Bound (ELBO):
  $$\mathcal{L}_{\text{ELBO}}(\theta) = \mathbb{E}_{q_\theta(\mathbf{w})}[\log p(\mathcal{D} \mid \mathbf{w})] - \beta \cdot \text{D}_{\text{KL}}(q_\theta(\mathbf{w}) \parallel p(\mathbf{w}))$$
  is highly sensitive to the scaling factor $\beta$ and the choice of prior $p(\mathbf{w})$. Poor choices lead to the "cold posterior" effect or over-regularization, where the variational posterior collapses back to the prior.

#### 3. Inference Latency (Curse of Monte Carlo)
To obtain predictive mean and variance, both SWAG and BNNs must draw $S$ samples of weights $\mathbf{w}^{(s)}$ at test time, requiring $S$ sequential forward passes:
$$\bar\mu(x) \approx \frac{1}{S}\sum_{s=1}^S f(x; \mathbf{w}^{(s)})$$
This increases computational latency and memory consumption linearly with $S$ ($\mathcal{O}(S)$), which can be prohibitive for real-time applications. `BatchEnsemble` or deep ensembles with small size $M \approx 5$ are often preferred in production.

---

## Method Selection Matrix

| Method | Epistemic? | Aleatoric? | API Reference | Best For |
|:-------|:----------:|:----------:|:--------------|:---------|
| **`DeepEnsemble`** | ✅ | ❌ | [`DeepEnsemble`](../../api/ensemble.md#torchregress.ensemble.DeepEnsemble) | High-accuracy baseline |
| **`HeteroEnsemble`** | ✅ | ✅ | [`HeteroscedasticEnsembleModel`](../../api/ensemble.md#torchregress.ensemble.HeteroscedasticEnsembleModel) | Full uncertainty |
| **`BatchEnsemble`** | ✅ | ✅ | [`HeteroscedasticBatchEnsembleModel`](../../api/ensemble.md#torchregress.ensemble.HeteroscedasticBatchEnsembleModel) | Production (fast) |
| **Building blocks** | — | — | [`BatchEnsembleLinear`](../../api/ensemble.md#torchregress.ensemble.BatchEnsembleLinear), [`BatchEnsembleMLPBackbone`](../../api/ensemble.md#torchregress.ensemble.BatchEnsembleMLPBackbone) | Rank-1 layers / shared MLP backbone |
| **`BinnedPDFEnsemble`** | ✅ | ⚠️ | [`BinnedPDFEnsembleModel`](../../api/ensemble.md#torchregress.ensemble.BinnedPDFEnsembleModel) | Ordered-bin / non-Gaussian PDFs |
| **`RandomPartitionEnsemble`** | ✅ | ⚠️ | [`RandomPartitionEnsembleModel`](../../api/ensemble.md#torchregress.ensemble.RandomPartitionEnsembleModel) | Members on different bin edges; CDF-averaged PDF |
| **`MDNEnsemble`** | ✅ | ✅ | [`MDNEnsembleModel`](../../api/ensemble.md#torchregress.ensemble.MDNEnsembleModel) | Multimodal predictive densities |
| **`SWAG`** [2] | ✅ | ❌ | [`SWAG`](../../api/ensemble.md#torchregress.ensemble.SWAG) | Large-scale Bayesian |
| **`MCDropout`** [3] | ✅ | ❌ | [`MCDropoutModel`](../../api/ensemble.md#torchregress.ensemble.MCDropoutModel) | Legacy / Cheap UQ |

---

## Quick Start: Training a Deep Ensemble

**torchregress** makes it easy to train and evaluate ensembles with minimal boilerplate.

```python
import torch
import torch.nn as nn
from torchregress.ensemble import HeteroscedasticEnsembleModel
from torchregress.losses import GaussianNLLLoss

# 1. Define your base model (predicts [mean, log_var])
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x): return self.net(x)

# 2. Wrap it in an ensemble
ensemble = HeteroscedasticEnsembleModel(base_model=MyModel, ensemble_size=5)

# 3. Train each member (parallel or sequential)
# torchregress provides utility for this
ensemble.train_members(dataloader, loss_fn=GaussianNLLLoss(), epochs=100)

# 4. Predict with uncertainty decomposition
result = ensemble.predict(x_test)
# result contains: 'mean', 'aleatoric_variance', 'epistemic_variance'
```

---

## Out-of-Distribution (OOD) Detection

Ensembles excel at detecting when a test point is far from the training data. In such cases, different members will extrapolate differently, leading to high **epistemic variance**.

!!! tip "OOD Strategy"

    Use `epistemic_variance` as a score for OOD detection. If it exceeds a threshold (calibrated on a validation set), consider the prediction "unreliable" or "selective".

→ See [Selective Prediction Examples](../../examples/ood_selective_prediction_comparison.md) for a practical guide.

---

## Advanced: Bayesian Model Averaging (BMA)

Instead of simple averaging, **torchregress** supports weighting members by their likelihood on a held-out validation set. This ensures that "better" models have more influence on the final prediction.

```python
from torchregress.ensemble import BayesianModelAveraging

bma = BayesianModelAveraging(ensemble)
bma.calibrate(val_dataloader) # weights members by validation NLL
```

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Lakshminarayanan et al. ["Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles."](https://arxiv.org/abs/1612.01474) *NeurIPS*, 2017. |
| 2 | Maddox et al. ["A Simple Baseline for Bayesian Deep Learning."](https://arxiv.org/abs/1902.02476) *NeurIPS*, 2019. |
| 3 | Gal & Ghahramani. ["Dropout as a Bayesian Approximation."](https://arxiv.org/abs/1506.02142) *ICML*, 2016. |
| 4 | Wen et al. ["BatchEnsemble: An Alternative Approach to Efficient Ensemble."](https://arxiv.org/abs/2002.06715) *ICLR*, 2020. |

---

## Next Steps
- Explore [Ensemble Methods Detail](methods.md)
- Learn about [Calibration Metrics](../../metrics/calibration.md)
- View the [Ensemble Tutorial](../../examples/ensemble_methods.md)
