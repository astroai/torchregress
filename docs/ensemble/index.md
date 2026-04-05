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

When ensemble members predict both a mean $\mu_m(x)$ and a variance $\sigma_m^2(x)$, the total predictive uncertainty can be decomposed into two components [1]:

$$\boxed{\; \sigma_{\text{total}}^2(x) \;=\; \underbrace{\frac{1}{M}\sum_{m=1}^{M}\sigma_m^2(x)}_{\text{Aleatoric (Data Noise)}} \;+\; \underbrace{\frac{1}{M}\sum_{m=1}^{M}\bigl(\mu_m(x) - \bar\mu(x)\bigr)^2}_{\text{Epistemic (Model Disagreement)}} \;}$$

where $\bar\mu(x) = \frac{1}{M}\sum_m \mu_m(x)$ is the ensemble mean.

- **Aleatoric**: Irreducible noise inherent in the data. Does **not** shrink with more data.
- **Epistemic**: Model's lack of knowledge. **Shrinks** as you add more training data.

---

## Method Selection Matrix

| Method | Epistemic? | Aleatoric? | API Reference | Best For |
|:-------|:----------:|:----------:|:--------------|:---------|
| **`DeepEnsemble`** | ✅ | ❌ | [`DeepEnsemble`](../api/ensemble.md#torchregress.ensemble.DeepEnsemble) | High-accuracy baseline |
| **`HeteroEnsemble`** | ✅ | ✅ | [`HeteroscedasticEnsembleModel`](../api/ensemble.md#torchregress.ensemble.HeteroscedasticEnsembleModel) | Full uncertainty |
| **`BatchEnsemble`** | ✅ | ✅ | [`HeteroscedasticBatchEnsembleModel`](../api/ensemble.md#torchregress.ensemble.HeteroscedasticBatchEnsembleModel) | Production (fast) |
| **Building blocks** | — | — | [`BatchEnsembleLinear`](../api/ensemble.md#torchregress.ensemble.BatchEnsembleLinear), [`BatchEnsembleMLPBackbone`](../api/ensemble.md#torchregress.ensemble.BatchEnsembleMLPBackbone) | Rank-1 layers / shared MLP backbone |
| **`BinnedPDFEnsemble`** | ✅ | ⚠️ | [`BinnedPDFEnsembleModel`](../api/ensemble.md#torchregress.ensemble.BinnedPDFEnsembleModel) | Ordered-bin / non-Gaussian PDFs |
| **`RandomPartitionEnsemble`** | ✅ | ⚠️ | [`RandomPartitionEnsembleModel`](../api/ensemble.md#torchregress.ensemble.RandomPartitionEnsembleModel) | Members on different bin edges; CDF-averaged PDF |
| **`MDNEnsemble`** | ✅ | ✅ | [`MDNEnsembleModel`](../api/ensemble.md#torchregress.ensemble.MDNEnsembleModel) | Multimodal predictive densities |
| **`SWAG`** [2] | ✅ | ❌ | [`SWAG`](../api/ensemble.md#torchregress.ensemble.SWAG) | Large-scale Bayesian |
| **`MCDropout`** [3] | ✅ | ❌ | [`MCDropoutModel`](../api/ensemble.md#torchregress.ensemble.MCDropoutModel) | Legacy / Cheap UQ |

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

→ See [Selective Prediction Examples](../examples/ood_selective_prediction_comparison.md) for a practical guide.

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
- Learn about [Calibration Metrics](../metrics/calibration.md)
- View the [Ensemble Tutorial](../examples/ensemble_methods.md)
