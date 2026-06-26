# Evidential Regression

> ← [Poisson-Gaussian](poisson_gaussian.md) | [Mixture Density Networks](mdn.md) →

Evidential regression provides **aleatoric + epistemic uncertainty** from a **single forward pass**, without ensembles or Monte Carlo sampling.  It does this by placing a higher-order prior (Normal-Inverse-Gamma) over the Gaussian likelihood parameters.

See the [EvidentialRegressionLoss API](../api/losses.md) for constructor parameters and the NIG loss definition. For ensemble-based decomposition, compare with [`uncertainty_decomposition`](../api/metrics.md).

!!! abstract "Key idea"
    Instead of predicting just $(\mu, \sigma^2)$, the model predicts the parameters of a **distribution over distributions**: $(\gamma, \nu, \alpha, \beta)$ defining a Normal-Inverse-Gamma prior.

---

## Mathematical Background

### The Normal-Inverse-Gamma Prior

The model places a NIG prior on the mean and variance of the observation:

$$(\mu, \sigma^2) \sim \text{NIG}(\gamma, \nu, \alpha, \beta)$$

which factors as:

$$\sigma^2 \sim \text{Inv-Gamma}(\alpha, \beta), \qquad \mu \mid \sigma^2 \sim \mathcal{N}\!\left(\gamma,\; \frac{\sigma^2}{\nu}\right)$$

### Uncertainty Decomposition

$$\boxed{\;\underbrace{\mathbb{E}[\sigma^2]}_{\text{aleatoric}} = \frac{\beta}{\alpha - 1}\;, \qquad \underbrace{\text{Var}[\mu]}_{\text{epistemic}} = \frac{\beta}{\nu(\alpha - 1)}\;}$$

### Loss Function

The total loss combines NIG negative log-likelihood with a regulariser that penalises overconfident wrong predictions:

$$\mathcal{L} = \mathcal{L}_{\text{NIG-NLL}}(\gamma, \nu, \alpha, \beta \mid y) + \lambda \cdot |y - \gamma| \cdot (2\nu + \alpha)$$

Matches [`EvidentialRegressionLoss`](../api/losses.md) with regularisation coefficient `coeff_nig`.

---

## Usage

The model must output **4 values per target feature**: $[\gamma, \nu, \alpha, \beta]$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchregress.losses import EvidentialRegressionLoss

class EvidentialModel(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 4 * out_dim),  # γ, ν, α, β per target
        )

    def forward(self, x):
        out = self.net(x)
        # Enforce constraints: ν > 0, α > 1, β > 0
        gamma = out[..., 0:1]
        nu    = F.softplus(out[..., 1:2]) + 1e-2
        alpha = F.softplus(out[..., 2:3]) + 1.01
        beta  = F.softplus(out[..., 3:4]) + 1e-2
        return torch.cat([gamma, nu, alpha, beta], dim=-1)

model = EvidentialModel(in_dim=10)
loss_fn = EvidentialRegressionLoss(coeff_nig=0.01)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `coeff_nig` | `float` | `0.01` | Regularisation strength $\lambda$ |

---

## Inference Methods

### `predict_with_uncertainty`

Decompose predictions into mean + aleatoric + epistemic:

```python
model.eval()
with torch.no_grad():
    params = model(x_test)
    mean, aleatoric, epistemic = loss_fn.predict_with_uncertainty(params)

    total_unc = aleatoric + epistemic
    lower = mean - 1.96 * torch.sqrt(total_unc)
    upper = mean + 1.96 * torch.sqrt(total_unc)
```

### `predict_interval`

Get proper prediction intervals using the Student-t predictive distribution:

```python
lower, upper = loss_fn.predict_interval(params, confidence=0.95)
```

!!! tip "Prefer `predict_interval` over Gaussian approximation"
    The true predictive distribution is **Student-t** with $2\alpha$ degrees of freedom, not Gaussian.  `predict_interval` uses the exact distribution; `predict_interval_gaussian` is an approximation valid only when $\alpha \gg 1$.

### `sample_predictions`

Sample from the NIG prior for Monte Carlo analysis:

```python
samples = loss_fn.sample_predictions(params, n_samples=100)
# samples shape: [n_samples, batch, n_features]
```

---

## When to Use

!!! success "Good fit"
    - Need uncertainty decomposition without ensemble cost
    - Want OOD detection (epistemic uncertainty spikes on novel inputs)
    - Single forward pass latency constraint

!!! warning "Limitations"
    - Assumes unimodal Gaussian observations — for multimodal data, use [`MDNLoss`](mdn.md) or [`NormalizingFlowLoss`](nflows.md)
    - Regularisation coefficient $\lambda$ requires tuning; too large and the model underfits by forcing all predictions toward a uniform prior; too small and the epistemic uncertainty estimates collapse to near-zero
    - Can be poorly calibrated without careful training; the NIG-derived Student-t predictive distribution has $2\alpha$ degrees of freedom — when $\alpha$ is small (close to 1), the predictive distribution has heavy tails and infinite variance, making interval estimates unreliable
    - The NIG regularizer is computed **per sample**; very small batch sizes (< 16) produce noisy regularizer estimates that can destabilize training
    - The model outputs 4 parameters per target dimension ($\gamma, \nu, \alpha, \beta$), so the output dimension is $4 \cdot d$, which becomes large for high-dimensional targets

---

## Recommendations

1. **Start with `coeff_nig=0.01`** and tune in log-space over `{0.001, 0.01, 0.1, 1.0}` on a validation set. Monitor epistemic uncertainty — if it collapses below 1% of aleatoric, increase $\lambda$; if the model underfits (train loss plateaus high), decrease $\lambda$.
2. **Verify $\alpha > 1$ consistently**: $\alpha$ (the NIG shape) must stay above 1 for finite variance. Enforce with `F.softplus(...) + 1.01`. If $\alpha$ drifts near 1 during training, the Student-t predictive distribution has infinite variance — increase `coeff_nig` to penalise overconfident wrong predictions harder.
3. **Use `predict_interval` over `predict_interval_gaussian`**: The true predictive is Student-t with $2\alpha$ degrees of freedom. The Gaussian approximation is only reliable when $\alpha \gg 1$ (typically $\alpha > 5$). For safety, default to the exact Student-t method.
4. **Batch size $\geq 32$**: The per-sample NIG regulariser is noisy at small batch sizes. Below 16, gradient variance destabilises the $\nu$ and $\alpha$ parameters. Use gradient accumulation if memory-constrained.
5. **Monitor OOD calibration, not just in-distribution error**: Evidential models can achieve low in-distribution RMSE while producing severely overconfident epistemic estimates on shifted data. Validate on covariate-shifted or worst-slice hold-out sets.
6. **Compare with ensembles for critical applications**: Evidential single-pass uncertainty is fast but less reliable than [DeepEnsemble](../methods/ensemble/index.md) disagreement for epistemic decomposition. When safety-critical, use both and flag disagreements.

## Next steps

- [Ensemble methods](../methods/ensemble/index.md) — more reliable (but costlier) epistemic uncertainty
- [Gaussian losses](gaussian.md) — standard heteroscedastic regression (aleatoric only)
- [Uncertainty decomposition](../guide/uncertainty-decomposition.md) — taxonomy of aleatoric vs. epistemic contracts
- [Evidential example](../examples/evidential_regression.md) — runnable comparison with ensembles

---

## References

| # | Reference |
|:-:|:----------|
| 1 | A. Amini, W. Schwarting, A. Soleimany, D. Rus. ["Deep Evidential Regression."](https://arxiv.org/abs/1910.02600) *NeurIPS*, **2020**. |
| 2 | M. Sensoy, L. Kaplan, M. Kandemir. ["Evidential Deep Learning to Quantify Classification Uncertainty."](https://arxiv.org/abs/1806.01768) *NeurIPS*, **2018**. |
