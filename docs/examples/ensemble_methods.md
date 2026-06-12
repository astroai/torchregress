# Ensemble Methods for Uncertainty Quantification

This guide covers different ensemble methods in torchregress for uncertainty estimation.

| # | Reference |
|:-:|:----------|
| 1 | Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles**. *Advances in Neural Information Processing Systems (NeurIPS)*. |

---

## Task-First Context

Use this page when your main requirement is **epistemic uncertainty**, **uncertainty
decomposition**, or **OOD/selective prediction**. For broader method selection (including
`SWAG`, `BNN`, `MDN`, conformal, and flows), start with the
[Task-First Method Selection Matrix](../guide/method-selection.md).

## Evidence / Tradeoff Notes

- Deep and heteroscedastic ensembles are strong practical defaults, but train/inference cost scales with ensemble size.
- `SWAG` and `BNN` should be evaluated as peer methods when weight-posterior uncertainty is a better fit for your constraints.
- Always compare calibration and decision metrics (coverage, risk-coverage, OOD scores), not just point error.
- Use the comparison examples and benchmark smoke/sweep reports to validate runtime assumptions for your deployment budget.

## Overview

Ensemble methods combine multiple models to improve predictions and quantify uncertainty. torchregress supports several ensemble approaches, each with different trade-offs and use cases.

## Types of Uncertainty

Before diving into ensembles, it's important to understand two types of uncertainty:

| Uncertainty Type | Description | Source | Can be reduced? |
|-----------------|-------------|---------|-----------------|
| **Epistemic** | Model uncertainty | Lack of knowledge/data | Yes (with more data) |
| **Aleatoric** | Data uncertainty | Inherent noise | No (irreducible) |

### Why This Matters

- **Epistemic uncertainty** tells you where the model is uncertain (e.g., OOD regions)
- **Aleatoric uncertainty** tells you where the data is noisy
- Knowing the source helps make better decisions (collect more data vs. accept noise)

## Ensemble Methods Comparison

### 1. Single Model (Baseline)

```python
import torch.nn as nn
from torchregress.losses import WeightedMSELoss

model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

loss_fn = WeightedMSELoss()
```

**Characteristics:**
- Fastest to train
- Point predictions only
- No uncertainty estimates

**Use when:**
- Speed is critical
- Uncertainty not needed
- Baseline comparison

### 2. Deep Ensemble

**Concept:** Train multiple independent models with different random initializations.

```python
from torchregress.ensemble import DeepEnsemble
from torchregress.utils import set_seed

# Train multiple models independently
models = []
for i in range(5):
    set_seed(42 + i)  # Different initialization
    model = create_model()
    train_model(model, data, epochs=100)
    models.append(model)

ensemble = DeepEnsemble(models)

# Get predictions
predictions = ensemble(x_test)  # List of predictions from each model

# Calculate uncertainty (epistemic only)
from torchregress.metrics import ensemble_mean, ensemble_std
pred_mean = ensemble_mean(predictions)
pred_std = ensemble_std(predictions)  # Epistemic uncertainty
```

**Characteristics:**
- ✅ Epistemic uncertainty from model disagreement
- ❌ No aleatoric uncertainty (assumes homoscedastic noise)
- Training time: N × single model
- Inference time: N × single model

**Uncertainty interpretation:**
- High `pred_std` → Models disagree → Epistemic uncertainty
- Low `pred_std` → Models agree → Low epistemic uncertainty
- Cannot tell if data is noisy in regions where models agree

**Use when:**
- Need epistemic uncertainty estimates
- Data noise is roughly constant
- Can afford N× computational cost
- Out-of-distribution detection
- Active learning

### 3. Heteroscedastic Ensemble

**Concept:** Each model predicts both mean AND variance, then ensemble the models.

```python
from torchregress.ensemble import HeteroscedasticEnsembleModel
from torchregress.losses import GaussianNLLLoss

# Each model outputs (mean, log_variance)
def create_heteroscedastic_model():
    return nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Linear(64, 2)  # Output: [mean, log_var]
    )

models = []
for i in range(5):
    set_seed(42 + i)
    model = create_heteroscedastic_model()
    loss_fn = GaussianNLLLoss()  # Learns variance
    train_model(model, data, loss_fn, epochs=100)
    models.append(model)

ensemble = HeteroscedasticEnsembleModel(models)

# Get predictions with uncertainty decomposition
predictions = ensemble(x_test)  # List of (mean, log_var) tuples

# Decompose uncertainty
from torchregress.metrics import ensemble_variance_decomposition
means = torch.stack([pred[0] for pred in predictions])
log_vars = torch.stack([pred[1] for pred in predictions])

epistemic, aleatoric = ensemble_variance_decomposition(means, log_vars)
total_uncertainty = torch.sqrt(epistemic + aleatoric)
```

**Characteristics:**
- ✅ Both epistemic AND aleatoric uncertainty
- ✅ Can decompose total uncertainty into sources
- ✅ Handles heteroscedastic (varying) noise
- Training time: N × single model
- Inference time: N × single model

**Uncertainty interpretation:**
- `epistemic` → Model disagreement → Where to collect data
- `aleatoric` → Predicted data noise → Irreducible uncertainty
- `total_uncertainty` → Combined prediction interval

**Use when:**
- Need full uncertainty quantification
- Data has varying noise levels
- Need to separate model vs. data uncertainty
- Safety-critical applications
- Calibrated prediction intervals

### 4. Batch Ensemble (Efficient Alternative)

**Concept:** Share most weights, only randomize rank-1 perturbations.

```python
from torchregress.ensemble import BatchEnsembleLinear

# Replace Linear layers with BatchEnsembleLinear
model = nn.Sequential(
    BatchEnsembleLinear(1, 64, ensemble_size=5),
    nn.ReLU(),
    BatchEnsembleLinear(64, 1, ensemble_size=5)
)
```

**Characteristics:**
- ✅ Much faster than full ensemble (~1.2× single model)
- ✅ Epistemic uncertainty estimates
- ⚠️ Lower quality uncertainty than Deep Ensemble
- Training time: ~1.2× single model
- Memory: ~1.2× single model

**Use when:**
- Need epistemic uncertainty but limited compute
- Real-time or embedded applications
- Rapid prototyping

For a single module that combines a [`BatchEnsembleMLPBackbone`](https://github.com/sfabbro/torchregress/blob/main/src/torchregress/ensemble/models.py) with a heteroscedastic batch head, optional `alpha` scaling of fast weights, and structured access to `mean` / `std_epistemic` via `predict_output()`, see [`PackedEnsembleRegressor`](https://github.com/sfabbro/torchregress/blob/main/src/torchregress/ensemble/packed.py).

## Uncertainty Decomposition Math

For heteroscedastic ensembles:

$$
\text{Total Variance} = \underbrace{\mathbb{E}[\sigma^2]}_{\text{Aleatoric}} + \underbrace{\text{Var}[\mu]}_{\text{Epistemic}}
$$

Where:
- $\mu_i$ = mean prediction from model $i$
- $\sigma_i^2$ = variance prediction from model $i$
- Aleatoric = Average of individual model variances (expected noise)
- Epistemic = Variance of model means (model disagreement)

## Decision Tree: Which Ensemble Method?

```
Do you need uncertainty estimates?
├─ No → Single Model (fastest)
└─ Yes
   ├─ Does your data have varying noise levels?
   │  ├─ Yes → Heteroscedastic Ensemble
   │  └─ No → Deep Ensemble or Batch Ensemble
   │
   ├─ Do you need to separate epistemic/aleatoric?
   │  ├─ Yes → Heteroscedastic Ensemble
   │  └─ No → Deep Ensemble
   │
   └─ Computational budget?
      ├─ Limited → Batch Ensemble
      └─ No constraints → Deep Ensemble or Heteroscedastic Ensemble
```

## Common Applications

### Out-of-Distribution Detection

Use **epistemic uncertainty** to detect OOD:

```python
# High epistemic = OOD
threshold = epistemic.quantile(0.95)
ood_mask = epistemic > threshold
```

### Active Learning

Query points with high epistemic uncertainty:

```python
# Select points where model is most uncertain
query_indices = torch.argsort(epistemic, descending=True)[:k]
```

### Prediction Intervals

Use **total uncertainty** for calibrated intervals:

```python
# 95% prediction interval
lower = pred_mean - 1.96 * total_std
upper = pred_mean + 1.96 * total_std
```

### Safety-Critical Decisions

Use **uncertainty decomposition** for risk assessment:

```python
# High aleatoric = inherently uncertain, proceed with caution
# High epistemic = collect more data before deploying
risk_score = epistemic + aleatoric
```

## Complete Example

See [`examples/ensemble_tutorial.py`](https://github.com/sfabbro/torchregress/blob/main/examples/ensemble_tutorial.py) for a complete working example comparing all ensemble methods.

## Metrics for Ensembles

torchregress provides specialized metrics:

```python
from torchregress.metrics import (
    ensemble_mean,
    ensemble_std,
    ensemble_variance_decomposition,
    prediction_interval_coverage,
    calibration_score,
)

# Basic statistics
mean = ensemble_mean(predictions)
std = ensemble_std(predictions)

# Coverage (should be ~95% for 95% intervals)
coverage = prediction_interval_coverage(y_true, lower, upper, confidence=0.95)

# Calibration (are uncertainties well-calibrated?)
calibration = calibration_score(y_true, pred_mean, pred_std)
```

## Best Practices

1. **Always use different random seeds** for ensemble members
2. **Use at least 5 models** for reliable uncertainty (10+ for production)
3. **Check calibration** on validation set before trusting uncertainties
4. **Visualize uncertainty** to verify it makes sense
5. **Consider computational budget** early in design

## When Ensembles Can Fail

- Overconfident uncertainty can still occur if all ensemble members share the same data leakage or misspecification.
- Small ensembles can produce unstable epistemic estimates.
- Heteroscedastic heads can improve decomposition but still require calibration checks.
- OOD detection should use multiple signals (uncertainty + OOD metrics + decision metrics), not a single threshold.

## Comparison with Other Methods

| Method | Epistemic | Aleatoric | Multimodal | Computational Cost |
|--------|-----------|-----------|------------|-------------------|
| Deep Ensemble | ✅ | ❌ | ❌ | High (N models) |
| Heteroscedastic Ensemble | ✅ | ✅ | ❌ | High (N models) |
| Batch Ensemble | ✅ | ❌ | ❌ | Low (~1.2×) |
| MDN | ❌ | ✅ | ✅ | Low (1 model) |
| Normalizing Flows | ❌ | ✅ | ✅ | Medium (1 model) |
| Quantile Regression | ❌ | ❌ | ❌ | Low (1 model) |
| Conformal Prediction | ❌ | ❌ | ❌ | Low (calibration) |

**Note:** MDN and Normalizing Flows can be ensembled to get epistemic uncertainty too.

## Further Reading

- [Core Concepts Guide](../getting-started/concepts.md)
- [Ensemble Metrics](../metrics/ensemble.md)
- [Best Practices](../guide/best-practices.md)
- Paper: "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" (Lakshminarayanan et al., 2017)

## See Also

- [Evidential Regression](evidential_regression.md) - Alternative uncertainty decomposition
- [Conformal Prediction](conformal_regression_example.md) - Distribution-free intervals
- [Normalizing Flows](normalizing_flows_multitarget.md) - Flexible distributions
