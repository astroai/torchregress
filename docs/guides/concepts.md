# Core Concepts in torchregress

This guide introduces the key concepts in torchregress for beginners. If you're new to uncertainty quantification and robust regression, start here!

## What is torchregress?

torchregress is a PyTorch library that extends standard regression to handle:

1. **Uncertainty Estimation** - Know when your model is uncertain
2. **Robust Regression** - Handle outliers gracefully
3. **Missing Data** - Work with incomplete datasets
4. **Advanced Distributions** - Model complex data patterns

## Core Concepts

### 1. Point Predictions vs. Probabilistic Predictions

**Point Prediction** (traditional):
```python
model(x) → y_pred  # Single number
```

Example: "The house will cost $500,000"

**Probabilistic Prediction** (torchregress):
```python
model(x) → (y_mean, y_variance)  # Distribution
```

Example: "The house will cost $500,000 ± $50,000 (95% confident)"

**Why it matters:** Probabilistic predictions tell you both *what* the model predicts and *how confident* it is.

### 2. Types of Uncertainty

<center>

| Type | What it means | Physics Analogy | Can reduce? |
|------|---------------|-----------------|-------------|
| **Aleatoric** | Irreducible data noise | Detector resolution, photon shot noise, thermal noise | No |
| **Epistemic** | Model/knowledge uncertainty | Systematic error, limited calibration data, extrapolation | Yes (more data) |

</center>

**For physicists:** Think of aleatoric uncertainty as the intrinsic measurement noise you cannot eliminate (like quantum shot noise in photon counting, or thermal Johnson noise in electronics). Epistemic uncertainty is like systematic error from an incomplete calibration curve—it can be reduced with more calibration data or a better model.

**Real-world example:**

```python
# Astronomy: Photometric redshift estimation
# High aleatoric: Photon noise in faint galaxies, CCD readout noise
# High epistemic: Rare galaxy types not in training set, extrapolating beyond calibration range

# Particle physics: Energy measurement
# High aleatoric: Shower fluctuations in calorimeter, electronic noise
# High epistemic: Unknown systematic from detector aging, pile-up effects
```

**Which uncertainty do you need?**

- **Active Learning:** Use epistemic (query where model is uncertain)
- **Safety-critical:** Use both (understand all sources of risk)
- **Prediction intervals:** Use total (aleatoric + epistemic)
- **Out-of-distribution detection:** Use epistemic (detect novelty)

### 3. Robustness to Outliers

**Problem:** One bad data point can ruin your model.

```python
# Data: [1, 2, 3, 4, 5, 1000]  ← outlier!
# MSE: Heavily influenced by 1000
# MAE: Less influenced (uses absolute error)
# Huber: Balanced approach
```

**Loss function comparison:**

| Loss | Sensitivity to Outliers | When to Use |
|------|------------------------|-------------|
| **MSE** | Very High | Clean data, normally distributed |
| **MAE** | Low | Many outliers, want robust predictions |
| **Huber** | Medium | Some outliers, balanced approach |
| **Cauchy** | Very Low | Heavy outliers, extreme robustness |

**Example:**

```python
from torchregress.losses import WeightedMSELoss, WeightedMAELoss, HuberLoss

# Clean data → use MSE (fastest, most efficient)
loss = WeightedMSELoss()

# Data with outliers → use MAE or Huber
loss = WeightedMAELoss()  # or HuberLoss(delta=1.0)
```

### 4. Heteroscedastic vs. Homoscedastic

**Homoscedastic:** Noise is constant everywhere

```
y = f(x) + ε,  where ε ~ N(0, σ²) and σ is constant
```

Example: A well-calibrated thermometer with fixed resolution across its range.

**Heteroscedastic:** Noise varies with input

```
y = f(x) + ε,  where ε ~ N(0, σ(x)²) and σ depends on x
```

**Physics examples of heteroscedasticity:**

- **CCD imaging:** Photon noise (Poisson statistics) means σ ∝ √signal—bright regions have higher absolute noise
- **Photometric redshift:** Faint, high-z galaxies have much larger uncertainties than bright, nearby ones
- **Spectrometer:** Signal-to-noise ratio varies with wavelength depending on detector sensitivity and source spectrum
- **Particle tracking:** Position uncertainty depends on track curvature and number of hits

**How to handle:**

```python
# Homoscedastic (constant noise)
from torchregress.losses import WeightedMSELoss
loss = WeightedMSELoss()

# Heteroscedastic (varying noise)
from torchregress.losses import GaussianNLLLoss
loss = GaussianNLLLoss()  # Learns variance as function of x

# Model outputs: (mean, log_variance)
model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 2)  # 2 outputs: mean and log_var
)
```

### 5. Interval Predictions

**Goal:** Provide a range instead of a point.

```python
# Instead of: "y = 100"
# Provide:    "y = 100 ± 10 with 95% confidence"
#             → interval: [90, 110]
```

**Types of intervals:**

| Method | Assumption | Guarantee | When to Use |
|--------|------------|-----------|-------------|
| **Gaussian** | Assumes normality | Approximate | Fast, data is Gaussian |
| **Quantile** | Distribution-free | Approximate | Non-Gaussian data |
| **Conformal** | Distribution-free | **Exact coverage** | Need guaranteed coverage |

**Example:**

```python
# Method 1: Gaussian intervals (assumes normal distribution)
from torchregress.losses import GaussianNLLLoss
loss = GaussianNLLLoss()
# → Get (mean, variance), compute mean ± 1.96*std

# Method 2: Quantile regression (no distributional assumption)
from torchregress.losses import QuantileLoss
loss = QuantileLoss(quantiles=[0.025, 0.5, 0.975])  # 95% interval
# → Directly predict lower, median, upper

# Method 3: Conformal prediction (guaranteed coverage)
from torchregress.losses import ConformalLoss
loss = ConformalLoss(method="split")
# → Calibrates any model to achieve exact coverage
```

### 6. Missing Data

Real-world data often has missing values. torchregress handles this automatically.

```python
# Data with missing values
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[5.0], [np.nan], [15.0]])  # Middle value missing

# Create mask: True = valid, False = missing
mask = ~torch.isnan(y)

# All torchregress losses support masks
loss_fn = WeightedMSELoss()
loss = loss_fn(y_pred, y, mask=mask)  # Automatically ignores missing values
```

**No need to:**
- Impute missing values
- Filter out incomplete samples
- Write custom masking logic

### 7. Sample Weighting

Give different importance to different samples.

```python
# Some samples are more important
weights = torch.tensor([1.0, 1.0, 10.0])  # Last sample 10x more important

loss_fn = WeightedMSELoss()
loss = loss_fn(y_pred, y, weights=weights)
```

**Use cases:**
- **Imbalanced data:** Weight rare samples higher
- **Data quality:** Weight reliable measurements higher
- **Cost-sensitive:** Weight expensive mistakes higher

### 8. Ensemble Methods

**Idea:** Multiple models are better than one.

```python
# Instead of one model:
model = create_model()

# Train multiple models:
ensemble = [create_model() for _ in range(5)]

# Predictions: average or analyze disagreement
predictions = [model(x) for model in ensemble]
mean = torch.mean(torch.stack(predictions), dim=0)
std = torch.std(torch.stack(predictions), dim=0)  # Uncertainty!
```

**Benefits:**
- More robust predictions
- Uncertainty estimates from disagreement
- Better generalization

**Types:**
- **Deep Ensemble:** Independent models (epistemic uncertainty)
- **Heteroscedastic Ensemble:** Each model predicts mean + variance (both uncertainties)
- **Batch Ensemble:** Efficient variant (shared weights)

See [Ensemble Methods Guide](../examples/ensemble_methods.md) for details.

## Comparison Table: Key Methods

### Loss Functions for Different Scenarios

| Scenario | Recommended Loss | Why |
|----------|-----------------|-----|
| Clean data, normally distributed | `WeightedMSELoss` | Optimal for Gaussian noise |
| Data with outliers | `HuberLoss`, `WeightedMAELoss` | Robust to extreme values |
| Varying noise levels | `GaussianNLLLoss` | Learns heteroscedastic variance |
| Need prediction intervals | `QuantileLoss` | Directly predicts quantiles |
| Multiple modes in data | `MDNLoss` | Models mixture of distributions |
| Count data (non-negative integers) | `PoissonNLLLoss` | Proper likelihood for counts |
| Guaranteed coverage | `ConformalLoss(method="split")` | Calibrates for exact coverage |

### Uncertainty Methods Comparison

| Method | Type | Epistemic | Aleatoric | Multimodal | Cost |
|--------|------|-----------|-----------|------------|------|
| Heteroscedastic Gaussian | Parametric | ❌ | ✅ | ❌ | Low |
| Deep Ensemble | Non-parametric | ✅ | ❌ | ❌ | High |
| Heteroscedastic Ensemble | Hybrid | ✅ | ✅ | ❌ | High |
| MDN | Parametric | ❌ | ✅ | ✅ | Low |
| Normalizing Flows | Flexible | ❌ | ✅ | ✅ | Medium |
| Quantile Regression | Distribution-free | ❌ | ❌* | ❌ | Low |
| Conformal Prediction | Distribution-free | ❌ | ❌* | ❌ | Low |

*Provides intervals but not explicit uncertainty decomposition

## Decision Trees

### Which loss function should I use?

```
What's your main concern?

├─ Outliers in data?
│  ├─ Many outliers → MAELoss or CauchyLoss
│  ├─ Some outliers → HuberLoss
│  └─ Few outliers → MSELoss (fastest)
│
├─ Need uncertainty estimates?
│  ├─ Simple (Gaussian) → GaussianNLLLoss
│  ├─ Multimodal data → MDNLoss or NormalizingFlowLoss
│  └─ Distribution-free → QuantileLoss
│
├─ Need guaranteed coverage?
│  └─ Yes → Conformal Prediction (`ConformalLoss(method="split")` or `ConformalLoss(method="cqr")`)
│
├─ Special data types?
│  ├─ Count data → PoissonNLLLoss or TweedieLoss
│  ├─ Heavy-tailed → CauchyLoss or StudentTLoss
│  └─ Compositional → LogitNormalLoss
│
└─ Default for clean data → MSELoss
```

### Which uncertainty method should I use?

```
What do you need?

├─ Decompose epistemic vs. aleatoric?
│  ├─ Yes → Heteroscedastic Ensemble or Evidential Regression
│  └─ No → Continue below
│
├─ Detect out-of-distribution samples?
│  └─ Yes → Deep Ensemble or SWAG (epistemic uncertainty)
│
├─ Guaranteed interval coverage?
│  └─ Yes → Conformal Prediction
│
├─ Model complex/multimodal distributions?
│  └─ Yes → MDN or Normalizing Flows
│
├─ Computational budget?
│  ├─ Limited → GaussianNLLLoss or QuantileLoss
│  └─ No constraints → Heteroscedastic Ensemble
│
└─ Simple uncertainty estimates → GaussianNLLLoss
```

## Getting Started: Recommended Path

### 1. Start Simple (Day 1)

```python
from torchregress.losses import WeightedMSELoss

# Basic regression
loss = WeightedMSELoss()
```

**Learn:** Basic usage, training loop, evaluation

**Example:** [`examples/basic_usage.py`](../../examples/basic_usage.py)

### 2. Add Uncertainty (Day 2)

```python
from torchregress.losses import GaussianNLLLoss

# Model outputs (mean, log_variance)
loss = GaussianNLLLoss()
```

**Learn:** Heteroscedastic uncertainty, prediction intervals

**Example:** [`examples/basic_usage.py`](../../examples/basic_usage.py) (Example 2)

### 3. Handle Outliers (Day 3)

```python
from torchregress.losses import HuberLoss

# Robust to outliers
loss = HuberLoss(delta=1.0)
```

**Learn:** Robust regression, loss comparison

**Example:** [`examples/loss_comparison.py`](../../examples/loss_comparison.py)

### 4. Advanced Uncertainty (Week 2)

```python
from torchregress.ensemble import DeepEnsemble

# Train ensemble for epistemic uncertainty
ensemble = train_ensemble(n_models=5)
```

**Learn:** Ensemble methods, uncertainty decomposition

**Example:** [`examples/ensemble_tutorial.py`](../../examples/ensemble_tutorial.py)

### 5. Specialized Methods (Week 3+)

Explore based on your needs:

- **Conformal Prediction:** [`examples/conformal_regression_example.py`](../../examples/conformal_regression_example.py)
- **Evidential Regression:** [`examples/evidential_regression.py`](../../examples/evidential_regression.py)
- **Normalizing Flows:** [`examples/normalizing_flows_multitarget.py`](../../examples/normalizing_flows_multitarget.py)
- **Imbalanced Data:** [`examples/imbalanced_regression.py`](../../examples/imbalanced_regression.py)

## Common Pitfalls

### 1. Using MSE with Outliers

❌ **Wrong:**
```python
# Data has outliers, but using MSE
loss = WeightedMSELoss()  # Will be dominated by outliers!
```

✅ **Right:**
```python
# Use robust loss
loss = HuberLoss(delta=1.0)  # or MAELoss
```

### 2. Ignoring Uncertainty Calibration

❌ **Wrong:**
```python
# Train model, trust uncertainties immediately
```

✅ **Right:**
```python
# Check calibration on validation set
from torchregress.metrics import calibration_score
cal = calibration_score(y_val, pred_mean, pred_std)
# If poorly calibrated, use conformal prediction to recalibrate
```

### 3. Confusing Conformal Prediction with Uncertainty Decomposition

❌ **Wrong:**
```python
# Trying to get epistemic uncertainty from conformal prediction
```

✅ **Right:**
```python
# Conformal prediction: guaranteed coverage (not uncertainty decomposition)
# Use ensembles or evidential regression for epistemic/aleatoric split
```

### 4. Not Using Masks for Missing Data

❌ **Wrong:**
```python
# Impute missing values with mean (loses information)
y[y.isnan()] = y.nanmean()
```

✅ **Right:**
```python
# Use mask to properly handle missing data
mask = ~torch.isnan(y)
loss = loss_fn(y_pred, y, mask=mask)
```

## Further Reading

### Tutorials
- [Quick Start](../usage/quickstart.md) - 3 examples to get started
- [Basic Usage Examples](../examples/basic_usage.md) - 4 detailed examples
- [Loss Comparison](../examples/loss_comparison.md) - When to use which loss

### Guides
- [Best Practices](best-practices.md) - 7-phase development workflow
- [Practical Usage](../usage/practical_usage.md) - Decision trees and tips
- [Ensemble Methods](../examples/ensemble_methods.md) - Complete ensemble guide

### API Reference
- [Losses API](../api/losses.md) - All available loss functions
- [Metrics API](../api/metrics.md) - Evaluation metrics
- [Utilities API](../api/utils.md) - Helper functions

## Glossary

| Term | Definition | Physics Analogy |
|------|------------|-----------------|
| **Aleatoric Uncertainty** | Irreducible noise in the data | Detector noise, shot noise, thermal fluctuations |
| **Epistemic Uncertainty** | Model uncertainty, reducible with more data | Systematic error, incomplete calibration |
| **Heteroscedastic** | Noise varies across input space | SNR varies with signal level (Poisson statistics) |
| **Homoscedastic** | Noise is constant | Fixed detector resolution |
| **Conformal Prediction** | Method providing guaranteed coverage | Distribution-free error bar calibration |
| **Quantile Regression** | Predicting percentiles of distribution | Estimating confidence intervals directly |
| **MDN** | Mixture Density Network for multimodal outputs | Modeling multi-peaked probability distributions |
| **NLL** | Negative Log-Likelihood (minimize = maximize probability) | Like minimizing free energy in stat mech |
| **CRPS** | Continuous Ranked Probability Score | Proper scoring rule for probabilistic forecasts |
| **PICP** | Prediction Interval Coverage Probability | Fraction of truth values within error bars |
| **MPIW** | Mean Prediction Interval Width | Average size of error bars |
| **OOD** | Out-of-Distribution (unseen data) | Test points outside training calibration range |

## Questions?

- Check [FAQ](../usage/faq.md) (if exists)
- See [Examples Index](../examples/index.md) for all examples
- Read [Best Practices](best-practices.md) for detailed guidance

Happy regressing! 🎯
