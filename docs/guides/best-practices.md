# Best Practices for Regression with torchregress

This guide provides essential best practices, sanity checks, and common pitfalls when building regression models with uncertainty quantification.

## Table of Contents

1. [Model Development Workflow](#model-development-workflow)
2. [Essential Sanity Checks](#essential-sanity-checks)
3. [Loss Function Selection](#loss-function-selection)
4. [Uncertainty Quantification](#uncertainty-quantification)
5. [Evaluation Strategy](#evaluation-strategy)
6. [Common Pitfalls](#common-pitfalls)
7. [Production Deployment](#production-deployment)

## Model Development Workflow

### Recommended Pipeline

```
1. Data Understanding
   ↓
2. Baseline Model (MSE)
   ↓
3. Sanity Checks
   ↓
4. Advanced Models
   ↓
5. Uncertainty Quantification
   ↓
6. Calibration Assessment
   ↓
7. Production Validation
   ↓
8. Deployment
```

### Phase 1: Data Understanding

**Before any modeling:**

```python
import torch
import torchregress as tr
import matplotlib.pyplot as plt
import numpy as np

# 1. Examine target distribution
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(y_train, bins=50, edgecolor='black')
plt.title('Target Distribution')
plt.xlabel('Target Value')

plt.subplot(132)
plt.boxplot(y_train)
plt.title('Box Plot (Check for outliers)')

plt.subplot(133)
from scipy import stats
stats.probplot(y_train.numpy(), dist="norm", plot=plt)
plt.title('QQ Plot (Check normality)')

plt.tight_layout()
plt.show()

# 2. Check for data issues
print(f"Missing values: {torch.isnan(y_train).sum().item()}")
print(f"Infinite values: {torch.isinf(y_train).sum().item()}")
print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"Target std: {y_train.std():.2f}")

# 3. Check feature-target relationships
for i, feature_name in enumerate(feature_names):
    correlation = np.corrcoef(X_train[:, i], y_train)[0, 1]
    print(f"{feature_name}: correlation = {correlation:.3f}")
```

**Key Questions:**
- Is the target distribution skewed? → Consider transform losses
- Are there outliers? → Use robust losses
- Is variance constant? → If not, use heteroscedastic models
- Are there exact zeros in continuous data? → Consider Compound Poisson

### Phase 2: Baseline Model

**Always start simple!**

```python
import torch.nn as nn
import torchregress as tr

# Simple baseline
class BaselineModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = BaselineModel(X_train.shape[1])
loss_fn = tr.losses.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
for epoch in range(100):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val)
        print(f"Epoch {epoch+1}: Train={loss:.4f}, Val={val_loss:.4f}")

# Evaluate baseline
with torch.no_grad():
    test_pred = model(X_test)
    baseline_rmse = tr.metrics.rmse(test_pred, y_test)
    baseline_mae = tr.metrics.mae(test_pred, y_test)
    baseline_r2 = tr.metrics.r2_score(test_pred, y_test)

print(f"\nBaseline Performance:")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"MAE: {baseline_mae:.4f}")
print(f"R²: {baseline_r2:.4f}")
```

**Why start with MSE?**
- Establishes performance floor
- Quick to train
- Easy to debug
- Identifies major issues

## Essential Sanity Checks

### ✅ Sanity Check 1: Can the Model Overfit?

**Test:** Train on tiny subset (10-50 samples)

```python
# Use 20 samples
tiny_X = X_train[:20]
tiny_y = y_train[:20]

model = BaselineModel(tiny_X.shape[1])
optimizer = torch.optim.Adam(model.parameters())
loss_fn = tr.losses.MSELoss()

# Train until near-zero loss
for epoch in range(1000):
    y_pred = model(tiny_X)
    loss = loss_fn(y_pred, tiny_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

# Check if model can memorize
final_pred = model(tiny_X)
overfit_rmse = tr.metrics.rmse(final_pred, tiny_y)

print(f"\nOverfit test RMSE: {overfit_rmse:.6f}")
if overfit_rmse > 0.01:
    print("⚠️  WARNING: Model cannot overfit! Check:")
    print("  - Learning rate (try 1e-2)")
    print("  - Model capacity (add layers/units)")
    print("  - Loss function implementation")
else:
    print("✓ Model can overfit (good!)")
```

**Expected:** RMSE should be near zero. If not, your model/training has issues.

### ✅ Sanity Check 2: Prediction Sanity

```python
with torch.no_grad():
    test_pred = model(X_test)

# 1. Check for invalid predictions
assert not torch.isnan(test_pred).any(), "NaN in predictions!"
assert not torch.isinf(test_pred).any(), "Inf in predictions!"

# 2. Check prediction range vs target range
pred_min, pred_max = test_pred.min(), test_pred.max()
target_min, target_max = y_test.min(), y_test.max()

print(f"Predictions range: [{pred_min:.2f}, {pred_max:.2f}]")
print(f"Targets range: [{target_min:.2f}, {target_max:.2f}]")

if pred_max < target_min or pred_min > target_max:
    print("⚠️  WARNING: Prediction range doesn't overlap with target range!")

# 3. Check if model is just predicting mean
pred_std = test_pred.std()
target_std = y_test.std()

print(f"\nPrediction std: {pred_std:.4f}")
print(f"Target std: {target_std:.4f}")

if pred_std < 0.1 * target_std:
    print("⚠️  WARNING: Model predicts nearly constant values!")
    print("  Check: learning rate, model capacity, feature scaling")
```

### ✅ Sanity Check 3: Residual Analysis

```python
import torchregress.viz as viz

with torch.no_grad():
    y_pred = model(X_test)

# Plot residuals
viz.plot_residuals(y_pred, y_test, title="Residual Analysis")

# Check for patterns
residuals = (y_pred - y_test).numpy()

# 1. Mean should be near zero
mean_residual = residuals.mean()
print(f"Mean residual: {mean_residual:.4f}")
if abs(mean_residual) > 0.1 * y_test.std().item():
    print("⚠️  WARNING: Residuals not centered at zero (bias!)")

# 2. Check heteroscedasticity
# Split predictions into quartiles
quartiles = np.quantile(y_pred.numpy(), [0.25, 0.5, 0.75])
q1_residuals = residuals[y_pred.numpy() <= quartiles[0]]
q4_residuals = residuals[y_pred.numpy() >= quartiles[2]]

std_ratio = q4_residuals.std() / q1_residuals.std()
print(f"Std ratio (Q4/Q1): {std_ratio:.2f}")

if std_ratio > 2.0 or std_ratio < 0.5:
    print("⚠️  Heteroscedasticity detected!")
    print("  → Consider: HeteroscedasticGaussianLoss")

# 3. Check for normality (for Gaussian assumptions)
viz.plot_qq_plot(y_pred, y_test, title="QQ Plot")
```

### ✅ Sanity Check 4: Train/Val/Test Consistency

```python
with torch.no_grad():
    train_pred = model(X_train)
    val_pred = model(X_val)
    test_pred = model(X_test)

    train_rmse = tr.metrics.rmse(train_pred, y_train)
    val_rmse = tr.metrics.rmse(val_pred, y_val)
    test_rmse = tr.metrics.rmse(test_pred, y_test)

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Val RMSE: {val_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# Check for major discrepancies
if val_rmse > 2 * train_rmse:
    print("⚠️  Severe overfitting!")
    print("  → Add regularization, dropout, or reduce model size")

if test_rmse > 1.5 * val_rmse:
    print("⚠️  Test performance much worse than validation!")
    print("  → Possible data distribution shift")
    print("  → Check: feature distributions, label quality")
```

## Loss Function Selection

### Decision Tree

```
Is your target continuous?
├─ NO (counts) → PoissonNLLLoss or NegativeBinomialNLLLoss
└─ YES
   └─ Is the distribution normal?
      ├─ YES
      │  └─ Is variance constant?
      │     ├─ YES → MSELoss or MAELoss
      │     └─ NO → HeteroscedasticGaussianLoss
      └─ NO
         └─ What's the issue?
            ├─ Outliers → HuberLoss or CauchyLoss
            ├─ Skewed → LogTransformLoss or BoxCoxTransformLoss
            ├─ Multi-modal → MixtureDensityLoss
            ├─ Exact zeros → CompoundPoissonLoss
            └─ Unknown → QuantileLoss (distribution-free)
```

### Loss Function Checklist

**For each loss, verify:**

1. **Input/Output shapes match**
   ```python
   # Example: HeteroscedasticGaussianLoss expects (mean, log_var)
   mean, log_var = model(X_test)
   print(f"Mean shape: {mean.shape}")
   print(f"Log_var shape: {log_var.shape}")
   print(f"Target shape: {y_test.shape}")

   # All should match!
   ```

2. **Loss doesn't explode**
   ```python
   losses = []
   for epoch in range(100):
       # ... training ...
       losses.append(loss.item())

   plt.plot(losses)
   plt.yscale('log')
   plt.title('Loss Curve (should decrease)')
   plt.show()

   if losses[-1] > losses[0]:
       print("⚠️  Loss not decreasing!")
   ```

3. **Predictions are reasonable**
   ```python
   # For Gaussian NLL
   mean, log_var = model(X_test)
   var = torch.exp(log_var)

   print(f"Mean predictions: {mean.min():.2f} to {mean.max():.2f}")
   print(f"Variance predictions: {var.min():.4f} to {var.max():.4f}")

   if var.min() < 0:
       print("⚠️  Negative variance! Check log_var output")
   if var.max() > 100 * y_test.std()**2:
       print("⚠️  Unreasonably large variance!")
   ```

## Uncertainty Quantification

### When to Use What

| Scenario | Method | Provides |
|----------|--------|----------|
| Quick uncertainty | HeteroscedasticGaussianLoss | Aleatoric only |
| Distribution-free intervals | QuantileLoss | Prediction intervals |
| Guaranteed coverage | ConformalLoss | Calibrated intervals |
| Model uncertainty | DeepEnsemble | Epistemic only |
| Full decomposition | Heteroscedastic Ensemble | Epistemic + Aleatoric |
| Multi-modal | MixtureDensityLoss | Full distribution |

### Uncertainty Validation

**1. Visual Inspection**

```python
import torchregress.viz as viz

# Prediction intervals
with torch.no_grad():
    mean, log_var = model(X_test)
    std = torch.exp(0.5 * log_var)
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std

viz.plot_prediction_intervals(
    y_pred=mean,
    y_lower=lower,
    y_upper=upper,
    y_true=y_test
)
```

**2. Calibration Metrics**

```python
# Check if 95% intervals actually contain 95% of data
picp = tr.metrics.picp(y_test, lower, upper)
print(f"Prediction Interval Coverage: {picp:.2%}")
print(f"Target: 95%")

if abs(picp - 0.95) > 0.05:
    print("⚠️  Poorly calibrated!")
    print("  → Use ConformalLoss for calibration")

# Check interval widths
mpiw = tr.metrics.mpiw(lower, upper)
print(f"Mean Interval Width: {mpiw:.4f}")
print(f"Target std: {y_test.std():.4f}")
```

**3. Reliability Diagram**

```python
viz.plot_reliability_diagram(
    y_pred_quantiles={
        0.05: quantile_05,
        0.25: quantile_25,
        0.50: quantile_50,
        0.75: quantile_75,
        0.95: quantile_95
    },
    y_true=y_test
)
```

## Evaluation Strategy

### Metric Selection

**Point Prediction Metrics:**
```python
# Basic metrics
mse = tr.metrics.mse(y_pred, y_true)
rmse = tr.metrics.rmse(y_pred, y_true)
mae = tr.metrics.mae(y_pred, y_true)
r2 = tr.metrics.r2_score(y_pred, y_true)

# Robust metrics (for outliers)
median_ae = tr.metrics.median_absolute_error(y_pred, y_true)
trimmed_mse = tr.metrics.trimmed_mean_squared_error(y_pred, y_true, trim=0.1)
```

**Probabilistic Metrics:**
```python
# Gaussian NLL
nll = tr.metrics.gaussian_nll(mean, y_true, var)

# CRPS (works for any distribution)
crps = tr.metrics.continuous_ranked_probability_score(samples, y_true)

# Interval metrics
picp = tr.metrics.picp(y_true, lower, upper)
interval_score = tr.metrics.interval_score(y_true, lower, upper, alpha=0.1)
```

### Cross-Validation Best Practices

```python
from sklearn.model_selection import KFold

def evaluate_with_cv(create_model_fn, X, y, n_splits=5):
    """Proper cross-validation with torchregress"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {
        'rmse': [],
        'mae': [],
        'r2': [],
        'picp': []  # If using uncertainty
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create fresh model
        model = create_model_fn()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = tr.losses.HeteroscedasticGaussianLoss()

        # Train
        for epoch in range(100):
            mean, log_var = model(X_train)
            var = torch.exp(log_var)
            loss = loss_fn(mean, y_train, var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        with torch.no_grad():
            mean, log_var = model(X_val)
            var = torch.exp(log_var)
            std = torch.sqrt(var)

            results['rmse'].append(tr.metrics.rmse(mean, y_val).item())
            results['mae'].append(tr.metrics.mae(mean, y_val).item())
            results['r2'].append(tr.metrics.r2_score(mean, y_val).item())

            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
            results['picp'].append(tr.metrics.picp(y_val, lower, upper).item())

    # Report results
    for metric, values in results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

    return results
```

## Common Pitfalls

### ❌ Pitfall 1: Not Scaling Features

```python
# BAD: Unscaled features
X_train = torch.randn(1000, 10) * torch.tensor([1, 1000, 0.01, ...])

# GOOD: Scale features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = torch.FloatTensor(scaler.fit_transform(X_train))
X_test_scaled = torch.FloatTensor(scaler.transform(X_test))
```

### ❌ Pitfall 2: Leaking Validation Data

```python
# BAD: Fit scaler on all data
scaler.fit(torch.cat([X_train, X_val, X_test]))

# GOOD: Fit only on training
scaler.fit(X_train)
```

### ❌ Pitfall 3: Ignoring Missing Data

```python
# BAD: Drop or fill without thought
X_train = X_train[~torch.isnan(X_train).any(dim=1)]

# GOOD: Use masks
mask = ~torch.isnan(X_train)
loss = loss_fn(y_pred, y_true, mask=mask)
```

### ❌ Pitfall 4: Wrong Metric for the Task

```python
# If you care about worst-case errors
# BAD: Use RMSE (sensitive to average)
# GOOD: Use max absolute error or quantile loss at 0.95

max_error = torch.abs(y_pred - y_true).max()
```

### ❌ Pitfall 5: Not Checking Calibration

```python
# BAD: Trust 95% intervals without checking
lower = mean - 1.96 * std
upper = mean + 1.96 * std

# GOOD: Always verify coverage
picp = tr.metrics.picp(y_test, lower, upper)
print(f"Actual coverage: {picp:.2%} (target: 95%)")
```

## Production Deployment

### Pre-Deployment Checklist

- [ ] Model passes all sanity checks
- [ ] Cross-validation results are stable
- [ ] Test set performance matches validation
- [ ] Uncertainty is calibrated
- [ ] Model handles edge cases (empty inputs, all-missing, extreme values)
- [ ] Predictions are deterministic (same input → same output)
- [ ] Model inference is fast enough
- [ ] Model size is acceptable
- [ ] Monitoring metrics are defined

### Monitoring in Production

```python
class ModelMonitor:
    """Monitor model performance in production"""

    def __init__(self, model, threshold_rmse=1.0):
        self.model = model
        self.threshold_rmse = threshold_rmse
        self.predictions = []
        self.errors = []

    def predict_and_monitor(self, X, y_true=None):
        """Make prediction and log for monitoring"""
        with torch.no_grad():
            y_pred = self.model(X)
            self.predictions.append(y_pred)

            if y_true is not None:
                error = tr.metrics.rmse(y_pred, y_true)
                self.errors.append(error.item())

                if error > self.threshold_rmse:
                    print(f"⚠️  HIGH ERROR: {error:.4f}")

        return y_pred

    def check_drift(self):
        """Check for distribution drift"""
        if len(self.predictions) < 100:
            return

        recent = torch.cat(self.predictions[-100:])
        historical = torch.cat(self.predictions[:-100])

        recent_mean = recent.mean()
        historical_mean = historical.mean()

        drift = abs(recent_mean - historical_mean) / historical.std()

        if drift > 2.0:
            print(f"⚠️  DISTRIBUTION DRIFT DETECTED: {drift:.2f} std")
```

### A/B Testing New Models

```python
def ab_test_models(model_a, model_b, X_test, y_test, n_bootstrap=1000):
    """Statistical comparison of two models"""

    with torch.no_grad():
        pred_a = model_a(X_test)
        pred_b = model_b(X_test)

    # Bootstrap confidence intervals
    rmse_diff = []

    for _ in range(n_bootstrap):
        idx = torch.randint(0, len(X_test), (len(X_test),))

        rmse_a = tr.metrics.rmse(pred_a[idx], y_test[idx])
        rmse_b = tr.metrics.rmse(pred_b[idx], y_test[idx])

        rmse_diff.append((rmse_b - rmse_a).item())

    rmse_diff = np.array(rmse_diff)

    print(f"RMSE difference (B - A): {rmse_diff.mean():.4f}")
    print(f"95% CI: [{np.percentile(rmse_diff, 2.5):.4f}, "
          f"{np.percentile(rmse_diff, 97.5):.4f}]")

    if np.percentile(rmse_diff, 97.5) < 0:
        print("✓ Model B is significantly better!")
    elif np.percentile(rmse_diff, 2.5) > 0:
        print("✓ Model A is significantly better!")
    else:
        print("= No significant difference")
```

## Summary: Your Regression Checklist

**Before training:**
- [ ] Understand data distribution
- [ ] Check for missing values, outliers, inf/nan
- [ ] Scale features appropriately
- [ ] Split data properly (train/val/test)

**During training:**
- [ ] Start with simple baseline (MSE)
- [ ] Verify model can overfit small dataset
- [ ] Monitor train/val curves
- [ ] Check for NaN/Inf in losses

**After training:**
- [ ] Analyze residuals
- [ ] Check prediction range vs target range
- [ ] Verify train/val/test consistency
- [ ] Assess uncertainty calibration (if applicable)

**Before deployment:**
- [ ] Cross-validate results
- [ ] Test on held-out set
- [ ] Check edge cases
- [ ] Set up monitoring
- [ ] Define success metrics

## Further Reading

- [Model Evaluation Guide](model-evaluation.md) - Comprehensive evaluation workflows
- [Debugging Checklist](debugging-checklist.md) - When things go wrong
- [Visualization Guide](../visualization/index.md) - Diagnostic plots
- [Loss Function Comparison](../examples/loss_comparison.md) - Choosing the right loss
