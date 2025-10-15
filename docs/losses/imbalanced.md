# Imbalanced Regression

Imbalanced regression addresses problems where the target distribution is highly non-uniform, with some regions having many samples (dense) and others having very few samples (sparse). This is common in real-world scenarios like predicting rare events, extreme values, or skewed distributions.

## Quick Start

Here's a quick example of how to use `DensityWeightedLoss` to handle imbalanced regression:

```python
import torch
import torch.nn as nn
from torchregress.losses import DensityWeightedLoss

# 1. Create synthetic imbalanced data
X = torch.randn(1000, 10)
y = torch.cat([torch.randn(900, 1), 5 + torch.randn(100, 1)])

# 2. Create a model
model = nn.Sequential(nn.Linear(10, 1))

# 3. Create the loss function
loss_fn = DensityWeightedLoss(kernel_width=0.5)

# 4. Fit the density on the training targets
loss_fn.fit_density(y)

# 5. Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
```

## The Imbalance Problem

Traditional regression methods optimize average performance across all samples, which can lead to:

- **Poor performance on rare values**: The model focuses on dense regions
- **Biased predictions**: Systematic underestimation or overestimation of extreme values
- **Loss of important information**: Rare cases may be more critical (e.g., high-value transactions, extreme weather)

## Mathematical Background

In imbalanced regression, the target density $p(y)$ is highly non-uniform:

$$p(y) = \begin{cases}
\text{high} & \text{for common values (dense region)} \\
\text{low} & \text{for rare values (sparse region)}
\end{cases}$$

Standard losses treat all samples equally:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \ell(f(x_i), y_i)$$

Imbalanced regression methods reweight samples based on their rarity:

$$\mathcal{L}_{\text{weighted}} = \frac{1}{N}\sum_{i=1}^N w(y_i) \cdot \ell(f(x_i), y_i)$$

where $w(y_i)$ is inversely proportional to the density at $y_i$.

## Critical Consideration: Calibration

**WARNING**: Many imbalanced regression methods can break model calibration. This is a fundamental trade-off:

- **Reweighting samples**: Changes the empirical distribution, affecting calibration
- **Changing targets**: Directly breaks calibration properties

**Best Practice**: Always validate calibration after training with imbalanced methods and consider post-hoc calibration techniques.

## Available Methods

### DensityWeightedLoss (Calibration-Safe ✅)

```python
class DensityWeightedLoss(RegressionLoss)
```

Weights samples inversely proportional to their local density in target space. This is the **recommended** method as it **preserves calibration** by only reweighting training samples without changing the conditional distribution $p(y|x)$.

**Parameters**:

- `kernel_width` (float, optional): Bandwidth for kernel density estimation. Default: 0.5
  - Smaller values → more local density estimation
  - Larger values → smoother density estimation
- `base_loss` (str, optional): Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
- `reweight_factor` (float, optional): Strength of reweighting (0-1). Default: 1.0
  - 0 = no reweighting (uniform weights)
  - 1 = full inverse density weighting
- `reduction` (str, optional): Loss reduction method. Default: 'mean'

**Methods**:

- `fit_density(train_targets)`: Estimate target density and compute inverse density weights
- `forward(y_pred, y_true, sample_indices=None, mask=None, weights=None)`: Compute density-weighted loss

**Example**:

```python
import torch
import torch.nn as nn
from torchregress.losses import DensityWeightedLoss

# Create synthetic imbalanced data
# Most samples around y=0, few samples at extreme values
torch.manual_seed(42)
X_dense = torch.randn(800, 5)
y_dense = torch.randn(800, 1)

X_rare = torch.randn(200, 5)
y_rare = 5 + 2 * torch.randn(200, 1)  # Rare extreme values

X_train = torch.cat([X_dense, X_rare])
y_train = torch.cat([y_dense, y_rare])

# Shuffle
indices = torch.randperm(len(X_train))
X_train, y_train = X_train[indices], y_train[indices]

# Create model
model = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Create density-weighted loss
loss_fn = DensityWeightedLoss(
    kernel_width=0.5,
    base_loss='mse',
    reweight_factor=1.0  # Full reweighting
)

# IMPORTANT: Fit density on training targets before training
loss_fn.fit_density(y_train)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()

    # Option 1: Pass target values directly (slower, computes density on-the-fly)
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    # Option 2: Pass sample indices (faster, uses precomputed weights)
    # Create DataLoader that returns indices
    # for x_batch, y_batch, indices in train_loader:
    #     y_pred = model(x_batch)
    #     loss = loss_fn(y_pred, y_batch, sample_indices=indices)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Evaluate on rare values
model.eval()
with torch.no_grad():
    y_pred_rare = model(X_rare)
    mae_rare = torch.abs(y_pred_rare - y_rare).mean()
    print(f"MAE on rare values: {mae_rare:.4f}")

# Validate calibration (IMPORTANT!)
from torchregress.metrics import calibration_error
# cal_err = calibration_error(model, val_loader)
# print(f"Calibration error: {cal_err:.4f}")
```

**When to Use**:

- ✅ Imbalanced target distribution (rare extreme values)
- ✅ Need to maintain calibration (critical for uncertainty)
- ✅ Acceptable to fit density before training
- ✅ Have enough samples to estimate density reliably

**Tuning Tips**:

1. **kernel_width**: Start with 0.5, decrease if too smooth, increase if too noisy
2. **reweight_factor**: Start with 1.0, reduce to 0.5-0.7 if overfitting to rare samples
3. **base_loss**: Use 'huber' for robustness to outliers

### LDSLoss (Label Distribution Smoothing) ⚠️

```python
class LDSLoss(RegressionLoss)
```

Smooths the label distribution using kernel smoothing and reweights samples based on effective label frequency. This is more aggressive than DensityWeightedLoss but **CAN BREAK CALIBRATION**.

**WARNING**: This method modifies the effective training distribution through smoothing, which can affect calibration. Always validate calibration and apply post-hoc calibration if needed.

**Parameters**:

- `kernel` (str, optional): Kernel type for smoothing ('gaussian', 'triang', 'laplace'). Default: 'gaussian'
- `kernel_width` (float, optional): Bandwidth for kernel smoothing. Default: 2.0
- `reweight_factor` (float, optional): Strength of reweighting (0-1). Default: 1.0
- `base_loss` (str, optional): Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
- `reduction` (str, optional): Loss reduction method. Default: 'mean'

**Methods**:

- `fit(train_targets, n_bins=100)`: Compute LDS weights from training targets
- `forward(y_pred, y_true, sample_indices=None, mask=None, weights=None)`: Compute LDS-weighted loss

**Example**:

```python
import torch
import torch.nn as nn
from torchregress.losses import LDSLoss
from torchregress.metrics import expected_calibration_error

# Create LDS loss
loss_fn = LDSLoss(
    kernel='gaussian',
    kernel_width=2.0,
    reweight_factor=0.8  # Moderate reweighting
)

# Fit on training targets
loss_fn.fit(y_train, n_bins=100)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    loss.backward()
    optimizer.step()

# CRITICAL: Validate calibration after training
model.eval()
with torch.no_grad():
    y_pred_val = model(X_val)
    # For calibration, you need prediction intervals
    # If using heteroscedastic model:
    # mean, std = y_pred_val
    # ece = expected_calibration_error(mean, std, y_val)

    # Check if predictions are systematically biased
    bias = (y_pred_val - y_val).mean()
    print(f"Prediction bias: {bias:.4f}")

    # Check performance on rare values
    rare_mask = torch.abs(y_val) > 3.0
    if rare_mask.sum() > 0:
        mae_rare = torch.abs(y_pred_val[rare_mask] - y_val[rare_mask]).mean()
        print(f"MAE on rare values: {mae_rare:.4f}")

# If calibration is poor, apply post-hoc calibration:
# 1. Temperature scaling for uncertainty
# 2. Isotonic regression for prediction correction
# 3. Platt scaling for probabilistic outputs
```

**When to Use**:

- ✅ Extremely imbalanced data (very few rare samples)
- ✅ Performance on rare values is more important than calibration
- ✅ Can afford post-hoc calibration validation/correction
- ⚠️ Understand calibration may be affected

**Post-Hoc Calibration**:

```python
# Example: Simple bias correction
from sklearn.isotonic import IsotonicRegression

# Fit on validation set
model.eval()
with torch.no_grad():
    y_pred_val = model(X_val).cpu().numpy()
    y_val_np = y_val.cpu().numpy()

# Isotonic regression for monotonic correction
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(y_pred_val.flatten(), y_val_np.flatten())

# Apply correction at test time
y_pred_test = model(X_test).cpu().numpy()
y_pred_calibrated = iso_reg.predict(y_pred_test.flatten())
```

## When Does Imbalance Matter?

Not all non-uniform distributions require special treatment. Use imbalanced regression methods when:

### ✅ Use Imbalanced Methods When:

1. **Extreme skew**: >80% of data in <20% of target range
2. **Sparse critical regions**: <5% of samples in important range
3. **Performance gaps**: Model performs 2-3× worse on rare values
4. **Domain requirements**: Rare cases are more valuable/critical

### ❌ Don't Use When:

1. **Mild imbalance**: Reasonably uniform distribution
2. **Natural rarity**: Rare values are genuinely rare in population
3. **Sufficient samples**: Even "rare" regions have 100+ samples
4. **Calibration critical**: Uncertainty estimates must be accurate

## Decision Guide

```
┌─ Is Your Regression Imbalanced? ────────────────────────┐
│                                                          │
│  Check your target distribution:                         │
│  1. Plot histogram of y_train                            │
│  2. Calculate density at different quantiles             │
│  3. Measure performance across target ranges             │
│                                                          │
│  Density Ratio > 10:1 (dense vs sparse)?                │
│  ├─ No → Standard losses sufficient                     │
│  └─ Yes → Continue below                                 │
│                                                          │
│  Is calibration critical?                                │
│  ├─ Yes → DensityWeightedLoss (calibration-safe)       │
│  └─ No → Continue below                                  │
│                                                          │
│  Extremely sparse rare regions (<1% of data)?            │
│  ├─ Yes → LDSLoss + post-hoc calibration               │
│  └─ No → DensityWeightedLoss                            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Comparison: Method Selection

| Method | Calibration | Performance on Rare | Complexity | Best For |
|--------|------------|-------------------|-----------|----------|
| **Standard Loss** | ✅ Perfect | ❌ Poor | Simple | Balanced data |
| **DensityWeightedLoss** | ✅ Preserved | ⭐⭐ Good | Medium | General imbalance, need calibration |
| **LDSLoss** | ⚠️ May break | ⭐⭐⭐ Excellent | Medium | Extreme imbalance, can tolerate miscalibration |

## Complete Example: Comparison

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchregress.losses import DensityWeightedLoss, LDSLoss

# Generate highly imbalanced data
torch.manual_seed(42)
n_common = 900
n_rare = 100

# Common values: y ~ N(0, 1)
X_common = torch.randn(n_common, 5)
y_common = torch.randn(n_common, 1)

# Rare extreme values: y ~ N(5, 0.5)
X_rare = torch.randn(n_rare, 5)
y_rare = 5 + 0.5 * torch.randn(n_rare, 1)

X_train = torch.cat([X_common, X_rare])
y_train = torch.cat([y_common, y_rare])

# Shuffle
idx = torch.randperm(len(X_train))
X_train, y_train = X_train[idx], y_train[idx]

# Separate test data
X_test_common = torch.randn(200, 5)
y_test_common = torch.randn(200, 1)

X_test_rare = torch.randn(50, 5)
y_test_rare = 5 + 0.5 * torch.randn(50, 1)

def train_model(loss_fn, name):
    """Train a model with given loss function."""
    model = nn.Sequential(
        nn.Linear(5, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"{name} - Epoch {epoch+1}: Loss = {loss.item():.4f}")

    return model

# Method 1: Standard MSE Loss
print("=== Training with Standard MSE ===")
mse_loss = nn.MSELoss()
model_mse = train_model(mse_loss, "MSE")

# Method 2: DensityWeightedLoss
print("\n=== Training with DensityWeightedLoss ===")
density_loss = DensityWeightedLoss(kernel_width=0.5, reweight_factor=1.0)
density_loss.fit_density(y_train)
model_density = train_model(density_loss, "DensityWeighted")

# Method 3: LDSLoss
print("\n=== Training with LDSLoss ===")
lds_loss = LDSLoss(kernel='gaussian', kernel_width=2.0, reweight_factor=0.8)
lds_loss.fit(y_train, n_bins=50)
model_lds = train_model(lds_loss, "LDS")

# Evaluate all models
print("\n=== Evaluation Results ===")

models = {
    'MSE': model_mse,
    'DensityWeighted': model_density,
    'LDS': model_lds
}

for name, model in models.items():
    model.eval()
    with torch.no_grad():
        # Common values
        y_pred_common = model(X_test_common)
        mae_common = torch.abs(y_pred_common - y_test_common).mean()

        # Rare values
        y_pred_rare = model(X_test_rare)
        mae_rare = torch.abs(y_pred_rare - y_test_rare).mean()

        # Overall
        X_test_all = torch.cat([X_test_common, X_test_rare])
        y_test_all = torch.cat([y_test_common, y_test_rare])
        y_pred_all = model(X_test_all)
        mae_all = torch.abs(y_pred_all - y_test_all).mean()

        print(f"\n{name}:")
        print(f"  MAE (common values): {mae_common:.4f}")
        print(f"  MAE (rare values):   {mae_rare:.4f}")
        print(f"  MAE (overall):       {mae_all:.4f}")
        print(f"  Rare/Common ratio:   {mae_rare/mae_common:.2f}x")

# Expected output:
# MSE: Good on common, poor on rare (high ratio)
# DensityWeighted: Balanced performance (lower ratio)
# LDS: Best on rare, may sacrifice common slightly
```

## Best Practices

### 1. Always Check If You Need It

```python
import matplotlib.pyplot as plt
import numpy as np

# Visualize target distribution
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(y_train.numpy(), bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.title('Target Distribution')

# Check density ratio
plt.subplot(132)
sorted_y = np.sort(y_train.numpy().flatten())
percentiles = np.percentile(sorted_y, [10, 25, 50, 75, 90])
counts, bins = np.histogram(y_train.numpy(), bins=50)
plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black')
for p in percentiles:
    plt.axvline(p, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Target Value')
plt.ylabel('Count')
plt.title('Density Across Quantiles')

# Performance across target ranges
plt.subplot(133)
# Train baseline model and check errors by target range
# (code to compute errors by range)
plt.xlabel('Target Range')
plt.ylabel('Mean Absolute Error')
plt.title('Performance vs Target Value')

plt.tight_layout()
plt.show()

# Quantitative check
from scipy.stats import iqr
target_iqr = iqr(y_train.numpy())
target_range = y_train.max() - y_train.min()
sparsity_ratio = target_range / target_iqr

print(f"Target range / IQR: {sparsity_ratio:.2f}")
if sparsity_ratio > 5:
    print("→ Consider imbalanced regression methods")
else:
    print("→ Standard methods likely sufficient")
```

### 2. Validate Calibration

```python
def check_calibration(model, X_test, y_test):
    """Check if model is calibrated on test set."""
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

        # Residuals should be centered around zero
        residuals = y_pred - y_test
        bias = residuals.mean()
        std = residuals.std()

        print(f"Residual bias: {bias:.4f} (should be ~0)")
        print(f"Residual std: {std:.4f}")

        # Check across target ranges
        for quantile in [0.1, 0.5, 0.9]:
            threshold = y_test.quantile(quantile)
            mask = y_test < threshold
            if mask.sum() > 10:
                bias_q = residuals[mask].mean()
                print(f"Bias at {quantile*100}th percentile: {bias_q:.4f}")

check_calibration(model, X_val, y_val)
```

### 3. Hyperparameter Tuning

```python
# Grid search for best kernel_width and reweight_factor
from sklearn.model_selection import ParameterGrid

param_grid = {
    'kernel_width': [0.3, 0.5, 0.7, 1.0],
    'reweight_factor': [0.5, 0.7, 0.9, 1.0]
}

best_score = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    # Train with these parameters
    loss_fn = DensityWeightedLoss(**params)
    loss_fn.fit_density(y_train)

    model = train_model(loss_fn)

    # Evaluate on validation set (focus on rare values)
    val_mae_rare = evaluate_on_rare(model, X_val, y_val)

    if val_mae_rare < best_score:
        best_score = val_mae_rare
        best_params = params

print(f"Best params: {best_params}")
print(f"Best MAE on rare: {best_score:.4f}")
```

## Common Pitfalls

### ❌ Pitfall 1: Over-weighting Rare Samples

```python
# Too aggressive reweighting
loss_fn = DensityWeightedLoss(reweight_factor=1.0, kernel_width=0.1)
# → Model overfits to rare samples, poor overall performance
```

**Solution**: Start with `reweight_factor=0.7-0.8` and tune on validation set.

### ❌ Pitfall 2: Ignoring Calibration

```python
# Using LDSLoss without calibration check
loss_fn = LDSLoss()
# Train model
# Deploy to production ← DANGER: predictions may be biased!
```

**Solution**: Always validate calibration and apply correction if needed.

### ❌ Pitfall 3: Wrong Kernel Width

```python
# Kernel too wide → doesn't capture local density
loss_fn = DensityWeightedLoss(kernel_width=5.0)

# Kernel too narrow → overfits to noise
loss_fn = DensityWeightedLoss(kernel_width=0.01)
```

**Solution**: Visualize density estimate and tune on validation performance.

### ❌ Pitfall 4: Insufficient Rare Samples

```python
# Only 10 samples in rare region
# → Density estimation unreliable
```

**Solution**: Need at least 50-100 samples per region for reliable density estimation. Consider data augmentation or different approach if too sparse.

## References

- Yang et al. "Delving into Deep Imbalanced Regression" (ICML 2021)
- Steininger et al. "Density-based weighting for imbalanced regression" (Machine Learning 2021)
- Branco et al. "A Survey of Predictive Modeling on Imbalanced Domains" (ACM Computing Surveys 2016)
