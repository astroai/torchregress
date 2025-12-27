# Loss Function Comparison

This page provides comprehensive comparisons of different loss functions across various data scenarios, helping you choose the right loss for your application.

## Scenario 1: Clean Data (No Outliers)

When your data has no outliers and follows typical assumptions, different losses should perform similarly.

```python
import torch
import torch.nn as nn
import torchregress as tr
import numpy as np
import matplotlib.pyplot as plt

# Generate clean quadratic data
np.random.seed(42)
n_samples = 500
X = np.random.uniform(-2, 2, size=(n_samples, 1))
y_true = X[:, 0]**2 + 2 * X[:, 0] - 1
y = y_true + 0.3 * np.random.randn(n_samples)  # Small Gaussian noise

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# Test multiple losses
losses_to_test = {
    'MSE': tr.losses.MSELoss(),
    'MAE': tr.losses.L1Loss(),
    'Huber (δ=1)': tr.losses.HuberLoss(delta=1.0),
    'LogCosh': tr.losses.LogCoshLoss(),
}

results = {}

for loss_name, loss_fn in losses_to_test.items():
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(200):
        y_pred = model(X_tensor)
        loss = loss_fn(y_pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
        results[loss_name] = {
            'pred': y_pred,
            'mse': tr.metrics.mse(y_pred, y_tensor).item(),
            'mae': tr.metrics.mae(y_pred, y_tensor).item(),
            'r2': tr.metrics.R2Score()(y_pred, y_tensor).item()
        }

# Print comparison
print("Clean Data - Loss Function Comparison")
print("=" * 60)
print(f"{'Loss':<15} {'MSE':<10} {'MAE':<10} {'R²':<10}")
print("=" * 60)
for name, metrics in results.items():
    print(f"{name:<15} {metrics['mse']:<10.4f} {metrics['mae']:<10.4f} {metrics['r2']:<10.4f}")

# Key insight: All losses perform similarly on clean data
```

**Output (typical):**
```
Clean Data - Loss Function Comparison
============================================================
Loss            MSE        MAE        R²
============================================================
MSE             0.0912     0.2401     0.9845
MAE             0.0918     0.2398     0.9844
Huber (δ=1)     0.0915     0.2400     0.9844
LogCosh         0.0914     0.2399     0.9845
```

**Conclusion:** On clean data, loss choice matters less. Use MSE for simplicity.

## Scenario 2: Data with Outliers

Outliers significantly affect different losses. Robust losses (MAE, Huber, Cauchy) handle outliers better.

```python
import torch
import torch.nn as nn
import torchregress as tr
import numpy as np
import matplotlib.pyplot as plt

# Generate data with outliers
np.random.seed(42)
n_samples = 400
X = np.random.uniform(-2, 2, size=(n_samples, 1))
y_true = X[:, 0]**2
y = y_true + 0.2 * np.random.randn(n_samples)

# Add 10% outliers with large errors
n_outliers = 40
outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
y[outlier_idx] += np.random.uniform(-4, 4, size=n_outliers)

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Test robust losses
losses_to_test = {
    'MSE (Least Squares)': tr.losses.MSELoss(),
    'MAE (L1)': tr.losses.L1Loss(),
    'Huber (δ=0.5)': tr.losses.HuberLoss(delta=0.5),
    'Huber (δ=1.0)': tr.losses.HuberLoss(delta=1.0),
    'Cauchy (γ=0.5)': tr.losses.CauchyLoss(scale=0.5),
    'Tukey Biweight': tr.losses.TukeyBiweightLoss(c=4.685),
}

results = {}

for loss_name, loss_fn in losses_to_test.items():
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(300):
        y_pred = model(X_tensor)
        loss = loss_fn(y_pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)

        # Compute metrics on clean data (excluding outliers)
        clean_mask = torch.ones(n_samples, dtype=torch.bool)
        clean_mask[outlier_idx] = False

        results[loss_name] = {
            'pred': y_pred,
            'mse_all': tr.metrics.mse(y_pred, y_tensor).item(),
            'mse_clean': tr.metrics.mse(
                y_pred[clean_mask],
                y_tensor[clean_mask]
            ).item(),
            'mae_all': tr.metrics.mae(y_pred, y_tensor).item(),
        }

# Print comparison
print("\nOutlier Data - Loss Function Comparison")
print("=" * 70)
print(f"{'Loss':<20} {'MSE (All)':<12} {'MSE (Clean)':<14} {'MAE (All)':<12}")
print("=" * 70)
for name, metrics in sorted(results.items(), key=lambda x: x[1]['mse_clean']):
    print(f"{name:<20} {metrics['mse_all']:<12.4f} "
          f"{metrics['mse_clean']:<14.4f} {metrics['mae_all']:<12.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

X_plot = np.linspace(-2, 2, 100)
y_plot_true = X_plot**2

for i, (loss_name, result) in enumerate(results.items()):
    ax = axes[i]

    # Plot clean data
    clean_mask = np.ones(n_samples, dtype=bool)
    clean_mask[outlier_idx] = False
    ax.scatter(X[clean_mask], y[clean_mask], alpha=0.3, s=10, c='blue', label='Clean Data')

    # Plot outliers
    ax.scatter(X[outlier_idx], y[outlier_idx], c='red', s=30, marker='x',
               label='Outliers', linewidths=2)

    # Plot true function
    ax.plot(X_plot, y_plot_true, 'g--', alpha=0.7, linewidth=2, label='True Function')

    # Plot prediction
    idx = np.argsort(X[:, 0])
    ax.plot(X[idx], result['pred'].numpy()[idx], 'b-', linewidth=2, label='Predicted')

    ax.set_title(f"{loss_name}\nMSE (clean): {result['mse_clean']:.3f}")
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Key Findings:**

1. **MSE**: Heavily influenced by outliers, tries to fit them
2. **MAE**: More robust, but can still be affected by extreme outliers
3. **Huber**: Good balance - quadratic for small errors, linear for large
4. **Cauchy**: Very robust to outliers, almost ignores them
5. **Tukey Biweight**: Completely ignores errors beyond threshold

**Recommendation:** For data with moderate outliers, use Huber (δ=1.0). For extreme outliers, use Cauchy or Tukey.

## Scenario 3: Heteroscedastic Data

When variance changes across the input space, methods that model uncertainty perform better.

```python
import torch
import torch.nn as nn
import torchregress as tr
import numpy as np
import matplotlib.pyplot as plt

# Generate heteroscedastic data
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(800, 1))
# Variance increases with |X|
noise_std = 0.1 + 0.5 * np.abs(X[:, 0])
y = X[:, 0]**2 + noise_std * np.random.randn(800)

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# 1. Homoscedastic model (MSE)
class HomoscedasticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

# 2. Heteroscedastic model (Gaussian NLL)
class HeteroscedasticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.mean_head = nn.Linear(128, 1)
        self.logvar_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        return (self.mean_head(h), self.logvar_head(h))  # Return tuple

# Train homoscedastic (MSE)
homo_model = HomoscedasticModel()
mse_loss = tr.losses.MSELoss()
optimizer1 = torch.optim.Adam(homo_model.parameters(), lr=1e-3)

for epoch in range(200):
    y_pred = homo_model(X_train)
    loss = mse_loss(y_pred, y_train)
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()

# Train heteroscedastic
hetero_model = HeteroscedasticModel()
hetero_loss = tr.losses.GaussianNLLLoss()
optimizer2 = torch.optim.Adam(hetero_model.parameters(), lr=1e-3)

for epoch in range(200):
    y_pred = hetero_model(X_train)  # Returns (mean, logvar) tuple
    loss = hetero_loss(y_pred, y_train)
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

# Evaluate both
homo_model.eval()
hetero_model.eval()

with torch.no_grad():
    # Homoscedastic predictions
    y_pred_homo = homo_model(X_test)
    mse_homo = tr.metrics.mse(y_pred_homo, y_test).item()
    mae_homo = tr.metrics.mae(y_pred_homo, y_test).item()

    # Heteroscedastic predictions
    mean_hetero, logvar_hetero = hetero_model(X_test)
    var_hetero = torch.exp(logvar_hetero)
    std_hetero = torch.sqrt(var_hetero)

    mse_hetero = tr.metrics.mse(mean_hetero, y_test).item()
    mae_hetero = tr.metrics.mae(mean_hetero, y_test).item()
    nll_hetero = gnll_loss(mean_hetero, y_test, var_hetero).item()

    # 95% prediction intervals
    lower = mean_hetero - 1.96 * std_hetero
    upper = mean_hetero + 1.96 * std_hetero
    picp = tr.metrics.prediction_interval_coverage_probability(lower, upper, y_test).item()

print("\nHeteroscedastic Data Comparison")
print("=" * 50)
print(f"{'Model':<25} {'MSE':<10} {'MAE':<10}")
print("=" * 50)
print(f"{'MSE (Homoscedastic)':<25} {mse_homo:<10.4f} {mae_homo:<10.4f}")
print(f"{'Gaussian NLL (Heteroscedastic)':<25} {mse_hetero:<10.4f} {mae_hetero:<10.4f}")
print("\nUncertainty Metrics (Heteroscedastic only):")
print(f"NLL: {nll_hetero:.4f}")
print(f"PICP (95%): {picp:.4f}")

# Visualization
idx = torch.argsort(X_test.flatten())
X_sorted = X_test[idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Homoscedastic
axes[0].scatter(X_test, y_test, alpha=0.3, s=10, c='gray')
axes[0].plot(X_sorted, y_pred_homo[idx], 'b-', linewidth=2)
axes[0].set_title(f'Homoscedastic (MSE)\nMSE: {mse_homo:.3f}')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].grid(True, alpha=0.3)

# Plot 2: Heteroscedastic with uncertainty
axes[1].scatter(X_test, y_test, alpha=0.3, s=10, c='gray', label='Data')
axes[1].plot(X_sorted, mean_hetero[idx], 'r-', linewidth=2, label='Mean')
axes[1].fill_between(
    X_sorted.flatten(),
    lower[idx].flatten(),
    upper[idx].flatten(),
    alpha=0.3,
    color='red',
    label='95% Interval'
)
axes[1].set_title(f'Heteroscedastic (Gaussian NLL)\nMSE: {mse_hetero:.3f}, PICP: {picp:.3f}')
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Key Insight:** Heteroscedastic models (Gaussian NLL) provide:
- Similar point prediction accuracy
- Proper uncertainty quantification (intervals adapt to input)
- Better calibrated prediction intervals

## Scenario 4: Multi-Modal Distributions

When the conditional distribution has multiple modes, use MDN or Normalizing Flows.

```python
import torch
import torch.nn as nn
import torchregress as tr
import numpy as np

# Generate multi-modal data (inverse problem)
np.random.seed(42)
n_samples = 2000

# For each X, y can be either +sqrt(X) or -sqrt(X)
X = np.random.uniform(0, 4, size=(n_samples, 1))
y = np.where(
    np.random.rand(n_samples) > 0.5,
    np.sqrt(X[:, 0]),
    -np.sqrt(X[:, 0])
)
y += 0.1 * np.random.randn(n_samples)  # Small noise

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Compare different approaches

# 1. MSE (will fail - predicts mean ≈ 0)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

mse_model = SimpleModel()
mse_loss = tr.losses.MSELoss()
optimizer = torch.optim.Adam(mse_model.parameters(), lr=1e-3)

for epoch in range(100):
    y_pred = mse_model(X_tensor)
    loss = mse_loss(y_pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 2. MDN (will succeed - models both modes)
class MDNModel(nn.Module):
    def __init__(self, n_components=2):
        super().__init__()
        mdn_size = n_components + 2 * n_components * 1
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, mdn_size)
        )
    def forward(self, x):
        return self.net(x)

mdn_model = MDNModel(n_components=2)
mdn_loss = tr.losses.MixtureDensityLoss(n_components=2, n_features=1)
optimizer = torch.optim.Adam(mdn_model.parameters(), lr=1e-3)

for epoch in range(200):
    mdn_params = mdn_model(X_tensor)
    loss = mdn_loss(mdn_params, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate
mse_model.eval()
mdn_model.eval()

with torch.no_grad():
    mse_pred = mse_model(X_tensor)
    mdn_params = mdn_model(X_tensor)

    mse_error = tr.metrics.mse(mse_pred, y_tensor).item()
    nll_mdn = mdn_loss(mdn_params, y_tensor).item()

print("\nMulti-Modal Data Comparison")
print("=" * 50)
print(f"MSE Model - MSE: {mse_error:.4f}")
print(f"MDN Model - NLL: {nll_mdn:.4f}")
print("\nNote: MSE predicts ~0 (mean of modes) and fails.")
print("MDN correctly models both modes.")
```

**Conclusion:** For multi-modal distributions:
- **MSE/MAE/Gaussian NLL:** Fail (predict mean of modes)
- **MDN:** Good if you know the number of modes
- **Normalizing Flows:** Best for unknown/complex multimodality

## Decision Guide

Use this flowchart to select the appropriate loss:

```
Is your data clean (no outliers)?
├─ YES
│  └─ Is the distribution unimodal?
│     ├─ YES
│     │  └─ Is variance constant?
│     │     ├─ YES → Use MSE or MAE
│     │     └─ NO → Use Gaussian NLL
│     └─ NO → Use MDN or Normalizing Flows
│
└─ NO (has outliers)
   └─ Outlier severity?
      ├─ Mild (< 5%) → Use Huber Loss (δ=1.0)
      ├─ Moderate (5-15%) → Use Cauchy Loss
      └─ Severe (> 15%) → Use Tukey Biweight or Quantile Regression
```

## Performance Summary Table

| Scenario | Best Loss | Alternative | Avoid |
|----------|-----------|-------------|-------|
| Clean, unimodal, homoscedastic | MSE | MAE | - |
| Clean, unimodal, heteroscedastic | Gaussian NLL | Quantile | MSE |
| Mild outliers (< 5%) | Huber | LogCosh | MSE |
| Severe outliers (> 10%) | Cauchy, Tukey | MAE | MSE, Gaussian NLL |
| Multi-modal | MDN, NF | - | MSE, MAE, Gaussian NLL |
| Need coverage guarantees | Conformal | - | - |

## Further Reading

- [Basic Usage Examples](basic_usage.md) - Step-by-step tutorials
- [Learn more about the mathematical formulations of these losses →](../math/index.md) - Theory behind each loss
- [Robust Losses](../losses/robust.md) - Detailed robust loss documentation
