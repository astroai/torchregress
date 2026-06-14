# Basic Usage Examples

This page provides complete, runnable examples for common regression tasks using `torchregress`.

| # | Reference |
|:-:|:----------|
| 1 | Nix, D. A., & Weigend, A. S. (1994). **Estimating the mean and variance of target distributions**. *IEEE International Conference on Neural Networks*. |
| 2 | Huber, P. J. (1964). **Robust Estimation of a Location Parameter**. *Annals of Mathematical Statistics*. |
| 3 | Koenker, R., & Bassett Jr, G. (1978). **Regression Quantiles**. *Econometrica*. |

---

## Mathematical Formulations

*   **Mean Squared Error (MSE)**:
    $$L_{\text{MSE}}(y, \hat{y}) = (y - \hat{y})^2$$
*   **Huber Loss (Robust)**:
    $$L_{\text{Huber}}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
*   **Pinball / Quantile Loss**:
    $$L_q(y, \hat{y}) = \max(q(y - \hat{y}), (q-1)(y - \hat{y}))$$
*   **Gaussian Negative Log-Likelihood (NLL)**:
    $$L_{\text{NLL}}(y, \hat{y}, \sigma^2) = \frac{1}{2}\log\sigma^2 + \frac{(y - \hat{y})^2}{2\sigma^2} + \frac{1}{2}\log(2\pi)$$

---


## Example 1: Simple MSE Regression

The most basic regression task - predicting a continuous value with Mean Squared Error loss.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchregress as tr
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(1000, 1))
y = X[:, 0]**2 + 0.5 * np.random.randn(1000)

# Convert to tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Split data
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, X_val, X_test = (
    X_tensor[:train_size],
    X_tensor[train_size:train_size+val_size],
    X_tensor[train_size+val_size:]
)
y_train, y_val, y_test = (
    y_tensor[:train_size],
    y_tensor[train_size:train_size+val_size],
    y_tensor[train_size+val_size:]
)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define model
class SimpleRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

# Create model, loss, optimizer
model = SimpleRegressor()
loss_fn = tr.losses.WeightedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
n_epochs = 100
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    # Training
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        # Forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = loss_fn(val_pred, y_val)

    train_losses.append(epoch_loss / len(train_loader))
    val_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}")

# Evaluation on test set
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)

    # Calculate metrics
    mse = tr.metrics.mse(y_pred_test, y_test)
    rmse = tr.metrics.rmse(y_pred_test, y_test)
    mae = tr.metrics.mae(y_pred_test, y_test)
    r2 = tr.metrics.R2Score()(y_pred_test, y_test)

print(f"\nTest Set Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Learning curves
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses, label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Learning Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Predictions vs True
idx = torch.argsort(X_test.flatten())
axes[1].scatter(X_test, y_test, alpha=0.3, label='True', s=20)
axes[1].plot(X_test[idx], y_pred_test[idx], 'r-', linewidth=2, label='Predicted')
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].set_title('Predictions on Test Set')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Example 2: Heteroscedastic Regression with Uncertainty

Predict both mean and variance to capture input-dependent uncertainty.

```python
import torch
import torch.nn as nn
import torchregress as tr
import numpy as np
import matplotlib.pyplot as plt

# Generate heteroscedastic data (variance increases with |x|)
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(1000, 1))
noise_std = 0.1 + 0.3 * np.abs(X[:, 0])  # Heteroscedastic noise
y = X[:, 0]**2 + noise_std * np.random.randn(1000)

# Convert and split
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Model that predicts mean and log variance
class HeteroscedasticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mean_head = nn.Linear(128, 1)
        self.logvar_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return (mean, logvar)  # Return as tuple for GaussianNLLLoss

# Create model and loss
model = HeteroscedasticModel()
loss_fn = tr.losses.GaussianNLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    y_pred = model(X_train)  # Returns (mean, logvar)

    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_test)
            val_loss = loss_fn(y_pred_val, y_test)
        print(f"Epoch {epoch+1}/{n_epochs}, "
              f"Train Loss: {loss.item():.4f}, "
              f"Val Loss: {val_loss.item():.4f}")

# Evaluation with uncertainty quantification
model.eval()
with torch.no_grad():
    mean_pred, logvar_pred = model(X_test)
    var_pred = torch.exp(logvar_pred)
    std_pred = torch.sqrt(var_pred)

    # 95% prediction intervals
    lower = mean_pred - 1.96 * std_pred
    upper = mean_pred + 1.96 * std_pred

    # Calculate metrics
    rmse = tr.metrics.rmse(mean_pred, y_test)
    nll = loss_fn((mean_pred, torch.log(var_pred)), y_test)
    picp = tr.metrics.prediction_interval_coverage_probability(lower, upper, y_test)
    mpiw = torch.mean(upper - lower)

print(f"\nTest Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"NLL: {nll:.4f}")
print(f"PICP (95%): {picp:.4f} (should be ~0.95)")
print(f"MPIW: {mpiw:.4f}")

# Visualization with uncertainty bands
idx = torch.argsort(X_test.flatten())
X_sorted = X_test[idx]
mean_sorted = mean_pred[idx]
lower_sorted = lower[idx]
upper_sorted = upper[idx]

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.3, s=20, label='True Data')
plt.plot(X_sorted, mean_sorted, 'r-', linewidth=2, label='Predicted Mean')
plt.fill_between(
    X_sorted.flatten(),
    lower_sorted.flatten(),
    upper_sorted.flatten(),
    alpha=0.3,
    color='red',
    label='95% Prediction Interval'
)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Heteroscedastic Regression with Uncertainty')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Example 3: Robust Regression with Outliers

Handle outliers using robust loss functions.

```python
import torch
import torch.nn as nn
import torchregress as tr
import numpy as np
import matplotlib.pyplot as plt

# Generate data with outliers
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(500, 1))
y_clean = X[:, 0]**2

# Add normal noise
y = y_clean + 0.2 * np.random.randn(500)

# Add outliers (10% of data)
n_outliers = 50
outlier_idx = np.random.choice(500, n_outliers, replace=False)
y[outlier_idx] += np.random.uniform(-8, 8, size=n_outliers)

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Define simple model
class RobustModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# Compare different loss functions
loss_functions = {
    'MSE': tr.losses.WeightedMSELoss(),
    'MAE (L1)': tr.losses.WeightedL1Loss(),
    'Huber': tr.losses.WeightedHuberLoss(delta=1.0),
    'Cauchy': tr.losses.CauchyLoss(scale=1.0),
    'LogCosh': tr.losses.LogCoshLoss()
}

results = {}

for loss_name, loss_fn in loss_functions.items():
    print(f"\nTraining with {loss_name}...")

    # Create fresh model
    model = RobustModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Train
    for epoch in range(100):
        y_pred = model(X_tensor)
        loss = loss_fn(y_pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Store results
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
        results[loss_name] = {
            'predictions': y_pred.clone(),
            'rmse': tr.metrics.rmse(y_pred, y_tensor).item(),
            'mae': tr.metrics.mae(y_pred, y_tensor).item()
        }

    print(f"{loss_name} - RMSE: {results[loss_name]['rmse']:.4f}, "
          f"MAE: {results[loss_name]['mae']:.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# True function for reference
X_true = np.linspace(-3, 3, 100).reshape(-1, 1)
y_true = X_true[:, 0]**2

for i, (loss_name, result) in enumerate(results.items()):
    ax = axes[i]

    # Plot data
    ax.scatter(X, y, alpha=0.3, s=10, c='gray', label='Data (with outliers)')
    ax.scatter(X[outlier_idx], y[outlier_idx], c='red', s=30,
               marker='x', label='Outliers', linewidths=2)

    # Plot true function
    ax.plot(X_true, y_true, 'g--', linewidth=2, alpha=0.7, label='True Function')

    # Plot prediction
    idx = torch.argsort(X_tensor.flatten())
    ax.plot(X_tensor[idx], result['predictions'][idx],
            'b-', linewidth=2, label='Predicted')

    ax.set_title(f"{loss_name}\nRMSE: {result['rmse']:.3f}, MAE: {result['mae']:.3f}")
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Remove extra subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.show()

# Print comparison
print("\n" + "="*50)
print("Performance Comparison (lower is better):")
print("="*50)
for loss_name, result in sorted(results.items(),
                                 key=lambda x: x[1]['rmse']):
    print(f"{loss_name:15s}: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}")
```

## Example 4: Quantile Regression for Prediction Intervals

Predict multiple quantiles to obtain prediction intervals without distributional assumptions.

```python
import torch
import torch.nn as nn
import torchregress as tr
import numpy as np
import matplotlib.pyplot as plt

# Generate heteroscedastic data
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(1000, 1))
noise_std = 0.1 + 0.4 * np.abs(X[:, 0])
y = X[:, 0]**2 + noise_std * np.random.randn(1000)

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Model predicts multiple quantiles simultaneously
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
n_quantiles = len(quantiles)

class QuantileModel(nn.Module):
    def __init__(self, n_quantiles=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_quantiles)  # One output per quantile
        )

    def forward(self, x):
        return self.network(x)

model = QuantileModel(n_quantiles=n_quantiles)
loss_fn = tr.losses.MultiQuantileLoss(quantiles=quantiles)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(200):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test)
            val_loss = loss_fn(val_pred, y_test)
        print(f"Epoch {epoch+1}/200, Train Loss: {loss.item():.4f}, "
              f"Val Loss: {val_loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    quantile_preds = model(X_test)

    # Extract individual quantiles
    q05 = quantile_preds[:, 0:1]  # 5th percentile
    q25 = quantile_preds[:, 1:2]  # 25th percentile
    q50 = quantile_preds[:, 2:3]  # Median (50th)
    q75 = quantile_preds[:, 3:4]  # 75th percentile
    q95 = quantile_preds[:, 4:5]  # 95th percentile

    # Calculate interval metrics
    picp_50 = tr.metrics.prediction_interval_coverage_probability(q25, q75, y_test)
    picp_90 = tr.metrics.prediction_interval_coverage_probability(q05, q95, y_test)
    mpiw_50 = torch.mean(q75 - q25)
    mpiw_90 = torch.mean(q95 - q05)

print(f"\nTest Metrics:")
print(f"50% Interval Coverage: {picp_50:.4f} (should be ~0.50)")
print(f"90% Interval Coverage: {picp_90:.4f} (should be ~0.90)")
print(f"50% Interval Width: {mpiw_50:.4f}")
print(f"90% Interval Width: {mpiw_90:.4f}")

# Visualization
idx = torch.argsort(X_test.flatten())
X_sorted = X_test[idx]

plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, alpha=0.3, s=20, label='True Data', c='gray')
plt.plot(X_sorted, q50[idx], 'b-', linewidth=2, label='Median (q50)')

# 50% interval
plt.fill_between(
    X_sorted.flatten(),
    q25[idx].flatten(),
    q75[idx].flatten(),
    alpha=0.3,
    color='blue',
    label='50% Interval (q25-q75)'
)

# 90% interval
plt.fill_between(
    X_sorted.flatten(),
    q05[idx].flatten(),
    q95[idx].flatten(),
    alpha=0.15,
    color='blue',
    label='90% Interval (q05-q95)'
)

plt.xlabel('X')
plt.ylabel('y')
plt.title('Quantile Regression with Prediction Intervals')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Next Steps

- [Loss Comparison Examples](loss_comparison.md) - Detailed comparison of different loss functions
- [API Reference](../api/index.md) - Complete API documentation
