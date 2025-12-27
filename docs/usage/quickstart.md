# Quick Start

This guide provides a quick introduction to torchregress for regression and uncertainty estimation.

## Basic Usage

Here's a minimal example to get started with torchregress:

```python
import torch
import torchregress as tr
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic data
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(1000, 1))
y = X[:, 0]**2 + 0.1 * np.random.randn(1000)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Define a model
class RegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.network(x)
        
# Create model, loss function, and optimizer
model = RegressionModel()
loss_fn = tr.losses.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    y_pred = model(X_test)
    mse = tr.metrics.mse(y_pred, y_test)
    mae = tr.metrics.mae(y_pred, y_test)
    r2 = tr.metrics.R2Score()(y_pred, y_test)
    
print(f"MSE: {mse.item():.4f}")
print(f"MAE: {mae.item():.4f}")
print(f"R²: {r2.item():.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.3, label="True")
plt.scatter(X_test, y_pred, alpha=0.3, label="Predicted")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Basic Regression Example")
plt.show()
```

## Adding Uncertainty Estimation

Now let's extend the example to include uncertainty estimation:

```python
import torch
import torchregress as tr
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic heteroscedastic data
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(1000, 1))
noise = 0.1 + 0.5 * np.abs(X[:, 0])  # Heteroscedastic noise
y = X[:, 0]**2 + noise * np.random.randn(1000)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Define a model that predicts both mean and variance
class UncertaintyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        )
        self.mean_head = torch.nn.Linear(64, 1)
        self.logvar_head = torch.nn.Linear(64, 1)
        
    def forward(self, x):
        features = self.shared(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)  # Predicting log variance for numerical stability
        return mean, logvar
        
# Create model, loss function, and optimizer
model = UncertaintyModel()
loss_fn = tr.losses.GaussianNLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    # Forward pass
    mean, logvar = model(X_train)
    var = torch.exp(logvar)
    loss = loss_fn(mean, y_train, var)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    mean, logvar = model(X_test)
    var = torch.exp(logvar)
    std = torch.sqrt(var)
    
    # Compute metrics
    mse = tr.metrics.mse(mean, y_test)
    nll = tr.metrics.gaussian_nll(mean, y_test, var)
    
    # Compute prediction intervals (95%)
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    
    # Coverage metrics
    picp = tr.metrics.prediction_interval_coverage_probability(lower, upper, y_test)
    mpiw = torch.mean(upper - lower)
    
print(f"MSE: {mse.item():.4f}")
print(f"NLL: {nll.item():.4f}")
print(f"PICP (95%): {picp.item():.4f}")
print(f"MPIW: {mpiw.item():.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.3, label="True")
plt.scatter(X_test, mean.detach(), alpha=0.3, label="Predicted")

# Sort for better visualization
idx = X_test.flatten().argsort()
X_plot = X_test[idx]
mean_plot = mean[idx]
lower_plot = lower[idx]
upper_plot = upper[idx]

plt.plot(X_plot, mean_plot, 'r-', linewidth=2)
plt.fill_between(X_plot.flatten(), 
                 lower_plot.flatten(), 
                 upper_plot.flatten(), 
                 alpha=0.2, color='r', 
                 label="95% Prediction Interval")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Regression with Uncertainty Estimation")
plt.show()

# Plot calibration curve
tr.viz.plot_calibration_curve(mean, std, y_test)
```

## Using Robust Losses

Here's how to use robust loss functions for datasets with outliers:

```python
import torch
import torchregress as tr
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic data with outliers
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(1000, 1))
y_clean = X[:, 0]**2
# Add normal noise to most points
y = y_clean + 0.1 * np.random.randn(1000)
# Add extreme outliers to 5% of the data
outlier_idx = np.random.choice(1000, 50, replace=False)
y[outlier_idx] = y_clean[outlier_idx] + np.random.uniform(-5, 5, size=50)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Compare different loss functions
loss_functions = {
    'MSE': tr.losses.MSELoss(),
    'MAE': tr.losses.L1Loss(),
    'Huber': tr.losses.HuberLoss(delta=1.0),
    'LogCosh': tr.losses.LogCoshLoss(),
    'Cauchy': tr.losses.CauchyLoss(scale=1.0)
}

results = {}

# Train models with different loss functions
for name, loss_fn in loss_functions.items():
    # Define model
    model = RegressionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(100):
        # Forward pass
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        y_pred = model(X_test)
        results[name] = y_pred.detach()

# Plot results
plt.figure(figsize=(12, 8))
plt.scatter(X_test, y_test, alpha=0.5, label="True (with outliers)")

# Create true function line for reference
X_sorted = torch.sort(X_test, dim=0)[0]
y_true_clean = X_sorted**2
plt.plot(X_sorted, y_true_clean, 'k--', label="True function (no outliers)")

# Plot predictions
for name, y_pred in results.items():
    # Sort for better visualization
    idx = X_test.flatten().argsort()
    X_plot = X_test[idx]
    y_pred_plot = y_pred[idx]
    
    plt.plot(X_plot, y_pred_plot, label=f"{name}")

plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Comparing Loss Functions with Outliers")
plt.show()

# Compare metrics
for name, y_pred in results.items():
    mse = tr.metrics.mse(y_pred, y_test)
    mae = tr.metrics.mae(y_pred, y_test)
    print(f"{name} - MSE: {mse.item():.4f}, MAE: {mae.item():.4f}")
```

## Next Steps

Now that you've seen the basics of torchregress, you can:

1. **Explore more loss functions** for your specific regression needs
2. **Implement uncertainty estimation** to provide prediction intervals
3. **Use evaluation metrics** to properly assess your models
4. **Visualize predictions and uncertainty** with the provided tools

For more advanced use cases, check out:

- [Practical Usage Guide](practical_usage.md) for detailed recommendations
- [Examples](../examples/index.md) for complete code examples
[Learn more about the mathematical formulations of these losses →](../math/index.md) for the theoretical background
