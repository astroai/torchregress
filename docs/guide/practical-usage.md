# Practical Usage Guide

This guide provides practical advice on using torchregress effectively for various regression tasks.

For a task-first shortlist before diving into implementation details, use the
[Task-First Method Selection Matrix](../guide/method-selection.md).

## Choosing the Right Loss Function

Selecting the appropriate loss function is crucial for successful regression modeling. Here's a decision tree to help you choose:

1. **Are there outliers in your data?**
   - **Yes**: Consider robust losses like `WeightedHuberLoss`, `LogCoshLoss`, or `CauchyLoss`
   - **No**: Standard losses like `WeightedMSELoss` or `WeightedL1Loss` may be sufficient

2. **Do you need uncertainty estimates?**
   - **Yes**:
     - **Is your uncertainty heteroscedastic (varies with input)?** Use `GaussianNLLLoss`
     - **Is your distribution multi-modal?** Use `MDNLoss` (Mixture Density Networks)
     - **Do you need flexible distribution shapes?** Use `NormalizingFlowLoss`
   - **No**: Standard point prediction losses are sufficient

3. **Are you working with count data or non-negative values?**
   - **Yes**: Consider `PoissonDevianceLoss` or `TweedieLoss`
   - **No**: Standard losses are appropriate

4. **Do you need prediction intervals without distributional assumptions?**
   - **Yes**: Use `QuantileLoss` with multiple quantiles (e.g., 0.05 and 0.95 for 90% intervals)
   - **No**: Parametric uncertainty methods may be more efficient

5. **Is there uncertainty in your input features (not just targets)?**
   - **Yes**: Consider error-in-variables methods like `OrthogonalDistanceRegressionLoss`
   - **No**: Standard methods are appropriate

## Common Loss Function Combinations

Here are some commonly used loss function combinations for different scenarios:

### General-Purpose Regression

```python
# Standard regression with robust loss
loss_fn = tr.losses.WeightedHuberLoss(delta=1.0)

# OR: With uncertainty estimation
loss_fn = tr.losses.GaussianNLLLoss()
```

### Financial Time Series

```python
# For asymmetric risk preferences (e.g., penalize underestimation more)
loss_fn = tr.losses.QuantileLoss(quantile=0.6)  # Biased toward overestimation

# OR: For Value-at-Risk prediction
q_loss = tr.losses.MultiQuantileLoss(quantiles=[0.01, 0.05, 0.1])
```

### Scientific Applications

```python
# For physical processes with known error characteristics
loss_fn = tr.losses.GaussianNLLLoss()

# OR: For processes with heavy tails
loss_fn = tr.losses.CauchyLoss(scale=1.0)
```

### Computer Vision Regression

```python
# For robust keypoint regression
loss_fn = tr.losses.CharbonnierLoss(eps=1e-3)

# OR: For multi-modal outputs (e.g., multiple possible hand positions)
loss_fn = tr.losses.MDNLoss(components=5)
```

## Training with Uncertainty

When training models to predict uncertainty, remember these principles:

1. **Proper Scoring Rules**: Always use proper scoring rules like NLL or CRPS to train uncertainty-aware models
2. **Calibration**: Check calibration of uncertainty estimates using reliability diagrams
3. **Multiple Sources**: Consider epistemic (model) and aleatoric (data) uncertainty sources
4. **Regularization**: Use appropriate regularization to prevent overconfidence or underconfidence

Example code for training with uncertainty:

```python
import torch
import torchregress as tr

# Define model that outputs mean and log_variance
class UncertaintyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        )
        self.mean_head = torch.nn.Linear(64, 1)
        self.logvar_head = torch.nn.Linear(64, 1)

    def forward(self, x):
        features = self.backbone(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)  # Predicting log variance for numerical stability
        return mean, logvar

# Create model and loss
model = UncertaintyModel()
loss_fn = tr.losses.GaussianNLLLoss()

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):
    for X_batch, y_batch in dataloader:
        # Forward pass
        mean, logvar = model(X_batch)
        # Convert log variance to variance
        var = torch.exp(logvar)
        # Calculate loss
        loss = loss_fn((mean, logvar), y_batch)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Evaluation Best Practices

To thoroughly evaluate regression models, especially those with uncertainty estimates:

1. **Multiple Metrics**: Use both point prediction metrics (RMSE, MAE) and distributional metrics (NLL, CRPS)
2. **Calibration Analysis**: Check if uncertainty estimates match empirical error rates
3. **Visual Diagnostics**: Create residual plots, Q-Q plots, and calibration curves
4. **OOD Detection**: Test if uncertainty increases appropriately for out-of-distribution samples

Example evaluation code:

```python
import torch
import torchregress as tr

# Evaluation mode
model.eval()
with torch.no_grad():
    # Get predictions
    mean, logvar = model(X_test)
    var = torch.exp(logvar)
    std = torch.sqrt(var)

    # Point prediction metrics
    rmse = tr.metrics.rmse(mean, y_test)
    mae = tr.metrics.mae(mean, y_test)
    r2 = tr.metrics.R2Score()(mean, y_test)

    # Distribution metrics
    nll = tr.metrics.gaussian_nll(mean, y_test, var)
    crps = tr.metrics.crps_gaussian(mean, y_test, std)

    # Interval metrics (90% prediction intervals)
    lower = mean - 1.645 * std  # 5th percentile
    upper = mean + 1.645 * std  # 95th percentile
    picp = tr.metrics.prediction_interval_coverage_probability(lower, upper, y_test)
    mpiw = torch.mean(upper - lower)

# Visualization
tr.viz.plot_prediction_intervals(mean, lower, upper, y_true=y_test)
tr.viz.plot_calibration_curve(mean, std, y_test)
tr.viz.plot_residuals(mean, y_test)
```

## When to Use Ensemble Methods

Ensemble methods are particularly useful for:

1. Capturing **epistemic uncertainty** (model uncertainty)
2. Improving robustness and generalization
3. Avoiding local minima during optimization

torchregress provides several ensemble approaches:

```python
# Deep ensemble (train multiple models with different initializations)
models = [create_model() for _ in range(5)]
ensemble = tr.ensemble.DeepEnsemble(models)

# SWAG (posterior approximation over weights)
ensemble = tr.ensemble.SWAG(base_model)

# MC-Dropout wrapper (cheap uncertainty baseline)
ensemble = tr.ensemble.MCDropoutWrapper(base_model, n_samples=30)
```

## Performance Optimization

To optimize performance when using torchregress:

1. **Batch Operations**: Use vectorized operations instead of loops
2. **Data Types**: Use appropriate precision (float32 is typically sufficient)
3. **GPU Acceleration**: Move tensors and models to GPU when available
4. **Numerical Stability**: Use stable implementations (e.g., log-space calculations)

Example of optimized code:

```python
# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X = X.to(device)
y = y.to(device)

# Use mixed precision for speed (PyTorch >= 2.0)
from torch.amp import GradScaler, autocast
scaler = GradScaler("cuda")

# Training loop with mixed precision
for epoch in range(100):
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Use mixed precision
        with autocast(device_type="cuda", dtype=torch.float16):
            mean, logvar = model(X_batch)
            var = torch.exp(logvar)
            loss = loss_fn((mean, logvar), y_batch)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## Common Pitfalls and Solutions

1. **Numerical Instability**
   - **Problem**: Infinity or NaN values in loss calculation
   - **Solution**: Use log-space computation, clamp small values, use stable implementations

2. **Overconfident Uncertainty**
   - **Problem**: Model predicts too small uncertainty
   - **Solution**: Add regularization, use ensembling, check for data leakage

3. **Underconfident Uncertainty**
   - **Problem**: Model predicts unnecessarily large uncertainty
   - **Solution**: Increase training data, check for model misspecification

4. **Poorly Calibrated Uncertainty**
   - **Problem**: Predictive intervals don't match empirical frequencies
   - **Solution**: Use post-hoc calibration, temperature scaling, or better loss functions

5. **Slow Training**
   - **Problem**: Training takes too long
   - **Solution**: Use GPU acceleration, optimize batch size, use mixed precision
