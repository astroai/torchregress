# Photometric Redshift Estimation

This example demonstrates using torchregress for a real-world astronomy application: estimating galaxy redshifts from photometric observations.

## Background

**Photometric redshift (photo-z)** estimation is a fundamental problem in astronomy where we predict the redshift (distance) of galaxies from broadband imaging data. This is a challenging regression problem because:

- The relationship between colors and redshift is complex and non-linear
- Measurement uncertainties vary significantly across observations
- Some galaxies have ambiguous color-redshift mappings (multi-modal distributions)
- Accurate uncertainty quantification is critical for cosmological analyses

## Problem Setup

We'll demonstrate several approaches:
1. Simple point prediction (MSE)
2. Heteroscedastic uncertainty (Gaussian NLL)
3. Deep ensemble for epistemic uncertainty
4. Mixture Density Network for multi-modal distributions

```python
import torch
import torch.nn as nn
import torchregress as tr
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic photo-z data
# In practice, you would use real survey data (e.g., SDSS, DES, LSST)
np.random.seed(42)

def generate_photoz_data(n_samples=5000):
    """
    Generate synthetic photometry and redshifts
    Features: 5 colors (u-g, g-r, r-i, i-z, z-y)
    Target: Redshift (0 to 2)
    """
    # True redshifts
    z_true = np.random.uniform(0, 2, size=n_samples)

    # Simulated colors (simplified model)
    # In reality, these come from galaxy spectral energy distributions
    colors = np.zeros((n_samples, 5))

    # u-g color increases with redshift
    colors[:, 0] = 1.0 + 0.8 * z_true + 0.2 * np.random.randn(n_samples)

    # g-r color has complex relationship
    colors[:, 1] = 0.5 + 0.3 * np.sin(3 * z_true) + 0.15 * np.random.randn(n_samples)

    # r-i, i-z colors
    colors[:, 2] = 0.3 + 0.5 * z_true + 0.1 * np.random.randn(n_samples)
    colors[:, 3] = 0.2 + 0.4 * z_true**2 + 0.1 * np.random.randn(n_samples)

    # z-y color
    colors[:, 4] = 0.1 + 0.2 * z_true + 0.1 * np.random.randn(n_samples)

    # Add photometric errors (heteroscedastic)
    # Fainter galaxies (higher redshift) have larger errors
    photo_errors = 0.02 + 0.08 * (z_true / 2.0)
    colors += photo_errors[:, np.newaxis] * np.random.randn(n_samples, 5)

    return colors, z_true, photo_errors

# Generate data
colors, redshifts, errors = generate_photoz_data(5000)

# Convert to tensors
X = torch.FloatTensor(colors)
y = torch.FloatTensor(redshifts).reshape(-1, 1)

# Split: 60% train, 20% calibration, 20% test
n_train = 3000
n_cal = 1000

X_train, X_cal, X_test = X[:n_train], X[n_train:n_train+n_cal], X[n_train+n_cal:]
y_train, y_cal, y_test = y[:n_train], y[n_train:n_train+n_cal], y[n_train+n_cal:]

print(f"Training set: {len(X_train)}")
print(f"Calibration set: {len(X_cal)}")
print(f"Test set: {len(X_test)}")
```

## Approach 1: Simple Point Estimation (MSE)

Start with a baseline model using MSE loss.

```python
class PhotoZModel(nn.Module):
    """Simple feed-forward network for photo-z prediction"""
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive redshift
        )

    def forward(self, x):
        return self.network(x)

# Train MSE model
mse_model = PhotoZModel()
mse_loss = tr.losses.MSELoss()
optimizer = torch.optim.Adam(mse_model.parameters(), lr=1e-3)

# Training loop
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

n_epochs = 100
for epoch in range(n_epochs):
    mse_model.train()
    for X_batch, y_batch in train_loader:
        y_pred = mse_model(X_batch)
        loss = mse_loss(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        mse_model.eval()
        with torch.no_grad():
            val_pred = mse_model(X_cal)
            val_loss = mse_loss(val_pred, y_cal)
        print(f"Epoch {epoch+1}/{n_epochs}, Val Loss: {val_loss:.4f}")

# Evaluate
mse_model.eval()
with torch.no_grad():
    y_pred_mse = mse_model(X_test)

    # Photo-z metrics
    rmse = tr.metrics.rmse(y_pred_mse, y_test).item()
    mae = tr.metrics.mae(y_pred_mse, y_test).item()

    # Catastrophic outlier rate (|z_pred - z_true| > 0.15(1+z))
    threshold = 0.15 * (1 + y_test)
    outlier_rate = (torch.abs(y_pred_mse - y_test) > threshold).float().mean().item()

    # Normalized median absolute deviation (NMAD)
    residuals = (y_pred_mse - y_test) / (1 + y_test)
    nmad = 1.48 * torch.median(torch.abs(residuals - torch.median(residuals))).item()

print(f"\nMSE Model Results:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"NMAD: {nmad:.4f}")
print(f"Outlier rate: {outlier_rate:.2%}")
```

## Approach 2: Uncertainty-Aware Prediction (Gaussian NLL)

Model both redshift and its uncertainty.

```python
class HeteroscedasticPhotoZModel(nn.Module):
    """Photo-z model with heteroscedastic uncertainty"""
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Positive redshift
        )
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return (mean, logvar)  # Return as tuple

# Train heteroscedastic model
hetero_model = HeteroscedasticPhotoZModel()
hetero_loss = tr.losses.HeteroscedasticGaussianLoss(learnable_variance=False)
optimizer = torch.optim.Adam(hetero_model.parameters(), lr=1e-3)

for epoch in range(100):
    hetero_model.train()
    for X_batch, y_batch in train_loader:
        y_pred = hetero_model(X_batch)  # Returns (mean, logvar) tuple
        loss = hetero_loss(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        hetero_model.eval()
        with torch.no_grad():
            y_pred_val = hetero_model(X_cal)
            val_loss = hetero_loss(y_pred_val, y_cal)
        print(f"Epoch {epoch+1}/{n_epochs}, Val Loss: {val_loss:.4f}")

# Evaluate with uncertainty
hetero_model.eval()
with torch.no_grad():
    mean_pred, logvar_pred = hetero_model(X_test)
    var_pred = torch.exp(logvar_pred)
    std_pred = torch.sqrt(var_pred)

    # Prediction intervals
    lower_68 = mean_pred - std_pred
    upper_68 = mean_pred + std_pred
    lower_95 = mean_pred - 1.96 * std_pred
    upper_95 = mean_pred + 1.96 * std_pred

    # Metrics
    rmse = tr.metrics.rmse(mean_pred, y_test).item()
    nll = gnll_loss(mean_pred, y_test, var_pred).item()
    picp_68 = tr.metrics.picp(y_test, lower_68, upper_68).item()
    picp_95 = tr.metrics.picp(y_test, lower_95, upper_95).item()

print(f"\nHeteroscedastic Model Results:")
print(f"RMSE: {rmse:.4f}")
print(f"NLL: {nll:.4f}")
print(f"PICP 68%: {picp_68:.2%} (target: 68%)")
print(f"PICP 95%: {picp_95:.2%} (target: 95%)")
```

## Approach 3: Deep Ensemble for Epistemic Uncertainty

Use an ensemble to quantify model uncertainty.

```python
class EnsemblePhotoZ:
    """Deep ensemble for photo-z with uncertainty decomposition"""
    def __init__(self, n_models=5):
        self.models = [HeteroscedasticPhotoZModel() for _ in range(n_models)]
        self.n_models = n_models

    def train(self, train_loader, n_epochs=100):
        loss_fn = tr.losses.HeteroscedasticGaussianLoss(learnable_variance=False)

        for i, model in enumerate(self.models):
            print(f"\nTraining ensemble member {i+1}/{self.n_models}")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            for epoch in range(n_epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    y_pred = model(X_batch)  # Returns (mean, logvar) tuple
                    loss = loss_fn(y_pred, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{n_epochs}")

    def predict(self, x):
        """
        Returns:
            mean: Ensemble mean prediction
            epistemic: Epistemic uncertainty (variance of means)
            aleatoric: Aleatoric uncertainty (mean of variances)
        """
        predictions = []
        variances = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                mean, logvar = model(x)
                var = torch.exp(logvar)
                predictions.append(mean)
                variances.append(var)

        predictions = torch.stack(predictions)
        variances = torch.stack(variances)

        # Ensemble mean
        ensemble_mean = predictions.mean(dim=0)

        # Epistemic uncertainty (disagreement between models)
        epistemic = predictions.var(dim=0)

        # Aleatoric uncertainty (average predicted variance)
        aleatoric = variances.mean(dim=0)

        # Total uncertainty
        total = epistemic + aleatoric

        return ensemble_mean, epistemic, aleatoric, total

# Train ensemble
ensemble = EnsemblePhotoZ(n_models=5)
ensemble.train(train_loader, n_epochs=80)

# Evaluate ensemble
mean_ens, epistemic, aleatoric, total = ensemble.predict(X_test)

rmse_ens = tr.metrics.rmse(mean_ens, y_test).item()
std_total = torch.sqrt(total)

# Prediction intervals using total uncertainty
lower = mean_ens - 1.96 * std_total
upper = mean_ens + 1.96 * std_total
picp = tr.metrics.picp(y_test, lower, upper).item()

print(f"\nDeep Ensemble Results:")
print(f"RMSE: {rmse_ens:.4f}")
print(f"PICP 95%: {picp:.2%}")
print(f"Mean Epistemic Std: {torch.sqrt(epistemic).mean():.4f}")
print(f"Mean Aleatoric Std: {torch.sqrt(aleatoric).mean():.4f}")
print(f"Mean Total Std: {std_total.mean():.4f}")
```

## Visualization and Analysis

```python
import matplotlib.pyplot as plt

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Scatter plot: True vs Predicted (MSE model)
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, y_pred_mse, alpha=0.3, s=10)
ax1.plot([0, 2], [0, 2], 'r--', linewidth=2)
ax1.set_xlabel('True Redshift')
ax1.set_ylabel('Predicted Redshift')
ax1.set_title('MSE Model')
ax1.grid(True, alpha=0.3)

# 2. Scatter plot: True vs Predicted (Heteroscedastic)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, mean_pred, alpha=0.3, s=10)
ax2.plot([0, 2], [0, 2], 'r--', linewidth=2)
ax2.set_xlabel('True Redshift')
ax2.set_ylabel('Predicted Redshift')
ax2.set_title('Heteroscedastic Model')
ax2.grid(True, alpha=0.3)

# 3. Scatter plot: True vs Predicted (Ensemble)
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(y_test, mean_ens, alpha=0.3, s=10)
ax3.plot([0, 2], [0, 2], 'r--', linewidth=2)
ax3.set_xlabel('True Redshift')
ax3.set_ylabel('Predicted Redshift')
ax3.set_title('Deep Ensemble')
ax3.grid(True, alpha=0.3)

# 4. Residuals (MSE)
ax4 = fig.add_subplot(gs[1, 0])
residuals_mse = y_pred_mse - y_test
ax4.scatter(y_test, residuals_mse, alpha=0.3, s=10)
ax4.axhline(0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('True Redshift')
ax4.set_ylabel('Residual')
ax4.set_title('MSE Residuals')
ax4.grid(True, alpha=0.3)

# 5. Uncertainty vs Error (Heteroscedastic)
ax5 = fig.add_subplot(gs[1, 1])
errors = torch.abs(mean_pred - y_test)
ax5.scatter(std_pred, errors, alpha=0.3, s=10)
ax5.plot([0, std_pred.max()], [0, std_pred.max()], 'r--', linewidth=2)
ax5.set_xlabel('Predicted Std Dev')
ax5.set_ylabel('Absolute Error')
ax5.set_title('Uncertainty Calibration')
ax5.grid(True, alpha=0.3)

# 6. Uncertainty Decomposition (Ensemble)
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(torch.sqrt(epistemic), torch.sqrt(aleatoric), alpha=0.3, s=10)
ax6.set_xlabel('Epistemic Std')
ax6.set_ylabel('Aleatoric Std')
ax6.set_title('Uncertainty Decomposition')
ax6.grid(True, alpha=0.3)

# 7. Distribution of errors (all methods)
ax7 = fig.add_subplot(gs[2, :])
errors_mse = (y_pred_mse - y_test).flatten()
errors_hetero = (mean_pred - y_test).flatten()
errors_ens = (mean_ens - y_test).flatten()

ax7.hist(errors_mse, bins=50, alpha=0.5, label='MSE', density=True)
ax7.hist(errors_hetero, bins=50, alpha=0.5, label='Heteroscedastic', density=True)
ax7.hist(errors_ens, bins=50, alpha=0.5, label='Ensemble', density=True)
ax7.set_xlabel('Prediction Error')
ax7.set_ylabel('Density')
ax7.set_title('Error Distributions')
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.suptitle('Photo-z Estimation: Method Comparison', fontsize=16, y=0.995)
plt.show()

# Print comparison table
print("\n" + "="*70)
print("Method Comparison Summary")
print("="*70)
print(f"{'Method':<25} {'RMSE':<10} {'NMAD':<10} {'PICP 95%':<10}")
print("="*70)
print(f"{'MSE (baseline)':<25} {rmse:.4f}     {nmad:.4f}     {'N/A':<10}")

hetero_nmad = 1.48 * torch.median(torch.abs(
    (mean_pred - y_test)/(1 + y_test) -
    torch.median((mean_pred - y_test)/(1 + y_test))
)).item()
print(f"{'Heteroscedastic':<25} {rmse:.4f}     {hetero_nmad:.4f}     {picp_95:.4f}")

ens_nmad = 1.48 * torch.median(torch.abs(
    (mean_ens - y_test)/(1 + y_test) -
    torch.median((mean_ens - y_test)/(1 + y_test))
)).item()
print(f"{'Deep Ensemble':<25} {rmse_ens:.4f}     {ens_nmad:.4f}     {picp:.4f}")
print("="*70)
```

## Key Takeaways for Photo-z

1. **Simple MSE**: Provides point estimates but no uncertainty quantification
2. **Heteroscedastic**: Captures varying uncertainty across redshift range
3. **Deep Ensemble**: Separates epistemic (model) and aleatoric (data) uncertainty

**For astronomical applications:**
- Use heteroscedastic models when photometric errors dominate
- Use ensembles when you need to identify regions of high model uncertainty
- Consider MDN for galaxies with ambiguous color-redshift mappings
- Always report calibration metrics (PICP, NMAD) alongside RMSE

## Extension: Conformal Prediction for Guaranteed Coverage

For critical downstream analyses, use conformal prediction:

```python
# Train quantile model
quantile_model = PhotoZModel()
quantile_loss = tr.losses.MultiQuantileLoss(quantiles=[0.05, 0.95])

# Training loop (similar to above)
# ...

# Conformalize on calibration set
conformal = tr.losses.ConformalLoss(method='cqr', alpha=0.1)
with torch.no_grad():
    cal_quantiles = quantile_model(X_cal)
conformal.calibrate(cal_quantiles, y_cal)

# Get guaranteed 90% coverage on test set
with torch.no_grad():
    test_quantiles = quantile_model(X_test)
    lower_conf, upper_conf = conformal.predict_interval(test_quantiles)

coverage_conf = ((y_test >= lower_conf) & (y_test <= upper_conf)).float().mean()
print(f"Conformal Coverage: {coverage_conf:.2%} (guaranteed >= 90%)")
```

## Further Reading

- [Loss Comparison](loss_comparison.md) - Detailed comparison across scenarios
- [Learn more about uncertainty estimation →](../math/index.md) - Theory of uncertainty decomposition
- [Calibration Metrics](../metrics/calibration.md) - Evaluating prediction intervals
