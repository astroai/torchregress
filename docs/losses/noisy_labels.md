# Noisy Label Learning for Regression

Noisy label learning addresses the problem of training regression models when the target values (labels) contain errors or corruption. Unlike measurement noise in predictors (EIV), noisy labels represent systematic or random errors in the ground truth values themselves.

## The Noisy Label Problem

In real-world scenarios, training labels may be noisy due to:

- **Human annotation errors**: Manual labeling introduces mistakes
- **Sensor drift**: Measuring devices degrade over time
- **Data corruption**: Storage or transmission errors
- **Approximation errors**: Labels from proxy measurements
- **Adversarial corruption**: Intentional label manipulation

Traditional regression methods assume clean labels:

$$y_{\text{observed}} = y_{\text{true}}$$

But with noisy labels:

$$y_{\text{observed}} = y_{\text{true}} + \eta$$

where $\eta$ represents label noise (which may not be random).

## Mathematical Background

### Types of Label Noise

**1. Symmetric Noise**: Random corruption with zero mean
$$\eta \sim \mathcal{N}(0, \sigma^2)$$

**2. Asymmetric Noise**: Systematic bias
$$\eta = \begin{cases} +\Delta & \text{with probability } p \\ 0 & \text{with probability } 1-p \end{cases}$$

**3. Instance-Dependent Noise**: Corruption depends on input features
$$\eta = f(x) + \epsilon$$

### Loss Function Perspective

Standard loss with noisy labels:
$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \ell(f(x_i), y_i^{\text{noisy}})$$

Robust loss downweights suspected noisy samples:
$$\mathcal{L}_{\text{robust}} = \frac{1}{N}\sum_{i=1}^N w_i \cdot \ell(f(x_i), y_i^{\text{noisy}})$$

where $w_i \in [0, 1]$ is a confidence score (lower for noisy samples).

## Available Methods

### NoiseAdaptiveLoss

```python
class NoiseAdaptiveLoss(RegressionLoss)
```

Learns sample-specific noise levels via meta-learning. Maintains a learnable weight for each training sample that is optimized to identify and downweight noisy labels automatically.

**Key Concept**: Each sample has a learnable confidence score that decreases for samples with inconsistent losses.

**Parameters**:

- `n_samples` (int): Total number of samples in the dataset
- `base_loss` (str, optional): Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
- `initial_weight` (float, optional): Initial value for sample weights (0-1). Default: 1.0
- `reduction` (str, optional): Loss reduction method. Default: 'mean'

**Methods**:

- `get_sample_weights()`: Get current sample weights (confidence scores)
- `forward(y_pred, target, sample_indices, mask=None, weights=None)`: Compute noise-adaptive loss

**Example**:

```python
import torch
import torch.nn as nn
from torchregress.losses import NoiseAdaptiveLoss
from torch.utils.data import TensorDataset, DataLoader

# Generate data with noisy labels
torch.manual_seed(42)
n_samples = 1000
X = torch.randn(n_samples, 5)
y_clean = 2 * X[:, 0] - X[:, 1] + 0.5 * torch.randn(n_samples)

# Add label noise to 20% of samples
y_noisy = y_clean.clone()
noise_mask = torch.rand(n_samples) < 0.2  # 20% corruption
y_noisy[noise_mask] += 5 * torch.randn(noise_mask.sum())  # Large noise

# Create DataLoader that returns indices
class IndexedDataset(TensorDataset):
    def __getitem__(self, idx):
        return (*super().__getitem__(idx), idx)

dataset = IndexedDataset(X, y_noisy)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 1)
)

# Create noise-adaptive loss
loss_fn = NoiseAdaptiveLoss(n_samples=n_samples, base_loss='mse')

# IMPORTANT: Optimize both model and sample weights
model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
weight_optimizer = torch.optim.Adam([loss_fn.sample_weight_logits], lr=0.01)

# Training
for epoch in range(50):
    for x_batch, y_batch, indices in train_loader:
        # Forward pass
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch, sample_indices=indices)

        # Update model
        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        # Update sample weights
        weight_optimizer.step()

    if (epoch + 1) % 10 == 0:
        # Check learned weights
        sample_weights = loss_fn.get_sample_weights()
        avg_weight_clean = sample_weights[~noise_mask].mean()
        avg_weight_noisy = sample_weights[noise_mask].mean()

        print(f"Epoch {epoch+1}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Avg weight (clean): {avg_weight_clean:.3f}")
        print(f"  Avg weight (noisy): {avg_weight_noisy:.3f}")

# Identify detected noisy samples
sample_weights = loss_fn.get_sample_weights()
detected_noisy = sample_weights < 0.5  # Low confidence = likely noisy

# Evaluation
precision = (detected_noisy & noise_mask).sum() / detected_noisy.sum()
recall = (detected_noisy & noise_mask).sum() / noise_mask.sum()

print(f"\nNoise Detection Performance:")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
```

**When to Use**:

✅ **Have sample indices**: DataLoader returns indices
✅ **Can train longer**: Needs epochs to learn sample weights
✅ **Unknown noise pattern**: Works for various noise types
✅ **Want to identify noisy samples**: Get confidence scores

**Advantages**:
- Automatically learns which samples are noisy
- No need to specify noise rate or pattern
- Can be combined with any base loss

**Limitations**:
- Requires sample indices in training loop
- Needs separate optimizer for sample weights
- May overfit with very limited data

### CoTeachingLoss

```python
class CoTeachingLoss(RegressionLoss)
```

Co-teaching trains two networks simultaneously, where each network selects small-loss samples (assumed clean) for the other network to learn from. Based on the assumption that clean samples have smaller losses.

**Key Concept**: "Each network teaches the other" - Network 1 learns from samples Network 2 finds easy, and vice versa. This prevents both networks from memorizing the same noisy samples.

**Parameters**:

- `forget_rate` (float, optional): Fraction of samples to exclude (assumed noisy). Default: 0.2
- `base_loss` (str, optional): Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
- `num_gradual` (int, optional): Number of epochs to gradually increase forget rate. Default: 10
- `reduction` (str, optional): Loss reduction method. Default: 'mean'

**Methods**:

- `forward(y_pred1, y_pred2, target, epoch=0, mask=None)`: Compute co-teaching loss for both networks

**Example**:

```python
import torch
import torch.nn as nn
from torchregress.losses import CoTeachingLoss

# Create TWO independent models
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model1 = Model()
model2 = Model()

# Initialize with different random seeds for diversity
torch.manual_seed(1)
model1.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
torch.manual_seed(2)
model2.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

# Create co-teaching loss
# forget_rate = estimated noise rate (e.g., 0.2 for 20% corruption)
loss_fn = CoTeachingLoss(
    forget_rate=0.2,  # Exclude 20% highest-loss samples
    base_loss='mse',
    num_gradual=10  # Gradually increase forget rate over 10 epochs
)

# Separate optimizers for each network
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    for x_batch, y_batch in train_loader:
        # Forward pass for BOTH networks
        y_pred1 = model1(x_batch)
        y_pred2 = model2(x_batch)

        # Co-teaching: each network teaches the other
        loss1, loss2 = loss_fn(y_pred1, y_pred2, y_batch, epoch=epoch)

        # Update network 1
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        # Update network 2
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss1={loss1.item():.4f}, Loss2={loss2.item():.4f}")

# Ensemble predictions at inference
model1.eval()
model2.eval()
with torch.no_grad():
    y_pred1 = model1(X_test)
    y_pred2 = model2(X_test)
    y_pred_ensemble = (y_pred1 + y_pred2) / 2  # Average predictions
```

**When to Use**:

✅ **Know approximate noise rate**: Can estimate forget_rate
✅ **Can train two models**: 2× computational cost acceptable
✅ **Symmetric noise**: Works best when noise is random
✅ **Want simplicity**: No need for sample indices

**Advantages**:
- Simple to implement (just need two models)
- Robust to various noise types
- No need for sample indices
- Ensemble predictions at inference

**Limitations**:
- Requires knowing/estimating noise rate
- 2× training cost (two networks)
- Both networks may memorize same hard samples eventually

### RENTLoss (Robust Ensemble Training)

```python
class RENTLoss(RegressionLoss)
```

Uses ensemble disagreement to identify noisy labels. Samples where ensemble members disagree strongly are downweighted as they're likely noisy or ambiguous.

**Key Concept**: Clean samples → ensemble agrees. Noisy samples → ensemble disagrees.

**Parameters**:

- `ensemble_size` (int, optional): Expected number of models in ensemble. Default: 5
- `noise_threshold` (float, optional): Threshold for downweighting based on disagreement. Default: 2.0
- `base_loss` (str, optional): Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
- `reduction` (str, optional): Loss reduction method. Default: 'mean'

**Methods**:

- `forward(ensemble_preds, target, mask=None, weights=None)`: Compute RENT loss using ensemble disagreement

**Example**:

```python
import torch
import torch.nn as nn
from torchregress.losses import RENTLoss

# Create ensemble of models
ensemble_size = 5
models = [Model() for _ in range(ensemble_size)]

# Initialize with different seeds
for i, model in enumerate(models):
    torch.manual_seed(i)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

# Create RENT loss
loss_fn = RENTLoss(
    ensemble_size=ensemble_size,
    noise_threshold=2.0,  # Controls sensitivity to disagreement
    base_loss='mse'
)

# Create separate optimizers for each model
optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in models]

# Training loop
for epoch in range(50):
    for x_batch, y_batch in train_loader:
        # Get predictions from all ensemble members
        ensemble_preds = torch.stack([model(x_batch) for model in models])
        # Shape: [ensemble_size, batch_size, output_dim]

        # RENT loss automatically weights based on disagreement
        loss = loss_fn(ensemble_preds, y_batch)

        # Update all models
        for optimizer in optimizers:
            optimizer.zero_grad()

        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# At inference, use ensemble mean
models = [m.eval() for m in models]
with torch.no_grad():
    ensemble_preds = torch.stack([model(X_test) for model in models])
    y_pred_mean = ensemble_preds.mean(0)
    y_pred_std = ensemble_preds.std(0)  # Uncertainty estimate

    # High std indicates disagreement (potential noise or ambiguity)
    uncertain_samples = y_pred_std > y_pred_std.median()
    print(f"Uncertain samples: {uncertain_samples.sum()}/{len(X_test)}")
```

**When to Use**:

✅ **Already using ensembles**: Natural fit if training ensemble
✅ **Want uncertainty estimates**: Get both prediction and confidence
✅ **Complex noise patterns**: Handles various noise types
✅ **No need for indices**: Just forward all ensemble predictions

**Advantages**:
- Natural for ensemble methods
- Provides uncertainty estimates
- No need for sample indices
- Robust to various noise patterns

**Limitations**:
- Requires training full ensemble (5× cost)
- Assumes ensemble diversity
- May struggle if all models learn same errors

## Method Comparison

| Method | Training Cost | Needs Indices | Needs Noise Rate | Identifies Noisy | Best For |
|--------|--------------|---------------|-----------------|-----------------|----------|
| **NoiseAdaptiveLoss** | 1× (+ weight opt) | Yes | No | ✅ Yes | Unknown noise, want identification |
| **CoTeachingLoss** | 2× | No | Yes (approx) | ⚠️ Implicit | Known noise rate, simplicity |
| **RENTLoss** | 5× | No | No | ✅ Via disagreement | Using ensembles anyway |

## Decision Guide

```
┌─ Do you have label noise? ──────────────────────────────┐
│                                                          │
│  Check residuals on validation set:                     │
│  - High residuals on random samples → likely noise      │
│  - Residuals don't improve with more data → noise       │
│                                                          │
│  Already training an ensemble?                           │
│  ├─ Yes → RENTLoss (natural fit)                       │
│  └─ No → Continue below                                  │
│                                                          │
│  Know the approximate noise rate?                        │
│  ├─ Yes → CoTeachingLoss (simplest)                    │
│  └─ No → Continue below                                  │
│                                                          │
│  Can provide sample indices in DataLoader?               │
│  ├─ Yes → NoiseAdaptiveLoss (most flexible)            │
│  └─ No → CoTeachingLoss or train ensemble for RENT     │
│                                                          │
│  Need to identify specific noisy samples?                │
│  └─ Yes → NoiseAdaptiveLoss or RENTLoss                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Complete Comparison Example

```python
import torch
import torch.nn as nn
from torchregress.losses import NoiseAdaptiveLoss, CoTeachingLoss, RENTLoss
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Generate clean data
torch.manual_seed(42)
n_samples = 1000
X = torch.randn(n_samples, 5)
y_clean = 2 * X[:, 0] - X[:, 1] + 0.5 * torch.randn(n_samples)

# Add 20% label noise
y_noisy = y_clean.clone()
noise_rate = 0.2
n_noisy = int(noise_rate * n_samples)
noisy_indices = torch.randperm(n_samples)[:n_noisy]
y_noisy[noisy_indices] += 5 * torch.randn(n_noisy)

# Split data
train_size = 800
X_train, y_train = X[:train_size], y_noisy[:train_size]
X_test, y_test = X[train_size:], y_clean[train_size:]  # Test on CLEAN labels!

# Model architecture
def create_model():
    return nn.Sequential(
        nn.Linear(5, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

# Method 1: Standard training (baseline)
print("=" * 60)
print("Method 1: Standard Training (Baseline)")
print("=" * 60)

model_standard = create_model()
optimizer_std = torch.optim.Adam(model_standard.parameters(), lr=0.001)
mse_loss = nn.MSELoss()

for epoch in range(50):
    for i in range(0, len(X_train), 32):
        x_batch = X_train[i:i+32]
        y_batch = y_train[i:i+32]

        optimizer_std.zero_grad()
        loss = mse_loss(model_standard(x_batch), y_batch)
        loss.backward()
        optimizer_std.step()

# Method 2: NoiseAdaptiveLoss
print("\nMethod 2: NoiseAdaptiveLoss")
print("=" * 60)

class IndexedDataset(TensorDataset):
    def __getitem__(self, idx):
        return (*super().__getitem__(idx), idx)

dataset = IndexedDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model_adaptive = create_model()
loss_adaptive = NoiseAdaptiveLoss(n_samples=train_size, base_loss='mse')
model_opt = torch.optim.Adam(model_adaptive.parameters(), lr=0.001)
weight_opt = torch.optim.Adam([loss_adaptive.sample_weight_logits], lr=0.01)

for epoch in range(50):
    for x_batch, y_batch, indices in train_loader:
        model_opt.zero_grad()
        loss = loss_adaptive(model_adaptive(x_batch), y_batch, sample_indices=indices)
        loss.backward()
        model_opt.step()
        weight_opt.step()

# Method 3: CoTeachingLoss
print("\nMethod 3: CoTeachingLoss")
print("=" * 60)

model_ct1 = create_model()
model_ct2 = create_model()
loss_ct = CoTeachingLoss(forget_rate=noise_rate, num_gradual=10)
opt_ct1 = torch.optim.Adam(model_ct1.parameters(), lr=0.001)
opt_ct2 = torch.optim.Adam(model_ct2.parameters(), lr=0.001)

regular_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

for epoch in range(50):
    for x_batch, y_batch in regular_loader:
        y_pred1 = model_ct1(x_batch)
        y_pred2 = model_ct2(x_batch)

        loss1, loss2 = loss_ct(y_pred1, y_pred2, y_batch, epoch=epoch)

        opt_ct1.zero_grad()
        loss1.backward()
        opt_ct1.step()

        opt_ct2.zero_grad()
        loss2.backward()
        opt_ct2.step()

# Evaluate all methods
print("\n" + "=" * 60)
print("Evaluation on Clean Test Set")
print("=" * 60)

models = {
    'Standard (Baseline)': model_standard,
    'NoiseAdaptive': model_adaptive,
    'CoTeaching (Model 1)': model_ct1,
    'CoTeaching (Model 2)': model_ct2,
}

for name, model in models.items():
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = ((y_pred - y_test) ** 2).mean()
        mae = torch.abs(y_pred - y_test).mean()

        print(f"\n{name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")

# CoTeaching ensemble
with torch.no_grad():
    y_pred_ensemble = (model_ct1(X_test) + model_ct2(X_test)) / 2
    mse_ensemble = ((y_pred_ensemble - y_test) ** 2).mean()
    mae_ensemble = torch.abs(y_pred_ensemble - y_test).mean()

    print(f"\nCoTeaching (Ensemble):")
    print(f"  MSE: {mse_ensemble:.4f}")
    print(f"  MAE: {mae_ensemble:.4f}")

# Expected output:
# Standard: Higher error (affected by noise)
# NoiseAdaptive: Lower error (learned to ignore noisy samples)
# CoTeaching: Lower error (filtered noisy samples)
```

## Best Practices

### 1. Detect If You Have Label Noise

```python
# Check for label noise indicators
def diagnose_label_noise(model, X_train, y_train, X_val, y_val):
    """Diagnose if label noise is present."""
    model.eval()

    with torch.no_grad():
        # Training residuals
        y_pred_train = model(X_train)
        residuals_train = torch.abs(y_pred_train - y_train)

        # Validation residuals
        y_pred_val = model(X_val)
        residuals_val = torch.abs(y_pred_val - y_val)

        # Check for outliers in training residuals
        train_median = residuals_train.median()
        train_q95 = torch.quantile(residuals_train, 0.95)

        potential_noisy = residuals_train > 3 * train_median
        pct_outliers = potential_noisy.float().mean() * 100

        print(f"Training residuals - Median: {train_median:.4f}, 95th percentile: {train_q95:.4f}")
        print(f"Potential noisy samples: {pct_outliers:.1f}%")

        # If validation residuals are much lower → likely label noise
        if residuals_train.median() > 1.5 * residuals_val.median():
            print("⚠️  Warning: Training residuals >> validation residuals")
            print("   This suggests label noise in training set")

        return potential_noisy

# Usage
potential_noisy = diagnose_label_noise(model, X_train, y_train, X_val, y_val)
```

### 2. Estimate Noise Rate

```python
# Cross-validation to estimate noise rate
from sklearn.model_selection import KFold

def estimate_noise_rate(X, y, n_folds=5):
    """Estimate label noise rate using cross-validation."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_residuals = []

    for train_idx, val_idx in kfold.split(X):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        # Train simple model
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        mse_loss = nn.MSELoss()

        for epoch in range(50):
            optimizer.zero_grad()
            loss = mse_loss(model(X_train_fold), y_train_fold)
            loss.backward()
            optimizer.step()

        # Compute residuals on validation fold
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val_fold)
            residuals = torch.abs(y_pred - y_val_fold)
            all_residuals.append(residuals)

    all_residuals = torch.cat(all_residuals)

    # Estimate noise rate as fraction of large residuals
    threshold = torch.quantile(all_residuals, 0.80)
    estimated_noise_rate = (all_residuals > threshold).float().mean()

    print(f"Estimated noise rate: {estimated_noise_rate:.2%}")
    return estimated_noise_rate.item()

# Usage
noise_rate_est = estimate_noise_rate(X_train, y_train)
```

### 3. Validate Results on Clean Data

```python
# Always test on clean labels if available
def evaluate_noise_robustness(model, X_test_clean, y_test_clean,
                              X_test_noisy, y_test_noisy):
    """Evaluate model on both clean and noisy test sets."""
    model.eval()

    with torch.no_grad():
        # Clean test set
        y_pred_clean = model(X_test_clean)
        mae_clean = torch.abs(y_pred_clean - y_test_clean).mean()

        # Noisy test set
        y_pred_noisy = model(X_test_noisy)
        mae_noisy = torch.abs(y_pred_noisy - y_test_noisy).mean()

        print(f"MAE on clean labels: {mae_clean:.4f}")
        print(f"MAE on noisy labels: {mae_noisy:.4f}")
        print(f"Robustness ratio: {mae_noisy / mae_clean:.2f}x")

        # Good models should have ratio close to 1.0
        if mae_noisy / mae_clean < 1.5:
            print("✅ Model is robust to label noise")
        else:
            print("⚠️  Model may be overfitting to noise")
```

## Common Pitfalls

### ❌ Pitfall 1: Using Wrong Forget Rate

```python
# Too high forget rate → discards clean samples
loss_fn = CoTeachingLoss(forget_rate=0.5)  # If actual noise is 20%
# → Throws away 30% clean samples!
```

**Solution**: Estimate noise rate first, use slightly higher forget_rate (e.g., 0.25 for 20% noise).

### ❌ Pitfall 2: Not Training Long Enough

```python
# NoiseAdaptiveLoss needs time to learn sample weights
for epoch in range(10):  # Too few epochs
    # Sample weights haven't converged yet
```

**Solution**: Train for at least 50-100 epochs for weight learning to stabilize.

### ❌ Pitfall 3: Same Initialization for Co-Teaching

```python
# Both models start identical → learn same errors
model1 = create_model()
model2 = create_model()
# → Both memorize same noisy samples
```

**Solution**: Use different random seeds for initialization.

### ❌ Pitfall 4: Testing on Noisy Labels

```python
# Evaluating on test set with label noise
test_mae = evaluate(model, X_test, y_test_noisy)
# → Can't tell if model learned noise or true function
```

**Solution**: Always evaluate on clean labels if possible, or use cross-validation.

## References

- Li et al. "Learning to Learn from Noisy Labeled Data" (CVPR 2019)
- Han et al. "Co-teaching: Robust Training of DNNs with Extremely Noisy Labels" (NeurIPS 2018)
- Chen et al. "Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels" (ICML 2019)
- Patrini et al. "Making Deep Neural Networks Robust to Label Noise" (CVPR 2017)
- Zhang et al. "Understanding Deep Learning Requires Rethinking Generalization" (ICLR 2017)
