# Transform Losses for Regression

Transform losses apply mathematical transformations to target variables before computing the loss. This approach is useful when the relationship between predictors and targets is more linear or has better properties (e.g., constant variance) in transformed space.

## Why Transform Targets?

Standard regression assumes:
$$y = f(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

But real data often violates these assumptions:

**Problem 1: Heteroscedasticity** - Error variance depends on target value
$$\text{Var}(\epsilon | y) = \sigma^2 \cdot g(y)$$

**Problem 2: Skewed Distribution** - Targets have long tails
$$y \in \mathbb{R}^+, \quad p(y) \text{ is heavily right-skewed}$$

**Problem 3: Multiplicative Noise** - Errors scale with magnitude
$$y_{\text{true}} \cdot (1 + \epsilon) \quad \text{instead of} \quad y_{\text{true}} + \epsilon$$

### Solution: Transform to Better Space

Apply transformation $T: \mathbb{R} \rightarrow \mathbb{R}$ such that:
$$z = T(y)$$

Train in transformed space:
$$z = f(x) + \epsilon', \quad \epsilon' \sim \mathcal{N}(0, \sigma^2)$$

At inference, apply inverse transform:
$$\hat{y} = T^{-1}(\hat{z})$$

## Mathematical Background

### Variance Stabilizing Transformations

Different data-generating processes require different transforms:

| Data Type | Distribution | Var(Y) | Transform | Result |
|-----------|-------------|--------|-----------|---------|
| Count data | Poisson | $\mu$ | $\sqrt{y}$ | Constant variance |
| Proportion | Binomial | $\mu(1-\mu)$ | $\arcsin(\sqrt{y})$ | Constant variance |
| Positive continuous | Log-normal | $\mu^2$ | $\log(y)$ | Constant variance |
| General positive | Power family | $\mu^\lambda$ | $(y^\lambda - 1)/\lambda$ | Tunable |

## Available Transform Losses

### LogTransformLoss

```python
class LogTransformLoss(RegressionLoss)
```

Applies logarithmic transformation to both predictions and targets before computing MSE. This is variance-stabilizing for **multiplicative noise** (proportional errors).

**Use When**:
- Targets are strictly positive
- Errors grow proportionally with magnitude
- Target distribution is log-normal or heavily right-skewed
- Interested in relative errors rather than absolute errors

**Parameters**:

- `eps` (float, optional): Small constant for numerical stability. Default: 1e-6
- `reduction` (str, optional): Loss reduction method. Default: 'mean'

**Methods**:

- `forward(y_pred, target, mask=None, weights=None)`: Compute loss in log space
- `inverse(y_transformed)`: Map transformed values back to original scale

**Mathematical Form**:

Transform:
$$z = \log(y + \epsilon)$$

Loss in transformed space:
$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N (\log(\hat{y}_i + \epsilon) - \log(y_i + \epsilon))^2$$

Inverse transform:
$$\hat{y} = \exp(z) - \epsilon$$

**Example**:

```python
import torch
import torch.nn as nn
from torchregress.losses import LogTransformLoss

# Generate data with multiplicative noise
# True relationship: y = 2^x (exponential growth)
torch.manual_seed(42)
X = torch.linspace(0, 5, 200).unsqueeze(1)
y_true = torch.exp(X)  # Exponential growth

# Add multiplicative noise (error proportional to magnitude)
y = y_true * (1 + 0.1 * torch.randn_like(y_true))

# Split data
train_size = 150
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Model
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Softplus()  # Ensure positive outputs
)

# Log transform loss
loss_fn = LogTransformLoss(eps=1e-6)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()

    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

    # Compute errors in original space
    mae = torch.abs(y_pred - y_test).mean()

    # Compute relative errors (%)
    relative_error = torch.abs(y_pred - y_test) / y_test
    mape = (relative_error * 100).mean()

    print(f"\nTest MAE: {mae:.2f}")
    print(f"Test MAPE: {mape:.2f}%")

# Compare with standard MSE
model_mse = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Softplus()
)
mse_loss = nn.MSELoss()
optimizer_mse = torch.optim.Adam(model_mse.parameters(), lr=0.01)

for epoch in range(200):
    optimizer_mse.zero_grad()
    loss = mse_loss(model_mse(X_train), y_train)
    loss.backward()
    optimizer_mse.step()

model_mse.eval()
with torch.no_grad():
    y_pred_mse = model_mse(X_test)
    mape_mse = (torch.abs(y_pred_mse - y_test) / y_test * 100).mean()

    print(f"\nComparison:")
    print(f"  LogTransform MAPE: {mape:.2f}%")
    print(f"  Standard MSE MAPE: {mape_mse:.2f}%")
    print(f"  Improvement: {(mape_mse - mape) / mape_mse * 100:.1f}%")
```

**When to Use**:

✅ **Multiplicative noise**: Errors proportional to target magnitude
✅ **Exponential growth**: y = exp(f(x))
✅ **Right-skewed targets**: Long tail in distribution
✅ **Relative errors matter**: Care about % error not absolute error

**When NOT to Use**:

❌ **Targets contain zeros**: log(0) is undefined
❌ **Negative values**: log(x) only defined for x > 0
❌ **Additive constant noise**: log doesn't help
❌ **Symmetric distribution**: Transform adds complexity without benefit

### BoxCoxTransformLoss

```python
class BoxCoxTransformLoss(RegressionLoss)
```

Applies Box-Cox power transformation, a flexible family of transformations parameterized by λ:

$$T(y; \lambda) = \begin{cases}
\frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(y) & \text{if } \lambda = 0
\end{cases}$$

This subsumes several common transforms:
- λ = 1: Identity (no transform)
- λ = 0.5: Square root
- λ = 0: Log
- λ = -1: Reciprocal

**Parameters**:

- `lam` (float, optional): Box-Cox transformation parameter. Default: 0.0
- `eps` (float, optional): Small constant for numerical stability. Default: 1e-6
- `reduction` (str, optional): Loss reduction method. Default: 'mean'

**Methods**:

- `forward(y_pred, target, mask=None, weights=None)`: Compute loss in transformed space
- `inverse(y_transformed)`: Map transformed values back to original scale

**Example**:

```python
import torch
import torch.nn as nn
from torchregress.losses import BoxCoxTransformLoss
import matplotlib.pyplot as plt

# Generate data with power-law relationship
torch.manual_seed(42)
X = torch.linspace(0.1, 10, 200).unsqueeze(1)
y_true = X ** 2.5  # Power law
y = y_true + y_true ** 0.5 * torch.randn_like(y_true)  # Heteroscedastic noise

train_size = 150
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Try different lambda values
lambdas = [0.0, 0.25, 0.5, 1.0]
results = {}

for lam in lambdas:
    print(f"\n=== Box-Cox λ = {lam} ===")

    model = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Softplus()  # Positive outputs
    )

    loss_fn = BoxCoxTransformLoss(lam=lam, eps=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = ((y_pred - y_test) ** 2).mean()
        mae = torch.abs(y_pred - y_test).mean()

        results[lam] = {'mse': mse.item(), 'mae': mae.item(), 'model': model}

        print(f"  Test MSE: {mse:.2f}")
        print(f"  Test MAE: {mae:.2f}")

# Find best lambda
best_lam = min(results.keys(), key=lambda k: results[k]['mse'])
print(f"\n✅ Best λ = {best_lam} with MSE = {results[best_lam]['mse']:.2f}")

# Visualize results
plt.figure(figsize=(12, 4))

for i, lam in enumerate(lambdas):
    plt.subplot(1, 4, i+1)
    model = results[lam]['model']
    with torch.no_grad():
        y_pred = model(X_test)

    plt.scatter(X_test, y_test, alpha=0.5, label='True')
    plt.plot(X_test, y_pred, 'r-', linewidth=2, label='Predicted')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'λ = {lam}\nMSE = {results[lam]["mse"]:.2f}')
    plt.legend()

plt.tight_layout()
plt.savefig('boxcox_comparison.png')
print("\nPlot saved as boxcox_comparison.png")
```

**Choosing λ**:

1. **λ = 0 (log)**: Heavy right skew, multiplicative noise
2. **λ = 0.5 (sqrt)**: Count data, Poisson-like
3. **λ = 1 (identity)**: No transformation needed
4. **Cross-validation**: Try multiple λ values, select best on validation set

**When to Use**:

✅ **Unknown optimal transform**: Try different λ values
✅ **Positive targets**: Box-Cox only works for y > 0
✅ **Non-linear relationships**: May linearize in transformed space
✅ **Heteroscedasticity**: Can stabilize variance

### SqrtTransformLoss

```python
class SqrtTransformLoss(RegressionLoss)
```

Applies square root transformation. This is variance-stabilizing for **Poisson-distributed data** (count data where variance equals mean).

**Use When**:
- Targets are count data (non-negative integers)
- Variance increases linearly with mean: Var(Y) = μ
- Poisson or Poisson-like distribution

**Parameters**:

- `eps` (float, optional): Small constant for numerical stability. Default: 1e-6
- `reduction` (str, optional): Loss reduction method. Default: 'mean'

**Methods**:

- `forward(y_pred, target, mask=None, weights=None)`: Compute loss in sqrt space
- `inverse(y_transformed)`: Map transformed values back to original scale

**Mathematical Form**:

Transform:
$$z = \sqrt{y + \epsilon}$$

Loss:
$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N (\sqrt{\hat{y}_i + \epsilon} - \sqrt{y_i + \epsilon})^2$$

Inverse:
$$\hat{y} = z^2 - \epsilon$$

**Example**:

```python
import torch
import torch.nn as nn
from torchregress.losses import SqrtTransformLoss

# Generate Poisson-like count data
torch.manual_seed(42)
X = torch.linspace(0, 5, 200).unsqueeze(1)
lambda_rate = torch.exp(0.5 * X)  # Poisson rate parameter

# Simulate counts (approximate Poisson with Normal)
y = torch.poisson(lambda_rate).float()

train_size = 150
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Model with sqrt transform loss
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Softplus()  # Ensure non-negative
)

loss_fn = SqrtTransformLoss(eps=1e-6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

    # Round to nearest integer for count data
    y_pred_rounded = torch.round(y_pred)

    mae = torch.abs(y_pred_rounded - y_test).mean()
    print(f"\nTest MAE (rounded): {mae:.2f}")

    # Check if variance is stabilized
    residuals = y_pred - y_test
    print(f"Residual mean: {residuals.mean():.2f}")
    print(f"Residual std: {residuals.std():.2f}")
```

**When to Use**:

✅ **Count data**: Number of events, occurrences
✅ **Poisson process**: Traffic counts, website visits
✅ **Non-negative integers**: Discrete counts
✅ **Variance increases with mean**: Var(Y) ≈ E[Y]

**Note**: For true count data, consider `PoissonNLLLoss` which directly models the Poisson distribution rather than transforming.

## Transform Selection Guide

```
┌─ Which Transform to Use? ────────────────────────────────┐
│                                                          │
│  What type of data do you have?                          │
│                                                          │
│  Count Data (non-negative integers)?                     │
│  ├─ Poisson-like (Var ≈ Mean) → SqrtTransformLoss      │
│  └─ Over-dispersed → Consider PoissonNLLLoss instead   │
│                                                          │
│  Positive Continuous?                                    │
│  ├─ Multiplicative noise → LogTransformLoss            │
│  ├─ Power-law relationship → BoxCoxTransformLoss       │
│  └─ Unknown → Try BoxCox with different λ              │
│                                                          │
│  Contains Zeros or Negative Values?                      │
│  └─ Don't use transforms → Use robust losses instead   │
│                                                          │
│  Constant additive noise?                                │
│  └─ No transform needed → Use standard MSE             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Comparison Table

| Transform | Formula | Best For | Inverse | Handles Zeros? |
|-----------|---------|----------|---------|----------------|
| **Log** | $\log(y + \epsilon)$ | Multiplicative noise, exponential growth | $\exp(z) - \epsilon$ | With eps only |
| **Box-Cox** | $(y^\lambda - 1)/\lambda$ | Flexible, tunable, power-law | $(z\lambda + 1)^{1/\lambda}$ | No (y > 0) |
| **Sqrt** | $\sqrt{y + \epsilon}$ | Count data, Poisson | $z^2 - \epsilon$ | With eps only |

## Complete Comparison Example

```python
import torch
import torch.nn as nn
from torchregress.losses import LogTransformLoss, BoxCoxTransformLoss, SqrtTransformLoss
import matplotlib.pyplot as plt

# Generate data with heteroscedastic multiplicative noise
torch.manual_seed(42)
X = torch.linspace(0.5, 5, 200).unsqueeze(1)
y_true = torch.exp(0.8 * X)  # Exponential
y = y_true * (1 + 0.15 * torch.randn_like(y_true))  # 15% multiplicative noise

train_size = 150
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Define model factory
def create_model():
    return nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Softplus()  # Positive outputs
    )

# Method 1: Standard MSE
print("=" * 60)
print("Method 1: Standard MSE (Baseline)")
print("=" * 60)

model_mse = create_model()
mse_loss = nn.MSELoss()
optimizer_mse = torch.optim.Adam(model_mse.parameters(), lr=0.01)

for epoch in range(200):
    optimizer_mse.zero_grad()
    loss = mse_loss(model_mse(X_train), y_train)
    loss.backward()
    optimizer_mse.step()

# Method 2: Log Transform
print("\nMethod 2: Log Transform")
print("=" * 60)

model_log = create_model()
log_loss = LogTransformLoss()
optimizer_log = torch.optim.Adam(model_log.parameters(), lr=0.01)

for epoch in range(200):
    optimizer_log.zero_grad()
    loss = log_loss(model_log(X_train), y_train)
    loss.backward()
    optimizer_log.step()

# Method 3: Box-Cox (λ=0.5)
print("\nMethod 3: Box-Cox (λ=0.5)")
print("=" * 60)

model_boxcox = create_model()
boxcox_loss = BoxCoxTransformLoss(lam=0.5)
optimizer_boxcox = torch.optim.Adam(model_boxcox.parameters(), lr=0.01)

for epoch in range(200):
    optimizer_boxcox.zero_grad()
    loss = boxcox_loss(model_boxcox(X_train), y_train)
    loss.backward()
    optimizer_boxcox.step()

# Method 4: Sqrt Transform
print("\nMethod 4: Sqrt Transform")
print("=" * 60)

model_sqrt = create_model()
sqrt_loss = SqrtTransformLoss()
optimizer_sqrt = torch.optim.Adam(model_sqrt.parameters(), lr=0.01)

for epoch in range(200):
    optimizer_sqrt.zero_grad()
    loss = sqrt_loss(model_sqrt(X_train), y_train)
    loss.backward()
    optimizer_sqrt.step()

# Evaluate all methods
print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)

models = {
    'Standard MSE': model_mse,
    'Log Transform': model_log,
    'Box-Cox (λ=0.5)': model_boxcox,
    'Sqrt Transform': model_sqrt,
}

for name, model in models.items():
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

        # Absolute metrics
        mae = torch.abs(y_pred - y_test).mean()
        mse = ((y_pred - y_test) ** 2).mean()

        # Relative metrics (more meaningful for multiplicative noise)
        mape = (torch.abs(y_pred - y_test) / y_test * 100).mean()

        print(f"\n{name}:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  MSE:  {mse:.2f}")
        print(f"  MAPE: {mape:.2f}%")

# Expected output:
# Log Transform should have lowest MAPE (best for multiplicative noise)
# Standard MSE may have lower MAE on large values but higher MAPE overall
```

## Best Practices

### 1. Check Data Distribution First

```python
import matplotlib.pyplot as plt

# Visualize target distribution
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(y_train.numpy(), bins=50, edgecolor='black')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.title('Original Distribution')

# Check for skewness
from scipy.stats import skew
skewness = skew(y_train.numpy())
print(f"Skewness: {skewness:.2f}")
if skewness > 1:
    print("→ Right-skewed: Consider log or Box-Cox transform")
elif skewness < -1:
    print("→ Left-skewed: Transform may not help")
else:
    print("→ Roughly symmetric: Transform may not be needed")

# Plot log-transformed
plt.subplot(132)
plt.hist(torch.log(y_train + 1e-6).numpy(), bins=50, edgecolor='black')
plt.xlabel('log(Target)')
plt.ylabel('Frequency')
plt.title('Log-Transformed Distribution')

# Plot sqrt-transformed
plt.subplot(133)
plt.hist(torch.sqrt(y_train + 1e-6).numpy(), bins=50, edgecolor='black')
plt.xlabel('sqrt(Target)')
plt.ylabel('Frequency')
plt.title('Sqrt-Transformed Distribution')

plt.tight_layout()
plt.show()
```

### 2. Validate Transform Helps

```python
# Always compare with untransformed baseline
def compare_transforms(X_train, y_train, X_val, y_val):
    """Compare different transforms on validation set."""

    transforms = {
        'None (MSE)': nn.MSELoss(),
        'Log': LogTransformLoss(),
        'Box-Cox (0.5)': BoxCoxTransformLoss(lam=0.5),
        'Sqrt': SqrtTransformLoss(),
    }

    results = {}

    for name, loss_fn in transforms.items():
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train
        for epoch in range(100):
            optimizer.zero_grad()
            if isinstance(loss_fn, nn.MSELoss):
                loss = loss_fn(model(X_train), y_train)
            else:
                loss = loss_fn(model(X_train), y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val)
            val_mae = torch.abs(y_pred - y_val).mean()
            results[name] = val_mae.item()

    # Print results
    print("\nValidation MAE:")
    for name, mae in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:20s}: {mae:.4f}")

    best_transform = min(results, key=results.get)
    print(f"\n✅ Best transform: {best_transform}")

    return best_transform
```

### 3. Handle Edge Cases

```python
# Properly handle zeros and small values
def safe_transform(y, transform_type='log', eps=1e-6):
    """Apply transform with proper handling of edge cases."""

    if transform_type == 'log':
        # Check for zeros or negatives
        if (y <= 0).any():
            print(f"⚠️  Found {(y <= 0).sum()} non-positive values")
            print(f"   Adding eps={eps} to all values")
        return torch.log(y + eps)

    elif transform_type == 'sqrt':
        if (y < 0).any():
            print(f"⚠️  Found {(y < 0).sum()} negative values")
            print(f"   Clipping to zero")
            y = torch.clamp(y, min=0)
        return torch.sqrt(y + eps)

    else:
        return y
```

## Common Pitfalls

### ❌ Pitfall 1: Forgetting Inverse Transform

```python
# Predictions in transformed space
y_pred_log = model(X_test)  # This is log(y), not y!

# Need to inverse transform
y_pred = torch.exp(y_pred_log)  # ✅ Correct
```

**Solution**: Use `loss_fn.inverse()` method or apply inverse manually.

### ❌ Pitfall 2: Transform When Not Needed

```python
# Data already has constant additive noise
# Transform adds complexity without benefit
loss_fn = LogTransformLoss()  # Unnecessary!
```

**Solution**: Always compare with untransformed baseline on validation set.

### ❌ Pitfall 3: Not Handling Zeros

```python
# Targets contain zeros
y_train = torch.tensor([0, 1, 2, 3])
loss_fn = LogTransformLoss(eps=0)  # eps=0!
# → log(0) = -inf, NaN losses
```

**Solution**: Always use `eps > 0` (e.g., 1e-6) for numerical stability.

### ❌ Pitfall 4: Wrong Transform for Data Type

```python
# Count data with Poisson distribution
# Using log transform instead of sqrt
loss_fn = LogTransformLoss()  # Wrong!
# → Sqrt is variance-stabilizing for Poisson
```

**Solution**: Match transform to data-generating process (see decision guide).

## References

- Box, G. E. P., & Cox, D. R. (1964). "An Analysis of Transformations". Journal of the Royal Statistical Society.
- Bartlett, M. S. (1947). "The Use of Transformations". Biometrics.
- Anscombe, F. J. (1948). "The Transformation of Poisson, Binomial and Negative-Binomial Data". Biometrika.
- Tukey, J. W. (1977). "Exploratory Data Analysis". Addison-Wesley.
