# Quick Start

Get started with **torchregress** in minutes. This guide covers the three most common workflows: basic regression, uncertainty estimation, and robust modeling.

---

## 1. Basic Regression

If you just need a standard point-prediction model, **torchregress** provides a drop-in replacement for standard PyTorch losses with added support for weights and masking.

```python
import torch
import torch.nn as nn
from torchregress.losses import MSELoss
from torchregress.metrics import mae, r2_score

# 1. Define a standard PyTorch model
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))

# 2. Use a torchregress loss (supports weights and masking)
loss_fn = MSELoss()

# 3. Standard training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for x, y in dataloader:
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward(); optimizer.step(); optimizer.zero_grad()

# 4. Evaluate with high-level metrics
with torch.no_grad():
    y_pred = model(x_test)
    print(f"MAE: {mae(y_pred, y_test):.4f}")
    print(f"R²: {r2_score(y_pred, y_test):.4f}")
```

---

## 2. Uncertainty Estimation (Probabilistic)

To estimate **aleatoric uncertainty** (data noise), your model should predict both a **mean** ($\mu$) and a **log-variance** ($\log \sigma^2$).

```python
from torchregress.losses import GaussianNLLLoss

# 1. Model with two outputs: [mean, log_variance]
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))

# 2. Use Gaussian Negative Log-Likelihood
loss_fn = GaussianNLLLoss()

# 3. Training
for x, y in dataloader:
    out = model(x) # [batch, 2]
    loss = loss_fn(out, y)
    loss.backward(); optimizer.step(); optimizer.zero_grad()

# 4. Inference: Extract mean and 95% Prediction Interval
with torch.no_grad():
    out = model(x_test)
    mu, logvar = out[:, 0], out[:, 1]
    std = torch.exp(0.5 * logvar)
    lower, upper = mu - 1.96 * std, mu + 1.96 * std
```

---

## 3. Robust Modeling (Outliers)

If your data contains outliers or heavy-tailed noise, standard MSE will fail. Use a **Robust Loss** like Huber or Cauchy.

```python
from torchregress.losses import HuberLoss, CauchyLoss

# Huber is a safe default for mild outliers
loss_fn = HuberLoss(delta=1.0)

# Cauchy is more aggressive for severe outliers (non-convex)
loss_fn = CauchyLoss(c=1.0)

# The rest of the training loop remains identical to Basic Regression
```

---

## 4. Conformal Prediction (Guaranteed Coverage)

Apply Conformal Prediction to **any** pre-trained model to get intervals with mathematically guaranteed coverage.

```python
from torchregress.losses import SplitConformal

# 1. Initialize (target 90% coverage)
cp = SplitConformal(alpha=0.1)

# 2. Calibrate on a held-out set
cp.calibrate(y_pred_cal, y_cal)

# 3. Predict guaranteed intervals at test time
lower, upper = cp.predict_interval(y_pred_test)
```

---

## Next Steps

<div class="grid cards" markdown>

-   __Method Selection__
    -   Which loss should I use? [Method Selection Matrix](../guide/method-selection.md)
    -   How do I handle imbalanced data? [Imbalanced Regression](../losses/imbalanced.md)

-   __Deep Dives__
    -   [Mathematical Foundations](../guide/math/index.md)
    -   [Ensemble Methods](../methods/ensemble/index.md)
    -   [Conformal Prediction](../methods/conformal/index.md)

</div>
