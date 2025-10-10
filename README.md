# torchregress

[![PyPI](https://img.shields.io/pypi/v/torchregress.svg)](https://pypi.org/project/torchregress/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive PyTorch library for regression and uncertainty estimation.

## Overview

**torchregress** provides a collection of regression loss functions, metrics, and uncertainty estimation techniques implemented in PyTorch. It's designed to make it easy to:

- Use different regression loss functions beyond MSE
- Estimate uncertainty in regression predictions
- Evaluate regression models with appropriate metrics
- Visualize regression results and uncertainty

## Key Features

- **Diverse Loss Functions**: From basic (MSE, MAE) to advanced (MDN, NF) regression losses
- **Uncertainty Quantification**: Built-in support for different uncertainty estimation techniques
- **Comprehensive Metrics**: Evaluation metrics for point predictions, predictive distributions, and uncertainty
- **Visualization Tools**: Ready-to-use visualization functions for regression diagnostics
- **PyTorch Integration**: Seamlessly integrates with PyTorch models and training loops
- **Statistical Rigor**: Mathematically sound implementation of statistical estimators

## Installation

```bash
pip install torchregress
```

## Quickstart

```python
import torch
from torchregress.losses import GaussianNLLLoss
from torchregress.metrics import rmse

# Define your model (any PyTorch model that outputs mean and variance)
# For example:
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2) # Output mean and log_var

    def forward(self, x):
        return self.fc(x).chunk(2, dim=-1)

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = GaussianNLLLoss()

# Example data
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)
X_test = torch.randn(50, 10)
y_test = torch.randn(50, 1)

# Training loop
for epoch in range(10):
    for i in range(len(X_train)):
        # Forward pass: get predictions (mean and variance)
        mean, log_var = model(X_train[i:i+1])

        # Calculate loss
        loss = loss_fn(mean, y_train[i:i+1], log_var=log_var.exp())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation
with torch.no_grad():
    mean, log_var = model(X_test)
    rmse_score = rmse(mean, y_test)
    print(f"RMSE: {rmse_score:.4f}")

```

## Common Usage

### Conformal Prediction

```python
from torchregress.losses.conformal import ConformalLoss

# Your model should be trained to predict quantiles for CQR
# For split conformal prediction, it can be a standard regression model
model = MyModel() # A model that outputs a point prediction or quantiles
loss_fn = ConformalLoss(method='cqr', alpha=0.1)  # 90% prediction intervals

# ... training loop for the model ...

# Calibrate on hold-out data
# For CQR, y_pred_cal should contain lower and upper quantile predictions
y_pred_cal = torch.randn(100, 2)
y_true_cal = torch.randn(100, 1)
loss_fn.calibrate(y_pred_cal, y_true_cal)

# Make predictions with calibrated intervals
y_pred_test = torch.randn(50, 2)
lower_interval, upper_interval = loss_fn.predict_interval(y_pred_test)
```

For more advanced usage and API details, refer to the [full documentation](https://github.com/sfabbro/torchregress).

## License

MIT License
