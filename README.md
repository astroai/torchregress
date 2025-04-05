# TorchRegress

PyTorch loss functions for regression with uncertainty estimation and missing data support.

[![PyPI](https://img.shields.io/pypi/v/torchregress.svg)](https://pypi.org/project/torchregress/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install torchregress
```

## Features

- **Uncertainty-aware losses**: Gaussian NLL, Mixture Density Networks, Normalizing Flows
- **Robust regression**: Huber, Log-cosh, Tukey losses for outlier resistance 
- **Quantile regression**: Predict specific percentiles and prediction intervals
- **Count data**: Poisson and Poisson-Gaussian mixture models
- **Missing data support**: All losses handle masked values gracefully
- **Error propagation**: Error-in-Variables regression for inputs with uncertainty
- **Algorithms**: IRLS for robust fitting, ensemble methods for uncertainty

## Common Usage

### Heteroscedastic Regression

```python
from torchregress.losses.gaussian import DiagonalGaussianNLL

# Create model that outputs mean and log_variance
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(64, output_dim)
        self.logvar_head = torch.nn.Linear(64, output_dim)
    
    def forward(self, x):
        features = self.backbone(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return mean, logvar

# Training with uncertainty estimation
model = Model()
loss_fn = DiagonalGaussianNLL(n_features=2)
loss = loss_fn((mean, logvar), target)
```

### Robust Regression with Missing Data

```python
from torchregress.losses.robust import HuberLoss

# Mask indicates missing values (False = missing)
mask = torch.ones_like(target).bool()
mask[:, 1] = False  # Second feature missing

# Apply mask-aware loss
loss_fn = HuberLoss(delta=1.0)
loss = loss_fn(predictions, target, mask=mask)
```

### Quantile Regression

```python
from torchregress.losses.quantile import MultiQuantileLoss

# Predict 10th, 50th and 90th percentiles simultaneously
quantiles = [0.1, 0.5, 0.9]
loss_fn = MultiQuantileLoss(quantiles=quantiles)
loss = loss_fn(predictions, target)  # predictions shape: [batch, num_quantiles, features]
```

### Iteratively Reweighted Least Squares

```python
from torchregress.algorithms.irls import IRLS

# Train with robust IRLS
result = IRLS(
    model=model,
    train_data=(X, y),
    weight_fn="tukey",  # Robust to outliers
    num_epochs=20
)
```

For more advanced usage and API details, refer to the [documentation](https://github.com/sfabbro/torchregress).

## License

MIT License
