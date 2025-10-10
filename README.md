# torchregress

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
- **Modern regression**: Simultaneous quantile regression, Barron loss, conformal prediction
- **Ensemble methods**: Deep ensembles with uncertainty propagation
- **Target transformations**: Built-in log, Box-Cox, and sqrt transformations
- **Algorithms**: IRLS for robust fitting, ensemble methods for uncertainty

## Development

This project uses modern Python tooling with [uv](https://github.com/astral-sh/uv) as the primary package manager.

### Quick Start

1. Install uv:
   ```bash
   # On macOS and Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone and install:
   ```bash
   git clone https://github.com/sfabbro/torchregress.git
   cd torchregress
   uv pip install -e .[all]
   ```

### Common Development Tasks

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=torchregress --cov-report=html

# Code formatting
uv run black .

# Code linting
uv run ruff check .

# Type checking
uv run mypy torchregress

# Build documentation
uv run mkdocs build

# Serve documentation locally
uv run mkdocs serve

# Build distribution packages
uv build

# Publish to PyPI
uv publish
```

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

### Conformal Prediction

```python
from torchregress.losses.conformal import ConformalLoss

# Train conformal prediction model
loss_fn = ConformalLoss(alpha=0.1)  # 90% prediction intervals
loss = loss_fn(predictions, target)

# Calibrate on hold-out data
lower_pred, upper_pred = predictions[:, :n_features], predictions[:, n_features:]
tau = loss_fn.calibrate(lower_pred, upper_pred, target)

# Make predictions with calibrated intervals
lower_interval, upper_interval = loss_fn.predict_interval(lower_pred, upper_pred)
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