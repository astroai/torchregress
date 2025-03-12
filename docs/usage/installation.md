# Installation

TorchRegression can be installed from PyPI or directly from the source code. The library requires Python 3.10 or newer and PyTorch.

## From PyPI

The simplest way to install TorchRegression is via pip:

```bash
pip install torchregression
```

## With Extra Dependencies

TorchRegression provides optional dependency sets for different use cases:

```bash
# For testing capabilities
pip install 'torchregression[test]'

# For development 
pip install 'torchregression[dev]'

# For documentation
pip install 'torchregression[docs]'

# For all optional dependencies
pip install 'torchregression[all]'
```

## From Source

To install the latest development version:

```bash
git clone https://github.com/sfabbro/torchregression.git
cd torchregression
pip install -e .
```

## Requirements

TorchRegression has the following dependencies:

### Core Dependencies

- Python >= 3.10
- PyTorch
- NumPy

### Optional Dependencies

Testing:
- pytest >= 7.0.0
- pytest-cov >= 4.0.0

Development:
- black
- ruff
- mypy
- pre-commit

Documentation:
- sphinx
- sphinx-rtd-theme
- myst-parser

## Verifying Installation

To verify that TorchRegression is installed correctly, you can run:

```python
import torchregression
print(torchregression.__version__)
```

Or use the following minimal example:

```python
import torch
import torchregression as tr

# Create some dummy data
X = torch.randn(100, 1)
y = 2 * X.squeeze() + 1 + 0.1 * torch.randn(100)

# Define a simple model
model = torch.nn.Linear(1, 1)

# Choose a loss function
loss_fn = tr.losses.HuberLoss()

# Train the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(100):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate
with torch.no_grad():
    y_pred = model(X)
    rmse = tr.metrics.rmse(y_pred, y)
    print(f"RMSE: {rmse.item():.4f}")
```

If this runs without errors, TorchRegression is correctly installed.
