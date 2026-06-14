# Installation

torchregress can be installed from PyPI or directly from the source code. The library requires Python 3.12 to <3.16 and PyTorch 2.4 or newer.

## From PyPI

The simplest way to install torchregress is via pip:

```bash
pip install torchregress
```

## With Extra Dependencies

torchregress provides optional dependency sets for different use cases:

```bash
# For testing capabilities
pip install 'torchregress[test]'

# For development
pip install 'torchregress[dev]'

# For documentation
pip install 'torchregress[docs]'

# Compatibility extra for conformal workflows
pip install 'torchregress[conformal]'

# For normalizing-flow losses
pip install 'torchregress[flows]'

# For CANFAR Science Platform launch helpers
pip install 'torchregress[canfar]'

# For CLAUDS data tooling
pip install 'torchregress[clauds]'

# For TabReD data tooling
pip install 'torchregress[tabred]'

# For foundation-model experiments
pip install 'torchregress[foundation-models]'

# For all optional dependencies
pip install 'torchregress[all]'
```

## From Source

To install the latest development version:

```bash
git clone https://github.com/sfabbro/torchregress.git
cd torchregress
pip install -e .
```

## Requirements

torchregress has the following dependencies:

### Core Dependencies

- Python >= 3.12, < 3.16
- PyTorch >= 2.4.0
- NumPy >= 2.0.0
- matplotlib >= 3.8.0
- pandas >= 2.2.0
- torchmetrics >= 1.4.0
- scikit-learn >= 1.5.0
- scipy >= 1.11.0
- tqdm >= 4.66.0

### Optional Dependencies

Testing:
- pytest >= 8.0.0
- pytest-cov >= 5.0.0
- polars >= 0.20.0
- pyarrow >= 18.0.0
- PyYAML >= 6.0.0

Development:
- black
- ruff
- mypy

Documentation:
- mkdocs
- mkdocs-material
- mkdocstrings[python]
- pymdown-extensions

Normalizing flows:
- zuko >= 1.6.0

Science/data tooling extras:
- `canfar`: CANFAR Science Platform launcher dependencies
- `clauds`: CLAUDS data tooling dependencies
- `tabred`: TabReD data tooling dependencies
- `foundation-models`: TabPFN-backed experiment dependencies

## Verifying Installation

To verify that torchregress is installed correctly, you can run:

```python
import torchregress
print(torchregress.__version__)
```

Or use the following minimal example:

```python
import torch
import torchregress as tr

# Create some dummy data
X = torch.randn(100, 1)
y = 2 * X.squeeze() + 1 + 0.1 * torch.randn(100)

# Define a simple model
model = torch.nn.Linear(1, 1)

# Choose a loss function
loss_fn = tr.losses.WeightedHuberLoss()

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

If this runs without errors, torchregress is correctly installed.
