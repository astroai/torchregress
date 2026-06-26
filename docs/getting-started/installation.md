# Installation

torchregress can be installed from PyPI or directly from the source code. The library requires Python 3.12 to <3.16 and PyTorch 2.4 or newer. **Python 3.13 is recommended** for new environments; 3.12 and 3.14 are also supported.

## Python version

Supported runtimes: **3.12, 3.13, and 3.14** (see `requires-python` in `pyproject.toml` for the exact upper bound). Use **3.13** by default unless you have a reason to stay on 3.12 or try 3.14.

```bash
# Recommended for new machines (uv)
uv python install 3.13
uv venv --python 3.13 .venv
source .venv/bin/activate

# Or with pyenv / system Python 3.13
python3.13 -m venv .venv
source .venv/bin/activate
```

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

# For documentation
pip install 'torchregress[docs]'

# For normalizing-flow losses
pip install 'torchregress[flows]'

# For all extras
pip install 'torchregress[all]'
```

Development tools (ruff, mypy) are provided as a **dependency group** (PEP 735) instead of
an extra. If you are using `uv`, add `--dev` to your sync command:

```bash
uv sync --extra test --dev          # tests + lint + typecheck
uv sync --all-extras --dev          # everything including docs
```

## From Source

To install the latest development version:

```bash
git clone https://github.com/sfabbro/torchregress.git
cd torchregress
uv sync
```

## Requirements

torchregress has the following dependencies:

### Core Dependencies

- Python >= 3.12, < 3.16 (3.13 recommended)
- PyTorch >= 2.4.0
- NumPy >= 2.0.0
- matplotlib >= 3.8.0
- torchmetrics >= 1.4.0
- scipy >= 1.11.0
- tqdm >= 4.66.0

### Optional Dependencies

Testing:
- pytest >= 8.0.0
- pytest-cov >= 5.0.0
- polars >= 0.20.0
- pyarrow >= 18.0.0
- PyYAML >= 6.0.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0

Development (dependency group, not an extra):
- ruff
- mypy

Documentation:
- zensical

Normalizing flows:
- zuko >= 1.6.0


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
