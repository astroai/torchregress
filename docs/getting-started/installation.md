# Installation

> ← [Getting Started](index.md) | [Quick Start](quickstart.md) →

torchregress can be installed from PyPI or from source. Requires Python 3.12 to <3.16 and PyTorch 2.0 or newer. **Python 3.13 is recommended**.

## From PyPI

```bash
pip install torchregress
```

## With Extra Dependencies

```bash
pip install 'torchregress[test]'    # testing stack
pip install 'torchregress[docs]'    # documentation (zensical)
pip install 'torchregress[flows]'   # normalizing-flow losses (zuko)
pip install 'torchregress[viz]'     # matplotlib
pip install 'torchregress[all]'     # test + docs + viz + flows
```

## From Source (pixi)

Development uses [pixi](https://pixi.sh):

```bash
git clone https://github.com/astroai/torchregress.git
cd torchregress
pixi install

pixi run test
pixi run lint
pixi run typecheck
pixi run docs
pixi run ci
```

End users who only need the library can still use `pip install -e .` from a clone.

## Requirements

### Core Dependencies

- Python >= 3.12, < 3.16 (3.13 recommended)
- PyTorch
- NumPy
- torchmetrics
- scipy

### Optional Dependencies

- **viz:** matplotlib
- **flows:** zuko
- **docs:** zensical
- **test:** pytest, pytest-cov, polars, pyarrow, PyYAML, scikit-learn, pandas

Dev tools (ruff, ty) are provided via the pixi `dev` feature, not as package extras.

## Verifying Installation

```python
import torchregress
print(torchregress.__version__)
```

Or a minimal training loop:

```python
import torch
import torchregress as tr

X = torch.randn(100, 1)
y = 2 * X.squeeze() + 1 + 0.1 * torch.randn(100)

model = torch.nn.Linear(1, 1)
loss_fn = tr.losses.WeightedHuberLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(100):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    print(f"RMSE: {tr.metrics.rmse(model(X), y).item():.4f}")
```
