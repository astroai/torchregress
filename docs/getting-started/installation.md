# Installation

> ← [Getting Started](index.md) | [Quick Start](quickstart.md) →

torchregress needs Python 3.12 or newer (tested on 3.12, 3.13 and 3.14;
3.13 recommended) and PyTorch 2.13 or newer.

## From PyPI

```bash
pip install torchregress
```

!!! tip "GPU builds of PyTorch"
    `pip install torchregress` pulls the default PyTorch wheel for your
    platform. For a specific CUDA version, install PyTorch first with the
    command from [pytorch.org](https://pytorch.org/get-started/locally/), then
    install torchregress.

## Optional extras

| Extra | Installs | Needed for |
|:--|:--|:--|
| `flows` | zuko | normalizing-flow losses (`NormalizingFlowLoss`, `ContrastiveFlowLoss`) |
| `viz` | matplotlib | `torchregress.viz` plots |
| `test` | pytest, scikit-learn, pandas, polars, pyarrow, scoringrules, properscoring, hypothesis | running the test suite |
| `docs` | zensical | building this site |
| `all` | all of the above | development |

```bash
pip install "torchregress[flows,viz]"
```

`import torchregress` works without the extras; a module that needs one
raises `ImportError` when you use it.

## With pixi

In a [pixi](https://pixi.sh) project, add torchregress from PyPI next to the
conda-forge PyTorch build:

```bash
pixi add pytorch
pixi add --pypi torchregress
```

## From source

Development uses pixi:

```bash
git clone https://github.com/astroai/torchregress.git
cd torchregress
pixi install

pixi run test        # pytest + coverage
pixi run lint        # ruff
pixi run typecheck   # ty
pixi run docs        # zensical build --strict
pixi run ci          # all of the above
```

`pip install -e ".[all]"` from a clone also works if you do not use pixi.

## Requirements

| Package | Minimum |
|:--|:--|
| Python | 3.12 |
| torch | 2.13 |
| numpy | 2.5 |
| scipy | 1.18 |
| torchmetrics | 1.9 |

## Check the installation

The package installs a health check that prints the torchregress, PyTorch
and Python versions and the device, imports the main modules, and runs one
training step and one metric on random data (on the GPU when one is
available):

```bash
torchregress-health
```

Or check the version from Python:

```python
import torchregress
print(torchregress.__version__)
```

A minimal training loop:

```python
import torch
import torchregress as tr

X = torch.randn(100, 1)
y = 2 * X + 1 + 0.1 * torch.randn(100, 1)

model = torch.nn.Linear(1, 1)
loss_fn = tr.losses.WeightedHuberLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(100):
    loss = loss_fn(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    print(f"RMSE: {tr.metrics.rmse(model(X), y).item():.4f}")
```
