# torchregress

<a href="https://pypi.org/project/torchregress/" aria-label="PyPI package version"><img src="https://img.shields.io/pypi/v/torchregress.svg" alt="PyPI"></a>
<a href="https://opensource.org/licenses/MIT" aria-label="License"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://github.com/sfabbro/torchregress/actions/workflows/ci.yml" aria-label="CI status"><img src="https://img.shields.io/github/actions/workflow/status/sfabbro/torchregress/ci.yml?branch=main&label=CI" alt="CI"></a>
<a href="https://github.com/sfabbro/torchregress/blob/main/pyproject.toml" aria-label="Python 3.12+"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python 3.12+"></a>

PyTorch losses, metrics, and utilities for regression. The focus is on
uncertainty-aware and robust training: masks and sample weights on losses,
probabilistic heads, conformal intervals, ensembles, and related evaluation.

## What's included

- **Losses** — point losses (MSE, Huber, …), Gaussian and mixture-density
  (MDN) heads, quantiles, normalizing flows, conformal wrappers, and
  error-in-variables (EIV) objectives for noisy inputs
- **Metrics** — point error, calibration, interval coverage, distribution
  scores, out-of-distribution (OOD) checks, ensemble disagreement
- **Methods** — deep ensembles, post-hoc calibration, conformal prediction
  (split, conformalized quantile regression / CQR, adaptive conformal inference
  / ACI), and training algorithms such as iteratively reweighted least squares
  (IRLS)
- **Test-time helpers** — adapters for label shift, subspace alignment, and
  related utilities under `torchregress.test_time`

All modules follow PyTorch conventions (`forward(y_pred, target, mask=…,
weights=…)`). See the docs for method selection, formulas, and runnable
examples.

## Installation

```bash
pip install torchregress
```

Optional extras:

```bash
pip install torchregress[flows]      # normalizing flows (requires zuko)
pip install torchregress[external]   # MAPIE, BoTorch, scikit-lego baselines for comparison examples
pip install torchregress[all]        # dev, docs, tests, and optional deps
```

Requires Python 3.12+ and PyTorch 2.4+.

## Minimal example

```python
import torch
from torchregress.losses import WeightedMSELoss
from torchregress.metrics import rmse, r2_score

torch.manual_seed(0)
x = torch.randn(200, 4)
w = torch.tensor([1.0, -2.0, 0.5, 3.0])
y = x @ w + 0.1 * torch.randn(200)

model = torch.nn.Sequential(
    torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
)
loss_fn = WeightedMSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(150):
    opt.zero_grad()
    loss_fn(model(x), y).backward()
    opt.step()

with torch.no_grad():
    pred = model(x)
print(f"RMSE = {rmse(pred, y):.4f}  R² = {r2_score(pred, y):.4f}")
```

Swap `WeightedMSELoss` for another loss from `torchregress.losses` (robust,
Gaussian negative log-likelihood, quantile, conformal, …) and use the matching
metrics from `torchregress.metrics`.

## Documentation

Full guides, API reference, and comparison examples live under [`docs/`](docs/):

| Start here | |
|:--|:--|
| [Core concepts](docs/getting-started/concepts.md) | Point vs probabilistic prediction, uncertainty types |
| [Method selection](docs/guide/method-selection.md) | Pick a method by problem |
| [Examples index](docs/examples/index.md) | Runnable scripts and comparisons |
| [Losses](docs/losses/index.md) · [Metrics](docs/metrics/index.md) · [Methods](docs/methods/index.md) | Category overviews |

Build locally: `uv run mkdocs serve`

Repository examples: [`examples/`](examples/)

## License

MIT License
