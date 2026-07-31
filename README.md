# torchregress

<p align="center">
    <em>A PyTorch library for regression, uncertainty quantification, and robust estimation</em>
</p>

<p align="center">
<a href="https://pypi.org/project/torchregress/" aria-label="PyPI package version"><img src="https://img.shields.io/pypi/v/torchregress.svg" alt="PyPI"></a>
<a href="https://opensource.org/licenses/MIT" aria-label="License"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://github.com/sfabbro/torchregress/actions/workflows/ci.yml" aria-label="CI status"><img src="https://img.shields.io/github/actions/workflow/status/sfabbro/torchregress/ci.yml?branch=main&label=CI" alt="CI"></a>
<a href="https://github.com/sfabbro/torchregress/blob/main/pyproject.toml" aria-label="Python 3.12–3.14"><img src="https://img.shields.io/badge/python-3.12%20|%203.13%20|%203.14-blue.svg" alt="Python 3.12 | 3.13 | 3.14"></a>
</p>

**torchregress** is a PyTorch library of regression losses, metrics, and calibration tools for problems where you need more than a single point prediction — uncertainty, robustness, and messy real-world data included.

It plugs into standard training loops and supports missing data and sample weights out of the box.

---

## Features & Capabilities

torchregress is built for regression problems that are messy. In plain terms, it helps you:

- **Predict distributions** — Train models that output means, spreads, quantiles, or full predictive distributions when a single point estimate is not enough.
- **Measure and separate uncertainty** — Distinguish irreducible data noise from model uncertainty, and evaluate whether predicted ranges are trustworthy.
- **Stay robust on dirty data** — Handle missing values, sample weights, outliers, noisy inputs and labels, censored targets, and imbalanced or rare outcomes without bolting on ad hoc fixes.
- **Calibrate and guarantee coverage** — Turn a trained model into well-calibrated intervals or conformal prediction sets with stated coverage properties.
- **Adapt to distribution shifts** — Lightweight test-time tools when deployment data differs from what you trained on.
- **Evaluate the right metrics** — Metrics for point error, interval quality, distributional accuracy, and calibration — not just average squared error.

Every loss and metric drops into a normal PyTorch training loop. For method names, API details, and worked examples, see the [documentation index](docs/index.md).

## Installation

### From PyPI

```bash
pip install torchregress
```

For normalizing flows support for multi-target distribution predictions, install with the `flows` extra:

```bash
pip install torchregress[flows]
```

### Dev / Source Setup (pixi)

This project uses [pixi](https://pixi.sh) for development. Supports Python 3.12–3.15 (3.13 recommended).

```bash
git clone https://github.com/sfabbro/torchregress.git
cd torchregress
pixi install

pixi run test        # pytest + coverage
pixi run lint        # ruff check + format --check
pixi run typecheck   # ty
pixi run docs        # zensical build --strict
pixi run ci          # lint + typecheck + test + docs
```

---

## Onboarding Recipes

### 1. Parametric Aleatoric UQ (Heteroscedastic Gaussian)
Train a network that outputs both a mean $\mu(x)$ and log-variance $\log\sigma^2(x)$ using a proper scoring rule (Gaussian NLL), and evaluate with the Continuous Ranked Probability Score (CRPS).

```python
import torch
import torch.nn as nn
from torchregress.losses import GaussianNLLLoss
from torchregress.metrics import crps_gaussian

# Model outputs [mean, log_variance]
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = GaussianNLLLoss()

# Training loop
for x, y in train_loader:
    optimizer.zero_grad()
    pred = model(x)  # Shape: [Batch, 2]
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    out = model(x_val)
    mu, logvar = out[:, 0], out[:, 1]
    std = torch.exp(0.5 * logvar)
    val_crps = crps_gaussian(mu, std, y_val)
    print(f"Validation CRPS: {val_crps:.4f}")
```

### 2. Non-Parametric Quantile Regression
Estimate multiple target quantiles simultaneously with crossover penalties to prevent quantile crossing.

```python
import torch
import torch.nn as nn
from torchregress.losses import MultiQuantileLoss, QuantileCrossoverLoss

# Predict 3 quantiles: 10%, 50%, 90%
quantiles = [0.1, 0.5, 0.9]
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

pinball_loss = MultiQuantileLoss(quantiles=quantiles)
crossover_penalty = QuantileCrossoverLoss(quantiles=quantiles)

# Training loop
for x, y in train_loader:
    optimizer.zero_grad()
    pred = model(x)  # Shape: [Batch, 3]
    loss = pinball_loss(pred, y) + 0.1 * crossover_penalty(pred, y)
    loss.backward()
    optimizer.step()
```

### 3. Distribution-Free Conformal Prediction
Wrap a pre-trained regression model to output interval predictions with guaranteed $90\%$ coverage.

```python
import torch
from torchregress.losses import SplitConformal

# Assume base_model is a pre-trained point estimator model(x) -> y_pred
base_model.eval()

# Calibrate conformal thresholds
conformal = SplitConformal(alpha=0.1)
x_cal = torch.randn(200, 10)  # Calibration features
y_cal = torch.randn(200, 1)   # Calibration targets
with torch.no_grad():
    y_cal_pred = base_model(x_cal)
conformal.calibrate(y_cal_pred, y_cal)

# Predict 90% prediction intervals on new test targets
x_test = torch.randn(10, 10)
with torch.no_grad():
    y_test_pred = base_model(x_test)
lower, upper = conformal.predict_interval(y_test_pred)
```

---

## Development & Contribution

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, quality checks, tests, and documentation guidelines.


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
