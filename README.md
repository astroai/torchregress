# torchregress

<a href="https://pypi.org/project/torchregress/" aria-label="PyPI package version"><img src="https://img.shields.io/pypi/v/torchregress.svg" alt="PyPI"></a>
<a href="https://opensource.org/licenses/MIT" aria-label="License"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://github.com/sfabbro/torchregress/actions/workflows/ci.yml" aria-label="CI status"><img src="https://img.shields.io/github/actions/workflow/status/sfabbro/torchregress/ci.yml?branch=main&label=CI" alt="CI"></a>
<a href="https://github.com/sfabbro/torchregress/blob/main/pyproject.toml" aria-label="Python 3.12–3.14"><img src="https://img.shields.io/badge/python-3.12%20|%203.13%20|%203.14-blue.svg" alt="Python 3.12 | 3.13 | 3.14"></a>

**torchregress** is a PyTorch library providing regression losses, metrics, and calibration layers for deep learning regression. The focus is on proper scoring rules, uncertainty quantification (UQ), robust estimation, and handling real-world data pathologies (missing data masks, measurement errors, censoring, imbalance, and target shifts).

All modules are designed to integrate seamlessly into native PyTorch training loops, supporting mask arrays and sample weights out of the box.

---

## Capabilities Directory

| Category | Available Features | Documentation |
|:---|:---|:---|
| **Loss Primitives** | Point & M-estimators (MSE, MAE, Huber, Cauchy, Tukey, Barron, AdaptiveRobust, CVaR); Parametric Probabilistic (Gaussian NLL, Beta-NLL, Full Covariance, Low-Rank, Evidential NIG); Generative & Non-Parametric (MDN, Normalizing Flows) | [Losses Guide](docs/losses/index.md) |
| **Data Pathologies** | Error-in-Variables (Functional, Structural, ODR, MC marginalization); Censored Targets (AFT, Censored Gaussian/Quantile NLL); Continuous Target Imbalance (LDS, FDS); Label Noise (Consistency Regularization) | [Concepts Guide](docs/getting-started/concepts.md) |
| **Uncertainty & Ensembles** | Deep Ensembles, Batch Ensembles, Last-Layer Laplace Regressor, SWAG / MultiSWAG, Variational Inference & Bayesian Neural Networks (IVON) | [Ensembles Guide](docs/methods/ensemble/index.md) |
| **Conformal Calibration** | Split Conformal, Conformalized Quantile Regression (CQR), Mondrian/Group-conditioned, Local Conformal (MAD), Monte-Carlo Conformal, PPI inference | [Conformal Guide](docs/methods/conformal/index.md) |
| **Test-Time Adaptation** | Bayesian Linear Head adapters, Feature Stat Normalization, Significant Subspace Aligner (SSA), OT Conformal shift reweighting | [Test-Time API](docs/api/test_time.md) |
| **Metrics** | Point errors (RMSE, MAE, R², Huber); Distributional scores (CRPS, Energy Score, GNLL); Interval metrics (Interval Score, PICP, MPIW); Calibration (ECE, MCE, PIT histograms, typicality) | [Metrics API](docs/api/metrics.md) |

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
    val_crps = crps_gaussian(mu, std, y_val).mean()
    print(f"Validation CRPS: {val_crps.item():.4f}")
```

### 2. Non-Parametric Quantile Regression
Estimate multiple target quantiles simultaneously with crossover penalties to prevent quantile crossing.

```python
import torch
import torch.nn as nn
from torchregress.losses import QuantileLoss, QuantileCrossoverLoss

# Predict 3 quantiles: 10%, 50%, 90%
quantiles = [0.1, 0.5, 0.9]
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

pinball_loss = QuantileLoss(quantiles=quantiles)
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
conformal = SplitConformal(base_model)
x_cal = torch.randn(200, 10)  # Calibration features
y_cal = torch.randn(200, 1)   # Calibration targets
conformal.calibrate(x_cal, y_cal)

# Predict 90% prediction intervals on new test targets
x_test = torch.randn(10, 10)
intervals = conformal.predict_interval(x_test, alpha=0.1)
# intervals contains lower and upper bounds of shape [10, 2]
```

---

## Documentation Index

Detailed mathematical derivations, guide overviews, and examples:

* [Core Concepts](docs/getting-started/concepts.md) — Point vs. probabilistic prediction, aleatoric vs. epistemic uncertainty, proper scoring rules.
* [Method Selection Guide](docs/guide/method-selection.md) — Task-first lookup table to find the right tool for specific data characteristics.
* [Mathematical Foundations](docs/guide/math/index.md) — LaTeX derivations of CRPS, scoring rules, and variance decomposition formulas.
* [Runnable Examples](docs/examples/index.md) — Practical scripts comparing `torchregress` methods with external baselines.

---

## Sibling Repositories

| Repository | Role |
|:---|:---|
| [torchregress-research](https://github.com/sfabbro/torchregress-research) | NeurIPS manuscripts and SAGE/SPT benchmarks |
| [torchregress-harness](https://github.com/sfabbro/torchregress-harness) | External software comparisons (MAPIE, LightGBM, botorch, etc.) |

---

## License

MIT License
