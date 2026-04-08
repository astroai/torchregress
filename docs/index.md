# torchregress

<p align="center">
  <img src="https://raw.githubusercontent.com/sfabbro/torchregress/main/docs/assets/logo.png" width="200" alt="torchregress logo">
</p>

<p align="center">
    <em>Deep Learning for Regression & Uncertainty Estimation in PyTorch</em>
</p>

<p align="center">
<a href="https://github.com/sfabbro/torchregress/actions/workflows/ci.yml" target="_blank">
    <img src="https://github.com/sfabbro/torchregress/workflows/CI/badge.svg" alt="CI">
</a>
<a href="https://pypi.org/project/torchregress" target="_blank">
    <img src="https://img.shields.io/pypi/v/torchregress?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/torchregress" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/torchregress.svg?color=%2334D058" alt="Supported Python versions">
</a>
<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</a>
</p>

---

**torchregress** is a comprehensive PyTorch library designed for researchers and practitioners working on complex regression problems. It goes beyond standard Mean Squared Error, providing a unified toolkit for **probabilistic modeling**, **uncertainty quantification**, and **robust estimation**.

---

## Why torchregress?

Regression in the real world is rarely just about predicting a single number. Data is noisy, distributions are skewed, and knowing *how much to trust* a prediction is often as important as the prediction itself. 

- **Probabilistic by Design**: Built-in support for estimating aleatoric and epistemic uncertainty.
- **Robust to Real-World Data**: Advanced losses for noisy labels, outliers, and imbalanced targets.
- **Scientifically Rigorous**: Implements seminal methods with verified mathematical correctness.
- **Production Ready**: Seamlessly integrates with existing PyTorch workflows, supporting `torch.compile`, AMP, and distributed training.

---

## Key Features

<div class="grid cards" markdown>

-   :material-chart-bell-curve-cumulative: __Probabilistic Losses__

    -   [Gaussian NLL (Heteroscedastic)](losses/gaussian.md)
    -   [Multivariate & Low-Rank Covariance](losses/gaussian.md#2-multivariate-full-covariance-multivariategaussianloss)
    -   [Mixture Density Networks (MDN)](losses/mdn.md)
    -   [Normalizing Flows](losses/nflows.md)

-   :material-shield-check: __Uncertainty Quantification__

    -   [**Deep Ensembles** & SWAG](ensemble/index.md)
    -   [**Conformal Prediction** (CQR, Dist-Free)](conformal/index.md)
    -   [**Evidential Regression**](examples/evidential_regression.md)
    -   [MC-Dropout & Bayesian Neural Networks](ensemble/index.md#method-selection-matrix)

-   :material-flask-outline: __Robust Estimation__

    -   [Quantile & Expectile Regression](losses/quantile_expectile.md)
    -   [Robust M-Estimators (Huber, Cauchy, Tukey, Barron)](losses/robust.md)
    -   [Noisy Label & Outlier Mitigation](losses/robust.md#redescending-losses-cauchy-tukey)
    -   [Measurement Error (EIV) Correction](math/index.md#specialized-regression-tasks)


-   :material-chart-line: __Advanced Metrics__

    -   [Proper Scoring Rules (CRPS, Energy Score)](metrics/distribution.md)
    -   [Calibration Curves & Sharpness](metrics/calibration.md)
    -   [OOD Detection & Selective Prediction](metrics/ood.md)
    -   [Interval Coverage & Efficiency](metrics/interval.md)

</div>

---

## Installation

```bash
pip install torchregress
```

*Requires PyTorch 2.0+ and Python 3.9+.*

---

## Quickstart in 30 Seconds

Train a model that predicts both the **mean** ($\mu$) and **uncertainty** ($\sigma^2$) using the Heteroscedastic Gaussian NLL loss.

```python
import torch
import torch.nn as nn
from torchregress.losses import GaussianNLLLoss

# 1. Define a model with two outputs (mean and log-variance)
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))

# 2. Use the specialized Gaussian NLL loss
loss_fn = GaussianNLLLoss()

# 3. Standard PyTorch training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for x, y in dataloader:
    pred = model(x) # [batch, 2]
    loss = loss_fn(pred, y)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

---

## Getting Started

<div class="grid cards" markdown>

-   __New to Uncertainty?__
    -   Read the [Core Concepts Guide](guides/concepts.md)
    -   Check the [Mathematical Foundations](math/index.md)
    -   Follow the [Basic Usage Tutorial](examples/basic_usage.md)

-   __Advanced Users__
    -   Explore [Conformal Prediction](conformal/index.md)
    -   Master [Deep Ensembles](ensemble/index.md)
    -   View the [Method Selection Matrix](guides/method_selection_matrix.md)

</div>

---

## Citation

If you use torchregress in your research, please cite:

```bibtex
@software{torchregress,
  title = {{torchregress: A PyTorch Library for Regression and Uncertainty Estimation}},
  author = {Fabbro, Sébastien},
  url = {https://github.com/sfabbro/torchregress},
  version = {0.1.0},
  year = {2024},
}
```
