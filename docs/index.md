# torchregress

<p align="center">
    <em>Deep Learning for Regression & Uncertainty Estimation in PyTorch</em>
</p>

<p align="center">
<a href="https://github.com/sfabbro/torchregress/actions/workflows/ci.yml" target="_blank" aria-label="GitHub Actions CI status">
    <img src="https://github.com/sfabbro/torchregress/workflows/CI/badge.svg" alt="CI">
</a>
<a href="https://pypi.org/project/torchregress" target="_blank" aria-label="PyPI package version">
    <img src="https://img.shields.io/pypi/v/torchregress?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/torchregress" target="_blank" aria-label="Supported Python versions">
    <img src="https://img.shields.io/pypi/pyversions/torchregress.svg?color=%2334D058" alt="Supported Python versions">
</a>
<a href="https://opensource.org/licenses/MIT" target="_blank" aria-label="License">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</a>
</p>

---

**torchregress** is a PyTorch library for researchers and practitioners working on complex regression problems. It provides a unified toolkit for **probabilistic modeling**, **uncertainty quantification**, and **robust estimation** — going far beyond standard Mean Squared Error.

---

## What can you do with torchregress?

<div class="grid cards" markdown>

-   :material-chart-bell-curve-cumulative: __Predict with Confidence__

    Go beyond point predictions. Model heteroscedastic noise, full probability distributions, and prediction intervals with guaranteed coverage.

    [:octicons-arrow-right-24: Browse Loss Functions](losses/index.md)

-   :material-shield-check: __Quantify Uncertainty__

    Decompose uncertainty into aleatoric (data noise) and epistemic (model ignorance). Use ensembles, Bayesian methods, or conformal prediction.

    [:octicons-arrow-right-24: Explore Methods](methods/index.md)

-   :material-flask-outline: __Handle Messy Data__

    Outliers, noisy labels, missing values, measurement errors, imbalanced targets — built-in losses and algorithms for real-world data.

    [:octicons-arrow-right-24: See Robust Losses](losses/robust.md)

-   :material-chart-line: __Evaluate Rigorously__

    Proper scoring rules, calibration diagnostics, out-of-distribution detection, and interval metrics — not just RMSE.

    [:octicons-arrow-right-24: View Metrics](metrics/index.md)

</div>

---

## Quick Example

Train a model that predicts both the **mean** and **uncertainty** in a few lines:

```python
import torch
import torch.nn as nn
from torchregress.losses import GaussianNLLLoss

# Model with two outputs: mean and log-variance
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))

# Gaussian negative log-likelihood — learns uncertainty from data
loss_fn = GaussianNLLLoss()

# Standard PyTorch training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for x, y in dataloader:
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

[:octicons-arrow-right-24: Full Quick Start](getting-started/quickstart.md)

---

## Installation

```bash
pip install torchregress
```

*Requires PyTorch 2.4+ and Python 3.12 – 3.15.*

---

## Where should I start?

<div class="grid cards" markdown>

-   :fontawesome-solid-graduation-cap: __New to Uncertainty Quantification?__

    ---

    Start with the [Core Concepts](getting-started/concepts.md) to learn the vocabulary, then follow the [Quick Start](getting-started/quickstart.md) to train your first uncertainty-aware model.

    [:octicons-arrow-right-24: Get Started](getting-started/index.md)

-   :fontawesome-solid-flask: __Experienced Practitioner__

    ---

    Jump to the [Method Selection Matrix](guide/method-selection.md) to find the right loss for your problem, or browse the [Examples](examples/index.md) for runnable comparison scripts.

    [:octicons-arrow-right-24: User Guide](guide/index.md)

-   :fontawesome-solid-microscope: __Researcher or Statistician__

    ---

    Study the [Mathematical Foundations](guide/math/index.md) for rigorous derivations, check the [Reports & Evidence](reports/index.md) for benchmarks, or dive into the [API Reference](api/index.md).

    [:octicons-arrow-right-24: API Reference](api/index.md)

</div>

---

## Library at a Glance

| Category | What's Included |
|:---------|:----------------|
| **Loss Functions** | Gaussian (heteroscedastic, multivariate, low-rank), robust losses (Huber, Cauchy, Tukey), quantile & expectile losses, mixture density networks, normalizing flows, evidential regression, super-level set regression, ordinal & censored losses, Poisson & Tweedie, input measurement error, imbalanced regression, noisy labels, uncertain ground truth, target transforms, conformal wrappers |
| **Uncertainty Methods** | Deep ensembles, batch ensembles, Monte Carlo dropout, SWAG, Bayesian neural networks, conformal prediction (split, quantile, distributional, density-aware) |
| **Algorithms** | Robust fitting (IRLS), measurement error correction (RC, SIMEX, latent input regression, error-aware encoding), covariance learning, Bayesian last layer, adaptive prior inference, Bayesian learning rule optimizer |
| **Calibration** | Variance temperature scaling, isotonic calibration, PIT calibration, semi-supervised conformal calibration, label shift estimation |
| **Metrics** | Point, distribution, interval, calibration, out-of-distribution detection, ensemble, multivariate, ordinal, censored |
| **Inference** | Causal inference (doubly-robust ATE/CATE), prediction-powered inference, test-time adaptation |

---

## Citation

If you use torchregress in your research, please cite:

```bibtex
@software{torchregress,
  title = {{torchregress: A PyTorch Library for Regression and Uncertainty Estimation}},
  author = {Fabbro, S{\'e}bastien},
  url = {https://github.com/sfabbro/torchregress},
  version = {0.1.0},
  year = {2024},
}
```
