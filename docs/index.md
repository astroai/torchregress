# torchregress

<p align="center">
    <em>A PyTorch library for regression, uncertainty quantification, and robust estimation</em>
</p>

<p align="center">
<a href="https://github.com/astroai/torchregress/actions/workflows/ci.yml" target="_blank" aria-label="GitHub Actions CI status">
    <img src="https://github.com/astroai/torchregress/workflows/CI/badge.svg" alt="CI">
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

**torchregress** is a PyTorch library that treats regression as a **probabilistic
inference problem** rather than a single-point estimation task. It provides a
research-grade toolkit for **likelihood-based losses**, **uncertainty
decomposition**, **robust estimation**, **error-in-variables modelling**,
**conformal prediction**, **causal inference**, and **test-time adaptation** —
all designed to compose with native PyTorch training loops and the broader
scientific Python ecosystem.

The library targets two audiences:

- **ML practitioners** who need production-ready losses, metrics, and methods
  for non-trivial regression problems (heteroscedastic noise, outliers, imbalanced
  targets, censored observations, measurement error, distribution shift).
- **Researchers and statisticians** who want a unified interface to
  probabilistic regression primitives with rigorous scoring rules, calibration
  diagnostics, and reproducible comparison harnesses.

!!! tip "Start here"
    New to torchregress? Follow the [Quick Start](getting-started/quickstart.md)
    for a 5-minute tour of the canonical workflows, then browse the
    [Capability matrix](#capability-matrix) below to find methods for your
    specific problem. If you prefer to understand the concepts first, read the
    [Core Concepts](getting-started/concepts.md) page — it defines the vocabulary
    used throughout the library.

---

## Research-grade design principles

1. **Task-first framing.** Methods are selected by the *problem you have* (the
   row in the [Method Selection Matrix](guide/method-selection.md)), not by
   modelling ideology. Bayesian, frequentist, and geometric methods are peer
   entries in the catalog.
2. **Likelihood contracts over point estimates.** Most losses accept a
   distribution family (Gaussian, Beta, Gamma, MDN, flow) and return both
   predictive samples and scoring-rule losses. Point-prediction losses are a
   special case of `GaussianNLLLoss(fixed_variance=σ²)`.
3. **Proper scoring rules as first-class citizens.** CRPS, energy score, NLL,
   interval score, and Brier are evaluated in the same units as the model's
   predictive distribution, not as point RMSE.
4. **Uncertainty decomposition is explicit.** Aleatoric (data noise) and
   epistemic (model ignorance) uncertainty are tracked through dedicated
   contracts (`uncertainty_decomposition`, `PredictiveBatch.extra`) rather
   than being conflated with point-error bars.
5. **Conformal prediction complements, not replaces, likelihood.** Coverage
   guarantees from split / CQR / density-aware conformal are reported alongside
   density-estimation metrics, not substituted for them.
6. **Evidence-based maturity.** Each method carries a `Core` / `Strong` /
   `Available` / `Advanced` label based on test depth, documentation
   coverage, and example availability — not family membership.

---

## Capability matrix

| Problem | Recommended starting point | Strong alternative | See |
|:--------|:---------------------------|:-------------------|:----|
| Clean regression baseline | `WeightedMSELoss` | `WeightedHuberLoss` | [Losses](losses/index.md) |
| Heteroscedastic noise (aleatoric UQ) | `GaussianNLLLoss` | `BetaNLLLoss`, heteroscedastic ensemble | [Gaussian](losses/gaussian.md) |
| Epistemic uncertainty | `DeepEnsemble` | `SWAG`, `BayesianNeuralNetwork`, `HeteroscedasticBNN` | [Ensembles](methods/ensemble/index.md) |
| Multimodal conditional distributions | `MDNLoss` | `NormalizingFlowLoss` | [MDN](losses/mdn.md) · [Flows](losses/nflows.md) |
| Imbalanced / rare targets | `QuantileLoss` + tail-slice evaluation | `DensityWeightedLoss`, `LDSLoss` | [Imbalanced](losses/imbalanced.md) |
| Noisy features / measurement error | `InputNoiseMarginalizationLoss` | `FunctionalEIVLoss`, `StructuralEIVLoss`, `OrthogonalDistanceRegressionLoss` | [EIV](losses/eiv.md) |
| Noisy labels / weak supervision | `NoisyTargetGaussianNLL` | `ConsistencyRegLoss`, `PseudoLabelConsistencyLoss` | [Noisy labels](losses/noisy_labels.md) |
| Censored / survival | `CensoredGaussianNLLLoss` | `AFTLoss`, `CensoredQuantileLoss` | [Censored](losses/censored.md) |
| Ordinal / ordered categories | `CumulativeLinkLoss` | `CORALLoss` | [Ordinal](losses/ordinal.md) |
| Count / Tweedie / positive-skewed | `TweedieLoss` | `NegativeBinomialNLLLoss`, `GammaLoss` | [Tweedie](losses/poisson_tweedie.md) |
| Robust to outliers | `WeightedHuberLoss` | `CauchyLoss`, `TukeyBiweightLoss` | [Robust](losses/robust.md) |
| Worst-case / tail-focused | `CVaRLoss` | robust losses + tail-slice evaluation | [Robust](losses/robust.md) |
| Coverage guarantees | `SplitConformal` | `CQR`, `DensityConformal`, `MonteCarloConformal` | [Conformal](methods/conformal/index.md) |
| Distribution shift (test-time) | `BayesianLinearHead` | `ShiftFactoredPredictiveTransport`, `ScoreCDFReweighter` | [Test-time](methods/test-time/bayesian-linear-regression.md) |
| Causal inference (ATE / CATE) | `dr_ate`, `dr_cate` | `PredictionPoweredInference` | [Causal](methods/causal.md) |
| OOD / selective prediction | `DeepEnsemble` + OOD metrics | `HeteroscedasticBatchEnsembleModel`, `SWAG` | [Ensembles](methods/ensemble/index.md) |

The full, code-driven catalog (maturity, capability flags, family peer
comparison) lives in the [Method Selection Matrix](guide/method-selection.md)
and the auto-generated
[Method Catalog report](reports/method_catalog_generated.md).

---

## Library at a glance

| Category | What's included |
|:---------|:----------------|
| **Losses** | Gaussian (diagonal, full covariance, low-rank, heteroscedastic), β-NLL, faithful NLL, Wasserstein bound surrogate; robust M-estimators (Huber, Pseudo-Huber, Cauchy, Tukey, Charbonnier, LogCosh, Barron, AdaptiveRobust, CVaR); quantile and expectile; mixture density networks; normalizing flows; evidential regression (NIG); super-level set regression; ordinal and censored; Poisson / Negative-Binomial / Tweedie / Gamma / Inverse-Gaussian; input measurement error (functional, structural, ODR, ensemble, input-noise marginalisation); imbalanced regression; noisy labels and uncertain ground truth; target transforms; conformal wrappers |
| **Uncertainty methods** | Deep ensembles, batch ensembles, MC dropout, SWAG / MultiSWAG, Bayesian neural networks (IVON, last-layer Laplace, VIDS), heteroscedastic ensembles, evidential regression |
| **Conformal prediction** | Split, CQR, UACQR, density-aware, Monte-Carlo, local (LVD), CTI, super-level set, multi-dimensional, prevalence-adjusted, SLS |
| **Algorithms** | Robust fitting (IRLS), measurement error correction (RC, SIMEX, latent-input regression, error-aware encoding), covariance learning (TicTac), Bayesian last layer, adaptive prior inference, Bayesian learning rule (IVON) |
| **Calibration** | Variance temperature scaling, isotonic mean calibration, PIT calibration, semi-supervised conformal calibration, binned label-shift estimation |
| **Metrics** | Point (RMSE, MAE, R², Huber, tail-slice), distributional (CRPS, energy score, GNLL, PIT, HPD), interval (interval score, PICP, MPIW), calibration (ECE, MCE), OOD (Mahalanobis, typicality, entropy, KDE), ensemble (uncertainty decomposition), multivariate, ordinal, censored, uncertain ground truth |
| **Inference** | Doubly-robust ATE / CATE / policy value, prediction-powered inference, test-time adaptation (Bayesian linear head, OT conformal, label shift, feature alignment, shift-factored transport) |

---

## Working with the library

The following snippet demonstrates the canonical workflow: a model that
outputs both a mean and a log-variance, trained with a proper scoring rule
(Gaussian NLL), and evaluated with the CRPS, NLL, and a reliability
diagnostic.

```python
import torch
import torch.nn as nn
from torchregress.losses import GaussianNLLLoss
from torchregress.metrics import crps_gaussian, gaussian_nll

# Heteroscedastic head: output [mean, log_var]
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))

loss_fn = GaussianNLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for x, y in dataloader:
    pred = model(x)                       # [B, 2]
    loss = loss_fn(pred, y)               # NLL — proper scoring rule
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Evaluation: density-aware metrics, not just RMSE
with torch.no_grad():
    out = model(x_test)
    mu, logvar = out[:, 0], out[:, 1]
    var = torch.exp(logvar)
    std = torch.sqrt(var)
    crps = crps_gaussian(mu, y_test, std)          # proper scoring rule
    gnll = gaussian_nll(mu, y_test, var)           # negative log-likelihood
```

For full end-to-end workflows, see the
[Quick Start](getting-started/quickstart.md) and the
[comprehensive comparison examples](examples/index.md).

---

## Installation

```bash
pip install torchregress
```

*Requires PyTorch 2.4+ and Python 3.12–3.14 (3.13 recommended).* For normalizing flows, install
the optional extra:

```bash
pip install torchregress[flows]
```

---

## Where to go next

<div class="grid cards" markdown>

-   :material-school: __New to uncertainty quantification?__

    ---

    Read the [Core Concepts](getting-started/concepts.md) to ground the
    vocabulary (aleatoric vs. epistemic, proper scoring rules, conformal
    coverage), then follow the [Quick Start](getting-started/quickstart.md) to
    train your first heteroscedastic model.

    [:octicons-arrow-right-24: Get Started](getting-started/index.md)

-   :material-flask: __Experienced practitioner__

    ---

    Open the [Method Selection Matrix](guide/method-selection.md) to shortlist
    losses by problem type, then validate the shortlist on the
    [comparison examples](examples/index.md).

    [:octicons-arrow-right-24: User Guide](guide/index.md)

-   :material-microscope: __Researcher or statistician__

    ---

    Study the [Mathematical Foundations](guide/math/index.md) for derivations
    of CRPS, NLL, interval score, and ensemble decomposition; check the
    [Reports & Evidence](reports/index.md) for benchmark matrices; consult
    the [API Reference](api/index.md) for the full surface.

    [:octicons-arrow-right-24: API Reference](api/index.md)

</div>

---

## Citation

If you use torchregress in your research, please cite:

```bibtex
@software{torchregress,
  title = {{torchregress: A PyTorch Library for Regression and Uncertainty Estimation}},
  author = {Fabbro, S{\'e}bastien},
  url = {https://github.com/astroai/torchregress},
  version = {0.1.0},
  year = {2024},
}
```
