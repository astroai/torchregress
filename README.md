# torchregress

<a href="https://pypi.org/project/torchregress/" aria-label="PyPI package version"><img src="https://img.shields.io/pypi/v/torchregress.svg" alt="PyPI"></a>
<a href="https://opensource.org/licenses/MIT" aria-label="License"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://github.com/sfabbro/torchregress/actions/workflows/ci.yml" aria-label="CI status"><img src="https://img.shields.io/github/actions/workflow/status/sfabbro/torchregress/ci.yml?branch=main&label=CI" alt="CI"></a>
<a href="https://github.com/sfabbro/torchregress/blob/main/pyproject.toml" aria-label="Python 3.12+"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python 3.12+"></a>
<a href="https://github.com/astral-sh/ruff" aria-label="Code style: ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Code style: ruff"></a>
<a href="https://codecov.io/gh/sfabbro/torchregress" aria-label="Codecov coverage"><img src="https://codecov.io/gh/sfabbro/torchregress/branch/main/graph/badge.svg" alt="Codecov"></a>

A comprehensive PyTorch library for regression, uncertainty estimation, calibration,
and hard regression settings (outliers, imbalance, noisy features, multimodal targets).

## Quick Links

- 📋 **[Task-First Method Selection Matrix](docs/guide/method-selection.md)** — pick a method by problem (clean / robust / multimodal / OOD / censored / causal)
- 💡 **[Choosing Methods by Constraint](docs/guide/choosing-by-constraint.md)** — pick by latency budget, coverage guarantees, decomposition, or operational complexity
- 📊 **[Comparative Evidence Matrix](docs/reports/comparative_evidence_matrix.md)** — see which tasks have decision-grade vs demo-only evidence
- 📚 **[Examples Index](docs/examples/index.md)** — runnable comparison examples for every category, including external baselines ([vs MAPIE / BoTorch / scikit-lego](docs/examples/external-comparison-vs-mapie-botorch-sklego.md))

## Hello world (30 lines)

```python
import torch
from torchregress.losses import WeightedMSELoss
from torchregress.metrics import rmse, r2_score

# 1. Synthetic data: y = x @ w + noise
torch.manual_seed(0)
x = torch.randn(200, 4)
w = torch.tensor([1.0, -2.0, 0.5, 3.0])
y = x @ w + 0.1 * torch.randn(200)

# 2. A simple 2-layer MLP
model = torch.nn.Sequential(
    torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
)

# 3. Pick a loss and optimizer
loss_fn = WeightedMSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

# 4. Train (swap loss_fn for WeightedHuberLoss, CauchyLoss, QuantileLoss, …)
for _ in range(150):
    opt.zero_grad()
    loss_fn(model(x), y).backward()
    opt.step()

# 5. Evaluate (in practice, evaluate on a held-out split)
with torch.no_grad():
    pred = model(x)
print(f"RMSE = {rmse(pred, y):.4f}  R² = {r2_score(pred, y):.4f}")
```

For uncertainty quantification, swap `WeightedMSELoss` → `GaussianNLLLoss` (returns a
`mean` + `log_var` head) or `QuantileLoss` (returns per-quantile outputs).
For prediction intervals, wrap any backbone with `ConformalLoss`.
For a full method-shortlist, jump to the
[Task-First Method Selection Matrix](docs/guide/method-selection.md).

## Next steps

After the hello world, the four highest-leverage pages for new users are:

1. 📋 **[Task-First Method Selection Matrix](docs/guide/method-selection.md)** — pick a method by problem (clean / robust / multimodal / OOD / censored / causal)
2. 💡 **[Choosing Methods by Constraint](docs/guide/choosing-by-constraint.md)** — pick by latency budget, coverage guarantees, decomposition, or operational complexity
3. 📊 **[Comparative Evidence Matrix](docs/reports/comparative_evidence_matrix.md)** — see which tasks have decision-grade vs demo-only evidence
4. 📚 **[Examples Index](docs/examples/index.md)** — runnable comparison examples for every category, including external baselines ([vs MAPIE / crepes / torchcp / BoTorch / scikit-lego](docs/examples/external-comparison-vs-mapie-botorch-sklego.md))

The external comparison page is the place to directly compare torchregress
against MAPIE / crepes / torchcp on conformal prediction regression, against
BoTorch on low-shot Bayesian linear regression, and against scikit-lego on
Tweedie regression — all on shared splits, fixed seeds, and the fairness
controls documented in the page. The numbers are an operational default, not
a horse race: capacity is intentionally not matched between libraries.

## Overview

**torchregress** provides a collection of regression loss functions, metrics, and uncertainty estimation techniques implemented in PyTorch. It's designed to make it easy to:

- Use different regression loss functions beyond MSE
- Estimate uncertainty in regression predictions
- Evaluate regression models with appropriate metrics
- Visualize regression results and uncertainty

## Start By Task (Recommended)

Use the library from the problem you need to solve, not from a method family:

- **Outliers / robust regression**: `WeightedHuberLoss`, `CauchyLoss`, `TukeyBiweightLoss`
- **Prediction intervals with coverage guarantees**: conformal prediction (`split`, `CQR`, `ACI`)
- **Uncertainty decomposition (epistemic + aleatoric)**: heteroscedastic ensembles
- **Well-calibrated Gaussian training**: `GaussianNLLLoss` for likelihood training, `GaussianCRPSLoss` when you want a proper scoring rule that directly rewards sharp calibrated predictive CDFs
- **Multimodal targets**: `MDN` first, then `MDNEnsembleModel` or `BinnedPDFEnsembleModel` when you want predictive-distribution averaging across members
- **Imbalanced / rare-target regression**: start with `GaussianCRPSLoss` or quantiles plus tail-slice evaluation; add density-aware methods only if they win on your benchmark
- **Noisy features / measurement error**: start with explicit input-noise marginalization and predictive averaging, then escalate to EIV / ODR losses if it clearly helps
- **OOD robustness / selective prediction**: ensemble uncertainty + OOD + decision metrics

For evidence-grade selection (what's decision-grade vs demo-only), pair this with the
[Comparative Evidence Matrix](docs/reports/comparative_evidence_matrix.md).

## Key Features

- **Diverse Loss Functions**: From basic (MSE, MAE) to advanced (MDN, NF) regression losses
- **Uncertainty Quantification**: Built-in support for different uncertainty estimation techniques
- **Comprehensive Metrics**: Evaluation metrics for point predictions, predictive distributions, and uncertainty
- **Visualization Tools**: Ready-to-use visualization functions for regression diagnostics
- **PyTorch Integration**: Seamlessly integrates with PyTorch models and training loops
- **Statistical Rigor**: Mathematically sound implementation of statistical estimators

## Installation

```bash
pip install torchregress
```

Optional extras:

- Flows (`zuko`): `pip install torchregress[flows]`
- Conformal: `pip install torchregress[conformal]`
- External comparison baselines (MAPIE / BoTorch / scikit-lego): `pip install torchregress[external]`
- Local dev/docs/tests: `uv pip install -e ".[all]"`

CI on `main` runs **pre-commit**, then **`uv sync` + pytest + mypy + `mkdocs build --strict` + codecov upload** in one test job (see `.github/workflows/ci.yml`).

## Test-Time Tooling

`torchregress` includes reusable test-time tooling designed to sit on top of
models owned by application repos such as `torchz`, rather than owning tabular
architectures itself.

Available building blocks:

- `PredictiveBatch`: a normalized prediction container for points, quantiles, bar
  distributions, samples, and support-grid densities
- label-shift correction via `PosteriorLabelShiftAdapter`
- confidence filtering and local-consistency weights for FTAT/PFT3A-style pipelines
- regression-oriented subspace alignment via `SignificantSubspaceAligner`
- lightweight dynamic ensembling via `ParameterEMA`

These tools live under `torchregress.prediction` and `torchregress.test_time` and
are meant to plug into existing predictors at test time:

- point predictions
- means and standard deviations
- quantiles
- bar logits plus bin edges
- support-grid densities
- samples

Minimal usage:

```python
from torchregress.prediction import PredictiveBatch
from torchregress.test_time import PosteriorLabelShiftAdapter, SignificantSubspaceAligner

adapter = PosteriorLabelShiftAdapter(source_prior=[0.5, 0.5, 0.0 + 1e-6])
aligner = SignificantSubspaceAligner(rank=8)
```

The intended stack is:

`application model -> PredictiveBatch -> test_time modules -> calibration / conformal / monitoring`

## Quickstart (Heteroscedastic Gaussian Regression)

```python
import torch
from torchregress.losses import GaussianNLLLoss
from torchregress.metrics import rmse

# Define your model (any PyTorch model that outputs mean and variance)
# For example:
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2) # Output mean and log_var

    def forward(self, x):
        return self.fc(x).chunk(2, dim=-1)

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = GaussianNLLLoss()

# Example data
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)
X_test = torch.randn(50, 10)
y_test = torch.randn(50, 1)

# Training loop
for epoch in range(10):
    for i in range(len(X_train)):
        # Forward pass: get predictions (mean and log variance)
        mean, log_var = model(X_train[i:i+1])

        # Calculate loss
        loss = loss_fn((mean, log_var), y_train[i:i+1])

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation
with torch.no_grad():
    mean, log_var = model(X_test)
    rmse_score = rmse(mean, y_test)
    print(f"RMSE: {rmse_score:.4f}")

```

## Common Usage

### Conformal Prediction

Install the conformal extra to use this module:

```bash
uv pip install -e ".[conformal]"
```

```python
from torchregress.losses.conformal import ConformalLoss

# Your model should be trained to predict quantiles for CQR
# For split conformal prediction, it can be a standard regression model
model = MyModel() # A model that outputs a point prediction or quantiles
loss_fn = ConformalLoss(method='cqr', alpha=0.1)  # 90% prediction intervals

# ... training loop for the model ...

# Calibrate on hold-out data
# For CQR, y_pred_cal should contain lower and upper quantile predictions
y_pred_cal = torch.randn(100, 2)
y_true_cal = torch.randn(100, 1)
loss_fn.calibrate(y_pred_cal, y_true_cal)

# Make predictions with calibrated intervals
y_pred_test = torch.randn(50, 2)
lower_interval, upper_interval = loss_fn.predict_interval(y_pred_test)
```

For more advanced usage and API details, refer to the [full documentation](https://github.com/sfabbro/torchregress).

## Choosing a Method (Examples by Problem)

- **Robustness + uncertainty + ensembles**:
  [`examples/comprehensive_comparison.py`](examples/comprehensive_comparison.py)
- **Loss comparison on outliers**:
  [`examples/comprehensive_loss_comparison.py`](examples/comprehensive_loss_comparison.py)
- **Imbalanced regression + calibration validation**:
  [`examples/imbalanced_regression.py`](examples/imbalanced_regression.py)
- **Conformal method comparison (coverage vs interval width)**:
  [`examples/evaluate_conformal_methods.py`](examples/evaluate_conformal_methods.py)
- **Multi-target multimodal regression (flows)**:
  [`examples/normalizing_flows_multitarget.py`](examples/normalizing_flows_multitarget.py)
- **External baselines (vs MAPIE / BoTorch / scikit-lego)**:
  [`docs/examples/external-comparison-vs-mapie-botorch-sklego.md`](docs/examples/external-comparison-vs-mapie-botorch-sklego.md)

Docs entry points:

- Concepts: [`docs/getting-started/concepts.md`](docs/getting-started/concepts.md)
- Method matrix: [`docs/guide/method-selection.md`](docs/guide/method-selection.md)
- Examples index: [`docs/examples/index.md`](docs/examples/index.md)

## Examples

- `examples/gaussian_full_covariance_regression.py` - full-covariance Gaussian regression
- `examples/gaussian_low_rank_regression.py` - low-rank Gaussian regression with diagonal correction

## Benchmarks

- `examples/benchmarks/tail_extremes_benchmark.py` - tail performance under noisy labels (robust, density-weighted, CVaR)
- `examples/benchmarks/tail_extremes_sweep.py` - sweep feature/label noise to identify best tail method
- `tools/benchmark_smoke.py` - fast smoke/sweep performance checks with CI threshold support

## License

MIT License
