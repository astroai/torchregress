# torchregress

[![PyPI](https://img.shields.io/pypi/v/torchregress.svg)](https://pypi.org/project/torchregress/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive PyTorch library for regression, uncertainty estimation, calibration,
and hard regression settings (outliers, imbalance, noisy features, multimodal targets).

## Overview

**torchregress** provides a collection of regression loss functions, metrics, and uncertainty estimation techniques implemented in PyTorch. It's designed to make it easy to:

- Use different regression loss functions beyond MSE
- Estimate uncertainty in regression predictions
- Evaluate regression models with appropriate metrics
- Visualize regression results and uncertainty

## Start By Task (Recommended)

Use the library from the problem you need to solve, not from a method family:

- **Outliers / robust regression**: `HuberLoss`, `CauchyLoss`, `TukeyBiweightLoss`
- **Prediction intervals with coverage guarantees**: conformal prediction (`split`, `CQR`, `ACI`)
- **Uncertainty decomposition (epistemic + aleatoric)**: heteroscedastic ensembles
- **Well-calibrated Gaussian training**: `GaussianNLLLoss` for likelihood training, `GaussianCRPSLoss` when you want a proper scoring rule that directly rewards sharp calibrated predictive CDFs
- **Multimodal targets**: `MDN` first, then `MDNEnsembleModel` or `BinnedPDFEnsembleModel` when you want predictive-distribution averaging across members
- **Imbalanced / rare-target regression**: start with `GaussianCRPSLoss` or quantiles plus tail-slice evaluation; add density-aware methods only if they win on your benchmark
- **Noisy features / measurement error**: start with explicit input-noise marginalization and predictive averaging, then escalate to EIV / ODR losses if it clearly helps
- **OOD robustness / selective prediction**: ensemble uncertainty + OOD + decision metrics

Task-first method matrix (recommended entry point):

- [`docs/guides/method_selection_matrix.md`](docs/guides/method_selection_matrix.md)

This matrix treats `SWAG`, `BNN`, `MDN`, ensembles, conformal, quantile, and flows as peer options.

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
- Local dev/docs/tests: `uv pip install -e ".[all]"`

CI on `main` runs **pre-commit**, then **`uv sync` + pytest + CPU benchmark smoke/sweep** in one test job (see `.github/workflows/ci.yml`).

## Test-Time Tooling

`torchregress` now includes reusable test-time tooling designed to sit on top of
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

Docs entry points:

- Concepts: [`docs/guides/concepts.md`](docs/guides/concepts.md)
- Method matrix: [`docs/guides/method_selection_matrix.md`](docs/guides/method_selection_matrix.md)
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
