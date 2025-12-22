# CLAUDE.md

This file provides merged guidance to coding agents (Claude, Gemini, Codex) when working in this repository.

## Project Overview

**torchregress** (lowercase) is a PyTorch library providing regression losses, metrics, and utilities with a focus on uncertainty estimation, robust regression, and missing data support.

**Naming Convention:** The library name is "torchregress" (all lowercase).

## Development Commands

This project uses [uv](https://github.com/astral-sh/uv) as the package manager.

### Setup
```bash
uv pip install -e .[all]
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=torchregress --cov-report=html

# Run single test file
uv run pytest tests/losses/test_gaussian.py

# Run specific test
uv run pytest tests/losses/test_gaussian.py::TestGaussianLosses::test_gaussian_nll_loss
```

If `uv` is not available, use the project venv directly:
```bash
.venv/bin/python -m pytest
```

### Code Quality
```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Type checking
uv run mypy torchregress
```

### Documentation
```bash
# Build docs
uv run mkdocs build

# Serve docs locally
uv run mkdocs serve
```

### Build & Publish
```bash
# Build distribution
uv build

# Publish to PyPI
uv publish
```

## Architecture

### Core Design Pattern

All losses inherit from a three-tier base class hierarchy in `torchregress/losses/base.py`:

1. **BaseLoss**: Root class providing reduction strategies (`mean`, `sum`, `none`) and mask/weight support
2. **RegressionLoss**: For point prediction losses (MSE, Huber, etc.)
3. **DistributionLoss**: For probabilistic losses that output distribution parameters (Gaussian NLL, MDN, etc.)

### Loss Function Convention

All loss functions follow PyTorch conventions:
- Parameter order: `forward(y_pred, target, mask=None, weights=None, **kwargs)`
- Support for missing data via boolean `mask` parameter (False = missing)
- Support for sample weighting via `weights` parameter
- Reductions handled by `_reduce()` method from BaseLoss

### Module Organization

```
torchregress/
├── losses/          # Loss functions (gaussian, robust, quantile, conformal, etc.)
├── ensemble/        # Ensemble models (DeepEnsemble, BatchEnsemble, etc.)
├── algorithms/      # Training algorithms (IRLS)
├── metrics/         # Evaluation metrics (point, interval, calibration, etc.)
├── utils/           # Utilities (masking, validation, transformations)
├── wrappers.py      # High-level factory functions
└── models/          # Pre-built model architectures
```

### Key Abstractions

**Wrapper Functions** (`torchregress/wrappers.py`): High-level factory functions that create (model, loss) tuples:
- `create_gaussian_model()`: Heteroscedastic regression with learned variance
- `create_quantile_model()`: Quantile regression
- `create_robust_model()`: Robust regression with outlier-resistant losses
- `create_mdn_model()`: Mixture Density Networks
- `create_deep_ensemble()`: Deep ensemble uncertainty estimation

**WeightedLossWrapper** (`torchregress/losses/base.py`): Wraps any PyTorch loss to add mask and weight support (e.g., `WeightedMSELoss`, `WeightedMAELoss`).

**Ensemble Models** (`torchregress/ensemble/`):
- `DeepEnsemble`: Multiple independently trained models
- `HeteroscedasticEnsembleModel`: Ensembles with aleatoric uncertainty
- `BatchEnsembleLinear`: Efficient batch ensemble layers

**IRLS Algorithm** (`torchregress/algorithms/irls.py`): Iteratively Reweighted Least Squares for robust regression.

### Distribution Parameters

Models that output distributions typically return tuples or concatenated tensors:
- **Gaussian (diagonal)**: `(mean, log_variance)` or concatenated `[mean, log_var]`
- **Gaussian (full covariance)**: `mean` plus `covariance_matrices` passed separately via `MultivariateGaussianLoss`
- **Gaussian (low-rank)**: `mean` plus `cov_factor` and `cov_diag` passed to `LowRankGaussianLoss`
- **MDN**: Raw output containing mixture weights, means, and log-variances
- **Quantile**: Multiple quantile predictions concatenated `[q1, q2, ..., qn]`

Use `create_gaussian_nll()` to pick the appropriate Gaussian loss based on covariance type.
For low-rank heads, `low_rank_output_dim()` and `split_low_rank_gaussian_output()` describe the output layout.

### Error-in-Variables (EIV) Losses

EIV losses treat `y_pred` as noisy inputs (`x_obs`) and require a model reference inside the loss:
- `FunctionalEIVLoss`, `StructuralEIVLoss`, `OrthogonalDistanceRegressionLoss`, `EnsembleEIVLoss`
- Call pattern: `loss_fn(x_obs, y_obs, mask=...)` (not `loss_fn(model(x), y)`).

### Uncertainty Decomposition

**Critical Distinction:** Not all uncertainty methods support epistemic/aleatoric decomposition.

| Method | Epistemic | Aleatoric | Use Case |
|--------|-----------|-----------|----------|
| Heteroscedastic Ensemble | ✅ | ✅ | Decomposed uncertainty with variance prediction |
| MDN (Mixture Density Network) | ✅ | ✅ | Multimodal distributions with decomposition |
| Normalizing Flows (ensemble) | ✅ | ✅ | Flexible distributions via flow ensemble |
| Deep Ensemble | ✅ | ❌ | Epistemic only (unless combined with variance prediction) |
| Quantile Regression | ❌ | ❌ | Distribution-free intervals, no decomposition |
| Conformal Prediction | ❌ | ❌ | Distribution-free coverage guarantees, NOT uncertainty decomposition |
| SWAG/MultiSWAG | ✅ | ⚠️ | Epistemic via weight posterior (aleatoric requires additional modeling) |

**Key Point:** Conformal prediction provides **coverage guarantees**, not uncertainty decomposition. Use it for calibrated intervals, not for separating epistemic/aleatoric uncertainty.

## Configuration

**pyproject.toml settings**:
- Python >= 3.10 required
- Black line length: 100
- Ruff: enforces E (pycodestyle), F (pyflakes), I (isort)
- MyPy: strict typing enabled with `disallow_untyped_defs`

**Test configuration**:
- Tests in `tests/` directory
- Pattern: `test_*.py` files with `test_*` functions
- Warnings for deprecation and user warnings are ignored

## Working with Loss Functions

When adding new loss functions:
1. Inherit from `RegressionLoss` (point predictions) or `DistributionLoss` (probabilistic)
2. Implement `forward(y_pred, target, mask=None, weights=None)`
3. Use `self._validate_inputs()` to check shapes
4. Use `self._reduce_with_mask()` or `self._reduce()` for reduction
5. Add to `torchregress/losses/__init__.py` exports
6. Add tests following patterns in `tests/`

## Dependencies

Core dependencies:
- torch >= 2.0.0
- numpy >= 1.21.0
- torchmetrics >= 1.0.0
- matplotlib, pandas (for visualization/data handling)
- scikit-learn (density weighting utilities)

Required (feature-specific) dependencies:
- **torchcp >= 1.2.0** (conformal prediction)
- **zuko >= 1.4.0** (normalizing flows)

### Import Policy

All imports must be direct/unconditional. NO conditional imports like:
```python
# ❌ WRONG - Do not use try/except for optional dependencies
try:
    import torchcp
    TORCHCP_AVAILABLE = True
except ImportError:
    TORCHCP_AVAILABLE = False
```

Instead:
```python
# ✅ CORRECT - Direct imports
import torchcp
from torchcp.regression import CQR, ACIPredictor
```

If a module is not installed, the import will fail immediately - this is the desired behavior. Users must install required dependencies for the features they use.
