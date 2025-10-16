# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

## Project Overview

**torchregress** (lowercase) is a PyTorch library providing loss functions and utilities for regression problems with focus on uncertainty estimation, robust regression, and missing data support.

The library name is "torchregress" (all lowercase), not "TorchRegress".

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
uv run pytest tests/test_metrics.py

# Run specific test
uv run pytest tests/test_metrics.py::test_function_name
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

All losses inherit from a three-tier base class hierarchy in [torchregress/losses/base.py](torchregress/losses/base.py):

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

**Wrapper Functions** ([wrappers.py](torchregress/wrappers.py)): High-level factory functions that create (model, loss) tuples:
- `create_gaussian_model()`: Heteroscedastic regression with learned variance
- `create_quantile_model()`: Quantile regression
- `create_robust_model()`: Robust regression with outlier-resistant losses
- `create_mdn_model()`: Mixture Density Networks
- `create_deep_ensemble()`: Deep ensemble uncertainty estimation

**WeightedLossWrapper** ([losses/base.py](torchregress/losses/base.py:289)): Wraps any PyTorch loss to add mask and weight support. All standard PyTorch losses are wrapped (e.g., `WeightedMSELoss`, `WeightedL1Loss`).

**Ensemble Models** ([ensemble/](torchregress/ensemble/)):
- `DeepEnsemble`: Multiple independently trained models
- `HeteroscedasticEnsembleModel`: Ensembles with aleatoric uncertainty
- `BatchEnsembleLinear`: Efficient batch ensemble layers

**IRLS Algorithm** ([algorithms/irls.py](torchregress/algorithms/irls.py)): Iteratively Reweighted Least Squares for robust regression with automatic weight function application.

### Distribution Parameters

Models that output distributions typically return tuples or concatenated tensors:
- **Gaussian**: `(mean, log_variance)` as tuple or `[mean, log_var]` concatenated
- **MDN**: Raw output containing mixture weights, means, and log-variances
- **Quantile**: Multiple quantile predictions concatenated `[q1, q2, ..., qn]`

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
5. Add to `losses/__init__.py` exports
6. Add tests following patterns in `tests/`

## Dependencies

Core dependencies:
- torch >= 2.0.0
- numpy >= 1.21.0
- torchmetrics >= 1.0.0
- matplotlib, pandas (for visualization/data handling)

Optional (required for specific features):
- **torchcp >= 1.2.0** (for conformal prediction - REQUIRED, no conditional imports)
- **zuko >= 1.4.0** (for normalizing flows - REQUIRED, no conditional imports)

### Import Policy

**IMPORTANT:** All imports must be direct/unconditional. NO conditional imports like:
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
