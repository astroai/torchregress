# API Reference

This section contains detailed documentation for the torchregress API.

## Package Structure

torchregress is organized into the following main modules:

```
torchregress/
│
├── losses/          # Loss functions for regression
│   ├── base.py      # Base loss classes
│   ├── gaussian.py  # Gaussian losses (MSE, MAE, NLL)
│   ├── robust.py    # Robust losses (Huber, Cauchy, etc.)
│   ├── quantile.py  # Quantile losses
│   ├── expectile.py # Expectile losses
│   ├── poisson.py   # Poisson losses
│   ├── tweedie.py   # Tweedie losses
│   ├── mdn.py       # Mixture Density Network losses
│   ├── nflows.py    # Normalizing Flows losses
│   ├── rag.py       # Residual Adaptive Gaussian losses
│   └── eiv.py       # Error-in-Variables losses
│
├── metrics/         # Evaluation metrics for regression
│   ├── point.py     # Metrics for point predictions
│   ├── distribution.py  # Metrics for distributional predictions
│   ├── interval.py  # Metrics for prediction intervals
│   ├── calibration.py   # Metrics for calibration
│   ├── ood.py       # Metrics for OOD detection
│   └── utils.py     # Utility functions for metrics
│
├── ensemble/        # Ensemble methods for regression
│   ├── deep.py      # Deep ensembles
│   ├── bootstrap.py # Bootstrapped ensembles
│   ├── snapshot.py  # Snapshot ensembles
│   └── utils.py     # Utility functions for ensembles
│
├── viz/             # Visualization tools
│   ├── diagnostic.py    # Diagnostic plots
│   ├── monitoring.py    # Training monitoring plots
│   ├── results.py       # Results visualization
│   └── utils.py         # Utility functions for visualization
│
└── utils/           # General utilities
    ├── validation.py    # Input validation
    └── tensor_ops.py    # Tensor operations
```

## Core Module: Losses

The `losses` module contains implementations of various loss functions for regression tasks.

```python
import torchregress.losses as losses

# Create a loss function
loss_fn = losses.HuberLoss(delta=1.0)

# Use in training
loss = loss_fn(y_pred, y_true)
```

[Learn more about loss functions →](losses.md)

## Core Module: Metrics

The `metrics` module provides functions to evaluate regression models.

```python
import torchregress.metrics as metrics

# Calculate metrics
rmse = metrics.rmse(y_pred, y_true)
nll = metrics.gaussian_nll(mean, y_true, variance)
picp = metrics.picp(y_true, lower, upper)
```

[Learn more about metrics →](metrics.md)

## Core Module: Visualization

The `viz` module contains functions for visualizing regression results and diagnostics.

```python
import torchregress.viz as viz

# Create visualization
viz.plot_predictions(X, y, y_pred)
viz.plot_residuals(y_pred, y_true)
viz.plot_calibration_curve(mean, std, y_true)
```

[Learn more about visualization →](viz.md)

## Core Module: Ensembles

The `ensemble` module provides tools for creating ensemble models.

```python
import torchregress.ensemble as ensemble

# Create an ensemble
models = [create_model() for _ in range(5)]
deep_ensemble = ensemble.DeepEnsemble(models)

# Get predictions with uncertainty
mean, variance = deep_ensemble.predict(X)
```

## Utilities

The `utils` module contains helper functions used throughout the library.

```python
import torchregress.utils as utils

# Validate input tensors
utils.validate_tensors(y_pred, y_true)

# Perform masked operations
masked_sum = utils.masked_sum(tensor, mask)
```

[Learn more about utilities →](utils.md)

## Complete Class Listing

Below is a complete listing of all public classes in torchregress:

### Loss Functions

- `MSELoss` - Mean Squared Error loss
- `L1Loss` - Mean Absolute Error loss
- `HuberLoss` - Huber loss
- `LogCoshLoss` - Log-Cosh loss
- `CauchyLoss` - Cauchy loss
- `BarronLoss` - General robust loss family with fixed shape
- `AdaptiveRobustLoss` - Trainable Barron-style robust loss
- `QuantileLoss` - Quantile regression loss
- `MultiQuantileLoss` - Multiple quantile regression loss
- `ExpectileLoss` - Expectile regression loss
- `GaussianNLLLoss` - Gaussian Negative Log-Likelihood loss
- `PoissonLoss` - Poisson regression loss
- `TweedieLoss` - Tweedie regression loss
- `MDNLoss` - Mixture Density Network loss
- `NormalizingFlowLoss` - Normalizing Flow loss
- `RAGLoss` - Residual Adaptive Gaussian loss
- `DemingLoss` - Deming regression loss
- `ODRLoss` - Orthogonal Distance Regression loss

### Metrics

- `mse` - Mean Squared Error
- `rmse` - Root Mean Squared Error
- `mae` - Mean Absolute Error
- `mape` - Mean Absolute Percentage Error
- `r2_score` - R² (coefficient of determination)
- `adjusted_r2_score` - Adjusted R²
- `gaussian_nll` - Gaussian Negative Log-Likelihood
- `crps_gaussian` - Continuous Ranked Probability Score for Gaussian distributions
- `energy_score` - Energy Score for multivariate distributions
- `picp` - Prediction Interval Coverage Probability
- `mpiw` - Mean Prediction Interval Width
- `interval_score` - Interval Score
- `expected_calibration_error` - Expected Calibration Error
- `auroc_ood` - AUROC for OOD detection

### Ensemble Methods

- `DeepEnsemble` - Deep ensemble of multiple models
- `BootstrappedEnsemble` - Ensemble trained on bootstrap samples
- `SnapshotEnsemble` - Ensemble using snapshots from training trajectory

### Visualization Functions

- `plot_predictions` - Plot predictions with uncertainty
- `plot_residuals` - Plot residuals
- `plot_calibration_curve` - Plot calibration curve
- `plot_reliability_diagram` - Plot reliability diagram for intervals
- `plot_learning_curve` - Plot learning curve during training
- `plot_comparison` - Compare multiple models' performance
