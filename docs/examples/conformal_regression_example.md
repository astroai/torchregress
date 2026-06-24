# Conformal Regression Example

This example demonstrates how to use the conformal prediction methods in torchregress to obtain prediction intervals with guaranteed coverage.

→ API: [`SplitConformal`](../api/losses.md#splitconformal), [`CQR`](../api/losses.md#cqr), [`ConformalLoss`](../api/losses.md#conformalloss). Guide: [Conformal predictors](../methods/conformal/predictors.md).

## Task-First Context

Use this example when you need **prediction intervals with coverage guarantees**. For
task-level comparison across conformal, quantile, Gaussian, and ensemble methods, start
with the [Task-First Method Selection Matrix](../guide/method-selection.md) and
the comparison script `examples/evaluate_conformal_methods.py`.

## Comparison / Fairness Notes

- Compare methods on the same train/calibration/test split.
- Report both empirical coverage and interval width (coverage alone is not enough).
- Keep base model capacity and training budget comparable across conformal variants.
- Re-run across multiple seeds when using learned quantile models (CQR).

## When This Approach Fails

- Conformal can produce very wide intervals if the base model is poor or the problem is highly shifted.
- Coverage guarantees do not provide epistemic/aleatoric decomposition.
- Multi-output conformal intervals can be conservative; check per-dimension coverage and width.

## Code

```python
import numpy as np
import torch
import torch.nn as nn

from torchregress.losses.conformal import (
    ConformalLoss,
    MultiDimensionalConformalLoss,
    CVPlus,
    EnsembleBatchCP,
)


class DummyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 2) -> tuple:
    """Generate synthetic regression data with heteroscedastic noise."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate features
    X = torch.randn(n_samples, n_features)

    # Generate targets with heteroscedastic noise
    if n_features >= 2:
        y_true = torch.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    else:
        y_true = torch.sin(X[:, 0])
    noise_std = 0.1 + 0.2 * torch.abs(X[:, 0])  # Heteroscedastic noise
    y = y_true + torch.randn(n_samples) * noise_std

    return X, y.unsqueeze(1), y_true.unsqueeze(1)


def demo_basic_conformal() -> None:
    """Demonstrate basic ConformalLoss (split conformal)."""
    print("=== Basic Conformal Prediction Demo (Split Conformal) ===")

    # Generate data
    X, y, _ = generate_synthetic_data(n_samples=1000, n_features=1)

    # Split into train/calibration/test
    n_train = 600
    n_cal = 200
    _X_train, _X_cal, _X_test = X[:n_train], X[n_train : n_train + n_cal], X[n_train + n_cal :]
    y_train, y_cal, y_test = y[:n_train], y[n_train : n_train + n_cal], y[n_train + n_cal :]

    # Create dummy predictions (point predictions for split conformal)
    y_pred_train = y_train + 0.1 * torch.randn_like(y_train)
    y_pred_cal = y_cal + 0.1 * torch.randn_like(y_cal)
    y_pred_test = y_test + 0.1 * torch.randn_like(y_test)

    # Create conformal loss
    loss_fn = ConformalLoss(method="split", alpha=0.1)

    # Train with the loss
    train_loss = loss_fn(y_pred_train, y_train)
    print(f"Training loss: {train_loss.item():.4f}")

    # Calibrate on calibration set
    loss_fn.calibrate(y_pred_cal, y_cal)
    print("Calibrated successfully.")

    # Get prediction intervals on test set
    lower_interval, upper_interval = loss_fn.predict_interval(y_pred_test)

    # Calculate coverage
    coverage = ((y_test >= lower_interval) & (y_test <= upper_interval)).float().mean()
    print(f"Test coverage: {coverage.item():.4f} (target: {1 - loss_fn.alpha:.2f})")

    # Calculate interval width
    avg_width = (upper_interval - lower_interval).mean()
    print(f"Average interval width: {avg_width.item():.4f}")
    print()


def demo_conformalized_quantile() -> None:
    """Demonstrate Conformalized Quantile Loss."""
    print("=== Conformalized Quantile Regression Demo ===")

    # Generate data
    X, y, _ = generate_synthetic_data(n_samples=1000, n_features=1)

    # Split data
    n_train = 600
    n_cal = 200
    _X_train, _X_cal, _X_test = X[:n_train], X[n_train : n_train + n_cal], X[n_train + n_cal :]
    y_train, y_cal, y_test = y[:n_train], y[n_train : n_train + n_cal], y[n_train + n_cal :]

    # Create dummy quantile predictions (lower and upper quantiles with noise)
    torch.manual_seed(123)
    y_pred_train = torch.cat([y_train - 0.5 + 0.15 * torch.randn_like(y_train), y_train + 0.5 + 0.15 * torch.randn_like(y_train)], dim=1)
    y_pred_cal = torch.cat([y_cal - 0.5 + 0.15 * torch.randn_like(y_cal), y_cal + 0.5 + 0.15 * torch.randn_like(y_cal)], dim=1)
    y_pred_test = torch.cat([y_test - 0.5 + 0.15 * torch.randn_like(y_test), y_test + 0.5 + 0.15 * torch.randn_like(y_test)], dim=1)

    # Create conformalized quantile loss
    loss_fn = ConformalLoss(method="cqr", alpha=0.1)

    # Train with the loss
    train_loss = loss_fn(y_pred_train, y_train)
    print(f"Training loss: {train_loss.item():.4f}")

    # Calibrate on calibration set (uses conformal correction for q-hat)
    loss_fn.calibrate(y_pred_cal, y_cal)
    print("Calibrated successfully.")

    # Get calibrated prediction intervals
    lower_interval, upper_interval = loss_fn.predict_interval(y_pred_test)

    # Calculate coverage
    coverage = ((y_test >= lower_interval) & (y_test <= upper_interval)).float().mean()
    print(f"Test coverage: {coverage.item():.4f} (target: {1 - loss_fn.alpha:.2f})")

    # Calculate interval width
    avg_width = (upper_interval - lower_interval).mean()
    print(f"Average interval width: {avg_width.item():.4f}")
    print()


def demo_width_adaptive_conformal() -> None:
    """Demonstrate UACQR (Width-Adaptive Conformal Quantile Regression)."""
    print("=== Width-Adaptive Conformal Prediction Demo (UACQR) ===")

    # Generate data
    X, y, _ = generate_synthetic_data(n_samples=1000, n_features=1)

    # Split into train/calibration/test
    n_train = 600
    n_cal = 200
    y_train, y_cal, y_test = y[:n_train], y[n_train : n_train + n_cal], y[n_train + n_cal :]

    # Create dummy quantile predictions (lower and upper quantiles with noise)
    torch.manual_seed(456)
    y_pred_train = torch.cat([y_train - 0.5 + 0.15 * torch.randn_like(y_train), y_train + 0.5 + 0.15 * torch.randn_like(y_train)], dim=1)
    y_pred_cal = torch.cat([y_cal - 0.5 + 0.15 * torch.randn_like(y_cal), y_cal + 0.5 + 0.15 * torch.randn_like(y_cal)], dim=1)
    y_pred_test = torch.cat([y_test - 0.5 + 0.15 * torch.randn_like(y_test), y_test + 0.5 + 0.15 * torch.randn_like(y_test)], dim=1)

    # Create conformal loss
    loss_fn = ConformalLoss(method="uacqr", alpha=0.1)

    # Train with the loss
    train_loss = loss_fn(y_pred_train, y_train)
    print(f"Training loss: {train_loss.item():.4f}")

    # Calibrate on calibration set (uses conformal correction for q-hat)
    loss_fn.calibrate(y_pred_cal, y_cal)
    print("Calibrated successfully.")

    # Get prediction intervals on test set
    lower_interval, upper_interval = loss_fn.predict_interval(y_pred_test)

    # Calculate coverage
    coverage = ((y_test >= lower_interval) & (y_test <= upper_interval)).float().mean()
    print(f"Test coverage: {coverage.item():.4f} (target: {1 - loss_fn.alpha:.2f})")

    # Calculate interval width
    avg_width = (upper_interval - lower_interval).mean()
    print(f"Average interval width: {avg_width.item():.4f}")
    print()


def demo_multidimensional_conformal() -> None:
    """Demonstrate MultiDimensionalConformalLoss."""
    print("=== Multi-Dimensional Conformal Prediction Demo ===")

    # Generate multi-output data
    X, y, _ = generate_synthetic_data(n_samples=1000, n_features=3)

    # Split data
    n_train = 600
    n_cal = 200
    X_train, X_cal, X_test = X[:n_train], X[n_train : n_train + n_cal], X[n_train + n_cal :]
    y_train, y_cal, y_test = y[:n_train], y[n_train : n_train + n_cal], y[n_train + n_cal :]

    # Create dummy predictions (point predictions)
    model = DummyModel(3, 3)
    y_pred_train = model(X_train)
    y_pred_cal = model(X_cal)
    y_pred_test = model(X_test)

    # Create multi-dimensional conformal loss
    loss_fn = MultiDimensionalConformalLoss(alpha=0.1)

    # Train with the loss
    train_loss = loss_fn(y_pred_train, y_train)
    print(f"Training loss: {train_loss.item():.4f}")

    # Calibrate on calibration set
    loss_fn.calibrate(y_pred_cal, y_cal)
    print("Calibrated successfully.")

    # Get prediction intervals
    lower_intervals, upper_intervals = loss_fn.predict_interval(y_pred_test)

    # Calculate coverage per dimension
    for i in range(y_test.shape[1]):
        coverage = (
            ((y_test[:, i] >= lower_intervals[:, i]) & (y_test[:, i] <= upper_intervals[:, i]))
            .float()
            .mean()
        )
        print(f"Dimension {i} coverage: {coverage.item():.4f} (target: {1 - loss_fn.alpha:.2f})")

    # Calculate average interval widths per dimension
    avg_widths = (upper_intervals - lower_intervals).mean(dim=0)
    print(f"Average interval widths: {avg_widths}")
    print()


def demo_cv_plus() -> None:
    """Demonstrate CV+ Conformal Predictor for cross-validation ensembles."""
    print("=== CV+ Ensemble Conformal Prediction Demo ===")

    # Generate data
    _, y, _ = generate_synthetic_data(n_samples=500, n_features=1)

    # We will simulate a 5-fold cross-validation
    n_folds = 5
    n_samples = y.shape[0]

    # Shuffle and split indices
    torch.manual_seed(42)
    shuffled_idx = torch.randperm(n_samples)
    fold_sizes = n_samples // n_folds

    # Out-of-fold predictions on calibration data (which we simulate as all training data)
    y_pred_oob = torch.zeros_like(y)
    fold_indices = torch.zeros(n_samples, dtype=torch.long)

    # Simulate ensemble members predictions for test samples (e.g. 50 test samples)
    n_test = 100
    _, y_test, _ = generate_synthetic_data(n_samples=n_test, n_features=1)

    # We will simulate member predictions on the test set: shape [n_folds, n_test, 1]
    y_pred_members = torch.zeros(n_folds, n_test, 1)

    for f in range(n_folds):
        start = f * fold_sizes
        end = (f + 1) * fold_sizes if f < n_folds - 1 else n_samples
        val_idx = shuffled_idx[start:end]

        # Out-of-fold predictions for validation fold
        # Simulate model f predicting on val_idx: true target + simulated fold-specific noise
        y_pred_oob[val_idx] = y[val_idx] + 0.15 * torch.randn(val_idx.shape[0], 1) + 0.05 * f
        fold_indices[val_idx] = f

        # Simulate model f predicting on test set
        y_pred_members[f] = y_test + 0.15 * torch.randn(n_test, 1) + 0.05 * f

    # Create CVPlus conformal predictor
    cp = CVPlus(alpha=0.1)

    # Calibrate using out-of-fold predictions
    cp.calibrate_ensemble(y_pred_oob, y, fold_indices)
    print("Calibrated CV+ ensemble successfully.")

    # Get CV+ prediction intervals
    lower_interval, upper_interval = cp.predict_interval(y_pred_members)

    # Calculate coverage
    coverage = ((y_test >= lower_interval) & (y_test <= upper_interval)).float().mean()
    print(f"Test coverage: {coverage.item():.4f} (target: {1 - cp.alpha:.2f})")

    # Calculate interval width
    avg_width = (upper_interval - lower_interval).mean()
    print(f"Average interval width: {avg_width.item():.4f}")
    print()


def demo_ensemble_batch_cp() -> None:
    """Demonstrate Ensemble Batch Conformal Prediction (EnbPI) / Bootstrap+."""
    print("=== Ensemble Batch Conformal Prediction (EnbPI) Demo ===")

    # Generate data
    _, y, _ = generate_synthetic_data(n_samples=600, n_features=1)

    # Split into calibration and test
    n_cal = 400
    y_cal, y_test = y[:n_cal], y[n_cal:]

    # Simulate out-of-bag ensemble predictions for calibration samples
    torch.manual_seed(42)
    y_pred_oob = y_cal + 0.15 * torch.randn_like(y_cal)

    # Simulate ensemble mean prediction for test samples
    y_pred_mean = y_test + 0.15 * torch.randn_like(y_test)

    # Create EnsembleBatchCP conformal predictor
    cp = EnsembleBatchCP(alpha=0.1)

    # Calibrate using OOB predictions
    cp.calibrate(y_pred_oob, y_cal)
    print("Calibrated EnsembleBatchCP successfully.")

    # Get prediction intervals
    lower_interval, upper_interval = cp.predict_interval(y_pred_mean)

    # Calculate coverage
    coverage = ((y_test >= lower_interval) & (y_test <= upper_interval)).float().mean()
    print(f"Test coverage: {coverage.item():.4f} (target: {1 - cp.alpha:.2f})")

    # Calculate interval width
    avg_width = (upper_interval - lower_interval).mean()
    print(f"Average interval width: {avg_width.item():.4f}")
    print()


def main() -> None:
    """Run all demos."""
    print("Conformal Prediction Examples")
    print("=" * 50)

    demo_basic_conformal()
    demo_conformalized_quantile()
    demo_width_adaptive_conformal()
    demo_multidimensional_conformal()
    demo_cv_plus()
    demo_ensemble_batch_cp()

    print("All demos completed successfully!")


if __name__ == "__main__":
    main()
```
