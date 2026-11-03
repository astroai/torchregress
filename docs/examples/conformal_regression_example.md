# Conformal Regression Example

This example demonstrates how to use the conformal prediction methods in torchregress to obtain prediction intervals with guaranteed coverage.

→ API: [`SplitConformal`](../api/losses.md), [`CQR`](../api/losses.md), [`ConformalLoss`](../api/losses.md). Guide: [Conformal predictors](../methods/conformal/predictors.md).

**Run the script** [`examples/conformal_regression_example.py`](https://github.com/astroai/torchregress/blob/main/examples/conformal_regression_example.py) (`--mode all` by default runs the three modes in turn):

```bash
pixi run python examples/conformal_regression_example.py --mode tour      # every conformal predictor on simulated predictions
pixi run python examples/conformal_regression_example.py --mode sweep     # trained models, coverage and width across alpha
pixi run python examples/conformal_regression_example.py --mode mondrian  # group-conditional vs global split conformal
```

| Mode | What it shows | Output |
|:--|:--|:--|
| `tour` | Split conformal, CQR, UACQR, per-dimension (`MultiTargetConformal`), `CVPlus` and `EnsembleBatchCP` on simulated predictions (code below) | Coverage and mean width per predictor |
| `sweep` | Point and two-quantile MLPs trained once, then split, CQR and UACQR at $\alpha \in \{0.05, 0.1, 0.2\}$ (details below) | Fairness notes, comparison table, `interval_width_vs_coverage.png`, `coverage_vs_target.png` |
| `mondrian` | `SplitConformal` with and without `groups=` on two noise regimes (details below) | Per-group coverage table |

## Task-First Context

Use this example when you need **prediction intervals with coverage guarantees**. For
task-level comparison across conformal, quantile, Gaussian, and ensemble methods, start
with the [Task-First Method Selection Matrix](../guide/method-selection.md) and
the `--mode sweep` comparison of this script.

## Comparison / Fairness Notes

- Compare methods on the same train/calibration/test split.
- Report both empirical coverage and interval width (coverage alone is not enough).
- Keep base model capacity and training budget comparable across conformal variants.
- Re-run across multiple seeds when using learned quantile models (CQR).

## When This Approach Fails

- Conformal can produce very wide intervals if the base model is poor or the problem is highly shifted.
- Coverage guarantees do not provide epistemic/aleatoric decomposition.
- Multi-output conformal intervals can be conservative; check per-dimension coverage and width.

## Code (`--mode tour`)

```python
import numpy as np
import torch
import torch.nn as nn

from torchregress.losses.conformal import (
    ConformalLoss,
    MultiTargetConformal,
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
    """Demonstrate MultiTargetConformal."""
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

    # Create per-dimension conformal predictor
    loss_fn = MultiTargetConformal(alpha=0.1)

    # Calibrate per-dimension thresholds on the calibration set
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

## Coverage Sweep (`--mode sweep`)

The sweep trains two `MLPRegressor` heads once on the same 1,000-point training split
(50 full-batch Adam epochs each): a point head with MSE for split conformal, and a
two-quantile head with `MultiQuantileLoss(quantiles=[0.05, 0.95])` shared by CQR and
UACQR. Each method is then calibrated on 500 points and evaluated on 500 test points at
every target level. For each method and $\alpha$ it reports target coverage $1-\alpha$,
empirical coverage, the coverage gap $|\widehat{\text{cov}} - (1-\alpha)|$, mean
interval width, and train/evaluation time.

!!! tip
    Read coverage and width together: at the same coverage, the narrower method is
    better. CQR and UACQR adapt width to the heteroscedastic noise; split conformal
    returns one constant half-width $\hat{q}$ for every input.

---

## Mondrian (Group-Conditional) Conformal (`--mode mondrian`)

This guide explains how to perform Mondrian (group-conditional) conformal prediction to guarantee valid coverage independently within subgroups of data using `SplitConformal` with group indicators.

→ API: [`SplitConformal`](../api/losses.md). Guide: [Conformal predictors](../methods/conformal/predictors.md).

| # | Reference |
|:-:|:----------|
| 1 | Vovk, V. (2002). [**Conditional Conformal Prediction**](https://link.springer.com/chapter/10.1007/3-540-36169-3_29). *International Conference on Algorithmic Learning Theory*. |
| 2 | Vovk, V., Gammerman, A., & Shafer, G. (2005). [**Algorithmic Learning in a Random World**](https://doi.org/10.1007/b106715). *Springer*. |

---

### Mathematical Formulations

Standard conformal prediction guarantees marginal coverage:
$$P(Y \in \hat{C}(X)) \ge 1 - \alpha$$
However, this overall average does not guarantee coverage conditional on group features $G$ (e.g., demographics, noise regimes). Mondrian conformal prediction guarantees:
$$P(Y \in \hat{C}(X) \mid G = g) \ge 1 - \alpha \quad \forall g$$

#### Mondrian Calibration

Let the calibration set $C_{\text{cal}}$ be partitioned into group-wise calibration sets:
$$C_g = \{i \in C_{\text{cal}} \mid G_i = g\}$$

For each group $g$:
1.  Compute nonconformity scores for $i \in C_g$:
    $$S_i = |y_i - \hat{y}_i|$$
2.  Compute the conformal threshold as the $(1-\alpha)\left(1 + \frac{1}{|C_g|}\right)$ quantile of $\{S_i\}_{i \in C_g}$:
    $$\hat{q}_g = \text{Quantile}\left(\{S_i\}_{i \in C_g}, (1-\alpha)\left(1 + \frac{1}{|C_g|}\right)\right)$$

#### Mondrian Prediction

For a new test point with prediction $\hat{y}$ and group label $G_{\text{test}} = g$, the prediction interval is:
$$[\hat{y} - \hat{q}_g, \hat{y} + \hat{q}_g]$$

---

### Code Example

Below is a condensed, self-contained version of `--mode mondrian` demonstrating group-conditional calibration (the script also trains on a separate split and prints per-group coverage against global split conformal).

```python
import numpy as np
import torch
import torch.nn as nn
from torchregress.losses.conformal import SplitConformal

# Generate synthetic grouped data with heteroscedastic noise
np.random.seed(42)
n_samples = 1000

x = np.random.uniform(-1.5, 1.5, size=(n_samples, 1)).astype(np.float32)
groups = np.random.choice([0, 1], size=(n_samples, 1)).astype(np.int64)
y_mean = 0.8 * x

# Group 0: low noise (0.2), Group 1: high noise (1.2)
noise_std = np.where(groups == 0, 0.2, 1.2).astype(np.float32)
y = y_mean + np.random.normal(0, noise_std).astype(np.float32)

x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)
groups_tensor = torch.from_numpy(groups).squeeze(1)

# Fit simple regression model
class MeanRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MeanRegressionModel()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    preds = model(x_tensor)
    loss = torch.nn.functional.mse_loss(preds, y_tensor)
    opt.zero_grad()
    loss.backward()
    opt.step()

# Calibrate Mondrian Conformal Predictor
model.eval()
with torch.no_grad():
    preds = model(x_tensor)

mondrian_cp = SplitConformal(alpha=0.1)
mondrian_cp.calibrate(preds, y_tensor, groups=groups_tensor)

print("Mondrian Conformal Predictor calibrated successfully.")
```
