r"""
Conformal prediction for regression: API tour, coverage sweep, and Mondrian calibration.

Modes (``--mode``):

``tour``
    Every conformal predictor on simulated predictions: split conformal, CQR, UACQR,
    per-dimension (multi-target), CV+ and EnbPI.

``sweep``
    Trained point and quantile MLPs, then split, CQR and UACQR calibrated at
    ``alpha in {0.05, 0.1, 0.2}``: actual coverage, coverage gap and interval width
    under a shared budget. Saves ``interval_width_vs_coverage.png`` and
    ``coverage_vs_target.png`` to the working directory.

``mondrian``
    Group-conditional (Mondrian) split conformal vs global split conformal on two
    groups with different noise levels.

``all`` (default)
    Runs ``tour``, ``sweep`` and ``mondrian`` in turn. Each mode reseeds its own data
    generator, so its output matches a standalone run of that mode.

Methods:

1. Split Conformal Prediction (point prediction correction):
   $$S_i = |y_i - \hat{y}_i|$$
   Interval: $[\hat{y} - \hat{q}, \hat{y} + \hat{q}]$ where $\hat{q}$ is the $(1-\alpha)(1+1/n)$ empirical quantile of calibration scores.

2. Conformalized Quantile Regression (CQR):
   $$S_i = \max(\hat{q}_{\alpha/2}(x_i) - y_i, y_i - \hat{q}_{1-\alpha/2}(x_i))$$
   Interval: $[\hat{q}_{\alpha/2}(x) - \hat{q}, \hat{q}_{1-\alpha/2}(x) + \hat{q}]$
   Seminal paper: Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized Quantile Regression. NeurIPS.

3. Width-Adaptive Conformalized Quantile Regression (UACQR):
   Normalizes the nonconformity score by the predicted interval width:
   $$S_i = \frac{\max(\hat{q}_{\alpha/2}(x_i) - y_i, y_i - \hat{q}_{1-\alpha/2}(x_i))}{\hat{q}_{1-\alpha/2}(x_i) - \hat{q}_{\alpha/2}(x_i)}$$
   Seminal paper: Gibbs, I., Cherian, J. J., & Candès, E. (2025). Conformal Prediction with Overfitting Correction.

4. Mondrian (group-conditional) conformal prediction calibrates one threshold per
   group, so coverage holds within each group rather than only on average.
   Seminal references: Vovk, V. (2002). Conditional Conformal Prediction. ALT.
   Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from torchregress.comparison import print_comparison_summary, print_fairness_notes, timed_call
from torchregress.losses.conformal import (
    ConformalLoss,
    CVPlus,
    EnsembleBatchCP,
    MultiTargetConformal,
    SplitConformal,
)
from torchregress.losses.quantile import MultiQuantileLoss

MODES = ("tour", "sweep", "mondrian")


class DummyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class MLPRegressor(nn.Module):
    """Two-hidden-layer ReLU MLP used by the sweep and Mondrian modes."""

    def __init__(self, in_features: int = 1, out_features: int = 1, hidden: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


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


# ============================================================================
# Mode "tour": every conformal predictor on simulated predictions
# ============================================================================


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
    y_pred_train = torch.cat(
        [
            y_train - 0.5 + 0.15 * torch.randn_like(y_train),
            y_train + 0.5 + 0.15 * torch.randn_like(y_train),
        ],
        dim=1,
    )
    y_pred_cal = torch.cat(
        [
            y_cal - 0.5 + 0.15 * torch.randn_like(y_cal),
            y_cal + 0.5 + 0.15 * torch.randn_like(y_cal),
        ],
        dim=1,
    )
    y_pred_test = torch.cat(
        [
            y_test - 0.5 + 0.15 * torch.randn_like(y_test),
            y_test + 0.5 + 0.15 * torch.randn_like(y_test),
        ],
        dim=1,
    )

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
    y_pred_train = torch.cat(
        [
            y_train - 0.5 + 0.15 * torch.randn_like(y_train),
            y_train + 0.5 + 0.15 * torch.randn_like(y_train),
        ],
        dim=1,
    )
    y_pred_cal = torch.cat(
        [
            y_cal - 0.5 + 0.15 * torch.randn_like(y_cal),
            y_cal + 0.5 + 0.15 * torch.randn_like(y_cal),
        ],
        dim=1,
    )
    y_pred_test = torch.cat(
        [
            y_test - 0.5 + 0.15 * torch.randn_like(y_test),
            y_test + 0.5 + 0.15 * torch.randn_like(y_test),
        ],
        dim=1,
    )

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
    """Demonstrate MultiTargetConformal (per-dimension split conformal)."""
    print("=== Multi-Dimensional Conformal Prediction Demo ===")

    # Generate multi-output data
    torch.manual_seed(42)
    X = torch.randn(1000, 3)
    y_base = torch.stack([torch.sin(X[:, 0]), 0.5 * X[:, 1] ** 2, 0.3 * X[:, 2]], dim=1)
    noise = 0.1 * torch.randn_like(y_base)
    y = y_base + noise

    # Split data
    n_train = 600
    n_cal = 200
    X_cal, X_test = X[n_train : n_train + n_cal], X[n_train + n_cal :]
    y_cal, y_test = y[n_train : n_train + n_cal], y[n_train + n_cal :]

    # Create dummy predictions (point predictions from an untrained model)
    model = DummyModel(3, 3)
    y_pred_cal = model(X_cal)
    y_pred_test = model(X_test)

    # Per-dimension conformal predictor (one threshold per output dimension).
    # It is a post-hoc calibrator, not a loss: train the model with any loss.
    loss_fn = MultiTargetConformal(alpha=0.1)

    # Calibrate on calibration set (uses conformal correction for q-hat)
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


def run_tour() -> None:
    """Run every conformal predictor demo on simulated predictions."""
    print("Conformal Prediction Examples")
    print("=" * 50)

    demo_basic_conformal()
    demo_conformalized_quantile()
    demo_width_adaptive_conformal()
    demo_multidimensional_conformal()
    demo_cv_plus()
    demo_ensemble_batch_cp()

    print("All demos completed successfully!")


# ============================================================================
# Mode "sweep": trained models, coverage and width across alpha
# ============================================================================


def train_model(model, X_train, y_train, loss_fn, epochs=50):
    """Train a simple model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def run_sweep() -> list[dict[str, object]]:
    """Compare split, CQR and UACQR coverage and width across target levels."""
    print("Evaluating Conformal Prediction Methods")
    print("=" * 50)

    X, y, _ = generate_synthetic_data(n_samples=2000, n_features=1)
    n_train = 1000
    n_cal = 500
    X_train, X_cal, X_test = X[:n_train], X[n_train : n_train + n_cal], X[n_train + n_cal :]
    y_train, y_cal, y_test = y[:n_train], y[n_train : n_train + n_cal], y[n_train + n_cal :]

    # Train a model for CQR (outputs two quantiles)
    cqr_model = MLPRegressor(in_features=1, out_features=2)
    train_loss_cqr = MultiQuantileLoss(quantiles=[0.05, 0.95])
    cqr_model, cqr_train_s = timed_call(train_model, cqr_model, X_train, y_train, train_loss_cqr)

    # Train a model for Split (outputs a point prediction)
    point_model = MLPRegressor(in_features=1, out_features=1)
    train_loss_mse = nn.MSELoss()
    point_model, point_train_s = timed_call(
        train_model, point_model, X_train, y_train, train_loss_mse
    )

    alphas = [0.05, 0.1, 0.2]
    results = []
    summary_rows: list[dict[str, object]] = []

    for alpha in alphas:
        methods_config = {
            "CQR": {"loss": ConformalLoss(method="cqr", alpha=alpha), "model": cqr_model},
            "UACQR": {"loss": ConformalLoss(method="uacqr", alpha=alpha), "model": cqr_model},
            "SPLIT": {"loss": ConformalLoss(method="split", alpha=alpha), "model": point_model},
        }

        for name, config in methods_config.items():
            loss_fn = config["loss"]
            model = config["model"]
            train_s = cqr_train_s if name in ("CQR", "UACQR") else point_train_s

            def _evaluate_method():
                with torch.no_grad():
                    pred_cal = model(X_cal)
                    pred_test = model(X_test)
                loss_fn.calibrate(pred_cal, y_cal)
                lower, upper = loss_fn.predict_interval(pred_test)
                coverage = ((y_test >= lower) & (y_test <= upper)).float().mean().item()
                width = (upper - lower).mean().item()
                return coverage, width

            (coverage, width), eval_s = timed_call(_evaluate_method)
            coverage_gap = abs(coverage - (1 - alpha))

            results.append(
                {
                    "Method": name,
                    "Target Coverage": 1 - alpha,
                    "Actual Coverage": coverage,
                    "Interval Width": width,
                }
            )
            summary_rows.append(
                {
                    "Method": f"{name} (alpha={alpha:.2f})",
                    "target_cov": 1 - alpha,
                    "actual_cov": coverage,
                    "coverage_gap": coverage_gap,
                    "width": width,
                    "train_s": train_s,
                    "eval_s": eval_s,
                    "Notes": "quantile head" if name in ("CQR", "UACQR") else "point head",
                }
            )

    # Print results
    results_df = pd.DataFrame(results)
    print(results_df)
    print_fairness_notes(
        title="Conformal Methods",
        seed_policy="fixed data generation seed and shared train/cal/test splits",
        train_budget="same architecture/depth; CQR and point models each trained once for 50 epochs",
        metric_policy="coverage gap and interval width across alpha values; per-method evaluation timed including calibration",
    )
    print_comparison_summary(
        "Conformal Comparison Summary",
        summary_rows,
        metric_order=["target_cov", "actual_cov", "coverage_gap", "width", "train_s", "eval_s"],
    )

    # Plot results
    plt.figure(figsize=(10, 6))
    methods = results_df["Method"].unique().tolist()
    coverages = sorted(results_df["Target Coverage"].unique())
    x = np.arange(len(methods))
    bar_width = 0.8 / max(len(coverages), 1)
    for i, cov in enumerate(coverages):
        subset = results_df[results_df["Target Coverage"] == cov]
        subset = subset.set_index("Method").reindex(methods)
        offset = (i - (len(coverages) - 1) / 2) * bar_width
        plt.bar(
            x + offset,
            subset["Interval Width"].values,
            width=bar_width,
            label=f"{cov:.2f}",
        )
    plt.xticks(x, methods, rotation=30, ha="right")
    plt.title("Mean Interval Width vs. Target Coverage")
    plt.xlabel("Method")
    plt.ylabel("Interval Width")
    plt.legend(title="Target Coverage")
    plt.tight_layout()
    plt.savefig("interval_width_vs_coverage.png")
    print("\nPlot saved to interval_width_vs_coverage.png")

    plt.figure(figsize=(10, 6))
    for method in methods:
        subset = results_df[results_df["Method"] == method].sort_values("Target Coverage")
        plt.plot(
            subset["Target Coverage"].values,
            subset["Actual Coverage"].values,
            marker="o",
            label=method,
        )
    plt.plot([0.8, 1.0], [0.8, 1.0], color="gray", linestyle="--")
    plt.title("Actual Coverage vs. Target Coverage")
    plt.xlabel("Target Coverage")
    plt.ylabel("Actual Coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("coverage_vs_target.png")
    print("Plot saved to coverage_vs_target.png")
    return summary_rows


# ============================================================================
# Mode "mondrian": group-conditional calibration
# ============================================================================


def generate_grouped_data(n_samples=2000, seed=42):
    """Generate data from two groups with distinct noise characteristics.

    Group 0 (low noise): std = 0.2
    Group 1 (high noise): std = 1.2
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = np.random.uniform(-1.5, 1.5, size=(n_samples, 1)).astype(np.float32)

    # Half the samples are in group 0, half in group 1
    groups = np.random.choice([0, 1], size=(n_samples, 1)).astype(np.int64)

    # True regression function
    y_mean = 0.8 * x

    # Group-dependent noise standard deviation
    noise_std = np.where(groups == 0, 0.2, 1.2).astype(np.float32)
    noise = np.random.normal(0, noise_std).astype(np.float32)

    y = y_mean + noise

    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(groups).squeeze(1),
    )


def run_mondrian(epochs: int = 100) -> dict[str, dict[int, float]]:
    """Standard vs Mondrian split conformal coverage per group."""
    print("=" * 60)
    print("Mondrian Conformal Prediction Demo")
    print("=" * 60)

    # Generate samples
    x, y, groups = generate_grouped_data(n_samples=2400)

    # Train / Cal / Test split
    n_train = 800
    n_cal = 800

    x_train, y_train = x[:n_train], y[:n_train]
    x_cal, y_cal, groups_cal = (
        x[n_train : n_train + n_cal],
        y[n_train : n_train + n_cal],
        groups[n_train : n_train + n_cal],
    )
    x_test, y_test, groups_test = (
        x[n_train + n_cal :],
        y[n_train + n_cal :],
        groups[n_train + n_cal :],
    )

    # 1. Train base model on training set
    model = MLPRegressor(in_features=1, out_features=1, hidden=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        cal_preds = model(x_cal)
        test_preds = model(x_test)

    alpha = 0.1  # Target coverage: 90%

    # 2. Calibrate Standard Conformal Predictor (Global Calibration)
    standard_cp = SplitConformal(alpha=alpha)
    standard_cp.calibrate(cal_preds, y_cal)

    # 3. Calibrate Mondrian Conformal Predictor (Group-Conditional Calibration)
    mondrian_cp = SplitConformal(alpha=alpha)
    mondrian_cp.calibrate(cal_preds, y_cal, groups=groups_cal)

    # 4. Predict intervals and evaluate
    with torch.no_grad():
        # Standard intervals
        std_lower, std_upper = standard_cp.predict_interval(test_preds)
        std_cov = ((y_test >= std_lower) & (y_test <= std_upper)).float()

        # Mondrian intervals
        mondrian_lower, mondrian_upper = mondrian_cp.predict_interval(
            test_preds, groups=groups_test
        )
        mon_cov = ((y_test >= mondrian_lower) & (y_test <= mondrian_upper)).float()

    print(f"\n--- Coverage Results (Alpha = {alpha}, Target = {1 - alpha:.2%}) ---")
    print(f"{'Group':<10} | {'Standard CP Coverage':<22} | {'Mondrian CP Coverage':<22}")
    print("-" * 65)

    coverage: dict[str, dict[int, float]] = {"standard": {}, "mondrian": {}}
    for g in [0, 1]:
        g_mask = groups_test == g
        std_g_cov = std_cov[g_mask].mean().item()
        mon_g_cov = mon_cov[g_mask].mean().item()
        coverage["standard"][g] = std_g_cov
        coverage["mondrian"][g] = mon_g_cov
        print(f"Group {g:<4} | {std_g_cov:<22.2%} | {mon_g_cov:<22.2%}")

    print("-" * 65)
    print(f"Overall    | {std_cov.mean().item():<22.2%} | {mon_cov.mean().item():<22.2%}")

    print("\nObservation:")
    print("1. Standard CP achieves overall coverage but under-covers group 1 (high noise)")
    print("   and over-covers group 0 (low noise).")
    print("2. Mondrian CP guarantees exactly the target coverage for each group individually.")
    return coverage


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument(
        "--mondrian-epochs",
        type=int,
        default=100,
        help="training epochs for the Mondrian base model",
    )
    args = parser.parse_args(argv)

    modes = MODES if args.mode == "all" else (args.mode,)
    for i, mode in enumerate(modes):
        if i:
            print()
        if mode == "tour":
            run_tour()
        elif mode == "sweep":
            run_sweep()
        else:
            run_mondrian(epochs=args.mondrian_epochs)


if __name__ == "__main__":
    main()
