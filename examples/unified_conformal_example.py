"""
Example demonstrating the unified conformal prediction module.
"""

import numpy as np
import torch

from torchregress.losses.conformal import (
    AdaptiveConformalLoss,
    ConformalizedQuantileLoss,
    ConformalLoss,
    MultiDimensionalConformalLoss,
)


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


def demo_cqr() -> None:
    """Demonstrate Conformalized Quantile Regression (CQR)."""
    print("=== Conformalized Quantile Regression (CQR) Demo ===")

    # Generate data
    X, y, y_true = generate_synthetic_data(n_samples=1000, n_features=1)

    # Split into train/calibration/test
    n_train = 600
    n_cal = 200

    _X_train, _X_cal, _X_test = X[:n_train], X[n_train : n_train + n_cal], X[n_train + n_cal :]
    y_train, y_cal, y_test = y[:n_train], y[n_train : n_train + n_cal], y[n_train + n_cal :]
    _y_true_train, _y_true_cal, _y_true_test = (
        y_true[:n_train],
        y_true[n_train : n_train + n_cal],
        y_true[n_train + n_cal :],
    )

    # Create dummy predictions (in practice, these would come from a trained model)
    # CQR expects lower and upper quantiles
    lower_train = y_train - 0.5 + 0.1 * torch.randn_like(y_train)
    upper_train = y_train + 0.5 + 0.1 * torch.randn_like(y_train)
    pred_train = torch.cat([lower_train, upper_train], dim=1)

    lower_cal = y_cal - 0.5 + 0.1 * torch.randn_like(y_cal)
    upper_cal = y_cal + 0.5 + 0.1 * torch.randn_like(y_cal)
    pred_cal = torch.cat([lower_cal, upper_cal], dim=1)

    lower_test = y_test - 0.5 + 0.1 * torch.randn_like(y_test)
    upper_test = y_test + 0.5 + 0.1 * torch.randn_like(y_test)
    pred_test = torch.cat([lower_test, upper_test], dim=1)

    # Create conformal loss
    loss_fn = ConformalizedQuantileLoss(alpha=0.1)

    # "Train" with the loss
    train_loss = loss_fn(pred_train, y_train)
    print(f"Training loss: {train_loss.item():.4f}")

    # Calibrate on calibration set
    loss_fn.calibrate(pred_cal, y_cal)
    print("Calibrated successfully.")

    # Get prediction intervals on test set
    lower_interval, upper_interval = loss_fn.predict_interval(pred_test)

    # Calculate coverage
    coverage = ((y_test >= lower_interval) & (y_test <= upper_interval)).float().mean()
    print(f"Test coverage: {coverage.item():.4f} (target: {1-loss_fn.alpha:.2f})")

    # Calculate interval width
    avg_width = (upper_interval - lower_interval).mean()
    print(f"Average interval width: {avg_width.item():.4f}")
    print()


def demo_adaptive() -> None:
    """Demonstrate Adaptive Conformal Inference (ACI)."""
    print("=== Adaptive Conformal Inference (ACI) Demo ===")

    # Generate data
    X, y, y_true = generate_synthetic_data(n_samples=1000, n_features=1)

    # Split into train/calibration/test
    n_train = 600
    n_cal = 200

    _X_train, _X_cal, _X_test = X[:n_train], X[n_train : n_train + n_cal], X[n_train + n_cal :]
    y_train, y_cal, y_test = y[:n_train], y[n_train : n_train + n_cal], y[n_train + n_cal :]

    # ACI works on point predictions
    pred_train = y_train + 0.1 * torch.randn_like(y_train)
    pred_cal = y_cal + 0.1 * torch.randn_like(y_cal)
    pred_test = y_test + 0.1 * torch.randn_like(y_test)

    # Create conformal loss
    loss_fn = AdaptiveConformalLoss(alpha=0.1)

    # "Train" with the loss
    train_loss = loss_fn(pred_train, y_train)
    print(f"Training loss: {train_loss.item():.4f}")

    # Calibrate on calibration set
    loss_fn.calibrate(pred_cal, y_cal)
    print("Calibrated successfully.")

    # Get prediction intervals on test set
    lower_interval, upper_interval = loss_fn.predict_interval(pred_test)

    # Calculate coverage
    coverage = ((y_test >= lower_interval) & (y_test <= upper_interval)).float().mean()
    print(f"Test coverage: {coverage.item():.4f} (target: {1-loss_fn.alpha:.2f})")

    # Calculate interval width
    avg_width = (upper_interval - lower_interval).mean()
    print(f"Average interval width: {avg_width.item():.4f}")
    print()


def main() -> None:
    """Run all demos."""
    print("Unified Conformal Prediction Examples")
    print("=" * 50)

    demo_cqr()
    demo_adaptive()

    print("All demos completed successfully!")


if __name__ == "__main__":
    main()
