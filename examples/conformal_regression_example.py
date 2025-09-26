"""
Example demonstrating conformal prediction methods for regression.
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
    y_true = torch.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    noise_std = 0.1 + 0.2 * torch.abs(X[:, 0])  # Heteroscedastic noise
    y = y_true + torch.randn(n_samples) * noise_std

    return X, y.unsqueeze(1), y_true.unsqueeze(1)


def demo_basic_conformal() -> None:
    """Demonstrate basic ConformalLoss."""
    print("=== Basic Conformal Prediction Demo ===")

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
    # For demo, we'll use noisy versions of true values
    lower_train = y_train - 0.5 + 0.1 * torch.randn_like(y_train)
    upper_train = y_train + 0.5 + 0.1 * torch.randn_like(y_train)

    lower_cal = y_cal - 0.5 + 0.1 * torch.randn_like(y_cal)
    upper_cal = y_cal + 0.5 + 0.1 * torch.randn_like(y_cal)

    lower_test = y_test - 0.5 + 0.1 * torch.randn_like(y_test)
    upper_test = y_test + 0.5 + 0.1 * torch.randn_like(y_test)

    # Create conformal loss
    loss_fn = ConformalLoss(alpha=0.1)

    # Train with the loss (in practice, this would be part of model training)
    train_loss = loss_fn(torch.cat([lower_train, upper_train], dim=1), y_train)
    print(f"Training loss: {train_loss.item():.4f}")

    # Calibrate on calibration set
    tau = loss_fn.calibrate(lower_cal, upper_cal, y_cal)
    print(f"Calibration tau: {tau.item():.4f}")

    # Get prediction intervals on test set
    lower_interval, upper_interval = loss_fn.predict_interval(lower_test, upper_test)

    # Calculate coverage
    coverage = ((y_test >= lower_interval) & (y_test <= upper_interval)).float().mean()
    print(f"Test coverage: {coverage.item():.4f} (target: {1-loss_fn.alpha:.2f})")

    # Calculate interval width
    avg_width = (upper_interval - lower_interval).mean()
    print(f"Average interval width: {avg_width.item():.4f}")
    print()


def demo_adaptive_conformal() -> None:
    """Demonstrate AdaptiveConformalLoss."""
    print("=== Adaptive Conformal Prediction Demo ===")

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
    # For demo, we'll use noisy versions of true values
    lower_train = y_train - 0.5 + 0.1 * torch.randn_like(y_train)
    upper_train = y_train + 0.5 + 0.1 * torch.randn_like(y_train)

    lower_cal = y_cal - 0.5 + 0.1 * torch.randn_like(y_cal)
    upper_cal = y_cal + 0.5 + 0.1 * torch.randn_like(y_cal)

    lower_test = y_test - 0.5 + 0.1 * torch.randn_like(y_test)
    upper_test = y_test + 0.5 + 0.1 * torch.randn_like(y_test)

    # Create adaptive conformal loss
    loss_fn = AdaptiveConformalLoss(alpha=0.1, gamma=0.01)

    # Train with the loss (in practice, this would be part of model training)
    train_loss = loss_fn(torch.cat([lower_train, upper_train], dim=1), y_train)
    print(f"Training loss: {train_loss.item():.4f}")

    # Calibrate on calibration set
    tau = loss_fn.calibrate(lower_cal, upper_cal, y_cal)
    print(f"Calibration tau: {tau.item():.4f}")
    print(f"Adapted alpha: {loss_fn.alpha:.4f}")

    # Get prediction intervals on test set
    lower_interval, upper_interval = loss_fn.predict_interval(lower_test, upper_test)

    # Calculate coverage
    coverage = ((y_test >= lower_interval) & (y_test <= upper_interval)).float().mean()
    print(f"Test coverage: {coverage.item():.4f} (target: {1-loss_fn.alpha:.2f})")

    # Calculate interval width
    avg_width = (upper_interval - lower_interval).mean()
    print(f"Average interval width: {avg_width.item():.4f}")
    print()


def demo_conformalized_quantile() -> None:
    """Demonstrate ConformalizedQuantileLoss."""
    print("=== Conformalized Quantile Regression Demo ===")

    # Generate data
    X, y, y_true = generate_synthetic_data(n_samples=1000, n_features=1)

    # Split data
    n_train = 600
    n_cal = 200

    _X_train, _X_cal, _X_test = X[:n_train], X[n_train : n_train + n_cal], X[n_train + n_cal :]
    y_train, y_cal, y_test = y[:n_train], y[n_train : n_train + n_cal], y[n_train + n_cal :]
    _y_true_train, _y_true_cal, _y_true_test = (
        y_true[:n_train],
        y_true[n_train : n_train + n_cal],
        y_true[n_train + n_cal :],
    )

    # Create dummy quantile predictions
    quantiles = [0.05, 0.5, 0.95]
    len(quantiles)

    # For demo, create predictions for each quantile
    quantile_preds_train = torch.stack(
        [
            y_train - 1.0 + 0.2 * torch.randn_like(y_train),  # 0.05 quantile
            y_train + 0.1 * torch.randn_like(y_train),  # 0.5 quantile (median)
            y_train + 1.0 + 0.2 * torch.randn_like(y_train),  # 0.95 quantile
        ],
        dim=-1,
    ).squeeze(
        -2
    )  # Shape: [batch, n_quantiles]

    quantile_preds_cal = torch.stack(
        [
            y_cal - 1.0 + 0.2 * torch.randn_like(y_cal),
            y_cal + 0.1 * torch.randn_like(y_cal),
            y_cal + 1.0 + 0.2 * torch.randn_like(y_cal),
        ],
        dim=-1,
    ).squeeze(-2)

    quantile_preds_test = torch.stack(
        [
            y_test - 1.0 + 0.2 * torch.randn_like(y_test),
            y_test + 0.1 * torch.randn_like(y_test),
            y_test + 1.0 + 0.2 * torch.randn_like(y_test),
        ],
        dim=-1,
    ).squeeze(-2)

    # Create conformalized quantile loss
    loss_fn = ConformalizedQuantileLoss(quantiles=(0.05, 0.95), alpha=0.1)

    # Train with the loss
    train_loss = loss_fn(quantile_preds_train, y_train)
    print(f"Training loss: {train_loss.item():.4f}")

    # Calibrate on calibration set
    tau = loss_fn.calibrate(quantile_preds_cal, y_cal)
    print(f"Calibration tau: {tau.item():.4f}")

    # Get calibrated prediction intervals
    lower_interval, upper_interval = loss_fn.predict_interval(quantile_preds_test)

    # Calculate coverage
    coverage = ((y_test >= lower_interval) & (y_test <= upper_interval)).float().mean()
    print(f"Test coverage: {coverage.item():.4f} (target: {1-loss_fn.alpha:.2f})")

    # Calculate interval width
    avg_width = (upper_interval - lower_interval).mean()
    print(f"Average interval width: {avg_width.item():.4f}")
    print()


def demo_multidimensional_conformal() -> None:
    """Demonstrate MultiDimensionalConformalLoss."""
    print("=== Multi-Dimensional Conformal Prediction Demo ===")

    # Generate multi-output data
    X, y, y_true = generate_synthetic_data(n_samples=1000, n_features=3)

    # Split data
    n_train = 600
    n_cal = 200

    _X_train, _X_cal, _X_test = X[:n_train], X[n_train : n_train + n_cal], X[n_train + n_cal :]
    y_train, y_cal, y_test = y[:n_train], y[n_train : n_train + n_cal], y[n_train + n_cal :]
    _y_true_train, _y_true_cal, _y_true_test = (
        y_true[:n_train],
        y_true[n_train : n_train + n_cal],
        y_true[n_train + n_cal :],
    )

    # Create dummy predictions
    y_train - 0.5 + 0.1 * torch.randn_like(y_train)
    y_train + 0.5 + 0.1 * torch.randn_like(y_train)

    lower_cal = y_cal - 0.5 + 0.1 * torch.randn_like(y_cal)
    upper_cal = y_cal + 0.5 + 0.1 * torch.randn_like(y_cal)

    lower_test = y_test - 0.5 + 0.1 * torch.randn_like(y_test)
    upper_test = y_test + 0.5 + 0.1 * torch.randn_like(y_test)

    # Create multi-dimensional conformal loss
    loss_fn = MultiDimensionalConformalLoss(alpha=0.1)

    # Train with the loss
    train_loss = loss_fn(y_train, y_train)  # Dummy training
    print(f"Training loss: {train_loss.item():.4f}")

    # Calibrate on calibration set
    taus = loss_fn.calibrate(lower_cal, upper_cal, y_cal)
    print(f"Calibration taus: {taus}")

    # Get prediction intervals
    lower_intervals, upper_intervals = loss_fn.predict_intervals(lower_test, upper_test)

    # Calculate coverage per dimension
    for i in range(y_test.shape[1]):
        coverage = (
            ((y_test[:, i] >= lower_intervals[:, i]) & (y_test[:, i] <= upper_intervals[:, i]))
            .float()
            .mean()
        )
        print(f"Dimension {i} coverage: {coverage.item():.4f} (target: {1-loss_fn.alpha:.2f})")

    # Calculate average interval widths per dimension
    avg_widths = (upper_intervals - lower_intervals).mean(dim=0)
    print(f"Average interval widths: {avg_widths}")
    print()


def main() -> None:
    """Run all demos."""
    print("Conformal Prediction Examples")
    print("=" * 50)

    demo_basic_conformal()
    demo_adaptive_conformal()
    demo_conformalized_quantile()
    demo_multidimensional_conformal()

    print("All demos completed successfully!")


if __name__ == "__main__":
    main()
