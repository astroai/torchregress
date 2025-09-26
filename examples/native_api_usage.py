"""
Examples demonstrating the PyTorch-native API usage patterns for torchregress.
"""

import torch
import torch.nn as nn

from torchregress.losses.gaussian import DiagonalGaussianNLL
from torchregress.losses.robust import HuberLoss


# Example 1: Standard regression model with MSE loss
def example_standard_regression():
    # Create a simple model manually - PyTorch native
    model = nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
    )

    # Use standard PyTorch loss
    loss_fn = nn.MSELoss()

    # Example usage
    x = torch.randn(32, 10)  # 32 samples, 10 features
    y = torch.randn(32, 1)  # 32 targets

    # Forward pass
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)
    print(f"Standard regression loss: {loss.item()}")

    return model, loss_fn


# Example 2: Heteroscedastic regression with Gaussian NLL
def example_heteroscedastic_regression():
    # Create model with 2*output_dim outputs (mean, log_var)
    class HeteroscedasticModel(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
            )
            self.mean_head = nn.Linear(64, out_features)
            self.logvar_head = nn.Linear(64, out_features)

        def forward(self, x):
            features = self.net(x)
            mean = self.mean_head(features)
            logvar = self.logvar_head(features)
            return mean, logvar

    # Create model and loss
    model = HeteroscedasticModel(in_features=10, out_features=1)
    loss_fn = DiagonalGaussianNLL(learnable_variance=False)

    # Example usage
    x = torch.randn(32, 10)  # 32 samples, 10 features
    y = torch.randn(32, 1)  # 32 targets

    # Forward pass
    mean, logvar = model(x)

    # Compute loss - directly pass the tuple of (mean, logvar)
    loss = loss_fn((mean, logvar), y)
    print(f"Heteroscedastic regression loss: {loss.item()}")

    return model, loss_fn


# Example 3: Robust regression with Huber loss
def example_robust_regression():
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
    )

    # Use Huber loss for robustness
    loss_fn = HuberLoss(delta=1.0)

    # Example usage
    x = torch.randn(32, 10)  # 32 samples, 10 features
    y = torch.randn(32, 1)  # 32 targets

    # Add some outliers
    y[0] = 100.0  # Add an outlier

    # Forward pass
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)
    print(f"Robust regression loss: {loss.item()}")

    # Compare with MSE
    mse_loss = nn.MSELoss()(y_pred, y)
    print(f"MSE on same data: {mse_loss.item()}")

    return model, loss_fn


if __name__ == "__main__":
    print("=== Example 1: Standard Regression ===")
    example_standard_regression()

    print("\n=== Example 2: Heteroscedastic Regression ===")
    example_heteroscedastic_regression()

    print("\n=== Example 3: Robust Regression ===")
    example_robust_regression()
