"""
torchregress basic usage example.

This example demonstrates how to use various regression loss functions
from the torchregress library in a simple neural network training scenario.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import (
    GaussianNLLLoss,  # For uncertainty estimation
    QuantileLoss,  # For quantile regression
    WeightedLossWrapper,  # Wrapper for standard losses
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# 1. Create a synthetic dataset with heteroscedastic noise
def generate_dataset(n_samples=1000):
    """Generate a synthetic regression dataset with heteroscedastic noise."""
    np.random.seed(42)
    x = np.random.uniform(-3, 3, size=(n_samples, 1)).astype(np.float32)

    # True function: y = x^2 + sin(x)
    y_true = x**2 + np.sin(x * 3)

    # Heteroscedastic noise (noise increases with x)
    noise_scale = 0.1 + 0.1 * np.abs(x)
    y = y_true + np.random.normal(0, noise_scale, size=y_true.shape)

    # Convert to PyTorch tensors
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    return x_tensor, y_tensor, x, y, y_true, noise_scale


# 2. Define a simple neural network model
class RegressionModel(nn.Module):
    """Simple MLP for regression."""

    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


# 3. Define a model with uncertainty estimation
class UncertaintyModel(nn.Module):
    """Regression model with aleatoric uncertainty estimation."""

    def __init__(self, input_dim=1, hidden_dim=50):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Mean prediction head
        self.mean_head = nn.Linear(hidden_dim, 1)

        # Log variance prediction head (for aleatoric uncertainty)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)

        # Ensure variance is positive and stable
        var = torch.exp(logvar)

        return mean, var


def train_model(
    model, loss_fn, x_train, y_train, epochs=200, lr=0.01, uncertainty_model=False, verbose=True
):
    """Train a model using the specified loss function."""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create data loader
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()

            if uncertainty_model:
                mean, var = model(x_batch)
                # GaussianNLLLoss expects y_pred as (mean, log_var) tuple
                # var is already in variance form, so convert to log_var
                log_var = torch.log(var)
                loss = loss_fn((mean, log_var), y_batch)
            else:
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses


def main():
    # Generate data
    x_train, y_train, x_np, y_np, y_true, noise_scale = generate_dataset()

    # Plot the dataset
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.scatter(x_np, y_np, alpha=0.5, label="Noisy data")
    plt.plot(x_np, y_true, "r-", label="True function")
    plt.title("Synthetic Dataset")
    plt.legend()

    # 4. Train with MSE loss (basic)
    model_mse = RegressionModel()
    mse_loss = WeightedLossWrapper(nn.MSELoss)
    mse_losses = train_model(model_mse, mse_loss, x_train, y_train, verbose=False)

    # 5. Train with Huber loss (robust)
    model_huber = RegressionModel()
    huber_loss = WeightedLossWrapper(nn.HuberLoss, delta=1.0)
    huber_losses = train_model(model_huber, huber_loss, x_train, y_train, verbose=False)

    # 6. Train with Quantile loss (asymmetric)
    model_quantile = RegressionModel()
    quantile_loss = QuantileLoss(quantile=0.5)  # Median regression
    quantile_losses = train_model(model_quantile, quantile_loss, x_train, y_train, verbose=False)

    # 7. Train with Heteroscedastic Gaussian loss (uncertainty aware)
    model_uncertainty = UncertaintyModel()
    # Model outputs (mean, log_var), so use learnable_variance=False to accept model's variance
    gaussian_nll = GaussianNLLLoss()
    gnll_losses = train_model(
        model_uncertainty, gaussian_nll, x_train, y_train, uncertainty_model=True, verbose=False
    )

    # Plot loss curves
    plt.subplot(2, 2, 2)
    plt.plot(mse_losses, label="MSE Loss")
    plt.plot(huber_losses, label="Huber Loss")
    plt.plot(quantile_losses, label="Quantile Loss")
    plt.plot(gnll_losses, label="Gaussian NLL")
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 8. Compare predictions
    x_sorted, indices = torch.sort(x_train, 0)

    # MSE model predictions
    with torch.no_grad():
        y_pred_mse = model_mse(x_sorted)

    # Huber model predictions
    with torch.no_grad():
        y_pred_huber = model_huber(x_sorted)

    # Quantile model predictions
    with torch.no_grad():
        y_pred_quantile = model_quantile(x_sorted)

    # Uncertainty model predictions
    with torch.no_grad():
        y_pred_mean, y_pred_var = model_uncertainty(x_sorted)
        y_pred_std = torch.sqrt(y_pred_var)
        lower_bound = y_pred_mean - 2 * y_pred_std
        upper_bound = y_pred_mean + 2 * y_pred_std

    # Plot model predictions
    plt.subplot(2, 2, 3)
    plt.scatter(x_np, y_np, alpha=0.2, label="Data")
    plt.plot(x_sorted.numpy(), y_pred_mse.numpy(), "r-", label="MSE")
    plt.plot(x_sorted.numpy(), y_pred_huber.numpy(), "g-", label="Huber")
    plt.plot(x_sorted.numpy(), y_pred_quantile.numpy(), "b-", label="Quantile")
    plt.title("Model Predictions")
    plt.legend()

    # Plot uncertainty estimation
    plt.subplot(2, 2, 4)
    plt.scatter(x_np, y_np, alpha=0.2, label="Data")
    plt.plot(x_sorted.numpy(), y_pred_mean.numpy(), "r-", label="Mean prediction")
    plt.fill_between(
        x_sorted.numpy().flatten(),
        lower_bound.numpy().flatten(),
        upper_bound.numpy().flatten(),
        alpha=0.3,
        color="r",
        label="95% prediction interval",
    )
    plt.title("Uncertainty Estimation with Gaussian NLL")
    plt.legend()

    plt.tight_layout()
    plt.savefig("regression_comparison.png")

    print("Example completed! Results saved to regression_comparison.png")


if __name__ == "__main__":
    main()
