"""
Poisson-Gaussian Mixture Loss for Shot and Readout Noise Demo.

This example demonstrates how to train models using Poisson-Gaussian mixture loss functions,
which are particularly useful in signal processing and scientific imaging (e.g., microscopy,
low-light photography) where observations suffer from both discrete Poisson counting (shot)
noise and continuous Gaussian electronic (readout) noise.

Supported losses:
1. PoissonGaussianMixtureLoss: Models y ~ Poisson(λ) + N(0, σ²).
2. EnhancedPoissonGaussianMixtureLoss: Models y ~ Poisson(g * λ + b) + N(0, σ_1² + σ_2² * λ)
   where g is gain, b is offset, σ_1² is read noise, and σ_2² is shot noise.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import (
    EnhancedPoissonGaussianMixtureLoss,
    PoissonGaussianMixtureLoss,
)


def generate_poisson_gaussian_data(n_samples=1000, sigma=0.5, seed=42):
    """Generate synthetic data contaminated with both Poisson and Gaussian noise."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = np.random.uniform(-1.5, 1.5, size=(n_samples, 1)).astype(np.float32)

    # True latent rate (lambda)
    true_rate = np.exp(0.8 * x + 1.2).astype(np.float32)

    # Shot noise (Poisson process)
    shot = np.random.poisson(true_rate).astype(np.float32)

    # Readout noise (Gaussian process)
    readout = np.random.normal(0, sigma, size=shot.shape).astype(np.float32)

    # Total observed signal
    y = shot + readout

    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(true_rate),
    )


class SignalModel(nn.Module):
    """Neural network predicting the latent rate (positive)."""

    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate(loss_name, loss_fn, x_train, y_train, x_test, true_rate_test, epochs=150):
    model = SignalModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            # Predict mean rate, which must be positive
            rate_pred = torch.exp(model(bx))
            loss = loss_fn(rate_pred, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        rate_pred = torch.exp(model(x_test))
        mae = (rate_pred - true_rate_test).abs().mean().item()

    return rate_pred, mae


def main():
    print("=" * 60)
    print("Poisson-Gaussian Mixture Regression Demo")
    print("=" * 60)

    sigma = 0.6
    x, y, true_rate = generate_poisson_gaussian_data(n_samples=1200, sigma=sigma)

    # Train / Test split
    n_train = 800
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, _y_test = y[:n_train], y[n_train:]
    true_rate_test = true_rate[n_train:]

    losses = {
        "PG-Mixture (Fixed Var)": PoissonGaussianMixtureLoss(
            initial_variance=sigma**2, log_input=False
        ),
        "PG-Mixture (Learn Var)": PoissonGaussianMixtureLoss(
            learn_variance=True, initial_variance=1.0, log_input=False
        ),
        "PG-Enhanced (Learn Gain)": EnhancedPoissonGaussianMixtureLoss(
            gain="learn", read_noise=sigma**2, shot_noise=0.1, log_input=False
        ),
    }

    results = {}
    for name, loss_fn in losses.items():
        rate_pred, mae = train_and_evaluate(name, loss_fn, x_train, y_train, x_test, true_rate_test)
        results[name] = {"pred": rate_pred, "mae": mae}
        print(f"Loss: {name:<25} MAE on true rate: {mae:.4f}")

    print("\nObservation:")
    print("1. Poisson-Gaussian mixture loss models both counting uncertainty and readout noise.")
    print(
        "2. Enhanced formulation allows learning additional physical parameters like camera gain."
    )


if __name__ == "__main__":
    main()
