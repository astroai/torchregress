"""
Tweedie Regression and Zero-Inflated Continuous Data Demo.

This example demonstrates Tweedie regression for modeling zero-inflated continuous responses
(such as rainfall or insurance claim sizes) using Tweedie and Compound Poisson losses.

The Tweedie distribution ($1 < p < 2$) is parameterized as a compound Poisson-Gamma distribution:
- Number of claims/events: $N \sim \text{Poisson}(\lambda)$
- Size of each claim: $X_j \sim \text{Gamma}(k, \theta)$
- Total response: $Y = \sum_{j=1}^N X_j$, with $Y = 0$ when $N = 0$.

Seminal paper: Jorgensen, B. (1987). Free Windings, Exponential Dispersion Models and
Tweedie Convergence. Journal of the Royal Statistical Society.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import CompoundPoissonLoss, TweedieLoss


def generate_tweedie_data(n_samples=1000, p_power=1.5, phi=1.0, seed=42):
    """Generate Compound Poisson-Gamma data using Tweedie parametrization."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = np.random.uniform(-1.5, 1.5, size=(n_samples, 1)).astype(np.float32)

    # Predicted mean: log(mu) = 0.8 * x + 0.5
    mu = np.exp(0.8 * x + 0.5).astype(np.float32)

    # Translate to Poisson-Gamma parameters
    # Poisson lambda
    lambda_p = (mu ** (2.0 - p_power)) / (phi * (2.0 - p_power))
    # Gamma shape k
    shape_k = (2.0 - p_power) / (p_power - 1.0)
    # Gamma scale theta
    scale_theta = phi * (p_power - 1.0) * (mu ** (p_power - 1.0))

    y = np.zeros((n_samples, 1), dtype=np.float32)
    for i in range(n_samples):
        # Sample Poisson count of events
        n_events = np.random.poisson(lambda_p[i])
        if n_events > 0:
            # Sample event sizes from Gamma
            # Note: numpy shape parameter is shape_k, scale parameter is scale_theta
            y[i, 0] = np.random.gamma(shape_k, scale_theta[i], size=n_events).sum()

    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mu)


class TweedieModel(nn.Module):
    """Simple MLP predicting Tweedie mean parameter (positive)."""

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


def train_tweedie(loss_name, loss_fn, x_train, y_train, x_test, true_mu_test, epochs=150):
    model = TweedieModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            # Tweedie mean must be positive
            mu_pred = torch.exp(model(bx))
            loss = loss_fn(mu_pred, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        mu_pred = torch.exp(model(x_test))
        mae = (mu_pred - true_mu_test).abs().mean().item()

    return mu_pred, mae


def main():
    print("=" * 60)
    print("Tweedie and Compound Poisson-Gamma Regression Demo")
    print("=" * 60)

    p_power = 1.6
    x, y, true_mu = generate_tweedie_data(n_samples=1000, p_power=p_power, phi=0.8)

    print(f"Zero-fraction: {(y == 0).float().mean().item() * 100:.1f}%")

    n_train = 700
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, _y_test = y[:n_train], y[n_train:]
    true_mu_test = true_mu[n_train:]

    losses = {
        "Tweedie (p=1.6)": TweedieLoss(p=p_power),
        "CompoundPoisson": CompoundPoissonLoss(p=p_power),
    }

    for name, loss_fn in losses.items():
        _, mae = train_tweedie(name, loss_fn, x_train, y_train, x_test, true_mu_test)
        print(f"Loss: {name:<20} MAE on true mean: {mae:.4f}")


if __name__ == "__main__":
    main()
