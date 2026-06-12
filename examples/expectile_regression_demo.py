"""
Expectile vs. Quantile Regression under Heteroscedasticity.

This example compares expectile regression and quantile regression under heteroscedastic noise,
motivated by the seminal paper of Newey and Powell (1987).

Unlike quantiles (which minimize asymmetric absolute error), expectiles minimize asymmetric
squared error. Expectiles are easier to optimize (differentiable everywhere) and provide
conditional tail expectations, whereas quantiles estimate conditional quantiles directly.

Seminal paper: Newey, W. K., & Powell, J. L. (1987). Asymmetric Least Squares Estimation and
Testing. Econometrica.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import ExpectileCrossoverLoss, QuantileCrossoverLoss


def generate_heteroscedastic_data(n_samples=1000, seed=42):
    """Generate synthetic heteroscedastic regression data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = np.random.uniform(-2.0, 2.0, size=(n_samples, 1)).astype(np.float32)

    # Nonlinear mean relationship
    mean = 0.5 * (x**3)

    # Noise variance grows with x^2 (heteroscedasticity)
    noise_std = 0.5 + 0.5 * (x**2)
    noise = np.random.normal(0.0, noise_std).astype(np.float32)

    y = mean + noise
    return torch.from_numpy(x), torch.from_numpy(y)


class MultiLevelModel(nn.Module):
    """Model predicting multiple target levels (e.g., 10th, 50th, 90th percentiles)."""

    def __init__(self, input_dim=1, hidden_dim=64, num_levels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_levels),
        )

    def forward(self, x):
        # Output shape: [batch_size, num_levels, 1]
        return self.net(x).unsqueeze(2)


def train_model(loss_fn, x_train, y_train, num_levels, epochs=200):
    model = MultiLevelModel(num_levels=num_levels)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = loss_fn(preds, by)
            loss.backward()
            optimizer.step()

    return model


def main():
    print("=" * 60)
    print("Expectile vs. Quantile Regression (Heteroscedasticity) Demo")
    print("=" * 60)

    x, y = generate_heteroscedastic_data(n_samples=1200)

    # Train / Test split
    n_train = 800
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    levels = [0.1, 0.5, 0.9]

    # Initialize crossover losses to prevent curve crossing
    expectile_loss = ExpectileCrossoverLoss(expectiles=levels, crossover_penalty=10.0)
    quantile_loss = QuantileCrossoverLoss(quantiles=levels, crossover_penalty=10.0)

    print("Training Expectile Regression Model...")
    expectile_model = train_model(expectile_loss, x_train, y_train, len(levels))

    print("Training Quantile Regression Model...")
    quantile_model = train_model(quantile_loss, x_train, y_train, len(levels))

    # Evaluate predictions
    expectile_model.eval()
    quantile_model.eval()

    with torch.no_grad():
        expectile_preds = expectile_model(x_test).squeeze(2)  # [n_test, 3]
        quantile_preds = quantile_model(x_test).squeeze(2)  # [n_test, 3]

        # Median/Mean predictions should be close to center
        # Compute empirical coverages
        print("\nEmpirical Coverages on Test Set:")
        for idx, level in enumerate(levels):
            exp_cov = (y_test.squeeze(1) <= expectile_preds[:, idx]).float().mean().item()
            q_cov = (y_test.squeeze(1) <= quantile_preds[:, idx]).float().mean().item()
            print(
                f"Level {level:.1f} -> Expected: {level:.2f} | "
                f"Quantile Cov: {q_cov:.2f} | Expectile Cov: {exp_cov:.2f}"
            )

    print("\nObservation:")
    print("1. Quantiles map exactly to cumulative probability levels (e.g., 0.1, 0.5, 0.9).")
    print("2. Expectiles generalize the mean using asymmetric least squares and do not map")
    print("   directly to percentiles, but they are fully differentiable and capture dispersion.")


if __name__ == "__main__":
    main()
