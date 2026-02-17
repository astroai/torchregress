"""
Comprehensive Loss Function Comparison for Regression

This example trains a simple MLP model on a synthetic dataset with outliers using a wide
variety of loss functions from the torchregress library and compares their performance
using point, distribution, and calibration metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import (
    CharbonnierLoss,
    DensityWeightedLoss,
    TukeyBiweightLoss,
    create_loss_from_config,
)
from torchregress.metrics import (
    ContinuousRankedProbabilityScore,
    ExpectedCalibrationError,
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)


class SimpleMLP(nn.Module):
    """A simple MLP for regression."""

    def __init__(self, input_dim=1, output_dim=1, hidden_sizes=[64, 32]):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def generate_data_with_outliers(n_samples=200, outlier_ratio=0.1, seed=42):
    """Generate synthetic data with outliers."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.linspace(-5, 5, n_samples).reshape(-1, 1)
    y_true = 2 * x + 1 + torch.sin(x * 2)
    noise = torch.randn_like(x) * 0.5
    n_outliers = int(n_samples * outlier_ratio)
    outlier_idx = torch.randperm(n_samples, device=x.device)[:n_outliers]
    noise[outlier_idx] = torch.randn(n_outliers, 1) * 5
    y = y_true + noise
    return x, y, y_true, outlier_idx


def train_model(model, loss_fn, train_loader, epochs=100, lr=0.01):
    """Train a model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()


def main():
    # --- 1. Data Preparation ---
    x, y, y_true, outlier_idx = generate_data_with_outliers()
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- 2. Model and Loss Definitions ---
    losses_to_compare = {
        "MSE": create_loss_from_config({"type": "mse"}),
        "MAE": create_loss_from_config({"type": "l1"}),
        "Huber": create_loss_from_config({"type": "huber", "delta": 1.0}),
        "LogCosh": create_loss_from_config({"type": "log_cosh"}),
        "GaussianNLL": create_loss_from_config({"type": "gaussian_nll"}),
        "TukeyBiweight": TukeyBiweightLoss(c=4.685),
        "Charbonnier": CharbonnierLoss(eps=1e-3),
        "DensityWeightedMSE": DensityWeightedLoss(base_loss="mse"),
    }

    # --- 3. Training and Evaluation ---
    results = {}
    for name, loss_fn in losses_to_compare.items():
        print(f"Training with {name}...")
        if name == "GaussianNLL":
            model = SimpleMLP(output_dim=2)  # Predict mean and variance
        else:
            model = SimpleMLP()

        if isinstance(loss_fn, DensityWeightedLoss):
            loss_fn.fit_density(y)

        train_model(model, loss_fn, train_loader)

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(x)
            if name == "GaussianNLL":
                mean, log_var = torch.chunk(y_pred, 2, dim=-1)
                var = torch.exp(log_var)
                samples = torch.distributions.Normal(mean, var.sqrt()).sample((100,))
                y_pred = mean
            else:
                samples = None

            # Point metrics
            mse = MeanSquaredError()(y_pred, y).item()
            mae = MeanAbsoluteError()(y_pred, y).item()
            r2 = R2Score()(y_pred, y).item()

            # Distribution and calibration metrics
            crps = None
            ece = None
            if samples is not None:
                quantiles = {q: torch.quantile(samples, q, dim=0) for q in np.arange(0.1, 1.0, 0.1)}
                crps = ContinuousRankedProbabilityScore()(quantiles, y).item()
                ece_quantiles = {
                    q: torch.quantile(samples, q, dim=0) for q in np.arange(0.05, 1.0, 0.05)
                }
                ece_metrics = ExpectedCalibrationError()(ece_quantiles, y)
                ece = ece_metrics["mean_absolute_calibration_error"].item()

            results[name] = {
                "model": model,
                "y_pred": y_pred,
                "metrics": {"MSE": mse, "MAE": mae, "R2": r2, "CRPS": crps, "ECE": ece},
            }

    # --- 4. Visualization ---
    n_losses = len(losses_to_compare)
    fig, axes = plt.subplots(n_losses, 2, figsize=(12, 5 * n_losses))

    for i, (name, result) in enumerate(results.items()):
        # Scatter plot
        ax = axes[i, 0]
        ax.scatter(x, y, alpha=0.5, label="Data")
        ax.scatter(x[outlier_idx], y[outlier_idx], color="red", label="Outliers")
        ax.plot(x, result["y_pred"], "g-", linewidth=2, label="Prediction")
        ax.plot(x, y_true, "k--", linewidth=1, label="True")
        ax.set_title(f"{name} - Predictions")
        ax.legend()

        # Metrics table
        ax = axes[i, 1]
        metrics = result["metrics"]
        table_data = [[k, f"{v:.4f}"] for k, v in metrics.items() if v is not None]
        table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.axis("off")
        ax.set_title(f"{name} - Metrics")

    plt.tight_layout()
    plt.savefig("comprehensive_loss_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
