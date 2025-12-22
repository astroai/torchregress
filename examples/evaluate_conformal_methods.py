"""
Evaluation script for comparing conformal prediction methods.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from torchregress.losses.conformal import ConformalLoss
from torchregress.losses.quantile import MultiQuantileLoss


class SimpleModel(nn.Module):
    """A simple neural network for regression."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


def generate_synthetic_data(n_samples: int = 2000, n_features: int = 1) -> tuple:
    """Generate synthetic regression data with heteroscedastic noise."""
    torch.manual_seed(42)
    np.random.seed(42)

    X = torch.randn(n_samples, n_features)
    y_true = torch.sin(X[:, 0])
    noise_std = 0.1 + 0.2 * torch.abs(X[:, 0])
    y = y_true + torch.randn(n_samples) * noise_std

    return X, y.unsqueeze(1), y_true.unsqueeze(1)


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


def main():
    """Run the evaluation."""
    print("Evaluating Conformal Prediction Methods")
    print("=" * 50)

    X, y, _ = generate_synthetic_data()
    n_train = 1000
    n_cal = 500
    X_train, X_cal, X_test = X[:n_train], X[n_train : n_train + n_cal], X[n_train + n_cal :]
    y_train, y_cal, y_test = y[:n_train], y[n_train : n_train + n_cal], y[n_train + n_cal :]

    # Train a model for CQR (outputs two quantiles)
    cqr_model = SimpleModel(in_features=1, out_features=2)
    train_loss_cqr = MultiQuantileLoss(quantiles=[0.05, 0.95])
    cqr_model = train_model(cqr_model, X_train, y_train, train_loss_cqr)

    # Train a model for Split and ACI (outputs a point prediction)
    point_model = SimpleModel(in_features=1, out_features=1)
    train_loss_mse = nn.MSELoss()
    point_model = train_model(point_model, X_train, y_train, train_loss_mse)

    alphas = [0.05, 0.1, 0.2]
    results = []

    for alpha in alphas:
        methods_config = {
            "CQR": {"loss": ConformalLoss(method="cqr", alpha=alpha), "model": cqr_model},
            "ACI": {
                "loss": ConformalLoss(method="aci", alpha=alpha, model=point_model),
                "model": point_model,
            },
            "SPLIT": {"loss": ConformalLoss(method="split", alpha=alpha), "model": point_model},
        }

        for name, config in methods_config.items():
            loss_fn = config["loss"]
            model = config["model"]

            # Generate predictions with the trained model
            with torch.no_grad():
                pred_cal = model(X_cal)
                pred_test = model(X_test)

            # Calibrate
            loss_fn.calibrate(pred_cal, y_cal)

            # Predict intervals
            lower, upper = loss_fn.predict_interval(pred_test)

            # Evaluate
            coverage = ((y_test >= lower) & (y_test <= upper)).float().mean().item()
            width = (upper - lower).mean().item()

            results.append(
                {
                    "Method": name,
                    "Target Coverage": 1 - alpha,
                    "Actual Coverage": coverage,
                    "Interval Width": width,
                }
            )

    # Print results
    results_df = pd.DataFrame(results)
    print(results_df)

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


if __name__ == "__main__":
    main()
