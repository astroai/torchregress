r"""
Loss function comparison on noisy data with outliers.

Suites (``--suite``):

``basic``
    A linear model trained with SGD on ``y = 2x + 1`` plus 15% strong outliers.
    Compares MSE, MAE, Huber, Log-Cosh, Barron, Adaptive Robust and the median
    (Quantile 0.5) loss by held-out loss value, and saves ``loss_comparison.png``.

``comprehensive``
    A small MLP trained with Adam on ``y = 2x + 1 + sin(2x)`` plus 10% outliers.
    Compares ten losses (adds Gaussian NLL, Tukey biweight, Charbonnier and
    density-weighted MSE) with point metrics (MSE, MAE, R2), CRPS/ECE for the
    probabilistic head, and runtime, under per-method fixed seeds. Saves
    ``docs/assets/images/examples/comprehensive_loss_comparison.png`` relative to the
    working directory.

``all`` (default)
    Runs ``basic`` then ``comprehensive``. Each suite reseeds its own data generator,
    so its output matches a standalone run.

Barron robust loss:
   $$\rho(x, \alpha, c) = \frac{|\alpha - 2|}{\alpha} \left( \left( \frac{(x/c)^2}{|\alpha - 2|} + 1 \right)^{\alpha/2} - 1 \right)$$
   Seminal paper: Barron, J. T. (2019). A General and Adaptive Robust Loss Function. CVPR.
The adaptive robust loss learns the shape parameter $\alpha$ (and scale $c$) jointly
with the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchregress.calibration import ExpectedCalibrationError
from torchregress.comparison import (
    compute_point_metrics,
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
)
from torchregress.losses import (
    AdaptiveRobustLoss,
    BarronLoss,
    CharbonnierLoss,
    DensityWeightedLoss,
    LogCoshLoss,
    QuantileLoss,
    TukeyBiweightLoss,
    WeightedHuberLoss,
    WeightedL1Loss,
    WeightedMSELoss,
    create_loss_from_config,
)
from torchregress.metrics import ContinuousRankedProbabilityScore

SUITES = ("basic", "comprehensive")

# ============================================================================
# Suite "basic": linear model, SGD
# ============================================================================


class SimpleModel(torch.nn.Module):
    """A simple linear model for demonstration purposes."""

    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def generate_linear_data_with_outliers(n_samples=100, outlier_ratio=0.1, seed=42):
    """Generate synthetic data with outliers."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate x values
    x = torch.linspace(-3, 3, n_samples).reshape(-1, 1)

    # True relationship: y = 2*x + 1 + noise
    y_true = 2 * x + 1

    # Add Gaussian noise
    noise = torch.randn_like(x) * 0.5

    # Add outliers
    n_outliers = int(n_samples * outlier_ratio)
    outlier_idx = torch.randperm(n_samples, device=x.device)[:n_outliers]
    noise[outlier_idx] = torch.randn(n_outliers, 1) * 5  # Strong outliers

    y = y_true + noise

    return x, y, y_true, outlier_idx


def train_and_evaluate(model, loss_fn, x_train, y_train, x_test, y_test, epochs=100):
    """Train a model using the specified loss function and evaluate it."""
    parameters = list(model.parameters())
    parameters.extend(param for param in loss_fn.parameters() if param.requires_grad)
    optimizer = optim.SGD(parameters, lr=0.01)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        test_loss = loss_fn(y_pred, y_test).item()

    return train_losses, test_loss, y_pred


def run_basic() -> dict[str, float]:
    # Generate data with outliers
    x, y, y_true, outlier_idx = generate_linear_data_with_outliers(
        n_samples=100, outlier_ratio=0.15
    )

    # Split into train and test
    train_size = int(0.8 * len(x))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    _y_true_train, _y_true_test = y_true[:train_size], y_true[train_size:]

    # Define loss functions to compare
    loss_functions = {
        "MSE": WeightedMSELoss(reduction="mean"),
        "MAE": WeightedL1Loss(reduction="mean"),
        "Huber": WeightedHuberLoss(reduction="mean"),
        "LogCosh": LogCoshLoss(reduction="mean"),
        "Barron": BarronLoss(alpha=1.0, scale=1.0, reduction="mean"),
        "AdaptiveRobust": AdaptiveRobustLoss(alpha_init=1.0, scale_init=1.0, reduction="mean"),
        "Quantile (0.5)": QuantileLoss(quantile=0.5, reduction="mean"),
    }

    # Train models with different loss functions
    results = {}
    plt.figure(figsize=(18, 12))

    for i, (name, loss_fn) in enumerate(loss_functions.items()):
        model = SimpleModel()
        train_losses, test_loss, y_pred = train_and_evaluate(
            model, loss_fn, x_train, y_train, x_test, y_test, epochs=200
        )

        results[name] = {"train_losses": train_losses, "test_loss": test_loss, "y_pred": y_pred}

        # Plot training curves
        plt.subplot(3, 3, 1)
        plt.plot(train_losses, label=f"{name}")
        plt.title("Training Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot predictions
        plt.subplot(3, 3, i + 2)
        plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.5, label="Train data")
        # Filter outlier indices to only those in training set
        train_outliers = outlier_idx[outlier_idx < train_size]
        if len(train_outliers) > 0:
            plt.scatter(
                x_train[train_outliers].numpy(),
                y_train[train_outliers].numpy(),
                color="red",
                label="Outliers",
            )
        plt.scatter(x_test.numpy(), y_test.numpy(), alpha=0.5, label="Test data")
        plt.plot(x_test.numpy(), y_pred.numpy(), "g-", linewidth=2, label="Prediction")
        plt.plot(x.numpy(), y_true.numpy(), "k--", linewidth=1, label="True relationship")
        plt.title(f"{name} Loss (Test Loss: {test_loss:.4f})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

    plt.tight_layout()
    plt.savefig("loss_comparison.png")
    plt.show()

    # Print numerical comparison
    print("Loss Function Comparison:")
    print("-" * 50)
    print(f"{'Loss Function':<15} {'Test Loss':<15}")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name:<15} {result['test_loss']:<15.6f}")
    return {name: result["test_loss"] for name, result in results.items()}


# ============================================================================
# Suite "comprehensive": MLP, Adam, point + distribution metrics
# ============================================================================


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


def generate_sine_data_with_outliers(n_samples=200, outlier_ratio=0.1, seed=42):
    """Generate synthetic data with outliers."""
    set_comparison_seed(seed)
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
    parameters = list(model.parameters())
    parameters.extend(param for param in loss_fn.parameters() if param.requires_grad)
    optimizer = optim.Adam(parameters, lr=lr)
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()


def run_comprehensive() -> list[dict[str, object]]:
    # --- 1. Data Preparation ---
    x, y, y_true, outlier_idx = generate_sine_data_with_outliers()
    dataset = TensorDataset(x, y)
    batch_size = 32
    epochs = 100
    learning_rate = 0.01

    # --- 2. Model and Loss Definitions ---
    losses_to_compare = {
        "MSE": create_loss_from_config({"type": "mse"}),
        "MAE": create_loss_from_config({"type": "l1"}),
        "Huber": create_loss_from_config({"type": "huber", "delta": 1.0}),
        "LogCosh": create_loss_from_config({"type": "log_cosh"}),
        "Barron": BarronLoss(alpha=1.0, scale=1.0),
        "AdaptiveRobust": AdaptiveRobustLoss(alpha_init=1.0, scale_init=1.0),
        "GaussianNLL": create_loss_from_config({"type": "gaussian_nll"}),
        "TukeyBiweight": TukeyBiweightLoss(c=4.685),
        "Charbonnier": CharbonnierLoss(eps=1e-3),
        "DensityWeightedMSE": DensityWeightedLoss(base_loss="mse"),
    }

    # --- 3. Training and Evaluation ---
    results = {}
    summary_rows = []
    for i, (name, loss_fn) in enumerate(losses_to_compare.items()):
        method_seed = 1000 + i
        set_comparison_seed(method_seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(method_seed),
        )
        print(f"Training with {name}...")
        if name == "GaussianNLL":
            model = SimpleMLP(output_dim=2)  # Predict mean and variance
        else:
            model = SimpleMLP()

        if isinstance(loss_fn, DensityWeightedLoss):
            loss_fn.fit_density(y)

        _, train_seconds = timed_call(
            train_model,
            model,
            loss_fn,
            loader,
            epochs=epochs,
            lr=learning_rate,
        )

        # Evaluation
        def _evaluate():
            model.eval()
            with torch.no_grad():
                raw_pred = model(x)
                if name == "GaussianNLL":
                    mean, log_var = torch.chunk(raw_pred, 2, dim=-1)
                    var = torch.exp(log_var)
                    # Set sampling RNG per method for stable comparison metrics.
                    torch.manual_seed(2025 + i)
                    samples = torch.distributions.Normal(mean, var.sqrt()).sample((100,))
                    y_pred_local = mean
                else:
                    samples = None
                    y_pred_local = raw_pred

                point_metrics = compute_point_metrics(y_pred_local, y)
                mse = point_metrics["MSE"]
                mae = point_metrics["MAE"]
                r2 = point_metrics["R2"]

                crps = None
                ece = None
                if samples is not None:
                    quantiles = {
                        q: torch.quantile(samples, q, dim=0) for q in np.arange(0.1, 1.0, 0.1)
                    }
                    crps = ContinuousRankedProbabilityScore()(quantiles, y).item()
                    ece_quantiles = {
                        q: torch.quantile(samples, q, dim=0) for q in np.arange(0.05, 1.0, 0.05)
                    }
                    ece_metrics = ExpectedCalibrationError()(ece_quantiles, y)
                    ece = ece_metrics["mean_absolute_calibration_error"].item()

                return y_pred_local, {"MSE": mse, "MAE": mae, "R2": r2, "CRPS": crps, "ECE": ece}

        (y_pred, metric_values), eval_seconds = timed_call(_evaluate)

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "metrics": {
                **metric_values,
                "train_s": train_seconds,
                "eval_s": eval_seconds,
            },
        }
        summary_rows.append(
            {
                "Method": name,
                **results[name]["metrics"],
                "Notes": (
                    "mean+variance head"
                    if name == "GaussianNLL"
                    else "jointly learned alpha/scale"
                    if name == "AdaptiveRobust"
                    else ""
                ),
            }
        )

    print_fairness_notes(
        title="Comprehensive Loss Comparison",
        seed_policy="per-method fixed seed; deterministic data generation and Gaussian sampling",
        train_budget=f"{epochs} epochs, batch_size={batch_size}, lr={learning_rate}",
        metric_policy="Common point metrics for all methods; CRPS/ECE only for probabilistic outputs",
    )
    print_comparison_summary("Comparison Summary (All Losses)", summary_rows)

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
    output_path = Path("docs/assets/images/examples/comprehensive_loss_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved comparison figure to {output_path}")
    plt.show()
    return summary_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    args = parser.parse_args(argv)

    suites = SUITES if args.suite == "all" else (args.suite,)
    for i, suite in enumerate(suites):
        if i:
            print()
        if suite == "basic":
            run_basic()
        else:
            run_comprehensive()


if __name__ == "__main__":
    main()
