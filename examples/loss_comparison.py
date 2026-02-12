import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim

from torchregress.losses import (
    LogCoshLoss,
    QuantileLoss,
    WeightedLossWrapper,
)


class SimpleModel(torch.nn.Module):
    """A simple linear model for demonstration purposes."""

    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def generate_data_with_outliers(n_samples=100, outlier_ratio=0.1, seed=42):
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
    outlier_idx = torch.randperm(n_samples)[:n_outliers]
    noise[outlier_idx] = torch.randn(n_outliers, 1) * 5  # Strong outliers

    y = y_true + noise

    return x, y, y_true, outlier_idx


def train_and_evaluate(model, loss_fn, x_train, y_train, x_test, y_test, epochs=100):
    """Train a model using the specified loss function and evaluate it."""
    optimizer = optim.SGD(model.parameters(), lr=0.01)

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


def main():
    # Generate data with outliers
    x, y, y_true, outlier_idx = generate_data_with_outliers(n_samples=100, outlier_ratio=0.15)

    # Split into train and test
    train_size = int(0.8 * len(x))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    _y_true_train, _y_true_test = y_true[:train_size], y_true[train_size:]

    # Define loss functions to compare
    loss_functions = {
        "MSE": WeightedLossWrapper(torch.nn.MSELoss, reduction="mean"),
        "MAE": WeightedLossWrapper(torch.nn.L1Loss, reduction="mean"),
        "Huber": WeightedLossWrapper(torch.nn.HuberLoss, reduction="mean"),
        "LogCosh": LogCoshLoss(reduction="mean"),
        "Quantile (0.5)": QuantileLoss(quantile=0.5, reduction="mean"),
    }

    # Train models with different loss functions
    results = {}
    plt.figure(figsize=(15, 10))

    for i, (name, loss_fn) in enumerate(loss_functions.items()):
        model = SimpleModel()
        train_losses, test_loss, y_pred = train_and_evaluate(
            model, loss_fn, x_train, y_train, x_test, y_test, epochs=200
        )

        results[name] = {"train_losses": train_losses, "test_loss": test_loss, "y_pred": y_pred}

        # Plot training curves
        plt.subplot(2, 3, 1)
        plt.plot(train_losses, label=f"{name}")
        plt.title("Training Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot predictions
        plt.subplot(2, 3, i + 2)
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


if __name__ == "__main__":
    main()
