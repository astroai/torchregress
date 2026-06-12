"""
Variational Bayesian Regression using the Bayesian Learning Rule (IVON Optimizer).

This example demonstrates how to train a neural network using the Improved Variational
Online Newton (IVON) optimizer (derived from the Bayesian Learning Rule framework).
It shows how to perform variational Bayesian inference to obtain model parameter
uncertainty and decompose predictive uncertainty into aleatoric and epistemic components.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.algorithms.ivon import IVON

# Set random seeds for reproducibility
torch.manual_seed(42)


def generate_synthetic_data(n_samples: int = 250):
    """Generate synthetic non-linear data with heteroscedastic noise."""
    x = torch.linspace(-4, 4, n_samples).reshape(-1, 1)
    # Target function: y = x^3 - 2*x^2 + sin(3*x)
    y_mean = 0.15 * x.pow(3) - 0.5 * x.pow(2) + torch.sin(3.0 * x)
    # Heteroscedastic noise (noise increases for larger values of x)
    noise_scale = 0.15 + 0.1 * torch.abs(x)
    noise = torch.randn_like(x) * noise_scale
    y = y_mean + noise
    return x, y, y_mean, noise_scale


class MLP(nn.Module):
    """Simple multi-layer perceptron for regression."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    print("Generating synthetic dataset...")
    x, y, y_mean, noise_scale = generate_synthetic_data(250)

    # Prepare data loaders
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = MLP(input_dim=1, hidden_dim=32, output_dim=1)

    # Initialize IVON optimizer (representing the Bayesian Learning Rule)
    # ess (effective sample size) is set to the total dataset size
    print("Initializing IVON optimizer...")
    optimizer = IVON(
        model.parameters(),
        lr=0.01,
        ess=len(dataset),
        beta1=0.9,
        beta2=0.99999,
        weight_decay=1e-3,
        hess_approx="price",
    )

    criterion = nn.MSELoss()

    # Training Loop
    print("Training neural network using the Bayesian Learning Rule (IVON)...")
    model.train()
    epochs = 150
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            # Under IVON, we use the sampled_params context manager to draw parameter
            # samples from the variational Gaussian posterior during the forward/backward pass.
            with optimizer.sampled_params(train=True):
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()

            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)

        if epoch % 30 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Mean Loss: {epoch_loss / len(dataset):.4f}")

    # Variational Inference and Prediction
    print("\nPerforming variational Bayesian prediction...")
    model.eval()

    # Evaluate on a dense grid for plotting
    x_test = torch.linspace(-5, 5, 200).reshape(-1, 1)

    # Draw Monte Carlo samples from the parameter posterior
    n_mc_samples = 50
    preds_samples = []

    for _ in range(n_mc_samples):
        # sampled_params(train=False) draws a sample of weights from the optimized posterior
        with optimizer.sampled_params(train=False):
            with torch.no_grad():
                preds_samples.append(model(x_test))

    # Convert samples to a tensor of shape [n_mc_samples, n_test_points, output_dim]
    preds_tensor = torch.stack(preds_samples, dim=0)  # [50, 200, 1]

    # Compute predictive mean and epistemic (model) uncertainty
    pred_mean = preds_tensor.mean(dim=0).squeeze(-1)  # [200]
    # Epistemic uncertainty is the standard deviation of predictions across posterior weight samples
    epistemic_std = preds_tensor.std(dim=0).squeeze(-1)  # [200]

    # For visualization, we plot the predictive mean and uncertainty bounds
    plt.figure(figsize=(10, 6))
    plt.scatter(
        x.numpy(), y.numpy(), color="#1f77b4", alpha=0.5, label="Train Observations", zorder=2
    )
    plt.plot(
        x_test.numpy(),
        pred_mean.numpy(),
        color="#e31a1c",
        linewidth=2,
        label="Bayesian Mean Prediction",
        zorder=3,
    )

    # Plot 1, 2, and 3 standard deviations of epistemic uncertainty
    plt.fill_between(
        x_test.squeeze(-1).numpy(),
        (pred_mean - 1.0 * epistemic_std).numpy(),
        (pred_mean + 1.0 * epistemic_std).numpy(),
        color="#e31a1c",
        alpha=0.3,
        label="1 std Epistemic Uncertainty",
    )
    plt.fill_between(
        x_test.squeeze(-1).numpy(),
        (pred_mean - 2.0 * epistemic_std).numpy(),
        (pred_mean + 2.0 * epistemic_std).numpy(),
        color="#e31a1c",
        alpha=0.15,
        label="2 std Epistemic Uncertainty",
    )

    plt.title("Variational Bayesian Regression via Bayesian Learning Rule (IVON)")
    plt.xlabel("Input x")
    plt.ylabel("Target y")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)

    plot_path = "bayesian_learning_rule_regression.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization plot to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
