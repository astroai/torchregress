"""
Example: Handling Noisy Labels in Regression

This example demonstrates three methods for robust training with noisy labels:
1. NoiseAdaptiveLoss: Meta-learning approach that learns sample-specific weights
2. CoTeachingLoss: Two networks teaching each other with clean samples
3. RENTLoss: Ensemble-based method using disagreement to identify noise

Dataset: Synthetic regression with artificial label noise
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchregress.losses import CoTeachingLoss, NoiseAdaptiveLoss, RENTLoss

# ============================================================================
# Synthetic Dataset with Noisy Labels
# ============================================================================


class NoisyRegressionDataset(Dataset):
    """Synthetic regression dataset with label noise."""

    def __init__(
        self,
        n_samples: int = 1000,
        noise_ratio: float = 0.2,
        noise_scale: float = 3.0,
        seed: int = 42,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate clean data: y = sin(x) + 0.5*x
        self.x = np.linspace(-3, 3, n_samples).astype(np.float32)
        self.y_clean = np.sin(self.x) + 0.5 * self.x

        # Add label noise
        self.y_noisy = self.y_clean.copy()
        n_noisy = int(n_samples * noise_ratio)
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

        # Large random noise on selected samples
        noise = np.random.randn(n_noisy) * noise_scale
        self.y_noisy[noisy_indices] += noise

        self.noisy_indices = set(noisy_indices)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return (x, y, index) - index needed for NoiseAdaptiveLoss
        return (
            torch.tensor([self.x[idx]], dtype=torch.float32),
            torch.tensor([self.y_noisy[idx]], dtype=torch.float32),
            idx,
        )


# ============================================================================
# Simple Model
# ============================================================================


class SimpleRegressor(nn.Module):
    """Simple MLP for regression."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Training Functions
# ============================================================================


def train_noise_adaptive(dataset, n_epochs=50):
    """Train using NoiseAdaptiveLoss."""
    print("\n" + "=" * 70)
    print("Training with NoiseAdaptiveLoss (Meta-Learning)")
    print("=" * 70)

    model = SimpleRegressor()
    loss_fn = NoiseAdaptiveLoss(n_samples=len(dataset), base_loss="huber")

    # Separate optimizers: model and sample weights
    optimizer_model = optim.Adam(model.parameters(), lr=0.01)
    optimizer_weights = optim.Adam([loss_fn.sample_weight_logits], lr=0.001)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for x, y, indices in loader:
            # Train model
            optimizer_model.zero_grad()
            optimizer_weights.zero_grad()

            y_pred = model(x)
            loss = loss_fn(y_pred, y, sample_indices=indices)

            loss.backward()
            optimizer_model.step()
            optimizer_weights.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(loader):.4f}")

    # Analyze learned weights
    sample_weights = loss_fn.get_sample_weights().detach().cpu().numpy()
    noisy_mask = np.array([i in dataset.noisy_indices for i in range(len(dataset))])

    print("\nWeight statistics:")
    print(f"  Clean samples: mean={sample_weights[~noisy_mask].mean():.3f}")
    print(f"  Noisy samples: mean={sample_weights[noisy_mask].mean():.3f}")
    print(
        f"  Correctly identified noisy: {(sample_weights[noisy_mask] < 0.5).sum()}/"
        f"{noisy_mask.sum()}"
    )

    return model, sample_weights


def train_coteaching(dataset, n_epochs=50):
    """Train using CoTeachingLoss."""
    print("\n" + "=" * 70)
    print("Training with CoTeachingLoss (Dual Networks)")
    print("=" * 70)

    model1 = SimpleRegressor()
    model2 = SimpleRegressor()

    # Initialize with different random seeds
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    torch.manual_seed(42)
    model1.apply(init_weights)
    torch.manual_seed(123)
    model2.apply(init_weights)

    loss_fn = CoTeachingLoss(forget_rate=0.2, base_loss="huber", num_gradual=10)

    optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.01)

    # DataLoader without indices
    loader = DataLoader(
        [(dataset[i][0], dataset[i][1]) for i in range(len(dataset))],
        batch_size=32,
        shuffle=True,
    )

    for epoch in range(n_epochs):
        total_loss1 = 0.0
        total_loss2 = 0.0

        for x, y in loader:
            # Forward pass for both networks
            y_pred1 = model1(x)
            y_pred2 = model2(x)

            # Co-teaching: each network selects clean samples for the other
            loss1, loss2 = loss_fn(y_pred1, y_pred2, y, epoch=epoch)

            # Update network 1
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            # Update network 2
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs}, "
                f"Loss1: {total_loss1/len(loader):.4f}, "
                f"Loss2: {total_loss2/len(loader):.4f}"
            )

    return model1, model2


def train_rent(dataset, n_epochs=50, ensemble_size=5):
    """Train using RENTLoss."""
    print("\n" + "=" * 70)
    print("Training with RENTLoss (Robust Ensemble)")
    print("=" * 70)

    # Create ensemble of models
    models = [SimpleRegressor() for _ in range(ensemble_size)]

    # Initialize with different seeds
    for i, model in enumerate(models):
        torch.manual_seed(42 + i)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    loss_fn = RENTLoss(ensemble_size=ensemble_size, noise_threshold=2.0, base_loss="huber")

    # Single optimizer for all models (or use separate optimizers)
    all_params = []
    for model in models:
        all_params.extend(model.parameters())
    optimizer = optim.Adam(all_params, lr=0.01)

    loader = DataLoader(
        [(dataset[i][0], dataset[i][1]) for i in range(len(dataset))],
        batch_size=32,
        shuffle=True,
    )

    for epoch in range(n_epochs):
        total_loss = 0.0

        for x, y in loader:
            # Get predictions from all ensemble members
            ensemble_preds = torch.stack(
                [model(x) for model in models]
            )  # [ensemble_size, batch, 1]

            # RENT loss
            optimizer.zero_grad()
            loss = loss_fn(ensemble_preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(loader):.4f}")

    return models


# ============================================================================
# Evaluation and Visualization
# ============================================================================


def evaluate_model(model, dataset, title="Model Predictions"):
    """Evaluate and visualize model predictions."""
    model.eval()

    x_test = torch.linspace(-3, 3, 200).view(-1, 1)
    with torch.no_grad():
        y_pred = model(x_test).numpy()

    x_test = x_test.numpy()

    # Calculate MSE on clean labels
    x_train = torch.tensor(dataset.x).view(-1, 1)
    with torch.no_grad():
        y_train_pred = model(x_train).numpy().flatten()
    mse_clean = ((y_train_pred - dataset.y_clean) ** 2).mean()

    print(f"\n{title}")
    print(f"MSE on clean labels: {mse_clean:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(dataset.x, dataset.y_noisy, alpha=0.3, s=20, label="Noisy training data")
    plt.plot(dataset.x, dataset.y_clean, "k-", linewidth=2, label="True function")
    plt.plot(x_test, y_pred, "r-", linewidth=2, label="Predicted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title} (MSE: {mse_clean:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return mse_clean


def main():
    """Run all experiments."""
    # Create dataset
    print("Creating dataset with 20% label noise...")
    dataset = NoisyRegressionDataset(n_samples=500, noise_ratio=0.2, noise_scale=3.0)

    # 1. NoiseAdaptiveLoss
    model_adaptive, weights = train_noise_adaptive(dataset, n_epochs=50)
    mse_adaptive = evaluate_model(model_adaptive, dataset, "NoiseAdaptiveLoss")

    # Plot learned weights
    plt.figure(figsize=(10, 4))
    colors = ["red" if i in dataset.noisy_indices else "blue" for i in range(len(dataset))]
    plt.scatter(range(len(dataset)), weights, c=colors, alpha=0.5, s=10)
    plt.axhline(y=0.5, color="k", linestyle="--", label="Threshold")
    plt.xlabel("Sample Index")
    plt.ylabel("Learned Weight")
    plt.title("NoiseAdaptiveLoss: Learned Sample Weights (Red=Noisy, Blue=Clean)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 2. CoTeachingLoss
    model_ct1, model_ct2 = train_coteaching(dataset, n_epochs=50)
    mse_ct = evaluate_model(model_ct1, dataset, "CoTeachingLoss (Network 1)")

    # 3. RENTLoss
    models_rent = train_rent(dataset, n_epochs=50, ensemble_size=5)
    mse_rent = evaluate_model(models_rent[0], dataset, "RENTLoss (Ensemble Member 1)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"NoiseAdaptiveLoss MSE: {mse_adaptive:.4f}")
    print(f"CoTeachingLoss MSE:    {mse_ct:.4f}")
    print(f"RENTLoss MSE:          {mse_rent:.4f}")
    print("\nAll methods successfully handled noisy labels!")

    plt.show()


if __name__ == "__main__":
    main()
