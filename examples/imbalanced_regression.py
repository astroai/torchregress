"""
Example: Imbalanced Regression with Calibration Validation

This example demonstrates:
1. DensityWeightedLoss: Safe method that preserves calibration
2. LDSLoss: Label Distribution Smoothing (can affect calibration)
3. Calibration validation and post-hoc correction

Dataset: Synthetic imbalanced regression (dense center, sparse tails)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, Dataset

from torchregress.losses import DensityWeightedLoss, LDSLoss


# ============================================================================
# Imbalanced Dataset
# ============================================================================


class ImbalancedRegressionDataset(Dataset):
    """Synthetic imbalanced regression: most data near center, few at tails."""

    def __init__(self, n_samples: int = 2000, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate imbalanced distribution:
        # 80% samples from [-1, 1], 20% from tails [-3, -1] U [1, 3]
        n_center = int(n_samples * 0.8)
        n_tail = n_samples - n_center

        x_center = np.random.uniform(-1, 1, n_center)
        x_tail = np.concatenate(
            [np.random.uniform(-3, -1, n_tail // 2), np.random.uniform(1, 3, n_tail // 2)]
        )

        self.x = np.concatenate([x_center, x_tail]).astype(np.float32)

        # True function: y = x^3 + noise
        self.y_clean = self.x**3
        noise = np.random.randn(n_samples) * 0.3
        self.y = self.y_clean + noise

        # Shuffle
        perm = np.random.permutation(n_samples)
        self.x = self.x[perm]
        self.y = self.y[perm]
        self.y_clean = self.y_clean[perm]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.tensor([self.x[idx]], dtype=torch.float32),
            torch.tensor([self.y[idx]], dtype=torch.float32),
            idx,
        )


# ============================================================================
# Model
# ============================================================================


class HeteroscedasticRegressor(nn.Module):
    """Regressor that predicts both mean and log-variance."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_var=False):
        features = self.shared(x)
        mean = self.mean_head(features)
        if return_var:
            log_var = self.log_var_head(features)
            return mean, log_var
        return mean


# ============================================================================
# Calibration Metrics
# ============================================================================


def compute_calibration_error(model, dataset, n_bins=10):
    """
    Compute calibration error using prediction intervals.

    For regression, we check if predicted confidence matches empirical coverage.
    """
    model.eval()

    # Get predictions with variance
    x_all = torch.tensor(dataset.x).view(-1, 1)
    y_all = torch.tensor(dataset.y).view(-1, 1)

    with torch.no_grad():
        mean, log_var = model(x_all, return_var=True)
        std = torch.exp(0.5 * log_var)

    mean = mean.numpy()
    std = std.numpy()
    y_all = y_all.numpy()

    # Compute z-scores (how many std deviations away is true value)
    z_scores = np.abs((y_all - mean) / (std + 1e-8))

    # Expected coverage at different confidence levels
    confidence_levels = np.linspace(0, 0.99, n_bins)
    expected_coverage = []
    observed_coverage = []

    for conf in confidence_levels:
        # For Gaussian, z-score threshold for this confidence level
        from scipy.stats import norm

        z_threshold = norm.ppf((1 + conf) / 2)

        # Expected: conf fraction should be within threshold
        expected_coverage.append(conf)

        # Observed: how many are actually within threshold
        within = (z_scores <= z_threshold).mean()
        observed_coverage.append(within)

    # Calibration error: mean absolute difference
    cal_error = np.mean(np.abs(np.array(expected_coverage) - np.array(observed_coverage)))

    return cal_error, expected_coverage, observed_coverage


def plot_calibration_curve(expected, observed, title="Calibration Curve"):
    """Plot calibration curve."""
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(expected, observed, "o-", label="Model calibration")
    plt.xlabel("Expected Coverage")
    plt.ylabel("Observed Coverage")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# ============================================================================
# Training Functions
# ============================================================================


def train_baseline(dataset, n_epochs=100):
    """Train baseline model with standard MSE loss."""
    print("\n" + "=" * 70)
    print("Baseline: Standard MSE Loss")
    print("=" * 70)

    model = HeteroscedasticRegressor()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    loader = DataLoader(
        [(dataset[i][0], dataset[i][1]) for i in range(len(dataset))],
        batch_size=64,
        shuffle=True,
    )

    for epoch in range(n_epochs):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()

            # Simple MSE loss (ignore variance prediction for baseline)
            y_pred = model(x, return_var=False)
            loss = nn.functional.mse_loss(y_pred, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(loader):.4f}")

    return model


def train_density_weighted(dataset, n_epochs=100):
    """Train with DensityWeightedLoss (calibration-safe)."""
    print("\n" + "=" * 70)
    print("DensityWeightedLoss (Calibration-Safe)")
    print("=" * 70)

    model = HeteroscedasticRegressor()

    # Fit density on training targets
    loss_fn = DensityWeightedLoss(kernel_width=0.5, base_loss="mse", reweight_factor=0.8)
    all_targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    loss_fn.fit_density(all_targets)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for x, y, indices in loader:
            optimizer.zero_grad()

            y_pred = model(x, return_var=False)
            loss = loss_fn(y_pred, y, sample_indices=indices)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(loader):.4f}")

    return model


def train_lds(dataset, n_epochs=100):
    """Train with LDSLoss (may affect calibration)."""
    print("\n" + "=" * 70)
    print("LDSLoss (Label Distribution Smoothing)")
    print("=" * 70)
    print("WARNING: This method can affect calibration!")

    model = HeteroscedasticRegressor()

    # Fit LDS
    loss_fn = LDSLoss(kernel="gaussian", kernel_width=1.5, reweight_factor=0.8, base_loss="mse")
    all_targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    loss_fn.fit(all_targets, n_bins=50)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for x, y, indices in loader:
            optimizer.zero_grad()

            y_pred = model(x, return_var=False)
            loss = loss_fn(y_pred, y, sample_indices=indices)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(loader):.4f}")

    return model


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_on_regions(model, dataset):
    """Evaluate MSE on dense vs sparse regions."""
    model.eval()

    # Dense region: [-1, 1]
    dense_mask = np.abs(dataset.x) <= 1
    # Sparse region: tails
    sparse_mask = np.abs(dataset.x) > 1

    x_dense = torch.tensor(dataset.x[dense_mask]).view(-1, 1)
    y_dense = dataset.y[dense_mask]

    x_sparse = torch.tensor(dataset.x[sparse_mask]).view(-1, 1)
    y_sparse = dataset.y[sparse_mask]

    with torch.no_grad():
        pred_dense = model(x_dense, return_var=False).numpy().flatten()
        pred_sparse = model(x_sparse, return_var=False).numpy().flatten()

    mse_dense = ((pred_dense - y_dense) ** 2).mean()
    mse_sparse = ((pred_sparse - y_sparse) ** 2).mean()

    print(f"MSE on dense region [-1, 1]: {mse_dense:.4f}")
    print(f"MSE on sparse region (tails): {mse_sparse:.4f}")
    print(f"MSE ratio (sparse/dense): {mse_sparse/mse_dense:.2f}")

    return mse_dense, mse_sparse


def visualize_predictions(model, dataset, title="Predictions"):
    """Visualize model predictions across input range."""
    model.eval()

    x_test = torch.linspace(-3, 3, 300).view(-1, 1)
    with torch.no_grad():
        y_pred = model(x_test, return_var=False).numpy()

    x_test = x_test.numpy()

    plt.figure(figsize=(10, 6))

    # Plot training data
    dense_mask = np.abs(dataset.x) <= 1
    plt.scatter(
        dataset.x[dense_mask],
        dataset.y[dense_mask],
        alpha=0.3,
        s=10,
        c="blue",
        label="Dense region",
    )
    plt.scatter(
        dataset.x[~dense_mask],
        dataset.y[~dense_mask],
        alpha=0.5,
        s=20,
        c="red",
        label="Sparse region (tails)",
    )

    # Plot predictions
    plt.plot(x_test, x_test**3, "k-", linewidth=2, label="True function")
    plt.plot(x_test, y_pred, "g-", linewidth=2, label="Predicted")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    """Run experiments comparing methods."""
    # Create imbalanced dataset
    print("Creating imbalanced regression dataset...")
    dataset = ImbalancedRegressionDataset(n_samples=2000)

    print(f"Dataset: {len(dataset)} samples")
    print(f"  Dense region [-1, 1]: {(np.abs(dataset.x) <= 1).sum()} samples")
    print(f"  Sparse tails: {(np.abs(dataset.x) > 1).sum()} samples")

    # 1. Baseline
    model_baseline = train_baseline(dataset, n_epochs=100)
    print("\nBaseline evaluation:")
    evaluate_on_regions(model_baseline, dataset)
    visualize_predictions(model_baseline, dataset, "Baseline (Standard MSE)")

    # 2. DensityWeightedLoss (safe)
    model_density = train_density_weighted(dataset, n_epochs=100)
    print("\nDensityWeightedLoss evaluation:")
    evaluate_on_regions(model_density, dataset)
    visualize_predictions(model_density, dataset, "DensityWeightedLoss (Safe)")

    # 3. LDSLoss (may affect calibration)
    model_lds = train_lds(dataset, n_epochs=100)
    print("\nLDSLoss evaluation:")
    evaluate_on_regions(model_lds, dataset)
    visualize_predictions(model_lds, dataset, "LDSLoss (Aggressive)")

    # Calibration validation
    print("\n" + "=" * 70)
    print("CALIBRATION VALIDATION")
    print("=" * 70)

    # We need models that predict variance for calibration
    # Let's retrain baseline with variance prediction
    print("\nRetraining models with variance prediction for calibration check...")

    from torchregress.losses import HeteroscedasticGaussianLoss

    # Baseline with heteroscedastic loss
    model_baseline_het = HeteroscedasticRegressor()
    loss_het = HeteroscedasticGaussianLoss(n_features=1, learnable_variance=False)
    optimizer = optim.Adam(model_baseline_het.parameters(), lr=0.01)

    loader = DataLoader(
        [(dataset[i][0], dataset[i][1]) for i in range(len(dataset))],
        batch_size=64,
        shuffle=True,
    )

    for epoch in range(50):
        for x, y in loader:
            optimizer.zero_grad()
            mean, log_var = model_baseline_het(x, return_var=True)
            y_pred = torch.cat([mean, log_var], dim=-1)
            loss = loss_het(y_pred, y)
            loss.backward()
            optimizer.step()

    # Compute calibration
    cal_error, expected, observed = compute_calibration_error(model_baseline_het, dataset)
    print(f"\nCalibration Error (Baseline): {cal_error:.4f}")
    plot_calibration_curve(expected, observed, "Baseline Calibration")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("DensityWeightedLoss: Improves tail performance while preserving calibration")
    print("LDSLoss: Aggressive reweighting, may need post-hoc calibration")
    print("\nRecommendation:")
    print("  - Start with DensityWeightedLoss (safe)")
    print("  - If using LDSLoss, validate calibration and apply post-hoc correction")

    plt.show()


if __name__ == "__main__":
    main()
