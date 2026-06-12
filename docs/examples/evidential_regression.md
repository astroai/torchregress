# Evidential Regression Example

This example demonstrates how to use evidential regression to decompose uncertainty into aleatoric and epistemic components.

| # | Reference |
|:-:|:----------|
| 1 | Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2020). **Deep Evidential Regression**. *Advances in Neural Information Processing Systems (NeurIPS)*. |

---

## Evidential Formulation

A neural network models the parameters $(\gamma, \nu, \alpha, \beta)$ of a Normal-Inverse-Gamma (NIG) distribution over the mean $\mu$ and variance $\sigma^2$ of the target:

$$\mu \sim \mathcal{N}(\gamma, \sigma^2 / \nu), \quad \sigma^2 \sim \text{Inv-Gamma}(\alpha, \beta)$$

*   **Predicted Mean**:
    $$\hat{y} = \gamma$$
*   **Aleatoric Uncertainty** (expected data noise):
    $$\text{Var}_{\text{aleatoric}} = \frac{\beta}{\alpha - 1}$$
*   **Epistemic Uncertainty** (model parameter uncertainty):
    $$\text{Var}_{\text{epistemic}} = \frac{\beta}{\nu(\alpha - 1)}$$
*   **Loss Function**:
    The loss function combines a Maximum Likelihood estimation of the NIG parameters with a regularization term scaled by `coeff_nig` to penalize overconfident predictions in regions with low evidence (low $\nu$):
    $$L = L_{\text{NLL}} + \lambda L_{\text{reg}}$$

---

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import EvidentialRegressionLoss

# ============================================================================
# Evidential Model
# ============================================================================


class EvidentialMLP(nn.Module):
    """
    MLP that outputs 4 NIG parameters: gamma, nu, alpha, beta.

    The output must satisfy:
    - nu > 0 (evidence for mean)
    - alpha > 1 (shape parameter)
    - beta > 0 (scale parameter)
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4 * output_dim),  # [gamma, nu, alpha, beta]
        )

    def forward(self, x):
        """
        Forward pass with activation functions to ensure constraints.

        Returns:
            Tensor of shape [batch_size, 4*output_dim]
            Format: [gamma, nu, alpha, beta] concatenated
        """
        out = self.net(x)

        # Split into 4 parts
        gamma = out[:, : self.output_dim]  # mean (no constraint)
        nu_raw = out[:, self.output_dim : 2 * self.output_dim]
        alpha_raw = out[:, 2 * self.output_dim : 3 * self.output_dim]
        beta_raw = out[:, 3 * self.output_dim :]

        # Apply constraints using softplus
        nu = F.softplus(nu_raw) + 0.01  # nu > 0
        alpha = F.softplus(alpha_raw) + 1.01  # alpha > 1
        beta = F.softplus(beta_raw) + 0.01  # beta > 0

        # Concatenate
        return torch.cat([gamma, nu, alpha, beta], dim=1)


# ============================================================================
# Dataset with Gaps (for OOD behavior)
# ============================================================================


def create_dataset_with_gaps(n_samples: int = 500):
    """
    Create 1D regression dataset with gaps to demonstrate OOD detection.

    Function: y = sin(2πx) + 0.3*x
    Data: x in [-1, -0.2] U [0.2, 1] (gap in middle)
    """
    # Training data: two regions with a gap in the middle
    n_left = n_samples // 2
    n_right = n_samples - n_left

    x_left = np.random.uniform(-1.0, -0.2, n_left)
    x_right = np.random.uniform(0.2, 1.0, n_right)

    x_train = np.concatenate([x_left, x_right]).astype(np.float32)

    # True function
    y_clean = np.sin(2 * np.pi * x_train) + 0.3 * x_train

    # Add heteroscedastic noise (increases with |x|)
    noise_std = 0.1 + 0.1 * np.abs(x_train)
    y_train = y_clean + np.random.randn(n_samples) * noise_std

    # Test data: full range including gap
    x_test = np.linspace(-1.2, 1.2, 500).astype(np.float32)
    y_test_clean = np.sin(2 * np.pi * x_test) + 0.3 * x_test

    return (
        torch.tensor(x_train).view(-1, 1),
        torch.tensor(y_train).view(-1, 1),
        torch.tensor(x_test).view(-1, 1),
        torch.tensor(y_test_clean).view(-1, 1),
    )


# ============================================================================
# Training
# ============================================================================


def train_evidential_model(x_train, y_train, n_epochs=500, coeff_nig=0.01):
    """Train evidential regression model."""
    print("\n" + "=" * 70)
    print("Training Evidential Regression Model")
    print("=" * 70)
    print(f"Regularization coefficient: {coeff_nig}")

    model = EvidentialMLP(input_dim=1, hidden_dim=64, output_dim=1)
    loss_fn = EvidentialRegressionLoss(coeff_nig=coeff_nig)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Create dataloader
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(batch_x)

            # Compute loss
            loss = loss_fn(y_pred, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return model, loss_fn


# ============================================================================
# Evaluation and Visualization
# ============================================================================


def evaluate_and_visualize(model, loss_fn, x_train, y_train, x_test, y_test_clean):
    """Evaluate model and visualize predictions with uncertainty."""
    model.eval()

    with torch.no_grad():
        # Get predictions and uncertainties on test set
        params = model(x_test)
        mean, aleatoric, epistemic = loss_fn.predict_with_uncertainty(params)

    # Convert to numpy
    x_test_np = x_test.numpy().flatten()
    mean_np = mean.numpy().flatten()
    ale_np = aleatoric.numpy().flatten()
    epi_np = epistemic.numpy().flatten()
    y_test_np = y_test_clean.numpy().flatten()

    total_unc = ale_np + epi_np
    std_total = np.sqrt(total_unc)

    # Calculate MSE on training regions
    in_region_mask = (x_test_np < -0.2) | (x_test_np > 0.2)
    mse_in = ((mean_np[in_region_mask] - y_test_np[in_region_mask]) ** 2).mean()

    # Calculate MSE in gap (OOD region)
    ood_mask = (x_test_np >= -0.2) & (x_test_np <= 0.2)
    mse_ood = ((mean_np[ood_mask] - y_test_np[ood_mask]) ** 2).mean()

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"MSE on training regions: {mse_in:.4f}")
    print(f"MSE in gap (OOD region): {mse_ood:.4f}")
    print(f"Average aleatoric uncertainty: {ale_np.mean():.4f}")
    print(f"Average epistemic uncertainty: {epi_np.mean():.4f}")
    print(
        f"Epistemic in gap vs training: "
        f"{epi_np[ood_mask].mean():.4f} vs {epi_np[in_region_mask].mean():.4f}"
    )

    # ========================================================================
    # Plot 1: Predictions with Total Uncertainty
    # ========================================================================
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    # Plot training data
    x_train_np = x_train.numpy().flatten()
    y_train_np = y_train.numpy().flatten()
    plt.scatter(x_train_np, y_train_np, alpha=0.3, s=10, label="Training data")

    # Plot true function
    plt.plot(x_test_np, y_test_np, "k-", linewidth=2, label="True function")

    # Plot predictions
    plt.plot(x_test_np, mean_np, "r-", linewidth=2, label="Prediction")

    # Plot uncertainty bands (95% CI)
    plt.fill_between(
        x_test_np,
        mean_np - 1.96 * std_total,
        mean_np + 1.96 * std_total,
        alpha=0.3,
        color="red",
        label="95% CI (total)",
    )

    # Highlight gap region
    plt.axvspan(-0.2, 0.2, alpha=0.1, color="yellow", label="OOD region (gap)")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Evidential Regression: Predictions with Total Uncertainty")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 2: Uncertainty Decomposition
    # ========================================================================
    plt.subplot(1, 2, 2)

    # Plot aleatoric uncertainty
    plt.plot(x_test_np, np.sqrt(ale_np), "b-", linewidth=2, label="Aleatoric (data noise)")

    # Plot epistemic uncertainty
    plt.plot(
        x_test_np,
        np.sqrt(epi_np),
        "g-",
        linewidth=2,
        label="Epistemic (model uncertainty)",
    )

    # Plot total uncertainty
    plt.plot(x_test_np, std_total, "r--", linewidth=2, label="Total uncertainty")

    # Highlight gap region
    plt.axvspan(-0.2, 0.2, alpha=0.1, color="yellow", label="OOD region")

    plt.xlabel("x")
    plt.ylabel("Uncertainty (std)")
    plt.title("Uncertainty Decomposition")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # ========================================================================
    # Plot 3: Epistemic Uncertainty for OOD Detection
    # ========================================================================
    plt.figure(figsize=(10, 6))

    plt.plot(x_test_np, epi_np, "g-", linewidth=2)
    plt.scatter(x_train_np, np.zeros_like(x_train_np), alpha=0.3, s=5, c="blue")

    plt.axvspan(-0.2, 0.2, alpha=0.2, color="red", label="OOD region (gap)")

    plt.xlabel("x")
    plt.ylabel("Epistemic Uncertainty")
    plt.title("Epistemic Uncertainty Increases in OOD Regions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def compare_regularization_strengths(x_train, y_train, x_test, y_test_clean):
    """Compare different regularization coefficients."""
    print("\n" + "=" * 70)
    print("Comparing Regularization Strengths")
    print("=" * 70)

    coeffs = [0.0, 0.001, 0.01, 0.1]
    models = []
    loss_fns = []

    for coeff in coeffs:
        print(f"\nTraining with coeff_nig = {coeff}")
        model, loss_fn = train_evidential_model(x_train, y_train, n_epochs=300, coeff_nig=coeff)
        models.append(model)
        loss_fns.append(loss_fn)

    # Visualize comparison
    plt.figure(figsize=(15, 10))
    x_test_np = x_test.numpy().flatten()
    y_test_np = y_test_clean.numpy().flatten()
    x_train_np = x_train.numpy().flatten()
    y_train_np = y_train.numpy().flatten()

    for idx, (model, loss_fn, coeff) in enumerate(zip(models, loss_fns, coeffs)):
        model.eval()
        with torch.no_grad():
            params = model(x_test)
            mean, ale, epi = loss_fn.predict_with_uncertainty(params)

        mean_np = mean.numpy().flatten()
        std_total = np.sqrt((ale + epi).numpy().flatten())

        plt.subplot(2, 2, idx + 1)

        # Training data
        plt.scatter(x_train_np, y_train_np, alpha=0.2, s=5, c="gray")

        # True function
        plt.plot(x_test_np, y_test_np, "k-", linewidth=1.5, label="True")

        # Predictions
        plt.plot(x_test_np, mean_np, "r-", linewidth=2, label="Predicted")

        # Uncertainty
        plt.fill_between(
            x_test_np,
            mean_np - 1.96 * std_total,
            mean_np + 1.96 * std_total,
            alpha=0.3,
            color="red",
        )

        # Gap region
        plt.axvspan(-0.2, 0.2, alpha=0.1, color="yellow")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Regularization coeff = {coeff}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-2, 2)

    plt.suptitle("Effect of NIG Regularization Coefficient", fontsize=14)
    plt.tight_layout()


def demonstrate_sampling(model, loss_fn, x_test):
    """Demonstrate sampling from the evidential distribution."""
    print("\n" + "=" * 70)
    print("Sampling from Evidential Distribution")
    print("=" * 70)

    model.eval()

    # Select a few test points
    test_indices = [100, 200, 250, 300, 400]  # Different regions
    x_selected = x_test[test_indices]

    with torch.no_grad():
        params = model(x_selected)

        # Get mean and uncertainties
        mean, ale, epi = loss_fn.predict_with_uncertainty(params)

        # Generate samples
        samples = loss_fn.sample_predictions(params, n_samples=1000)

    # Visualize distributions
    fig, axes = plt.subplots(1, 5, figsize=(18, 3))

    x_np = x_selected.numpy().flatten()
    mean_np = mean.numpy().flatten()
    ale_np = ale.numpy().flatten()
    epi_np = epi.numpy().flatten()

    for i, (ax, x_val) in enumerate(zip(axes, x_np)):
        samples_i = samples[:, i, 0].numpy()

        # Histogram of samples
        ax.hist(samples_i, bins=50, density=True, alpha=0.6, color="blue")

        # True mean
        ax.axvline(mean_np[i], color="red", linestyle="--", linewidth=2, label="Mean")

        ax.set_xlabel("Prediction")
        ax.set_ylabel("Density")
        ax.set_title(
            f"x={x_val:.2f}\n" f"Ale={ale_np[i]:.3f}, Epi={epi_np[i]:.3f}",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Predictive Distributions at Different Input Locations", fontsize=12)
    plt.tight_layout()


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all experiments."""
    print("=" * 70)
    print("Evidential Deep Learning for Regression")
    print("=" * 70)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Create dataset
    x_train, y_train, x_test, y_test_clean = create_dataset_with_gaps(n_samples=500)

    print("\nDataset:")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples: {len(x_test)}")
    print("  Gap region: [-0.2, 0.2] (no training data)")

    # 1. Train and evaluate main model
    model, loss_fn = train_evidential_model(x_train, y_train, n_epochs=500, coeff_nig=0.01)
    evaluate_and_visualize(model, loss_fn, x_train, y_train, x_test, y_test_clean)

    # 2. Compare regularization strengths
    compare_regularization_strengths(x_train, y_train, x_test, y_test_clean)

    # 3. Demonstrate sampling
    demonstrate_sampling(model, loss_fn, x_test)

    # Summary
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)
    print("1. Epistemic uncertainty increases in OOD regions (gap)")
    print("2. Aleatoric uncertainty reflects data noise (heteroscedastic)")
    print("3. Higher regularization → more conservative predictions")
    print("4. Single forward pass → fast inference (no ensembles needed)")
    print("5. Principled decomposition: epistemic + aleatoric = total")

    plt.show()


if __name__ == "__main__":
    main()
```
