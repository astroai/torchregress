# Normalizing Flows for Multi-Target Regression Example

This example demonstrates how to use normalizing flows for multi-target regression with complex, potentially multimodal distributions.

→ API: [`NormalizingFlowLoss`](../api/losses.md). Guide: [Normalizing flows](../losses/nflows.md).

## Task-First Context

Use this example when you need **multi-target** and **multimodal/non-Gaussian** predictive
distributions. Start with the [Task-First Method Selection Matrix](../guide/method-selection.md)
to compare against `MDN`, multivariate Gaussian (full/low-rank), and ensemble-based options.

## Comparison / Fairness Notes

- This page is a single-method deep dive, not a full decision-grade comparison.
- For fair comparison vs `MDN`/Gaussian baselines, keep backbone capacity, training epochs, and evaluation metrics aligned.
- Report runtime and calibration in addition to point metrics (`MSE`, `MAE`, `R2`) because expressive models can overfit tails.
- Use fixed seeds and the same synthetic data generation parameters when comparing flow types.

## When This Approach Fails

- Flows add complexity and optional dependencies; use `MDN` first unless Gaussian/MDN clearly miss structure.
- Small datasets can make flow models unstable or poorly calibrated.
- If latency/throughput is constrained, benchmark against Gaussian/MDN alternatives before adopting flows.

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

# ============================================================================
# 1. Generate Synthetic Multi-Target Data with Complex Distribution
# ============================================================================


def generate_complex_multitarget_data(n_samples=2000, noise_level=0.1):
    """
    Generate synthetic 2D target data with input-dependent multimodal distribution.

    The data has:
    - 2D targets (y1, y2) that are correlated
    - Distribution that changes with input x
    - Mixture of modes for some input regions
    """
    # Input features
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)

    # Create two correlated targets with complex relationship
    # Target 1: Sinusoidal with input-dependent amplitude
    y1 = torch.sin(x * 2) * (1 + 0.3 * x**2)

    # Target 2: Correlated with y1 but with quadratic relationship
    y2 = 0.5 * y1**2 + torch.cos(x * 3) * 0.5

    # Add input-dependent noise (aleatoric uncertainty)
    # Noise increases in certain regions
    noise_scale = noise_level * (1 + 0.5 * torch.abs(x))

    # Add multimodal noise in middle region
    multimodal_mask = (x.abs() < 1.5).float()
    mode_shift = (
        torch.where(torch.rand(n_samples, 1) < 0.3, torch.tensor(0.5), torch.tensor(-0.5))
        * multimodal_mask
    )

    # Combine targets with noise
    y1_noisy = y1 + torch.randn_like(y1) * noise_scale + mode_shift * 0.3
    y2_noisy = y2 + torch.randn_like(y2) * noise_scale + mode_shift * 0.2

    # Stack into [n_samples, 2] target matrix
    y = torch.cat([y1_noisy, y2_noisy], dim=1)

    return x, y


# ============================================================================
# 2. Build Model with Normalizing Flow
# ============================================================================


class MultiTargetFlowModel(nn.Module):
    """
    Neural network that outputs context for conditioning the normalizing flow.

    Architecture:
        Input -> Hidden Layers -> Context -> Flow -> Distribution over Targets
    """

    def __init__(self, input_dim, context_dim, hidden_dims=[64, 64]):
        super().__init__()

        # Feature extractor that outputs context vectors
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        # Final layer outputs context for flow
        layers.append(nn.Linear(prev_dim, context_dim))

        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        """Output context vectors that condition the flow."""
        return self.feature_extractor(x)


def create_multitarget_flow_model(
    input_dim=1, n_targets=2, context_dim=16, flow_type="nsf", n_transforms=5
):
    """
    Create a complete model for multi-target distributional regression.

    Args:
        input_dim: Dimension of input features
        n_targets: Number of target variables to predict
        context_dim: Dimension of context vectors for conditioning flow
        flow_type: Type of normalizing flow ('nsf', 'maf', 'realnvp')
        n_transforms: Number of transformation blocks in flow

    Returns:
        model: Neural network that outputs context
        flow: Normalizing flow conditioned on context
        loss_fn: Loss function combining model and flow
    """
    # Create the model that produces context
    model = MultiTargetFlowModel(
        input_dim=input_dim, context_dim=context_dim, hidden_dims=[128, 64]
    )

    # Create conditional normalizing flow
    # The flow learns to map simple Gaussian to complex target distribution
    flow = create_flow_model(
        n_features=n_targets,
        context_dim=context_dim,
        flow_type=flow_type,
        n_transforms=n_transforms,
        hidden_features=64,
        n_hidden_layers=2,
    )

    # Create loss function
    loss_fn = NormalizingFlowLoss(flow=flow, reduction="mean")

    return model, flow, loss_fn


# ============================================================================
# 3. Training
# ============================================================================


def train_flow_model(model, flow, loss_fn, train_loader, n_epochs=100, lr=1e-3):
    """
    Train the model and flow jointly.

    Both model parameters and flow parameters are optimized together.
    """
    # Combine parameters from both model and flow
    optimizer = torch.optim.Adam(list(model.parameters()) + list(flow.parameters()), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    losses = []

    for epoch in range(n_epochs):
        model.train()
        flow.train()

        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Model outputs context
            context = model(batch_x)

            # Flow computes negative log-likelihood given context
            loss = loss_fn(context, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return losses


# ============================================================================
# 4. Prediction and Sampling
# ============================================================================


def predict_with_flow(model, flow, loss_fn, x_test, n_samples=100):
    """
    Generate predictions from the learned distribution.

    For each input, the flow can:
    1. Sample multiple predictions from the learned distribution
    2. Compute the mean and uncertainty
    3. Capture multimodal distributions

    Args:
        model: Trained neural network
        flow: Trained normalizing flow
        loss_fn: Loss function containing the flow
        x_test: Test inputs [n_test, input_dim]
        n_samples: Number of samples to draw per input

    Returns:
        samples: Samples from predictive distribution [n_test, n_samples, n_targets]
        mean: Mean prediction [n_test, n_targets]
        std: Standard deviation [n_test, n_targets]
    """
    model.eval()
    flow.eval()

    with torch.no_grad():
        # Get context from model
        context = model(x_test)

        # Sample from flow's conditional distribution
        samples = loss_fn.sample(context, n_samples=n_samples)  # [n_test, n_samples, 2]

        # Compute statistics
        mean = samples.mean(dim=1)  # [n_test, 2]
        std = samples.std(dim=1)  # [n_test, 2]

    return samples, mean, std


# ============================================================================
# 5. Visualization
# ============================================================================


def visualize_predictions(x_train, y_train, x_test, y_test, samples, mean, std):
    """Visualize multi-target predictions with uncertainty."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot each target dimension
    for i, target_name in enumerate(["Target 1", "Target 2"]):
        ax = axes[i, 0]

        # Training data
        ax.scatter(
            x_train.numpy(),
            y_train[:, i].numpy(),
            alpha=0.3,
            s=10,
            label="Training data",
            color="gray",
        )

        # Test data
        ax.scatter(
            x_test.numpy(),
            y_test[:, i].numpy(),
            alpha=0.5,
            s=20,
            label="True test data",
            color="blue",
        )

        # Predictions with uncertainty
        ax.plot(x_test.numpy(), mean[:, i].numpy(), "r-", linewidth=2, label="Mean prediction")
        ax.fill_between(
            x_test.numpy().flatten(),
            (mean[:, i] - 2 * std[:, i]).numpy().flatten(),
            (mean[:, i] + 2 * std[:, i]).numpy().flatten(),
            alpha=0.3,
            color="red",
            label="±2 std",
        )

        ax.set_xlabel("Input x")
        ax.set_ylabel(target_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{target_name} Predictions")

    # 2D scatter: True vs Predicted
    ax = axes[0, 1]
    ax.scatter(y_test[:, 0].numpy(), mean[:, 0].numpy(), alpha=0.5)
    lim = [y_test[:, 0].min().item(), y_test[:, 0].max().item()]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set_xlabel("True Target 1")
    ax.set_ylabel("Predicted Target 1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Target 1: True vs Predicted")

    ax = axes[1, 1]
    ax.scatter(y_test[:, 1].numpy(), mean[:, 1].numpy(), alpha=0.5)
    lim = [y_test[:, 1].min().item(), y_test[:, 1].max().item()]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set_xlabel("True Target 2")
    ax.set_ylabel("Predicted Target 2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Target 2: True vs Predicted")

    plt.tight_layout()
    plt.savefig("normalizing_flows_multitarget_predictions.png", dpi=150)
    print("Saved predictions plot to 'normalizing_flows_multitarget_predictions.png'")

    # Joint distribution visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Select a few interesting points to visualize their distributions
    test_indices = [50, 100, 150]

    for idx, test_idx in enumerate(test_indices):
        ax = axes[idx]

        # Get samples for this input
        input_samples = samples[test_idx].numpy()  # [n_samples, 2]

        # 2D scatter of samples
        ax.scatter(input_samples[:, 0], input_samples[:, 1], alpha=0.3, s=10, label="Samples")

        # True point
        ax.scatter(
            y_test[test_idx, 0].numpy(),
            y_test[test_idx, 1].numpy(),
            color="red",
            s=100,
            marker="x",
            linewidths=3,
            label="True",
        )

        # Mean prediction
        ax.scatter(
            mean[test_idx, 0].numpy(),
            mean[test_idx, 1].numpy(),
            color="green",
            s=100,
            marker="o",
            label="Mean",
        )

        ax.set_xlabel("Target 1")
        ax.set_ylabel("Target 2")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Joint Distribution at x={x_test[test_idx].item():.2f}")

    plt.tight_layout()
    plt.savefig("normalizing_flows_joint_distributions.png", dpi=150)
    print("Saved joint distributions plot to 'normalizing_flows_joint_distributions.png'")


# ============================================================================
# 6. Main Execution
# ============================================================================


def main():
    """Run the complete multi-target normalizing flow example."""

    print("=" * 70)
    print("Multi-Target Distributional Regression with Normalizing Flows")
    print("=" * 70)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Generate data
    print("\n1. Generating synthetic multi-target data...")
    x_train, y_train = generate_complex_multitarget_data(n_samples=1500)
    x_test, y_test = generate_complex_multitarget_data(n_samples=500)

    print(f"   Training set: {x_train.shape[0]} samples")
    print(f"   Test set: {x_test.shape[0]} samples")
    print(f"   Input dim: {x_train.shape[1]}")
    print(f"   Target dim: {y_train.shape[1]}")

    # Create data loader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 2. Create model
    print("\n2. Creating multi-target flow model...")
    model, flow, loss_fn = create_multitarget_flow_model(
        input_dim=1,
        n_targets=2,
        context_dim=16,
        flow_type="nsf",  # Neural Spline Flow - most flexible
        n_transforms=5,
    )

    n_model_params = sum(p.numel() for p in model.parameters())
    n_flow_params = sum(p.numel() for p in flow.parameters())
    print(f"   Model parameters: {n_model_params:,}")
    print(f"   Flow parameters: {n_flow_params:,}")
    print(f"   Total parameters: {n_model_params + n_flow_params:,}")

    # 3. Train
    print("\n3. Training model and flow jointly...")
    print("   (Both model and flow parameters are optimized together)")
    losses = train_flow_model(model, flow, loss_fn, train_loader, n_epochs=100, lr=1e-3)

    # 4. Evaluate
    print("\n4. Generating predictions on test set...")
    samples, mean, std = predict_with_flow(model, flow, loss_fn, x_test, n_samples=200)

    # Compute metrics
    mse_target1 = ((mean[:, 0] - y_test[:, 0]) ** 2).mean().item()
    mse_target2 = ((mean[:, 1] - y_test[:, 1]) ** 2).mean().item()

    print(f"\n   Test MSE (Target 1): {mse_target1:.4f}")
    print(f"   Test MSE (Target 2): {mse_target2:.4f}")
    print(f"   Mean uncertainty (Target 1): {std[:, 0].mean().item():.4f}")
    print(f"   Mean uncertainty (Target 2): {std[:, 1].mean().item():.4f}")

    # 5. Visualize
    print("\n5. Creating visualizations...")
    visualize_predictions(x_train, y_train, x_test, y_test, samples, mean, std)

    print("\n" + "=" * 70)
    print("Key Advantages of Normalizing Flows for Multi-Target Regression:")
    print("=" * 70)
    print("✓ Models complex joint distributions over multiple targets")
    print("✓ Captures correlations between target variables")
    print("✓ Handles multimodal and non-Gaussian distributions")
    print("✓ Provides calibrated uncertainty estimates")
    print("✓ Input-dependent distributions (aleatoric uncertainty)")
    print("✓ Can sample from the learned distribution")
    print("✓ Gradients flow properly for end-to-end training")
    print("=" * 70)

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
```
