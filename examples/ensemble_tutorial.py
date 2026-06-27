"""
Ensemble Methods Tutorial for Uncertainty Quantification.

This example demonstrates different ensemble methods in torchregress for uncertainty
estimation, comparing:
1. Single Model (baseline)
2. Deep Ensemble (epistemic uncertainty)
3. Heteroscedastic Ensemble (epistemic + aleatoric uncertainty)

Seminal paper: Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable
Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS.

Mathematical Uncertainty Decomposition:
Given predictions $\mu_m(x)$ and predicted variances $\sigma_m^2(x)$ for members $m=1,\dots,M$:

- Predictive Mean:
  $$\bar{\mu}(x) = \frac{1}{M}\sum_{m=1}^M \mu_m(x)$$

- Epistemic Uncertainty (Model Disagreement):
  $$\text{Var}_{\text{epistemic}}(x) = \frac{1}{M}\sum_{m=1}^M (\mu_m(x) - \bar{\mu}(x))^2$$

- Aleatoric Uncertainty (Expected Data Noise):
  $$\text{Var}_{\text{aleatoric}}(x) = \frac{1}{M}\sum_{m=1}^M \sigma_m^2(x)$$

- Total Predictive Variance:
  $$\text{Var}_{\text{total}}(x) = \text{Var}_{\text{epistemic}}(x) + \text{Var}_{\text{aleatoric}}(x)$$
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.ensemble import BaseEnsembleModel, HeteroscedasticEnsembleModel
from torchregress.losses import GaussianNLLLoss, WeightedMSELoss
from torchregress.metrics import (
    ensemble_mean,
    ensemble_std,
    ensemble_variance_decomposition,
)
from torchregress.utils import set_all_seeds as set_seed


def generate_heteroscedastic_data(n_samples=1000, noise_scale=0.1):
    """Generate data with heteroscedastic (varying) noise."""
    np.random.seed(42)
    x = np.linspace(-5, 5, n_samples)
    # True function
    y_true = np.sin(x)
    # Noise that increases with |x|
    noise_std = noise_scale * (1 + 0.5 * np.abs(x))
    noise = np.random.normal(0, noise_std)
    y = y_true + noise

    return x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)


def create_mlp(input_dim=1, hidden_dim=64, output_dim=1):
    """Create a simple MLP for regression."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def create_heteroscedastic_mlp(input_dim=1, hidden_dim=64):
    """Create MLP that outputs both mean and log variance."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2),  # Output: [mean, log_variance]
    )


def train_model(model, dataloader, loss_fn, epochs=100, lr=0.01):
    """Train a single model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)

            # Handle heteroscedastic models (output is tuple or 2D)
            if isinstance(y_pred, tuple):
                loss = loss_fn(y_pred, y_batch)
            elif y_pred.shape[-1] == 2:
                # Split into mean and log variance
                mean, log_var = y_pred[:, 0:1], y_pred[:, 1:2]
                loss = loss_fn((mean, log_var), y_batch)
            else:
                loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    return model


def train_deep_ensemble(n_models, dataloader, hidden_dim=64, epochs=100, lr=0.01):
    """Train a deep ensemble (multiple independent models)."""
    print(f"\nTraining Deep Ensemble with {n_models} models...")
    set_seed(42)
    ensemble = BaseEnsembleModel(
        base_model=create_mlp(hidden_dim=hidden_dim),
        ensemble_size=n_models,
    )
    ensemble.fit(
        train_loader=dataloader,
        criterion=WeightedMSELoss(),
        epochs=epochs,
        lr=lr,
        verbose=True,
    )
    return ensemble


def train_heteroscedastic_ensemble(n_models, dataloader, hidden_dim=64, epochs=100, lr=0.01):
    """Train ensemble where each model predicts mean and variance."""
    print(f"\nTraining Heteroscedastic Ensemble with {n_models} models...")
    set_seed(42)
    ensemble = HeteroscedasticEnsembleModel(
        base_model=create_heteroscedastic_mlp(hidden_dim=hidden_dim),
        ensemble_size=n_models,
    )
    ensemble.fit(
        train_loader=dataloader,
        criterion=GaussianNLLLoss(),
        epochs=epochs,
        lr=lr,
        verbose=True,
    )
    return ensemble


def evaluate_ensemble(ensemble, x_test, y_test, model_type="Deep Ensemble"):
    """Evaluate ensemble predictions."""
    ensemble.eval()
    with torch.no_grad():
        predictions = ensemble(x_test)

        if isinstance(ensemble, HeteroscedasticEnsembleModel):
            # Predictions are either stacked tensors [M, B, 2] or list of tensors
            if isinstance(predictions, torch.Tensor):
                means = predictions[..., 0:1]
                log_vars = predictions[..., 1:2]
            else:
                means = torch.stack([pred[..., 0:1] for pred in predictions])
                log_vars = torch.stack([pred[..., 1:2] for pred in predictions])

            # Decompose uncertainty
            variances = torch.exp(log_vars)
            epistemic, aleatoric = ensemble_variance_decomposition(means, variances)
            total_std = torch.sqrt(epistemic + aleatoric)
            pred_mean = ensemble_mean(means)

            print(f"\n{model_type} Results:")
            print(f"  Mean Epistemic Uncertainty: {epistemic.mean().item():.4f}")
            print(f"  Mean Aleatoric Uncertainty: {aleatoric.mean().item():.4f}")
            print(f"  Total Uncertainty: {total_std.mean().item():.4f}")

            return pred_mean, total_std, epistemic, aleatoric

        else:
            # Standard ensemble - only epistemic uncertainty
            pred_mean = ensemble_mean(predictions)
            pred_std = ensemble_std(predictions)

            print(f"\n{model_type} Results:")
            print(f"  Mean Predictive Uncertainty (epistemic): {pred_std.mean().item():.4f}")

            return pred_mean, pred_std, None, None


def plot_results(x_train, y_train, x_test, results_dict):
    """Plot comparison of different ensemble methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    x_test_np = x_test.numpy()
    y_true = np.sin(x_test_np)

    for idx, (name, (pred_mean, pred_std, epistemic, aleatoric)) in enumerate(results_dict.items()):
        ax = axes[idx]

        pred_mean_np = pred_mean.numpy()
        pred_std_np = pred_std.numpy()

        # Plot data
        ax.scatter(x_train.numpy(), y_train.numpy(), alpha=0.3, s=10, label="Training data")

        # Plot predictions
        ax.plot(x_test_np, pred_mean_np, "r-", linewidth=2, label="Prediction")
        ax.plot(x_test_np, y_true, "g--", linewidth=1, label="True function")

        # Plot uncertainty
        ax.fill_between(
            x_test_np.flatten(),
            (pred_mean_np - 2 * pred_std_np).flatten(),
            (pred_mean_np + 2 * pred_std_np).flatten(),
            alpha=0.3,
            color="red",
            label="±2σ (95% interval)",
        )

        # If we have uncertainty decomposition, show it
        if epistemic is not None and aleatoric is not None:
            epistemic_np = torch.sqrt(epistemic).numpy()
            aleatoric_np = torch.sqrt(aleatoric).numpy()

            ax.fill_between(
                x_test_np.flatten(),
                (pred_mean_np - 2 * epistemic_np).flatten(),
                (pred_mean_np + 2 * epistemic_np).flatten(),
                alpha=0.2,
                color="blue",
                label="Epistemic",
            )
            ax.fill_between(
                x_test_np.flatten(),
                (pred_mean_np - 2 * aleatoric_np).flatten(),
                (pred_mean_np + 2 * aleatoric_np).flatten(),
                alpha=0.2,
                color="orange",
                label="Aleatoric",
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ensemble_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved as 'ensemble_comparison.png'")
    plt.show()


def main():
    """Main comparison of ensemble methods."""
    print("=" * 60)
    print("Ensemble Methods Tutorial")
    print("=" * 60)

    # Generate data
    x_train, y_train = generate_heteroscedastic_data(n_samples=300)
    x_test = np.linspace(-6, 6, 300).reshape(-1, 1).astype(np.float32)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test)

    dataset = TensorDataset(x_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train single model (baseline)
    print("\n" + "=" * 60)
    print("1. Training Single Model (Baseline)")
    print("=" * 60)
    set_seed(42)
    single_model = create_mlp()
    loss_fn = WeightedMSELoss()
    single_model = train_model(single_model, dataloader, loss_fn, epochs=100)

    single_model.eval()
    with torch.no_grad():
        single_pred = single_model(x_test_t)
    print("Single model provides point predictions only (no uncertainty)")

    # Train Deep Ensemble
    print("\n" + "=" * 60)
    print("2. Deep Ensemble (Epistemic Uncertainty Only)")
    print("=" * 60)
    deep_ensemble = train_deep_ensemble(n_models=5, dataloader=dataloader, epochs=100)
    deep_pred_mean, deep_pred_std, _, _ = evaluate_ensemble(
        deep_ensemble, x_test_t, None, "Deep Ensemble"
    )

    # Train Heteroscedastic Ensemble
    print("\n" + "=" * 60)
    print("3. Heteroscedastic Ensemble (Epistemic + Aleatoric)")
    print("=" * 60)
    hetero_ensemble = train_heteroscedastic_ensemble(n_models=5, dataloader=dataloader, epochs=100)
    hetero_pred_mean, hetero_total_std, hetero_epistemic, hetero_aleatoric = evaluate_ensemble(
        hetero_ensemble, x_test_t, None, "Heteroscedastic Ensemble"
    )

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print("\n1. Single Model:")
    print("   - Fastest to train (1 model)")
    print("   - No uncertainty estimates")
    print("   - Use when: Speed is critical, uncertainty not needed")

    print("\n2. Deep Ensemble:")
    print("   - Epistemic uncertainty from model disagreement")
    print("   - No aleatoric uncertainty (assumes homoscedastic noise)")
    print("   - Training time: 5x single model")
    print("   - Use when: Need epistemic uncertainty, noise is constant")

    print("\n3. Heteroscedastic Ensemble:")
    print("   - Both epistemic AND aleatoric uncertainty")
    print("   - Can decompose total uncertainty into sources")
    print("   - Training time: 5x single model")
    print("   - Use when: Need full uncertainty quantification")

    print("\n" + "=" * 60)
    print("When to use each method:")
    print("=" * 60)
    print("Deep Ensemble:")
    print("  ✓ Out-of-distribution detection")
    print("  ✓ Active learning (query high epistemic uncertainty)")
    print("  ✓ Model selection and validation")

    print("\nHeteroscedastic Ensemble:")
    print("  ✓ When data has varying noise levels")
    print("  ✓ Need to separate model vs. data uncertainty")
    print("  ✓ Safety-critical applications")
    print("  ✓ Calibrated prediction intervals")

    # Plot results
    results_dict = {
        "Single Model": (single_pred, torch.zeros_like(single_pred), None, None),
        "Deep Ensemble\n(Epistemic Only)": (
            deep_pred_mean,
            deep_pred_std,
            None,
            None,
        ),
        "Heteroscedastic Ensemble\n(Total Uncertainty)": (
            hetero_pred_mean,
            hetero_total_std,
            None,
            None,
        ),
        "Heteroscedastic Ensemble\n(Uncertainty Decomposition)": (
            hetero_pred_mean,
            hetero_total_std,
            hetero_epistemic,
            hetero_aleatoric,
        ),
    }

    plot_results(x_train_t, y_train_t, x_test_t, results_dict)

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
