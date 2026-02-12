"""
Comprehensive Comparison: Robustness, Uncertainty, and Ensembling

This example demonstrates the three main capabilities of torchregress:
1. Robust Regression - Handling outliers
2. Uncertainty Estimation - Quantifying prediction confidence
3. Ensemble Methods - Combining models for better predictions

We compare different approaches on challenging data scenarios to help you
understand when to use each method.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import (
    CauchyLoss,
    GaussianNLLLoss,
    HuberLoss,
    WeightedLossWrapper,
)
from torchregress.metrics import (
    ensemble_mean,
    ensemble_std,
    ensemble_variance_decomposition,
)
from torchregress.utils import set_seed

# ============================================================================
# Data Generation
# ============================================================================


def generate_clean_data(n_samples=200):
    """Generate clean sinusoidal data with Gaussian noise."""
    np.random.seed(42)
    x = np.linspace(-5, 5, n_samples)
    y = np.sin(x) + np.random.normal(0, 0.1, n_samples)
    return x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)


def generate_data_with_outliers(n_samples=200, outlier_fraction=0.1):
    """Generate data with outliers."""
    np.random.seed(42)
    x = np.linspace(-5, 5, n_samples)
    y = np.sin(x) + np.random.normal(0, 0.1, n_samples)

    # Add outliers
    n_outliers = int(n_samples * outlier_fraction)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    y[outlier_indices] += np.random.uniform(-2, 2, n_outliers)

    return x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)


def generate_heteroscedastic_data(n_samples=200):
    """Generate data with heteroscedastic (varying) noise."""
    np.random.seed(42)
    x = np.linspace(-5, 5, n_samples)
    y_true = np.sin(x)
    # Noise increases with |x|
    noise_std = 0.05 * (1 + 0.8 * np.abs(x))
    y = y_true + np.random.normal(0, noise_std)
    return x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)


# ============================================================================
# Model Architectures
# ============================================================================


def create_mlp(input_dim=1, hidden_dim=64, output_dim=1):
    """Create simple MLP for point prediction."""
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


# ============================================================================
# Training Functions
# ============================================================================


def train_model(model, dataloader, loss_fn, epochs=100, lr=0.01, verbose=False):
    """Train a single model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)

            # Handle heteroscedastic models
            if isinstance(y_pred, tuple):
                loss = loss_fn(y_pred, y_batch)
            elif y_pred.shape[-1] == 2:
                mean, log_var = y_pred[:, 0:1], y_pred[:, 1:2]
                loss = loss_fn((mean, log_var), y_batch)
            else:
                loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    return model


def train_ensemble(n_models, dataloader, model_fn, loss_fn, epochs=100, lr=0.01, verbose=False):
    """Train an ensemble of models."""
    models = []
    for i in range(n_models):
        if verbose:
            print(f"  Training model {i + 1}/{n_models}")
        set_seed(42 + i)
        model = model_fn()
        model = train_model(model, dataloader, loss_fn, epochs=epochs, lr=lr)
        models.append(model)
    return models


# ============================================================================
# Evaluation Functions
# ============================================================================


def evaluate_point_predictions(model, x_test, y_test, name="Model"):
    """Evaluate model with point predictions."""
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        if y_test is not None:
            mse = ((y_pred - y_test) ** 2).mean().item()
            mae = (torch.abs(y_pred - y_test)).mean().item()
        else:
            mse = None
            mae = None

    print(f"{name}:")
    if mse is not None:
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
    else:
        print("  Metrics skipped (no targets provided).")

    return y_pred, None


def evaluate_ensemble_predictions(ensemble_models, x_test, y_test, name="Ensemble"):
    """Evaluate ensemble predictions."""
    predictions = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            pred = model(x_test)
            predictions.append(pred)

    predictions = torch.stack(predictions)
    pred_mean = ensemble_mean(predictions)
    pred_std = ensemble_std(predictions)

    print(f"{name}:")
    if y_test is not None:
        mse = ((pred_mean - y_test) ** 2).mean().item()
        mae = (torch.abs(pred_mean - y_test)).mean().item()
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
    else:
        print("  Metrics skipped (no targets provided).")
    print(f"  Mean Uncertainty: {pred_std.mean().item():.4f}")

    return pred_mean, pred_std


def evaluate_heteroscedastic_ensemble(
    ensemble_models, x_test, y_test, name="Heteroscedastic Ensemble"
):
    """Evaluate heteroscedastic ensemble with uncertainty decomposition."""
    predictions = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            pred = model(x_test)
            mean, log_var = pred[:, 0:1], pred[:, 1:2]
            predictions.append((mean, log_var))

    means = torch.stack([p[0] for p in predictions])
    log_vars = torch.stack([p[1] for p in predictions])

    pred_mean = ensemble_mean(means)
    variances = torch.exp(log_vars)
    epistemic, aleatoric = ensemble_variance_decomposition(means, variances)
    total_std = torch.sqrt(epistemic + aleatoric)

    print(f"{name}:")
    if y_test is not None:
        mse = ((pred_mean - y_test) ** 2).mean().item()
        mae = (torch.abs(pred_mean - y_test)).mean().item()
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
    else:
        print("  Metrics skipped (no targets provided).")
    print(f"  Epistemic Uncertainty: {torch.sqrt(epistemic).mean().item():.4f}")
    print(f"  Aleatoric Uncertainty: {torch.sqrt(aleatoric).mean().item():.4f}")
    print(f"  Total Uncertainty: {total_std.mean().item():.4f}")

    return pred_mean, total_std, epistemic, aleatoric


# ============================================================================
# Plotting Functions
# ============================================================================


def plot_comparison(x_train, y_train, x_test, results, title, filename):
    """Plot comparison of different methods."""
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))

    if n_methods == 1:
        axes = [axes]

    x_test_np = x_test.numpy()
    y_true = np.sin(x_test_np)

    for ax, (name, result) in zip(axes, results.items()):
        pred_mean, pred_std, extra = result

        # Plot data
        ax.scatter(
            x_train.numpy(),
            y_train.numpy(),
            alpha=0.4,
            s=20,
            label="Training data",
            c="gray",
        )

        # Plot true function
        ax.plot(x_test_np, y_true, "g--", linewidth=1.5, label="True function")

        # Plot predictions
        ax.plot(
            x_test_np,
            pred_mean.numpy(),
            "r-",
            linewidth=2,
            label="Prediction",
        )

        # Plot uncertainty if available
        if pred_std is not None:
            ax.fill_between(
                x_test_np.flatten(),
                (pred_mean - 2 * pred_std).numpy().flatten(),
                (pred_mean + 2 * pred_std).numpy().flatten(),
                alpha=0.3,
                color="red",
                label="±2σ (95% interval)",
            )

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved as '{filename}'")
    plt.close()


# ============================================================================
# Scenarios
# ============================================================================


def scenario_1_clean_data():
    """Scenario 1: Clean data - comparing basic approaches."""
    print("\n" + "=" * 70)
    print("SCENARIO 1: Clean Data")
    print("=" * 70)
    print("Testing on clean data with minimal outliers")
    print()

    x_train, y_train = generate_clean_data(n_samples=200)
    x_test = np.linspace(-6, 6, 300).reshape(-1, 1).astype(np.float32)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test)

    dataset = TensorDataset(x_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    results = {}

    # MSE baseline
    print("\n1. Training MSE model...")
    set_seed(42)
    model_mse = create_mlp()
    model_mse = train_model(model_mse, dataloader, WeightedLossWrapper(nn.MSELoss), epochs=100)
    pred_mse, _ = evaluate_point_predictions(model_mse, x_test_t, None, "MSE")
    results["MSE (Baseline)"] = (pred_mse, None, None)

    # Heteroscedastic
    print("\n2. Training Heteroscedastic model...")
    set_seed(42)
    model_hetero = create_heteroscedastic_mlp()
    model_hetero = train_model(model_hetero, dataloader, GaussianNLLLoss(), epochs=100)
    model_hetero.eval()
    with torch.no_grad():
        pred = model_hetero(x_test_t)
        pred_mean, pred_log_var = pred[:, 0:1], pred[:, 1:2]
        pred_std = torch.exp(0.5 * pred_log_var)
    results["Heteroscedastic"] = (pred_mean, pred_std, None)

    # Deep Ensemble
    print("\n3. Training Deep Ensemble...")
    ensemble_models = train_ensemble(
        5, dataloader, create_mlp, WeightedLossWrapper(nn.MSELoss), epochs=100
    )
    pred_mean, pred_std = evaluate_ensemble_predictions(
        ensemble_models, x_test_t, None, "Deep Ensemble"
    )
    results["Deep Ensemble"] = (pred_mean, pred_std, None)

    plot_comparison(
        x_train_t,
        y_train_t,
        x_test_t,
        results,
        "Scenario 1: Clean Data",
        "comparison_clean.png",
    )

    print("\n📊 Key Insight:")
    print("   On clean data, all methods perform similarly.")
    print("   MSE is fastest, ensembles add uncertainty estimates.")


def scenario_2_outliers():
    """Scenario 2: Data with outliers - robust losses."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Data with Outliers")
    print("=" * 70)
    print("Testing robust regression methods (10% outliers)")
    print()

    x_train, y_train = generate_data_with_outliers(n_samples=200, outlier_fraction=0.1)
    x_test = np.linspace(-6, 6, 300).reshape(-1, 1).astype(np.float32)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test)

    dataset = TensorDataset(x_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    results = {}

    # MSE (sensitive to outliers)
    print("\n1. Training MSE model (sensitive to outliers)...")
    set_seed(42)
    model_mse = create_mlp()
    model_mse = train_model(model_mse, dataloader, WeightedLossWrapper(nn.MSELoss), epochs=100)
    pred_mse, _ = evaluate_point_predictions(model_mse, x_test_t, None, "MSE")
    results["MSE (Sensitive)"] = (pred_mse, None, None)

    # Huber (balanced)
    print("\n2. Training Huber model (balanced robustness)...")
    set_seed(42)
    model_huber = create_mlp()
    model_huber = train_model(model_huber, dataloader, HuberLoss(delta=1.0), epochs=100)
    pred_huber, _ = evaluate_point_predictions(model_huber, x_test_t, None, "Huber")
    results["Huber (Balanced)"] = (pred_huber, None, None)

    # Cauchy (very robust)
    print("\n3. Training Cauchy model (very robust)...")
    set_seed(42)
    model_cauchy = create_mlp()
    model_cauchy = train_model(model_cauchy, dataloader, CauchyLoss(c=0.5), epochs=100)
    pred_cauchy, _ = evaluate_point_predictions(model_cauchy, x_test_t, None, "Cauchy")
    results["Cauchy (Very Robust)"] = (pred_cauchy, None, None)

    plot_comparison(
        x_train_t,
        y_train_t,
        x_test_t,
        results,
        "Scenario 2: Robustness to Outliers",
        "comparison_outliers.png",
    )

    print("\n📊 Key Insight:")
    print("   MSE is pulled by outliers → poor fit")
    print("   Huber provides good balance")
    print("   Cauchy is most robust but can underfit clean regions")


def scenario_3_heteroscedastic():
    """Scenario 3: Heteroscedastic data - varying uncertainty."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Heteroscedastic Data")
    print("=" * 70)
    print("Testing with varying noise levels across input space")
    print()

    x_train, y_train = generate_heteroscedastic_data(n_samples=200)
    x_test = np.linspace(-6, 6, 300).reshape(-1, 1).astype(np.float32)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test)

    dataset = TensorDataset(x_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    results = {}

    # Simple Gaussian (assumes homoscedastic)
    print("\n1. Training simple Gaussian (assumes constant noise)...")
    set_seed(42)
    model_simple = create_heteroscedastic_mlp()
    model_simple = train_model(model_simple, dataloader, GaussianNLLLoss(), epochs=100)
    model_simple.eval()
    with torch.no_grad():
        pred = model_simple(x_test_t)
        pred_mean, pred_log_var = pred[:, 0:1], pred[:, 1:2]
        pred_std = torch.exp(0.5 * pred_log_var)
    results["Gaussian NLL"] = (pred_mean, pred_std, None)

    # Heteroscedastic Ensemble
    print("\n2. Training Heteroscedastic Ensemble...")
    ensemble_models = train_ensemble(
        5, dataloader, create_heteroscedastic_mlp, GaussianNLLLoss(), epochs=100
    )
    pred_mean, total_std, epistemic, aleatoric = evaluate_heteroscedastic_ensemble(
        ensemble_models, x_test_t, None
    )
    results["Hetero Ensemble"] = (pred_mean, total_std, None)

    # Show uncertainty decomposition separately
    results_decomp = {
        "Total Uncertainty": (pred_mean, total_std, None),
        "Epistemic (Model)": (pred_mean, torch.sqrt(epistemic), None),
        "Aleatoric (Data)": (pred_mean, torch.sqrt(aleatoric), None),
    }

    plot_comparison(
        x_train_t,
        y_train_t,
        x_test_t,
        results,
        "Scenario 3: Heteroscedastic Data",
        "comparison_heteroscedastic.png",
    )

    plot_comparison(
        x_train_t,
        y_train_t,
        x_test_t,
        results_decomp,
        "Scenario 3: Uncertainty Decomposition",
        "comparison_uncertainty_decomp.png",
    )

    print("\n📊 Key Insight:")
    print("   Heteroscedastic Ensemble captures varying noise")
    print("   Epistemic uncertainty is high where data is sparse")
    print("   Aleatoric uncertainty tracks the varying noise level")


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all comparison scenarios."""
    print("=" * 70)
    print("Comprehensive Comparison: Robustness, Uncertainty, Ensembling")
    print("=" * 70)

    scenario_1_clean_data()
    scenario_2_outliers()
    scenario_3_heteroscedastic()

    print("\n" + "=" * 70)
    print("SUMMARY: When to Use Each Method")
    print("=" * 70)

    print("\n📦 ROBUSTNESS (Scenario 2):")
    print("   MSE:    Fast, optimal for clean data")
    print("   Huber:  Good default for real-world data")
    print("   Cauchy: Extreme outliers (e.g., sensor failures)")

    print("\n🎲 UNCERTAINTY (Scenarios 1 & 3):")
    print("   Single Model:           No uncertainty")
    print("   Heteroscedastic:        Aleatoric (data noise)")
    print("   Deep Ensemble:          Epistemic (model uncertainty)")
    print("   Heteroscedastic Ensemble: BOTH (full decomposition)")

    print("\n🎯 RECOMMENDATION:")
    print("   Start: MSE (clean) or Huber (real-world)")
    print("   Need uncertainty: GaussianNLL")
    print("   Production/Safety-critical: Heteroscedastic Ensemble")

    print("\n" + "=" * 70)
    print("All scenarios complete! Check generated PNG files.")
    print("=" * 70)


if __name__ == "__main__":
    main()
