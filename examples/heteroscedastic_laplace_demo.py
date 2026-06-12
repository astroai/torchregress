"""
Heteroscedastic Laplace Regression and Uncertainty Decomposition Demo.

This example demonstrates how to use the Heteroscedastic Laplace Regressor to perform
Bayesian last-layer Laplace posterior approximation on deep heteroscedastic neural networks.

We decompose the predictive uncertainty into:
1. Aleatoric uncertainty: Data-dependent noise (inherent observation noise).
2. Epistemic uncertainty: Parameter uncertainty (lack of knowledge about model weights).

Seminal reference:
Immer, A., Postels, J., Lücke, J., Rätsch, G., & Mandt, S. (2023). Effective Bayesian
Heteroscedastic Regression with Deep Neural Networks. Advances in Neural Information
Processing Systems (NeurIPS).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.algorithms import (
    HeteroscedasticLaplaceRegressor,
    NaturalHeteroscedasticHead,
)


def generate_heteroscedastic_data(n_samples=1000, seed=42):
    """Generate synthetic heteroscedastic regression data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = np.random.uniform(-2.0, 2.0, size=(n_samples, 1)).astype(np.float32)

    # Nonlinear mean
    mean = 0.5 * (x**3)

    # Heteroscedastic noise: noise grows away from zero
    noise_std = 0.2 + 0.4 * (x + 1.0) ** 2
    noise = np.random.normal(0.0, noise_std).astype(np.float32)

    y = mean + noise
    return torch.from_numpy(x), torch.from_numpy(y)


def main():
    print("=" * 60)
    print("Heteroscedastic Laplace Regression & Uncertainty Decomposition")
    print("=" * 60)

    x, y = generate_heteroscedastic_data(n_samples=1200)

    # Train / Test split
    n_train = 800
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Define base model features representation
    hidden_dim = 16
    base_model = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, hidden_dim),
        nn.ReLU(),
    )

    # Define natural heteroscedastic output head mapping features to mean & log_var
    head = NaturalHeteroscedasticHead(in_features=hidden_dim, out_features=1, link_fn="exp")

    # Define Heteroscedastic Laplace Regressor wrapper
    laplace_reg = HeteroscedasticLaplaceRegressor(
        base_model=base_model,
        head=head,
        prior_precision=1.0,
        n_samples=50,
    )

    # Create DataLoader
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Fitting Heteroscedastic Laplace Regressor...")
    # Fits weights, then computes last-layer Laplace posterior approximation
    laplace_reg.fit(loader, lr=0.01, epochs=100)

    # Predict distribution on test set
    print("Predicting Distribution & Decomposing Uncertainty...")
    pred_dist = laplace_reg.predict_distribution(x_test)

    # Extract decomposed uncertainties
    pred_mean = pred_dist.mean.squeeze(1)
    epistemic_var = pred_dist.extra["epistemic_variance"].squeeze(1)
    aleatoric_var = pred_dist.extra["aleatoric_variance"].squeeze(1)
    pred_dist.std.squeeze(1)

    print("\n--- Predictions & Uncertainty Metrics on Test Set ---")
    print(
        f"{'Sample X':<10} | {'True Y':<10} | {'Pred Mean':<10} | {'Aleatoric Var':<13} | {'Epistemic Var':<13}"
    )
    print("-" * 68)

    # Print a few samples across the input range
    sorted_idx = torch.argsort(x_test.squeeze(1))
    sample_indices = [
        sorted_idx[0].item(),  # Left end
        sorted_idx[len(sorted_idx) // 2].item(),  # Center
        sorted_idx[-1].item(),  # Right end
    ]

    for idx in sample_indices:
        print(
            f"{x_test[idx, 0]:<10.3f} | "
            f"{y_test[idx, 0]:<10.3f} | "
            f"{pred_mean[idx]:<10.3f} | "
            f"{aleatoric_var[idx]:<13.4f} | "
            f"{epistemic_var[idx]:<13.4f}"
        )

    print("-" * 68)
    print("\nObservation:")
    print("1. Aleatoric uncertainty varies across inputs, capturing the noise standard deviation.")
    print("2. Epistemic uncertainty reflects parameter uncertainty from last-layer weights.")
    print(
        "3. Total variance is the sum of epistemic and aleatoric variance (law of total variance)."
    )


if __name__ == "__main__":
    main()
