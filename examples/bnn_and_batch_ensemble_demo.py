"""
Bayesian Neural Networks and Batch Ensemble Demo.

This script demonstrates torchregress's variational Bayesian neural networks,
batch ensembles, and Bayesian Model Averaging (BMA) for regression tasks.
It showcases how to fit these models, compute predictive intervals,
and decompose predictive uncertainty into aleatoric (data noise) and
epistemic (model parameter uncertainty) components.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.ensemble import (
    BatchEnsembleMLPBackbone,
    BayesianModelAveraging,
    HeteroscedasticBNN,
    PackedEnsembleRegressor,
)


def main():
    print("================================================================================")
    print("                 torchregress BNN & BatchEnsemble Showcase                      ")
    print("================================================================================")

    torch.manual_seed(42)

    # 1. Generate synthetic heteroscedastic training data
    # y = sin(x) + noise(x) where noise(x) increases with |x|
    n_samples = 300
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    noise = torch.randn(n_samples, 1) * (0.1 + 0.2 * torch.abs(x))
    y = torch.sin(x) + noise

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Generate out-of-distribution (OOD) test data to show epistemic uncertainty
    x_test = torch.linspace(-5, 5, 100).unsqueeze(1)
    torch.sin(x_test)

    # --------------------------------------------------------------------------------
    # 2. Heteroscedastic Bayesian Neural Network (BNN)
    # --------------------------------------------------------------------------------
    print("\n--- 1. Training Heteroscedastic BNN (Variational Inference) ---")
    bnn = HeteroscedasticBNN(
        input_dim=1,
        hidden_dims=[32, 16],
        output_dim=1,
        prior_sigma=1.0,
        n_samples=20,
    )

    # Train BNN using ELBO (NLL + KL)
    optimizer = torch.optim.Adam(bnn.parameters(), lr=0.01)
    bnn.train()

    for epoch in range(100):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            # Forward pass: BNN returns predicted mean and log variance
            mean, log_var = bnn(batch_x)

            # Compute Gaussian Negative Log Likelihood loss
            var = torch.exp(log_var)
            nll_loss = 0.5 * (torch.log(var) + (batch_y - mean) ** 2 / var).mean()

            # Compute KL divergence penalty for variational weights
            kl_div = bnn.kl_divergence()

            # ELBO loss = NLL + (KL weight / total samples)
            kl_weight = 1e-4
            loss = nll_loss + kl_weight * kl_div

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d} | ELBO Loss: {total_loss / len(loader):.6f}")

    # Evaluate BNN and decompose uncertainty
    bnn.eval()
    with torch.no_grad():
        mean_bnn, aleatoric_bnn, epistemic_bnn = bnn.predict_with_decomposition(x_test)
        lower_bnn, upper_bnn = bnn.predict_interval(x_test, confidence=0.95)

    print("\nBNN Uncertainty Decomposition on Test Points:")
    # Print stats for in-distribution (x=0) vs out-of-distribution (x=4.5)
    for idx_x in [50, 95]:  # approx x=0 and x=4.5
        print(f"  At x = {x_test[idx_x].item():.2f}:")
        print(f"    Predicted Mean : {mean_bnn[idx_x].item():.4f}")
        print(f"    Aleatoric Var  : {aleatoric_bnn[idx_x].item():.4f} (data noise)")
        print(f"    Epistemic Var  : {epistemic_bnn[idx_x].item():.4f} (model uncertainty)")
        print(
            f"    95% Interval   : [{lower_bnn[idx_x].item():.4f}, {upper_bnn[idx_x].item():.4f}]"
        )

    # --------------------------------------------------------------------------------
    # 3. Packed Ensemble Regressor (BatchEnsemble)
    # --------------------------------------------------------------------------------
    print("\n--- 2. Evaluating PackedEnsembleRegressor (BatchEnsemble) ---")

    # Define a BatchEnsemble backbone
    ensemble_size = 4
    backbone = BatchEnsembleMLPBackbone(
        input_dim=1,
        output_dim=16,
        ensemble_size=ensemble_size,
        hidden_dims=[32],
    )

    # Define the PackedEnsembleRegressor over the backbone
    packed_ensemble = PackedEnsembleRegressor(
        backbone=backbone,
        feature_dim=16,
        output_dim=1,
        ensemble_size=ensemble_size,
        heteroscedastic=True,
        alpha=1.2,  # multiplier for fast weight scaling
    )

    # Let's perform a forward pass (outputs mean, member_means, and uncertainty components)
    packed_output = packed_ensemble.predict_output(x_test)

    print("PackedEnsemble Output Shapes:")
    print("  Mean Shape               :", packed_output.mean.shape)
    print("  Member Means Shape       :", packed_output.member_means.shape)
    print("  Epistemic Variance Shape :", packed_output.epistemic_variance.shape)
    print("  Aleatoric Variance Shape :", packed_output.aleatoric_variance.shape)

    # --------------------------------------------------------------------------------
    # 4. Bayesian Model Averaging (BMA)
    # --------------------------------------------------------------------------------
    print("\n--- 3. Bayesian Model Averaging (BMA) Combiner ---")

    # Instantiate multiple candidate model architectures
    model1 = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    model2 = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
    model3 = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))

    bma = BayesianModelAveraging(models=[model1, model2, model3])

    # Get model weights (initial uniform probabilities)
    print("  Initial BMA Weights      :", bma.get_model_weights().detach().numpy())

    # We can optimize BMA weights directly using a dummy target or training loop
    dummy_x = torch.randn(10, 1)
    dummy_y = torch.randn(10, 1)
    optimizer_bma = torch.optim.Adam(bma.parameters(), lr=0.1)

    for _ in range(5):
        optimizer_bma.zero_grad()
        loss = nn.MSELoss()(bma(dummy_x), dummy_y)
        loss.backward()
        optimizer_bma.step()

    print("  Optimized BMA Weights    :", bma.get_model_weights().detach().numpy())

    # BMA prediction with uncertainty
    mean_bma, total_var_bma = bma.predict_with_uncertainty(x_test)
    print("  BMA Prediction Shape     :", mean_bma.shape)
    print("  BMA Variance Shape       :", total_var_bma.shape)

    print("================================================================================")
    print("                 BNN & BatchEnsemble Showcase completed!                        ")
    print("================================================================================")


if __name__ == "__main__":
    main()
