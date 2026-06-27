"""
Errors-in-Variables (EIV) Input Noise Correction Algorithms Demo.

This example demonstrates how to use the Regression Calibration (RC) and Simulation
Extrapolation (SIMEX) algorithms to correct for measurement error/noise in input features
(also known as Errors-in-Variables).

We compare:
1. Naive Baseline: Ignores input measurement error.
2. Regression Calibration (RC): A data-level preprocessing calibration that projects noisy
   inputs to their expected true values E[X | W] under a Gaussian assumption.
3. Simulation Extrapolation (SIMEX): A simulation-based training method that adds simulated
   noise of varying magnitudes, fits a trend curve, and extrapolates back to the zero-noise limit.

Seminal references:
1. Carroll, R. J., Ruppert, D., Crainiceanu, C. M., & Stefanski, L. A. (2006). Measurement Error
   in Nonlinear Models: A Modern Perspective. Chapman & Hall/CRC.
2. Cook, J. R., & Stefanski, L. A. (1994). Simulation-Extrapolation Estimation in Parametric
   Measurement Error Models. Journal of the American Statistical Association.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchregress.algorithms import SIMEX, LatentNN, RegressionCalibration


def generate_eiv_data(n_samples=800, sigma_u=0.4, seed=42):
    """Generate synthetic data with input measurement error (Errors-in-Variables)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # True latent features X
    x_true = np.random.uniform(-2.0, 2.0, size=(n_samples, 1)).astype(np.float32)

    # True relationship: y is a quadratic function of x
    y_true = (0.5 * x_true**2 + 0.8 * x_true).astype(np.float32)
    y_noise = np.random.normal(0, 0.1, size=y_true.shape).astype(np.float32)
    y = y_true + y_noise

    # Observed noisy features W = X + U
    u_noise = np.random.normal(0, sigma_u, size=x_true.shape).astype(np.float32)
    w_obs = x_true + u_noise

    return (
        torch.from_numpy(w_obs),
        torch.from_numpy(y),
        torch.from_numpy(x_true),
        torch.from_numpy(y_true),
    )


class Regressor(nn.Module):
    """Simple regression model."""

    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_baseline_model(x_train, y_train, epochs=100):
    """Train a standard regressor without correction."""
    model = Regressor()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(bx), by)
            loss.backward()
            optimizer.step()

    return model


def simex_train_wrapper(model, x, y, epochs=80):
    """Train wrapper used by SIMEX to update models on simulated noisy data."""
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(bx), by)
            loss.backward()
            optimizer.step()
    return model


def main():
    print("=" * 60)
    print("Errors-in-Variables (EIV) Input Noise Correction Demo")
    print("=" * 60)

    sigma_u = 0.45  # Standard deviation of input measurement error
    w, y, x_true, y_true = generate_eiv_data(n_samples=1000, sigma_u=sigma_u)

    # Train / Test split
    n_train = 600
    w_train, w_test = w[:n_train], w[n_train:]
    y_train, _y_test = y[:n_train], y[n_train:]
    x_true[n_train:]
    y_test_true = y_true[n_train:]

    # 1. Naive Baseline (ignores measurement error)
    print("Training Naive Baseline Model...")
    baseline_model = train_baseline_model(w_train, y_train)
    baseline_model.eval()
    with torch.no_grad():
        pred_naive = baseline_model(w_test)
        mae_naive = (pred_naive - y_test_true).abs().mean().item()

    # 2. Regression Calibration (RC)
    print("Fitting and Applying Regression Calibration...")
    rc = RegressionCalibration(sigma_u=sigma_u)
    # Fit on training observations, and transform both train and test observations
    w_train_cal = rc.fit(w_train).transform(w_train)
    w_test_cal = rc.transform(w_test)

    # Train model on calibrated features
    rc_model = train_baseline_model(w_train_cal, y_train)
    rc_model.eval()
    with torch.no_grad():
        pred_rc = rc_model(w_test_cal)
        mae_rc = (pred_rc - y_test_true).abs().mean().item()

    # 3. Simulation Extrapolation (SIMEX)
    print("Training and Extrapolating with SIMEX...")

    # Define model factory and training function wrapper for SIMEX
    def model_factory():
        return Regressor()

    simex = SIMEX(
        model_factory=model_factory,
        train_func=simex_train_wrapper,
        sigma_u=sigma_u,
        lambdas=[0.5, 1.0, 1.5, 2.0],
        n_simulations=3,
        extrapolation_order=2,
    )
    # Fits models across simulated error scales
    simex.fit(w_train, y_train)

    # Predicts using extrapolation back to lambda=-1 (zero measurement error)
    pred_simex = simex.predict(w_test)
    mae_simex = (pred_simex - y_test_true).abs().mean().item()

    # 4. Latent Input Neural Network (LatentNN)
    print("Training Jointly with LatentNN...")
    latent_nn = LatentNN(
        model_factory=model_factory,
        sigma_x=sigma_u,
        epochs=100,
        lr=0.01,
        batch_size=32,
    )
    # Fit the network and latent clean features jointly
    latent_nn.fit(w_train, y_train)

    # Predict on observations (LatentNN maps test-time inputs through the fitted model)
    pred_latent = latent_nn.predict(w_test)
    mae_latent = (pred_latent - y_test_true).abs().mean().item()

    print("\n--- Evaluation on True (Uncorrupted) Inputs ---")
    print(f"Naive Baseline MAE: {mae_naive:.4f}")
    print(f"Regression Calibr. MAE: {mae_rc:.4f}")
    print(f"SIMEX Extrapolation MAE: {mae_simex:.4f}")
    print(f"LatentNN Correction MAE: {mae_latent:.4f}")

    print("\nObservation:")
    print("1. Naive model suffers from attenuation bias (under-predicts high magnitude targets).")
    print("2. RC projects features to conditional expectations first, correcting the bias.")
    print(
        "3. SIMEX profiles the error's impact via simulation to extrapolate a corrected prediction."
    )
    print(
        "4. LatentNN jointly optimizes clean latent inputs and model weights with a quadratic penalty."
    )


if __name__ == "__main__":
    main()
