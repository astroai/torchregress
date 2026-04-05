"""
Comprehensive Loss Function Comparison for Photometric Redshift Estimation

This example trains and compares multiple models on the SDSS photo-z dataset using
a wide variety of loss functions from the torchregress library. It demonstrates how
to handle data with uncertainties and compares the performance of different approaches.
"""

import os
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

from torchregress.ensemble import BaseEnsembleModel, DeepEnsemble
from torchregress.losses import (
    BaseEIVLoss,
    FunctionalEIVLoss,
    QuantileLoss,
    TukeyBiweightLoss,
    create_loss_from_config,
)
from torchregress.metrics import (
    ContinuousRankedProbabilityScore,
    ExpectedCalibrationError,
    MeanAbsoluteError,
    MeanSquaredError,
    MedianAbsoluteError,
    NormalizedMedianAbsoluteDeviation,
    R2Score,
)
from torchregress.utils import GaussianNoise

# --- 1. Data Loading and Preparation ---

DATA_DIR = os.path.join("data", "sdss")


def download_sdss_data(force_download=False, sample_size=10000):
    os.makedirs(DATA_DIR, exist_ok=True)
    data_file = os.path.join(DATA_DIR, "sdss_photoz_real.csv")

    if os.path.exists(data_file) and not force_download:
        print("Loading existing SDSS data...")
        return pd.read_csv(data_file)

    print(f"Downloading {sample_size} galaxies from SDSS...")
    sql_query = f"""
    SELECT TOP {sample_size}
        p.objid, 
        p.modelMag_u as u, p.modelMagErr_u as u_err,
        p.modelMag_g as g, p.modelMagErr_g as g_err,
        p.modelMag_r as r, p.modelMagErr_r as r_err,
        p.modelMag_i as i, p.modelMagErr_i as i_err,
        p.modelMag_z as z_mag, p.modelMagErr_z as z_mag_err,
        s.z as spec_z, s.zErr as spec_z_err
    FROM PhotoObj AS p
    JOIN SpecObj AS s ON s.specobjid = p.specobjid
    WHERE 
        p.type = 3 AND s.class = 'GALAXY' AND s.zWarning = 0
        AND s.z BETWEEN 0.01 AND 0.8
        AND p.modelMag_u BETWEEN 10 AND 25 AND p.modelMag_g BETWEEN 10 AND 25
        AND p.modelMag_r BETWEEN 10 AND 25 AND p.modelMag_i BETWEEN 10 AND 25
        AND p.modelMag_z BETWEEN 10 AND 25
    ORDER BY NEWID()
    """
    try:
        params = {"cmd": sql_query, "format": "csv"}
        response = requests.post(
            "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch",
            data=params,
            timeout=120,
        )
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), comment="#")
        if len(df) == 0:
            raise ValueError("No data returned from SDSS API")
        df["u_g"] = df["u"] - df["g"]
        df["g_r"] = df["g"] - df["r"]
        df["r_i"] = df["r"] - df["i"]
        df["i_z"] = df["i"] - df["z_mag"]
        df["u_g_err"] = np.sqrt(df["u_err"] ** 2 + df["g_err"] ** 2)
        df["g_r_err"] = np.sqrt(df["g_err"] ** 2 + df["r_err"] ** 2)
        df["r_i_err"] = np.sqrt(df["r_err"] ** 2 + df["i_err"] ** 2)
        df["i_z_err"] = np.sqrt(df["i_err"] ** 2 + df["z_mag_err"] ** 2)
        df.to_csv(data_file, index=False)
        print(f"SDSS data saved to {data_file}")
        return df
    except Exception as e:
        print(f"Error downloading SDSS data: {e}")
        print("Falling back to simulated data...")
        return create_simulated_sdss_data(sample_size)


def create_simulated_sdss_data(n_galaxies=10000):
    """
    Create simulated SDSS-like data as a fallback when real data can't be downloaded.
    """
    print(f"Creating simulated SDSS data with {n_galaxies} galaxies...")

    np.random.seed(42)
    z_spec = np.random.lognormal(mean=-1.3, sigma=0.5, size=n_galaxies)
    z_spec = np.clip(z_spec, 0.01, 1.0)

    u = 20.0 + 2.0 * z_spec + np.random.normal(0, 0.15, n_galaxies)
    g = 19.0 + 1.8 * z_spec + np.random.normal(0, 0.08, n_galaxies)
    r = 18.5 + 1.6 * z_spec + np.random.normal(0, 0.06, n_galaxies)
    i = 18.0 + 1.4 * z_spec + np.random.normal(0, 0.05, n_galaxies)
    z = 17.5 + 1.2 * z_spec + np.random.normal(0, 0.07, n_galaxies)

    u_err = 0.01 + 0.05 * np.exp((u - 18) / 5)
    g_err = 0.01 + 0.03 * np.exp((g - 17) / 6)
    r_err = 0.01 + 0.02 * np.exp((r - 16) / 7)
    i_err = 0.01 + 0.02 * np.exp((i - 16) / 7)
    z_err = 0.01 + 0.03 * np.exp((z - 15) / 6)

    z_err_spec = 0.0001 + 0.0002 * z_spec

    df = pd.DataFrame(
        {
            "objid": np.arange(n_galaxies),
            "u": u,
            "u_err": u_err,
            "g": g,
            "g_err": g_err,
            "r": r,
            "r_err": r_err,
            "i": i,
            "i_err": i_err,
            "z_mag": z,
            "z_mag_err": z_err,
            "spec_z": z_spec,
            "spec_z_err": z_err_spec,
        }
    )

    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z_mag"]
    df["u_g_err"] = np.sqrt(df["u_err"] ** 2 + df["g_err"] ** 2)
    df["g_r_err"] = np.sqrt(df["g_err"] ** 2 + df["r_err"] ** 2)
    df["r_i_err"] = np.sqrt(df["r_err"] ** 2 + df["i_err"] ** 2)
    df["i_z_err"] = np.sqrt(df["i_err"] ** 2 + df["z_mag_err"] ** 2)

    os.makedirs(DATA_DIR, exist_ok=True)
    sim_file = os.path.join(DATA_DIR, "sdss_photoz_simulated.csv")
    df.to_csv(sim_file, index=False)
    print(f"Simulated SDSS data saved to {sim_file}")
    return df


class SDSSDataset(Dataset):
    def __init__(
        self,
        data,
        feature_cols,
        error_cols,
        target_col="spec_z",
        target_error_col="spec_z_err",
    ):
        self.data = data
        self.feature_cols = feature_cols
        self.error_cols = error_cols
        self.target_col = target_col
        self.target_error_col = target_error_col

        self.feature_scaler = StandardScaler()
        self.features = self.feature_scaler.fit_transform(data[feature_cols].values)
        self.feature_errors = data[error_cols].values / np.sqrt(self.feature_scaler.var_)
        self.targets = data[target_col].values.reshape(-1, 1)
        self.target_errors = data[target_error_col].values.reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.feature_errors[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.target_errors[idx], dtype=torch.float32),
        )


# --- 2. Model Definitions ---


class PhotoZMLP(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.network(x)


# --- 3. Training and Evaluation ---


def train_model(
    model,
    loss_fn,
    train_loader,
    epochs=50,
    lr=0.001,
    device="cpu",
    use_augmentation=False,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    augmenter = GaussianNoise(std=0.1, probability=0.5) if use_augmentation else None

    for epoch in range(epochs):
        for x_batch, x_err_batch, y_batch, y_err_batch in train_loader:
            if use_augmentation:
                x_batch, _ = augmenter(x_batch)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if isinstance(loss_fn, BaseEIVLoss):
                loss = loss_fn(x_batch, y_batch)
            else:
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()


def evaluate_model(model, loader, device="cpu"):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_batch, _, y_batch, _ in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds.append(model(x_batch))
            targets.append(y_batch)
    if isinstance(model, BaseEnsembleModel):
        return torch.cat(preds, dim=1), torch.cat(targets, dim=0)
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


# --- 4. Main Execution ---


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sdss_data = download_sdss_data()
    if sdss_data is None:
        return

    feature_cols = ["u_g", "g_r", "r_i", "i_z"]
    error_cols = ["u_g_err", "g_r_err", "r_i_err", "i_z_err"]
    dataset = SDSSDataset(sdss_data, feature_cols, error_cols)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    train_rows = sdss_data.iloc[train_dataset.indices]
    sigma_x_std = np.sqrt(np.mean(train_rows[error_cols].to_numpy() ** 2, axis=0))
    sigma_y_std = float(np.sqrt(np.mean(train_rows["spec_z_err"].to_numpy() ** 2)))

    def build_functional_eiv():
        model = PhotoZMLP(len(feature_cols))
        loss = FunctionalEIVLoss(
            model=model,
            sigma_x=torch.tensor(sigma_x_std, dtype=torch.float32),
            sigma_y=sigma_y_std,
        )
        return model, loss

    loss_builders = {
        "MSE": lambda: (PhotoZMLP(len(feature_cols)), create_loss_from_config({"type": "mse"})),
        "GaussianNLL": lambda: (
            PhotoZMLP(len(feature_cols), 2),
            create_loss_from_config({"type": "gaussian_nll"}),
        ),
        "FunctionalEIV": build_functional_eiv,
        "TukeyBiweight": lambda: (PhotoZMLP(len(feature_cols)), TukeyBiweightLoss()),
        "Quantile (0.5)": lambda: (PhotoZMLP(len(feature_cols)), QuantileLoss(quantile=0.5)),
        "Ensemble (MSE)": lambda: (
            DeepEnsemble(base_model=PhotoZMLP(len(feature_cols)), ensemble_size=5),
            create_loss_from_config({"type": "mse"}),
        ),
    }

    results = {}
    for i, (name, build) in enumerate(loss_builders.items()):
        print(f"\n--- Training {name} ---")
        model, loss_fn = build()
        if name == "Ensemble (MSE)":
            for i, sub_model in enumerate(model.models):
                print(f"Training ensemble member {i + 1}/{model.ensemble_size}")
                train_model(sub_model, loss_fn, train_loader, device=device)
        else:
            train_model(model, loss_fn, train_loader, device=device)

        y_pred, y_true = evaluate_model(model, test_loader, device)

        # For distributional models, get samples
        samples = None
        if name == "GaussianNLL":
            mean, log_var = torch.chunk(y_pred, 2, dim=-1)
            var = torch.exp(log_var)
            torch.manual_seed(2025 + i)
            samples = torch.distributions.Normal(mean, var.sqrt()).sample((100,))
            y_pred = mean
        elif name == "Ensemble (MSE)":
            samples = y_pred
            y_pred = y_pred.mean(0)

        metrics = {
            "MSE": MeanSquaredError()(y_pred, y_true).item(),
            "MAE": MeanAbsoluteError()(y_pred, y_true).item(),
            "R2": R2Score()(y_pred, y_true).item(),
            "MedAE": MedianAbsoluteError()(y_pred, y_true).item(),
            "NMAD": NormalizedMedianAbsoluteDeviation()(y_pred, y_true).item(),
        }
        if samples is not None:
            quantiles = {q: torch.quantile(samples, q, dim=0) for q in np.arange(0.1, 1.0, 0.1)}
            metrics["CRPS"] = ContinuousRankedProbabilityScore()(quantiles, y_true).item()
            ece_quantiles = {
                q: torch.quantile(samples, q, dim=0) for q in np.arange(0.05, 1.0, 0.05)
            }
            ece = ExpectedCalibrationError()(ece_quantiles, y_true)
            metrics["ECE"] = ece["mean_absolute_calibration_error"].item()

        results[name] = {"metrics": metrics, "model": model}

    # --- 5. Results ---
    results_df = pd.DataFrame({k: v["metrics"] for k, v in results.items()}).T
    print("\n--- Metric Comparison ---")
    print(results_df)

    # Visualization
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 1, figsize=(8, 6 * n_models), sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]

    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        y_pred, y_true = evaluate_model(result["model"], test_loader, device)
        if name == "Ensemble (MSE)":
            y_pred = y_pred.mean(0)
        elif name == "GaussianNLL":
            y_pred = torch.chunk(y_pred, 2, dim=-1)[0]

        ax.scatter(y_true.cpu(), y_pred.cpu(), alpha=0.1)
        ax.plot([0, 0.8], [0, 0.8], "k--")
        ax.set_xlabel("Spectroscopic Redshift")
        ax.set_ylabel("Photometric Redshift")
        ax.set_title(f"{name} (NMAD: {result['metrics']['NMAD']:.3f})")

    plt.tight_layout()
    plt.savefig("photoz_loss_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
