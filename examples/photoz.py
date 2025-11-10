"""
Photometric redshift estimation using SDSS data.

This example demonstrates using torchregress's error-in-variables (EIV) losses
for photometric redshift estimation from SDSS ugriz photometry, where both the
features (magnitudes/colors) and targets (spectroscopic redshifts) have measurement errors.
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

from torchregress.losses.eiv import EIVRegressionLoss, OrthogonalEIVLoss, WeightedEIVLoss

# Import torchregress losses and metrics
from torchregress.losses.gaussian import MSELoss
from torchregress.metrics.calibration import bias
from torchregress.metrics.point import mae, rmse

# Constants
DATA_DIR = os.path.join("data", "sdss")
BATCH_SIZE = 64
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SDSS SkyServer API URL
SDSS_API_URL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"


def download_sdss_data(force_download=False, sample_size=10000):
    """
    Download real SDSS photometric and spectroscopic data using the SDSS SkyServer API.

    Args:
        force_download: If True, force a new download even if file exists
        sample_size: Maximum number of galaxies to download (default: 10000)

    Returns:
        DataFrame containing SDSS data
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    data_file = os.path.join(DATA_DIR, "sdss_photoz_real.csv")

    # Return existing data if file exists and not forcing download
    if os.path.exists(data_file) and not force_download:
        print("Real SDSS data already exists. Loading from file...")
        return pd.read_csv(data_file)

    print(f"Downloading {sample_size} galaxies from SDSS database...")

    # SQL query to get galaxy data with clean spectra
    # We're selecting galaxies with good photometry and spectroscopy
    # and retrieving model magnitudes, errors, and spectroscopic redshift
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
        p.type = 3                     -- galaxies only
        AND s.class = 'GALAXY'         -- spectroscopically confirmed galaxies
        AND s.zWarning = 0             -- good redshift
        AND s.z BETWEEN 0.01 AND 0.8   -- reasonable redshift range
        AND p.modelMag_u BETWEEN 10 AND 25
        AND p.modelMag_g BETWEEN 10 AND 25
        AND p.modelMag_r BETWEEN 10 AND 25
        AND p.modelMag_i BETWEEN 10 AND 25
        AND p.modelMag_z BETWEEN 10 AND 25
    ORDER BY NEWID()                   -- random order
    """

    try:
        # Make request to SDSS API
        print("Sending query to SDSS SkyServer...")
        params = {"cmd": sql_query, "format": "csv"}

        response = requests.post(SDSS_API_URL, data=params, timeout=120)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Process response
        data = StringIO(response.text)
        df = pd.read_csv(data, comment="#")

        # Check if we got any data
        if len(df) == 0:
            raise ValueError("No data returned from SDSS API")

        print(f"Received {len(df)} galaxies from SDSS")

        # Add colors (which are often better features for photo-z estimation)
        df["u_g"] = df["u"] - df["g"]
        df["g_r"] = df["g"] - df["r"]
        df["r_i"] = df["r"] - df["i"]
        df["i_z"] = df["i"] - df["z_mag"]

        # Calculate color errors using error propagation
        df["u_g_err"] = np.sqrt(df["u_err"] ** 2 + df["g_err"] ** 2)
        df["g_r_err"] = np.sqrt(df["g_err"] ** 2 + df["r_err"] ** 2)
        df["r_i_err"] = np.sqrt(df["r_err"] ** 2 + df["i_err"] ** 2)
        df["i_z_err"] = np.sqrt(df["i_err"] ** 2 + df["z_mag_err"] ** 2)

        # Save the data
        df.to_csv(data_file, index=False)
        print(f"Real SDSS data with {len(df)} galaxies saved to {data_file}")
        return df

    except Exception as e:
        print(f"Error downloading data from SDSS API: {e}")
        print("Falling back to simulated data...")
        return create_simulated_sdss_data()


def create_simulated_sdss_data(n_galaxies=5000):
    """
    Create simulated SDSS-like data as a fallback when real data can't be downloaded.
    """
    print(f"Creating simulated SDSS data with {n_galaxies} galaxies...")

    # Generate redshifts with a realistic distribution
    np.random.seed(42)
    z_spec = np.random.lognormal(mean=-1.3, sigma=0.5, size=n_galaxies)
    z_spec = np.clip(z_spec, 0.01, 1.0)  # Clip to a reasonable range

    # Simulate ugriz magnitudes based on redshift
    # Using a simplified model where magnitudes depend on redshift
    u = 20.0 + 2.0 * z_spec + np.random.normal(0, 0.15, n_galaxies)
    g = 19.0 + 1.8 * z_spec + np.random.normal(0, 0.08, n_galaxies)
    r = 18.5 + 1.6 * z_spec + np.random.normal(0, 0.06, n_galaxies)
    i = 18.0 + 1.4 * z_spec + np.random.normal(0, 0.05, n_galaxies)
    z = 17.5 + 1.2 * z_spec + np.random.normal(0, 0.07, n_galaxies)

    # Generate errors for each band (typically errors increase with magnitude)
    u_err = 0.01 + 0.05 * np.exp((u - 18) / 5)
    g_err = 0.01 + 0.03 * np.exp((g - 17) / 6)
    r_err = 0.01 + 0.02 * np.exp((r - 16) / 7)
    i_err = 0.01 + 0.02 * np.exp((i - 16) / 7)
    z_err = 0.01 + 0.03 * np.exp((z - 15) / 6)

    # Spectroscopic redshift errors (typically much smaller)
    z_err = 0.0001 + 0.0002 * z_spec

    # Create a DataFrame
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
            "spec_z_err": z_err,
        }
    )

    # Add colors (which are often better features for photo-z estimation)
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z_mag"]

    # Calculate color errors using error propagation
    df["u_g_err"] = np.sqrt(df["u_err"] ** 2 + df["g_err"] ** 2)
    df["g_r_err"] = np.sqrt(df["g_err"] ** 2 + df["r_err"] ** 2)
    df["r_i_err"] = np.sqrt(df["r_err"] ** 2 + df["i_err"] ** 2)
    df["i_z_err"] = np.sqrt(df["i_err"] ** 2 + df["z_mag_err"] ** 2)

    # Save the simulated data
    os.makedirs(DATA_DIR, exist_ok=True)
    sim_file = os.path.join(DATA_DIR, "sdss_photoz_simulated.csv")
    df.to_csv(sim_file, index=False)
    print(f"Simulated SDSS data saved to {sim_file}")

    return df


class SDSSDataset(Dataset):
    """SDSS photometric dataset for redshift estimation."""

    def __init__(
        self, data, feature_cols, error_cols, target_col="spec_z", target_error_col="spec_z_err"
    ):
        """
        Initialize the dataset.

        Args:
            data: Pandas DataFrame with SDSS data
            feature_cols: List of column names to use as features
            error_cols: List of column names with feature errors
            target_col: Column name for the target variable (spectroscopic redshift)
            target_error_col: Column name for target errors
        """
        self.data = data
        self.feature_cols = feature_cols
        self.error_cols = error_cols
        self.target_col = target_col
        self.target_error_col = target_error_col

        # Scale features
        self.feature_scaler = StandardScaler()
        self.features = self.feature_scaler.fit_transform(data[feature_cols].values)

        # Scale feature errors (using the same scaling factors as features)
        self.feature_errors = (
            data[error_cols].values / np.sqrt(self.feature_scaler.var_)[:, np.newaxis].T
        )

        # Get targets and target errors
        self.targets = data[target_col].values.reshape(-1, 1)
        self.target_errors = data[target_error_col].values.reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        x_err = torch.tensor(self.feature_errors[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        y_err = torch.tensor(self.target_errors[idx], dtype=torch.float32)

        return x, x_err, y, y_err


class PhotoZModel(nn.Module):
    """Neural network for photometric redshift estimation."""

    def __init__(self, input_dim):
        super().__init__()

        # Simple MLP with dropout for regularization
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=50, device=DEVICE):
    """Train the photometric redshift model."""
    model.to(device)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for x, x_err, y, y_err in train_loader:
            x, x_err = x.to(device), x_err.to(device)
            y, y_err = y.to(device), y_err.to(device)

            optimizer.zero_grad()
            outputs = model(x)

            # Different losses handle errors differently
            if isinstance(loss_fn, (EIVRegressionLoss, OrthogonalEIVLoss, WeightedEIVLoss)):
                # EIV losses use both input and output errors
                loss = loss_fn(outputs, y, x_err=x_err, y_err=y_err)
            else:
                # Standard losses just use predictions and targets
                loss = loss_fn(outputs, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for x, x_err, y, y_err in val_loader:
                x, x_err = x.to(device), x_err.to(device)
                y, y_err = y.to(device), y_err.to(device)

                outputs = model(x)

                if isinstance(loss_fn, (EIVRegressionLoss, OrthogonalEIVLoss, WeightedEIVLoss)):
                    loss = loss_fn(outputs, y, x_err=x_err, y_err=y_err)
                else:
                    loss = loss_fn(outputs, y)

                val_running_loss += loss.item() * x.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}"
        )

        # Save the best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss

    return train_losses, val_losses


def evaluate_model(model, test_loader, device=DEVICE):
    """Evaluate the model on the test set."""
    model.eval()
    all_preds = []
    all_targets = []
    all_target_errors = []

    with torch.no_grad():
        for x, x_err, y, y_err in test_loader:
            x = x.to(device)
            outputs = model(x)

            all_preds.append(outputs.cpu())
            all_targets.append(y)
            all_target_errors.append(y_err)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_target_errors = torch.cat(all_target_errors, dim=0)

    # Calculate metrics
    rmse_value = rmse(all_preds, all_targets)
    mae_value = mae(all_preds, all_targets)
    bias_value = bias(all_preds, all_targets)

    # Calculate normalized metrics (important for photo-z)
    delta_z = all_preds - all_targets
    delta_z_norm = delta_z / (1 + all_targets)
    nmad = 1.48 * torch.median(torch.abs(delta_z_norm - torch.median(delta_z_norm)))

    return {
        "rmse": rmse_value.item(),
        "mae": mae_value.item(),
        "bias": bias_value.item(),
        "nmad": nmad.item(),
        "predictions": all_preds.numpy(),
        "targets": all_targets.numpy(),
        "target_errors": all_target_errors.numpy(),
    }


def plot_results(results, loss_names):
    """Plot comparison of different models."""
    plt.figure(figsize=(20, 16))

    # Plot training and validation losses
    plt.subplot(2, 2, 1)
    for name in loss_names:
        plt.plot(results[name]["train_losses"], label=f"{name} (train)")
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    for name in loss_names:
        plt.plot(results[name]["val_losses"], label=f"{name} (val)")
    plt.title("Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot photometric vs spectroscopic redshift for each model
    plt.subplot(2, 2, 3)

    # Plot identity line first
    max_z = 0
    for name in loss_names:
        metrics = results[name]["metrics"]
        max_z = max(max_z, np.max(metrics["targets"]))

    z_range = np.linspace(0, max_z, 100)
    plt.plot(z_range, z_range, "k--", label="Identity")

    # Plot results for each model
    for i, name in enumerate(loss_names):
        metrics = results[name]["metrics"]

        # Plot a random subset of points for clarity
        idx = np.random.choice(
            len(metrics["predictions"]), size=min(500, len(metrics["predictions"])), replace=False
        )

        plt.scatter(
            metrics["targets"][idx, 0],
            metrics["predictions"][idx, 0],
            alpha=0.5,
            label=f"{name} (NMAD: {metrics['nmad']:.4f})",
        )

    plt.title("Photometric vs Spectroscopic Redshift")
    plt.xlabel("Spec-z")
    plt.ylabel("Photo-z")
    plt.legend()

    # Plot error distribution
    plt.subplot(2, 2, 4)
    for name in loss_names:
        metrics = results[name]["metrics"]
        delta_z = metrics["predictions"] - metrics["targets"]
        delta_z_norm = delta_z / (1 + metrics["targets"])

        plt.hist(
            delta_z_norm.flatten(),
            bins=50,
            alpha=0.5,
            density=True,
            label=f"{name} (bias: {metrics['bias']:.4f})",
        )

    plt.title("Normalized Error Distribution")
    plt.xlabel("(Photo-z - Spec-z) / (1 + Spec-z)")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.savefig("photoz_comparison.png")
    plt.show()


def main():
    print(f"Using device: {DEVICE}")

    # Download real SDSS data (falls back to simulated if API fails)
    try:
        sdss_data = download_sdss_data(sample_size=5000)
    except Exception as e:
        print(f"Error obtaining SDSS data: {e}")
        return

    print(f"Loaded SDSS dataset with {len(sdss_data)} galaxies")

    # Print some basic statistics about the dataset
    print("\nDataset Statistics:")
    print(f"Magnitude range (r-band): {sdss_data['r'].min():.2f} - {sdss_data['r'].max():.2f}")
    print(f"Redshift range: {sdss_data['spec_z'].min():.3f} - {sdss_data['spec_z'].max():.3f}")
    print(f"Median r-band error: {sdss_data['r_err'].median():.4f}")
    print(f"Median spec-z error: {sdss_data['spec_z_err'].median():.6f}")

    # Show redshift distribution
    plt.figure(figsize=(10, 6))
    plt.hist(sdss_data["spec_z"], bins=50)
    plt.title("Spectroscopic Redshift Distribution")
    plt.xlabel("Redshift")
    plt.ylabel("Count")
    plt.savefig(os.path.join(DATA_DIR, "sdss_redshift_distribution.png"))

    # Define features and create dataset
    # We'll compare using magnitudes vs colors as input features
    feature_cols = ["u_g", "g_r", "r_i", "i_z"]  # Using colors as features
    error_cols = ["u_g_err", "g_r_err", "r_i_err", "i_z_err"]

    dataset = SDSSDataset(
        sdss_data,
        feature_cols=feature_cols,
        error_cols=error_cols,
        target_col="spec_z",
        target_error_col="spec_z_err",
    )

    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Define loss functions to compare
    loss_functions = {
        "MSE": MSELoss(reduction="mean"),  # Standard loss as baseline
        "EIV": EIVRegressionLoss(reduction="mean"),  # Basic EIV loss
        "OrthogonalEIV": OrthogonalEIVLoss(reduction="mean"),  # Orthogonal errors
        "WeightedEIV": WeightedEIVLoss(reduction="mean"),  # Weighted by uncertainties
    }

    # Dictionary to store results
    results = {}

    # Train models with different loss functions
    for loss_name, loss_fn in loss_functions.items():
        print(f"\n=== Training with {loss_name} Loss ===")

        # Initialize model
        model = PhotoZModel(input_dim=len(feature_cols))

        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Train model
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, loss_fn, optimizer, NUM_EPOCHS, DEVICE
        )

        # Evaluate model
        metrics = evaluate_model(model, test_loader, DEVICE)

        # Store results
        results[loss_name] = {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "metrics": metrics,
        }

    # Plot and compare results
    plot_results(results, loss_functions.keys())

    # Print results table
    print("\nPhotometric Redshift Estimation Results:")
    print("-" * 70)
    print(f"{'Loss Function':<15} {'MAE':<10} {'RMSE':<10} {'Bias':<10} {'NMAD':<10}")
    print("-" * 70)
    for loss_name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{loss_name:<15} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} "
            f"{metrics['bias']:<10.4f} {metrics['nmad']:<10.4f}"
        )

    print("\nNote: NMAD (Normalized Median Absolute Deviation) is a common metric in astronomy")
    print("for assessing photometric redshift accuracy. Lower values indicate better performance.")
    print("\nThis example highlights the value of error-in-variables losses for photometric")
    print("redshift estimation using real SDSS data with measurement uncertainties.")


if __name__ == "__main__":
    main()
