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

# Import torchregress losses and metrics
from torchregress.losses import (
    GaussianNLLLoss,
    HuberLoss,
    MSELoss,
    WeightedMAELoss,
)
from torchregress.losses.eiv import (
    BaseEIVLoss,
    EnsembleEIVLoss,
    FunctionalEIVLoss,
    OrthogonalDistanceRegressionLoss,
)
from torchregress.metrics.calibration import bias, calibration_metrics_report
from torchregress.metrics.interval import prediction_interval_coverage_probability
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

    The simulation properly models:
    - True underlying magnitudes that depend on redshift
    - Measurement noise added to observed magnitudes (with realistic heteroscedastic errors)
    - EIV scenario: model sees noisy observations, must predict true spec_z
    """
    print(f"Creating simulated SDSS data with {n_galaxies} galaxies...")

    # Generate redshifts with a realistic distribution
    np.random.seed(42)
    z_spec = np.random.lognormal(mean=-1.3, sigma=0.5, size=n_galaxies)
    z_spec = np.clip(z_spec, 0.01, 1.0)  # Clip to a reasonable range

    # Generate TRUE magnitudes based on redshift (intrinsic galaxy properties)
    # These have some intrinsic scatter representing galaxy diversity
    u_true = 20.0 + 2.0 * z_spec + np.random.normal(0, 0.10, n_galaxies)
    g_true = 19.0 + 1.8 * z_spec + np.random.normal(0, 0.06, n_galaxies)
    r_true = 18.5 + 1.6 * z_spec + np.random.normal(0, 0.04, n_galaxies)
    i_true = 18.0 + 1.4 * z_spec + np.random.normal(0, 0.03, n_galaxies)
    z_true = 17.5 + 1.2 * z_spec + np.random.normal(0, 0.05, n_galaxies)

    # Generate measurement errors (heteroscedastic - increase with magnitude/faintness)
    u_err = 0.02 + 0.08 * np.exp((u_true - 18) / 4)
    g_err = 0.015 + 0.05 * np.exp((g_true - 17) / 5)
    r_err = 0.01 + 0.03 * np.exp((r_true - 16) / 6)
    i_err = 0.01 + 0.03 * np.exp((i_true - 16) / 6)
    z_mag_err = 0.015 + 0.04 * np.exp((z_true - 15) / 5)

    # Add measurement noise to create OBSERVED magnitudes
    # This is what the telescope actually measures
    u = u_true + np.random.normal(0, 1, n_galaxies) * u_err
    g = g_true + np.random.normal(0, 1, n_galaxies) * g_err
    r = r_true + np.random.normal(0, 1, n_galaxies) * r_err
    i = i_true + np.random.normal(0, 1, n_galaxies) * i_err
    z = z_true + np.random.normal(0, 1, n_galaxies) * z_mag_err

    # Spectroscopic redshift errors (much smaller than photometric)
    # Typical SDSS spec-z errors are ~0.0001-0.001
    spec_z_err = 0.0005 + 0.001 * z_spec

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
            "z_mag_err": z_mag_err,
            "spec_z": z_spec,
            "spec_z_err": spec_z_err,
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


class HeteroscedasticPhotoZModel(nn.Module):
    """Neural network that outputs both mean and log-variance for uncertainty estimation."""

    def __init__(self, input_dim):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Separate heads for mean and log-variance
        self.mean_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        features = self.shared(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return mean, logvar


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

            if isinstance(loss_fn, BaseEIVLoss):
                # EIV losses use observed inputs and handle the model internally
                loss = loss_fn(x, y)
            else:
                outputs = model(x)
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

                if isinstance(loss_fn, BaseEIVLoss):
                    loss = loss_fn(x, y)
                else:
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                val_running_loss += loss.item() * x.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)

        message = (
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_epoch_loss:.4f}"
        )
        print(message)

        # Save the best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss

    return train_losses, val_losses


def gaussian_crps(mean, std, target):
    """
    Compute CRPS for Gaussian predictive distribution.

    CRPS(N(μ,σ²), y) = σ * [z*(2*Φ(z) - 1) + 2*φ(z) - 1/√π]
    where z = (y - μ) / σ

    Args:
        mean: Predicted mean
        std: Predicted standard deviation
        target: True values

    Returns:
        Mean CRPS over all samples
    """
    z = (target - mean) / (std + 1e-8)
    # Standard normal PDF and CDF
    phi = torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
    crps = std * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return torch.mean(crps)


def evaluate_model(model, test_loader, device=DEVICE, is_heteroscedastic=False):
    """Evaluate the model on the test set.

    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to use
        is_heteroscedastic: If True, model outputs (mean, logvar) tuple
    """
    model.eval()
    all_means = []
    all_stds = []
    all_targets = []
    all_target_errors = []

    with torch.no_grad():
        for x, x_err, y, y_err in test_loader:
            x = x.to(device)
            outputs = model(x)

            if is_heteroscedastic:
                mean, logvar = outputs
                std = torch.exp(0.5 * logvar)
                all_means.append(mean.cpu())
                all_stds.append(std.cpu())
            else:
                all_means.append(outputs.cpu())
                # For non-heteroscedastic models, estimate std from residuals
                all_stds.append(None)

            all_targets.append(y)
            all_target_errors.append(y_err)

    all_means = torch.cat(all_means, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_target_errors = torch.cat(all_target_errors, dim=0)

    if is_heteroscedastic:
        all_stds = torch.cat(all_stds, dim=0)
    else:
        # For point prediction models, use residual std as uncertainty estimate
        residuals = all_means - all_targets
        all_stds = torch.full_like(all_means, residuals.std().item())

    # Calculate point prediction metrics
    rmse_value = rmse(all_means, all_targets)
    mae_value = mae(all_means, all_targets)
    bias_value = bias(all_means, all_targets)

    # Calculate normalized metrics (important for photo-z)
    delta_z = all_means - all_targets
    delta_z_norm = delta_z / (1 + all_targets)
    nmad = 1.48 * torch.median(torch.abs(delta_z_norm - torch.median(delta_z_norm)))

    # Calculate probabilistic metrics
    # CRPS (Continuous Ranked Probability Score) - lower is better
    crps = gaussian_crps(
        all_means.squeeze(), all_stds.squeeze(), all_targets.squeeze()
    )

    # Prediction interval coverage (95% interval)
    lower_95 = all_means - 1.96 * all_stds
    upper_95 = all_means + 1.96 * all_stds
    picp_95 = prediction_interval_coverage_probability(lower_95, upper_95, all_targets)

    # Mean prediction interval width
    mpiw_95 = torch.mean(upper_95 - lower_95)

    # Calibration error using marginal calibration
    try:
        calib_report = calibration_metrics_report(
            {"mean": all_means.squeeze(), "std": all_stds.squeeze()},
            all_targets.squeeze(),
            n_samples=100,
        )
        mce = float(calib_report["marginal_calibration_error"])
    except Exception:
        mce = float("nan")  # May fail if sampling fails

    return {
        "rmse": float(rmse_value),
        "mae": float(mae_value),
        "bias": float(bias_value),
        "nmad": nmad.item(),
        "crps": float(crps),
        "picp_95": float(picp_95),
        "mpiw_95": float(mpiw_95),
        "mce": mce,
        "predictions": all_means.numpy(),
        "pred_stds": all_stds.numpy(),
        "targets": all_targets.numpy(),
        "target_errors": all_target_errors.numpy(),
    }


def plot_results(results, loss_names):
    """Plot comparison of different models."""
    plt.figure(figsize=(20, 16))

    # Plot each loss type on its own scale using separate line styles
    # Different loss functions measure different things:
    # - MSE/MAE/Huber/EnsembleEIV: error-based (~0.01-0.1)
    # - GaussianNLL/FunctionalEIV: log-likelihood (can be negative)
    # - OrthogonalEIV: Mahalanobis distance (~10-100)
    # We use twin axes to show error-based and likelihood-based losses together

    ax1 = plt.subplot(2, 2, 1)
    ax2 = ax1.twinx()

    # Group losses by scale
    error_based = ["MSE", "MAE", "Huber", "EnsembleEIV"]
    likelihood_based = ["GaussianNLL", "FunctionalEIV", "OrthogonalEIV"]

    # Color palette
    colors = {
        "MSE": "blue",
        "MAE": "cyan",
        "Huber": "navy",
        "GaussianNLL": "orange",
        "FunctionalEIV": "red",
        "OrthogonalEIV": "darkred",
        "EnsembleEIV": "green",
    }

    for name in loss_names:
        if name not in results:
            continue
        losses = np.array(results[name]["train_losses"])
        if name in error_based:
            ax1.plot(losses, label=f"{name}", color=colors.get(name, "gray"), linewidth=2)
        else:
            ax2.plot(losses, label=f"{name}", color=colors.get(name, "gray"), linewidth=2, linestyle="--")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error-based Loss", color="blue")
    ax2.set_ylabel("Likelihood/Distance Loss", color="orange")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.set_title("Training Loss (dual y-axes for different scales)")
    ax1.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 2)
    ax4 = ax3.twinx()

    for name in loss_names:
        if name not in results:
            continue
        losses = np.array(results[name]["val_losses"])
        if name in error_based:
            ax3.plot(losses, label=f"{name}", color=colors.get(name, "gray"), linewidth=2)
        else:
            ax4.plot(losses, label=f"{name}", color=colors.get(name, "gray"), linewidth=2, linestyle="--")

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Error-based Loss", color="blue")
    ax4.set_ylabel("Likelihood/Distance Loss", color="orange")
    ax3.tick_params(axis="y", labelcolor="blue")
    ax4.tick_params(axis="y", labelcolor="orange")

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="upper right", fontsize=8)
    ax3.set_title("Validation Loss (dual y-axes for different scales)")
    ax3.grid(True, alpha=0.3)

    # Plot photometric vs spectroscopic redshift for selected models
    plt.subplot(2, 2, 3)

    # Plot identity line first
    max_z = 0
    for name in loss_names:
        if name not in results:
            continue
        metrics = results[name]["metrics"]
        max_z = max(max_z, np.max(metrics["targets"]))

    z_range = np.linspace(0, max_z, 100)
    plt.plot(z_range, z_range, "k--", label="Identity", linewidth=2)

    # Only plot a subset of models to avoid clutter
    models_to_plot = ["MSE", "GaussianNLL", "Huber", "EnsembleEIV"]
    plot_colors = ["blue", "orange", "green", "red"]

    for name, color in zip(models_to_plot, plot_colors):
        if name not in results:
            continue
        metrics = results[name]["metrics"]

        # Plot a random subset of points for clarity
        idx = np.random.choice(
            len(metrics["predictions"]), size=min(200, len(metrics["predictions"])), replace=False
        )

        plt.scatter(
            metrics["targets"][idx, 0],
            metrics["predictions"][idx, 0],
            alpha=0.5,
            color=color,
            label=f"{name} (NMAD: {metrics['nmad']:.4f})",
            s=20,
        )

    plt.title("Photometric vs Spectroscopic Redshift")
    plt.xlabel("Spec-z")
    plt.ylabel("Photo-z")
    plt.legend(fontsize=8)

    # Plot error distribution for selected models
    plt.subplot(2, 2, 4)
    for name, color in zip(models_to_plot, plot_colors):
        if name not in results:
            continue
        metrics = results[name]["metrics"]
        delta_z = metrics["predictions"] - metrics["targets"]
        delta_z_norm = delta_z / (1 + metrics["targets"])

        plt.hist(
            delta_z_norm.flatten(),
            bins=50,
            alpha=0.4,
            density=True,
            color=color,
            label=f"{name} (bias: {metrics['bias']:.4f})",
        )

    plt.title("Normalized Error Distribution")
    plt.xlabel("(Photo-z - Spec-z) / (1 + Spec-z)")
    plt.ylabel("Density")
    plt.legend(fontsize=8)
    plt.xlim(-0.3, 0.3)

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

    # Estimate global measurement error levels from training data
    # IMPORTANT: Since features are standardized by StandardScaler, we must also
    # scale the errors accordingly. The EIV losses expect sigma in the same space
    # as the model inputs.
    train_rows = sdss_data.iloc[train_dataset.indices]
    raw_errors = train_rows[error_cols].to_numpy()
    feature_stds = np.sqrt(dataset.feature_scaler.var_)
    scaled_errors = raw_errors / feature_stds  # Scale errors same as features
    sigma_x_std = np.sqrt(np.mean(scaled_errors ** 2, axis=0))

    # For sigma_y, we need to estimate the TOTAL irreducible uncertainty, which includes:
    # 1. Spectroscopic measurement error (~0.001 for simulated data)
    # 2. Intrinsic scatter in the photo-z relationship not explained by colors
    #
    # The EIV formulation assumes: y_obs = f(x_true) + noise_y
    # where noise_y has std = sigma_y. If sigma_y is too small, the EIV loss
    # penalizes any model error as if it were from measurement noise, causing
    # gradient explosion.
    #
    # We estimate intrinsic scatter from a quick linear fit on training data:
    from sklearn.linear_model import LinearRegression

    X_train_raw = train_rows[feature_cols].values
    y_train_raw = train_rows["spec_z"].values
    lr = LinearRegression().fit(X_train_raw, y_train_raw)
    residual_std = float(np.std(y_train_raw - lr.predict(X_train_raw)))

    sigma_y_measurement = float(np.sqrt(np.mean(train_rows["spec_z_err"].to_numpy() ** 2)))
    # Total sigma_y combines measurement error and intrinsic scatter (in quadrature)
    sigma_y_std = float(np.sqrt(sigma_y_measurement**2 + residual_std**2))

    print(f"\nEIV parameters:")
    print(f"  sigma_x (scaled): {sigma_x_std}")
    print(f"  sigma_y: {sigma_y_std:.4f} (measurement: {sigma_y_measurement:.4f}, intrinsic: {residual_std:.4f})")

    # Define loss functions to compare
    # Each entry is (loss_builder, model_builder, is_heteroscedastic)
    loss_configs = {
        # Point prediction losses
        "MSE": (
            lambda _: MSELoss(reduction="mean"),
            lambda: PhotoZModel(input_dim=len(feature_cols)),
            False,
        ),
        "MAE": (
            lambda _: WeightedMAELoss(reduction="mean"),
            lambda: PhotoZModel(input_dim=len(feature_cols)),
            False,
        ),
        "Huber": (
            lambda _: HuberLoss(delta=0.1, reduction="mean"),
            lambda: PhotoZModel(input_dim=len(feature_cols)),
            False,
        ),
        # Heteroscedastic loss (learns uncertainty)
        "GaussianNLL": (
            lambda _: GaussianNLLLoss(reduction="mean"),
            lambda: HeteroscedasticPhotoZModel(input_dim=len(feature_cols)),
            True,
        ),
        # Error-in-variables losses
        "FunctionalEIV": (
            lambda model: FunctionalEIVLoss(
                model,
                sigma_x=torch.tensor(sigma_x_std, dtype=torch.float32),
                sigma_y=sigma_y_std,
                reduction="mean",
            ),
            lambda: PhotoZModel(input_dim=len(feature_cols)),
            False,
        ),
        "OrthogonalEIV": (
            lambda model: OrthogonalDistanceRegressionLoss(
                model,
                sigma_x=torch.tensor(sigma_x_std, dtype=torch.float32),
                sigma_y=sigma_y_std,
                reduction="mean",
            ),
            lambda: PhotoZModel(input_dim=len(feature_cols)),
            False,
        ),
        "EnsembleEIV": (
            lambda model: EnsembleEIVLoss(
                model,
                sigma_x=torch.tensor(sigma_x_std, dtype=torch.float32),
                n_samples=20,
                reduction="mean",
            ),
            lambda: PhotoZModel(input_dim=len(feature_cols)),
            False,
        ),
    }

    # Dictionary to store results
    results = {}

    # Train models with different loss functions
    for loss_name, (loss_builder, model_builder, is_heteroscedastic) in loss_configs.items():
        print(f"\n=== Training with {loss_name} Loss ===")

        # Initialize model
        model = model_builder()

        loss_fn = loss_builder(model)

        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Train model
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, loss_fn, optimizer, NUM_EPOCHS, DEVICE
        )

        # Evaluate model
        metrics = evaluate_model(model, test_loader, DEVICE, is_heteroscedastic=is_heteroscedastic)

        # Store results
        results[loss_name] = {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "metrics": metrics,
        }

    # Plot and compare results
    plot_results(results, loss_configs.keys())

    # Print results tables
    print("\n" + "=" * 90)
    print("PHOTOMETRIC REDSHIFT ESTIMATION RESULTS")
    print("=" * 90)

    # Point prediction metrics
    print("\n--- Point Prediction Metrics ---")
    print("-" * 70)
    print(f"{'Loss':<15} {'MAE':<10} {'RMSE':<10} {'Bias':<10} {'NMAD':<10}")
    print("-" * 70)
    for loss_name, result in results.items():
        m = result["metrics"]
        print(f"{loss_name:<15} {m['mae']:<10.4f} {m['rmse']:<10.4f} {m['bias']:<10.4f} {m['nmad']:<10.4f}")

    # Probabilistic metrics
    print("\n--- Probabilistic Metrics ---")
    print("-" * 70)
    print(f"{'Loss':<15} {'CRPS':<10} {'PICP@95%':<10} {'MPIW@95%':<10} {'MCE':<10}")
    print("-" * 70)
    for loss_name, result in results.items():
        m = result["metrics"]
        mce_str = f"{m['mce']:.4f}" if not np.isnan(m["mce"]) else "N/A"
        print(f"{loss_name:<15} {m['crps']:<10.4f} {m['picp_95']:<10.4f} {m['mpiw_95']:<10.4f} {mce_str:<10}")

    print("\n" + "=" * 90)
    print("METRIC DEFINITIONS")
    print("=" * 90)
    print("Point Metrics:")
    print("  MAE  = Mean Absolute Error (lower is better)")
    print("  RMSE = Root Mean Squared Error (lower is better)")
    print("  Bias = Mean signed error (closer to 0 is better)")
    print("  NMAD = Normalized Median Absolute Deviation (astronomy standard, lower is better)")
    print("\nProbabilistic Metrics:")
    print("  CRPS    = Continuous Ranked Probability Score (lower is better)")
    print("  PICP    = Prediction Interval Coverage Probability (should be ~0.95 for 95% intervals)")
    print("  MPIW    = Mean Prediction Interval Width (narrower is better, if PICP is correct)")
    print("  MCE     = Marginal Calibration Error (lower is better, measures CDF reliability)")
    print("\nNote: For point prediction models, uncertainty is estimated from residual std.")


if __name__ == "__main__":
    main()
