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

# Import visualization utilities
from photoz_utils import (
    compare_calibration,
    compute_binned_metrics,
    plot_binned_metrics,
    print_comprehensive_metrics_table,
    print_expected_metrics_guide,
    print_metrics_table,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

# Import torchregress losses and metrics
from torchregress.losses import (
    CauchyLoss,
    DensityWeightedLoss,
    EvidentialRegressionLoss,
    FocalRLoss,
    GaussianNLLLoss,
    LDSLoss,
    LogCoshLoss,
    MixtureDensityLoss,
    MultiQuantileLoss,
    WeightedLossWrapper,
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


class QuantilePhotoZModel(nn.Module):
    """
    Neural network for quantile regression.

    Outputs multiple quantiles simultaneously for constructing prediction intervals.
    For 95% intervals, we predict q=[0.025, 0.5, 0.975].
    """

    def __init__(self, input_dim, quantiles=(0.025, 0.5, 0.975)):
        super().__init__()
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_quantiles),  # Output one value per quantile
        )

    def forward(self, x):
        # Output shape: [batch_size, n_quantiles]
        return self.network(x)


class MDNPhotoZModel(nn.Module):
    """
    Mixture Density Network for multimodal photo-z estimation.

    MDNs can capture catastrophic outliers and multi-modal redshift posteriors
    that arise from color degeneracies.

    Output format: [mixture_logits, means, raw_log_stds]
    For n_components=3, n_features=1: output size = 3 + 3 + 3 = 9

    Note: MixtureDensityLoss applies softplus to raw_log_stds internally,
    so the model should output raw values (not exp-transformed).
    """

    def __init__(self, input_dim, n_components=3):
        super().__init__()
        self.n_components = n_components

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Output heads for mixture parameters
        # Mixture weights (logits)
        self.weights_head = nn.Linear(32, n_components)
        # Component means
        self.means_head = nn.Linear(32, n_components)
        # Raw log-stds (loss applies softplus)
        self.log_stds_head = nn.Linear(32, n_components)

    def forward(self, x):
        features = self.shared(x)

        # Get mixture parameters
        logits = self.weights_head(features)  # [batch, n_components]
        means = self.means_head(features)  # [batch, n_components]
        raw_log_stds = self.log_stds_head(features)  # [batch, n_components]

        # Output format: [batch, n_components + n_components + n_components]
        # MixtureDensityLoss applies softplus to raw_log_stds to get positive stds
        # Do NOT apply exp here - the loss handles the transformation
        return torch.cat([logits, means, raw_log_stds], dim=-1)


class EvidentialPhotoZModel(nn.Module):
    """
    Evidential regression model for uncertainty decomposition.

    Outputs Normal-Inverse-Gamma (NIG) parameters:
    - gamma: predicted mean
    - nu: virtual evidence for mean (> 0)
    - alpha: shape parameter (> 1)
    - beta: scale parameter (> 0)

    This enables principled decomposition into:
    - Aleatoric uncertainty: E[sigma^2] = beta / (alpha - 1)
    - Epistemic uncertainty: Var[mu] = beta / (nu * (alpha - 1))
    """

    def __init__(self, input_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Output 4 parameters for NIG distribution
        self.gamma_head = nn.Linear(32, 1)  # Mean (unconstrained)
        self.nu_head = nn.Linear(32, 1)  # Evidence for mean (> 0)
        self.alpha_head = nn.Linear(32, 1)  # Shape (> 1)
        self.beta_head = nn.Linear(32, 1)  # Scale (> 0)

    def forward(self, x):
        features = self.shared(x)

        gamma = self.gamma_head(features)  # Mean

        # Apply softplus to ensure positivity constraints
        nu = nn.functional.softplus(self.nu_head(features)) + 0.01  # nu > 0
        alpha = nn.functional.softplus(self.alpha_head(features)) + 1.01  # alpha > 1
        beta = nn.functional.softplus(self.beta_head(features)) + 0.01  # beta > 0

        # Concatenate: [gamma, nu, alpha, beta]
        return torch.cat([gamma, nu, alpha, beta], dim=-1)


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


def evaluate_model(
    model,
    test_loader,
    device=DEVICE,
    model_type="point",
    loss_fn=None,
):
    """Evaluate the model on the test set.

    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to use
        model_type: One of "point", "heteroscedastic", "quantile", "mdn", "evidential"
        loss_fn: Loss function (needed for evidential to extract uncertainties)
    """
    model.eval()
    all_means = []
    all_stds = []
    all_lower_95 = []
    all_upper_95 = []
    all_targets = []
    all_target_errors = []

    # For evidential: store decomposed uncertainties
    all_aleatoric = []
    all_epistemic = []

    with torch.no_grad():
        for x, x_err, y, y_err in test_loader:
            x = x.to(device)
            outputs = model(x)

            if model_type == "heteroscedastic":
                mean, logvar = outputs
                std = torch.exp(0.5 * logvar)
                all_means.append(mean.cpu())
                all_stds.append(std.cpu())

            elif model_type == "quantile":
                # Quantile model outputs [q_low, q_median, q_high]
                # For 95% intervals: q=[0.025, 0.5, 0.975]
                q_low = outputs[:, 0:1]
                q_median = outputs[:, 1:2]
                q_high = outputs[:, 2:3]
                all_means.append(q_median.cpu())
                all_lower_95.append(q_low.cpu())
                all_upper_95.append(q_high.cpu())
                # Estimate std from interval width (Gaussian approximation)
                # For 95% interval: width = 2 * 1.96 * std
                interval_width = q_high - q_low
                std_approx = interval_width / (2 * 1.96)
                all_stds.append(std_approx.cpu())

            elif model_type == "mdn":
                # MDN outputs: [logits, means, stds] for n_components
                # Use the proper sampling-based interval computation
                if loss_fn is not None and hasattr(loss_fn, "predict_interval"):
                    # Use proper mixture quantiles (not Gaussian approximation)
                    mixture_mean, mixture_std = loss_fn.predict_mean_std(outputs)
                    lower_95, upper_95 = loss_fn.predict_interval(outputs, confidence=0.95)
                    all_means.append(mixture_mean.cpu())
                    all_stds.append(mixture_std.cpu())
                    all_lower_95.append(lower_95.cpu())
                    all_upper_95.append(upper_95.cpu())
                else:
                    # Fallback: extract manually (may have calibration issues)
                    n_components = 3  # Assuming 3 components
                    logits = outputs[:, :n_components]
                    means = outputs[:, n_components : 2 * n_components]
                    stds = outputs[:, 2 * n_components :]

                    # Mixture weights
                    weights = torch.softmax(logits, dim=-1)

                    # Mixture mean: E[y] = sum(w_k * mu_k)
                    mixture_mean = (weights * means).sum(dim=-1, keepdim=True)

                    # Mixture variance: Var[y] = sum(w_k * (sigma_k^2 + mu_k^2)) - E[y]^2
                    mixture_var = (weights * (stds**2 + means**2)).sum(dim=-1, keepdim=True)
                    mixture_var = mixture_var - mixture_mean**2
                    mixture_std = torch.sqrt(mixture_var.clamp(min=1e-6))

                    all_means.append(mixture_mean.cpu())
                    all_stds.append(mixture_std.cpu())

            elif model_type == "evidential":
                # Evidential outputs: [gamma, nu, alpha, beta]
                if loss_fn is not None:
                    mean, ale_unc, epi_unc = loss_fn.predict_with_uncertainty(outputs)
                    all_means.append(mean.cpu())
                    # Total std = sqrt(aleatoric + epistemic)
                    total_var = ale_unc + epi_unc
                    all_stds.append(torch.sqrt(total_var.clamp(min=1e-6)).cpu())
                    all_aleatoric.append(ale_unc.cpu())
                    all_epistemic.append(epi_unc.cpu())
                    # Use proper Student-t intervals (not Gaussian approximation)
                    if hasattr(loss_fn, "predict_interval"):
                        lower_95, upper_95 = loss_fn.predict_interval(outputs, confidence=0.95)
                        all_lower_95.append(lower_95.cpu())
                        all_upper_95.append(upper_95.cpu())
                else:
                    # Fallback: just use gamma as mean
                    gamma = outputs[:, 0:1]
                    all_means.append(gamma.cpu())
                    all_stds.append(None)

            else:  # Point prediction
                all_means.append(outputs.cpu())
                all_stds.append(None)

            all_targets.append(y)
            all_target_errors.append(y_err)

    all_means = torch.cat(all_means, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_target_errors = torch.cat(all_target_errors, dim=0)

    # Handle stds
    if all_stds[0] is not None:
        all_stds = torch.cat(all_stds, dim=0)
    else:
        # For point prediction models, use residual std as uncertainty estimate
        residuals = all_means - all_targets
        all_stds = torch.full_like(all_means, residuals.std().item())

    # Handle direct quantile/proper intervals vs Gaussian approximation
    # For quantile, mdn, and evidential: we computed proper intervals in the loop
    if len(all_lower_95) > 0 and model_type in ["quantile", "mdn", "evidential"]:
        all_lower_95 = torch.cat(all_lower_95, dim=0)
        all_upper_95 = torch.cat(all_upper_95, dim=0)
    else:
        # For other models (point, heteroscedastic): use Gaussian approximation
        all_lower_95 = all_means - 1.96 * all_stds
        all_upper_95 = all_means + 1.96 * all_stds

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
    crps = gaussian_crps(all_means.squeeze(), all_stds.squeeze(), all_targets.squeeze())

    # Prediction interval coverage (95% interval)
    # Use pre-computed intervals for quantile models, or compute from std
    picp_95 = prediction_interval_coverage_probability(all_lower_95, all_upper_95, all_targets)

    # Mean prediction interval width
    mpiw_95 = torch.mean(all_upper_95 - all_lower_95)

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

    result = {
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

    # Add decomposed uncertainty for evidential models
    if model_type == "evidential" and all_aleatoric:
        result["aleatoric_uncertainty"] = torch.cat(all_aleatoric, dim=0).numpy()
        result["epistemic_uncertainty"] = torch.cat(all_epistemic, dim=0).numpy()

    return result


def plot_results(results, loss_names):
    """Plot comparison of different models."""
    plt.figure(figsize=(20, 16))

    # Plot each loss type on its own scale using separate line styles
    # Different loss functions measure different things:
    # - MSE/MAE/Huber/EnsembleEIV/Cauchy/LogCosh: error-based (~0.01-0.1)
    # - GaussianNLL/FunctionalEIV/Quantile/MDN/Evidential: likelihood (can be negative)
    # - OrthogonalEIV: Mahalanobis distance (~10-100)
    # We use twin axes to show error-based and likelihood-based losses together

    ax1 = plt.subplot(2, 2, 1)
    ax2 = ax1.twinx()

    # Group losses by scale
    error_based = [
        "EnsembleEIV",
        "LDS",
        "DensityWeighted",
        "FocalR",
    ]
    # likelihood_based = ["GaussianNLL", "FunctionalEIV", "OrthogonalEIV", "Quantile", "MDN", "Evidential"]

    # Extended color palette for all losses
    colors = {
        "MSE": "blue",
        "MAE": "cyan",
        "Huber": "navy",
        "Cauchy": "teal",
        "LogCosh": "steelblue",
        "GaussianNLL": "orange",
        "FunctionalEIV": "red",
        "OrthogonalEIV": "darkred",
        "EnsembleEIV": "green",
        "Quantile": "purple",
        "MDN": "magenta",
        "Evidential": "brown",
        "LDS": "olive",
        "DensityWeighted": "gold",
        "MCDropout": "coral",
        "FocalR": "crimson",
    }

    for name in loss_names:
        if name not in results:
            continue
        losses = np.array(results[name]["train_losses"])
        if name in error_based:
            ax1.plot(losses, label=f"{name}", color=colors.get(name, "gray"), linewidth=2)
        else:
            ax2.plot(
                losses, label=f"{name}", color=colors.get(name, "gray"), linewidth=2, linestyle="--"
            )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error-based Loss", color="blue")
    ax2.set_ylabel("Likelihood/Distance Loss", color="orange")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
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
            ax4.plot(
                losses, label=f"{name}", color=colors.get(name, "gray"), linewidth=2, linestyle="--"
            )

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Error-based Loss", color="blue")
    ax4.set_ylabel("Likelihood/Distance Loss", color="orange")
    ax3.tick_params(axis="y", labelcolor="blue")
    ax4.tick_params(axis="y", labelcolor="orange")

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="upper right", fontsize=7)
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
    models_to_plot = ["MSE", "GaussianNLL", "Quantile", "Evidential"]
    plot_colors = ["blue", "orange", "purple", "brown"]

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
    sigma_x_std = np.sqrt(np.mean(scaled_errors**2, axis=0))

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

    print("\nEIV parameters:")
    print(f"  sigma_x (scaled): {sigma_x_std}")
    print(
        f"  sigma_y: {sigma_y_std:.4f} (measurement: {sigma_y_measurement:.4f}, intrinsic: {residual_std:.4f})"
    )

    # Define loss functions to compare
    # Each entry is (loss_builder, model_builder, model_type)
    # model_type: "point", "heteroscedastic", "quantile", "mdn", "evidential"
    input_dim = len(feature_cols)
    quantiles = [0.025, 0.5, 0.975]  # For 95% prediction intervals

    loss_configs = {
        # === Point prediction losses ===
        "MSE": (
            lambda _: nn.MSELoss(reduction="mean"),
            lambda: PhotoZModel(input_dim=input_dim),
            "point",
        ),
        "MAE": (
            lambda _: WeightedLossWrapper(nn.L1Loss, reduction="mean"),
            lambda: PhotoZModel(input_dim=input_dim),
            "point",
        ),
        "Huber": (
            lambda _: WeightedLossWrapper(nn.HuberLoss, delta=0.1, reduction="mean"),
            lambda: PhotoZModel(input_dim=input_dim),
            "point",
        ),
        # === Robust losses (resistant to outliers) ===
        "Cauchy": (
            lambda _: CauchyLoss(c=1.0, reduction="mean"),
            lambda: PhotoZModel(input_dim=input_dim),
            "point",
        ),
        "LogCosh": (
            lambda _: LogCoshLoss(reduction="mean"),
            lambda: PhotoZModel(input_dim=input_dim),
            "point",
        ),
        # === Heteroscedastic loss (learns input-dependent uncertainty) ===
        "GaussianNLL": (
            lambda _: GaussianNLLLoss(reduction="mean"),
            lambda: HeteroscedasticPhotoZModel(input_dim=input_dim),
            "heteroscedastic",
        ),
        # === Quantile regression (distribution-free intervals) ===
        "Quantile": (
            lambda _: MultiQuantileLoss(quantiles=quantiles, reduction="mean"),
            lambda: QuantilePhotoZModel(input_dim=input_dim, quantiles=quantiles),
            "quantile",
        ),
        # === Mixture Density Network (multimodal distributions) ===
        "MDN": (
            lambda _: MixtureDensityLoss(n_components=3, n_features=1, reduction="mean"),
            lambda: MDNPhotoZModel(input_dim=input_dim, n_components=3),
            "mdn",
        ),
        # === Evidential regression (uncertainty decomposition) ===
        "Evidential": (
            lambda _: EvidentialRegressionLoss(coeff_nig=0.01, reduction="mean"),
            lambda: EvidentialPhotoZModel(input_dim=input_dim),
            "evidential",
        ),
        # === Error-in-variables losses ===
        "FunctionalEIV": (
            lambda model: FunctionalEIVLoss(
                model,
                sigma_x=torch.tensor(sigma_x_std, dtype=torch.float32),
                sigma_y=sigma_y_std,
                reduction="mean",
            ),
            lambda: PhotoZModel(input_dim=input_dim),
            "point",
        ),
        "OrthogonalEIV": (
            lambda model: OrthogonalDistanceRegressionLoss(
                model,
                sigma_x=torch.tensor(sigma_x_std, dtype=torch.float32),
                sigma_y=sigma_y_std,
                reduction="mean",
            ),
            lambda: PhotoZModel(input_dim=input_dim),
            "point",
        ),
        "EnsembleEIV": (
            lambda model: EnsembleEIVLoss(
                model,
                sigma_x=torch.tensor(sigma_x_std, dtype=torch.float32),
                n_samples=20,
                reduction="mean",
            ),
            lambda: PhotoZModel(input_dim=input_dim),
            "point",
        ),
    }

    # Dictionary to store results
    results = {}

    # Train models with different loss functions
    for loss_name, (loss_builder, model_builder, model_type) in loss_configs.items():
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
        metrics = evaluate_model(model, test_loader, DEVICE, model_type=model_type, loss_fn=loss_fn)

        # Store results
        results[loss_name] = {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "metrics": metrics,
        }

    # === ENSEMBLE METHODS ===
    print("\n" + "=" * 90)
    print("TRAINING ENSEMBLE METHODS")
    print("=" * 90)

    # Deep Ensemble: Train 5 MSE models with different seeds
    print("\n--- Training Deep Ensemble (5 MSE models) ---")
    ensemble_size = 5
    ensemble_models = []
    for i in range(ensemble_size):
        print(f"  Training member {i + 1}/{ensemble_size}...")
        torch.manual_seed(42 + i)  # Different seed for each member
        model = PhotoZModel(input_dim=input_dim)
        loss_fn = nn.MSELoss(reduction="mean")
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        train_model(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            num_epochs=NUM_EPOCHS // 2,
            device=DEVICE,  # Shorter training for demo
        )
        ensemble_models.append(model)

    # Evaluate deep ensemble
    print("  Evaluating ensemble...")
    ensemble_means = []
    ensemble_targets = []
    with torch.no_grad():
        for x, x_err, y, y_err in test_loader:
            x = x.to(DEVICE)
            preds = torch.stack([m(x).cpu() for m in ensemble_models])  # [5, batch, 1]
            ensemble_means.append(preds)
            ensemble_targets.append(y)

    all_preds = torch.cat(ensemble_means, dim=1)  # [5, total_samples, 1]
    all_targets = torch.cat(ensemble_targets, dim=0)

    # Ensemble mean and epistemic uncertainty (from disagreement)
    ens_mean = all_preds.mean(dim=0)  # [total_samples, 1]
    ens_std = all_preds.std(dim=0)  # Epistemic uncertainty

    # Compute metrics
    ens_rmse = rmse(ens_mean, all_targets)
    ens_mae = mae(ens_mean, all_targets)
    ens_crps = gaussian_crps(ens_mean.squeeze(), ens_std.squeeze(), all_targets.squeeze())
    ens_lower = ens_mean - 1.96 * ens_std
    ens_upper = ens_mean + 1.96 * ens_std
    ens_picp = prediction_interval_coverage_probability(ens_lower, ens_upper, all_targets)

    results["DeepEnsemble"] = {
        "metrics": {
            "rmse": float(ens_rmse),
            "mae": float(ens_mae),
            "bias": float(bias(ens_mean, all_targets)),
            "nmad": 1.48
            * float(torch.median(torch.abs((ens_mean - all_targets) / (1 + all_targets)))),
            "crps": float(ens_crps),
            "picp_95": float(ens_picp),
            "mpiw_95": float(torch.mean(ens_upper - ens_lower)),
            "mce": float("nan"),  # Skip for ensemble
            "predictions": ens_mean.numpy(),
            "pred_stds": ens_std.numpy(),
            "targets": all_targets.numpy(),
            "target_errors": np.zeros_like(all_targets.numpy()),
        },
        "train_losses": [],  # Not tracked for ensemble
        "val_losses": [],
    }
    print(f"  DeepEnsemble RMSE: {float(ens_rmse):.4f}, PICP@95%: {float(ens_picp):.3f}")

    # Heteroscedastic Ensemble: Train 3 GaussianNLL models
    print("\n--- Training Heteroscedastic Ensemble (3 GaussianNLL models) ---")
    het_ensemble_size = 3
    het_ensemble_models = []
    for i in range(het_ensemble_size):
        print(f"  Training member {i + 1}/{het_ensemble_size}...")
        torch.manual_seed(42 + i)
        model = HeteroscedasticPhotoZModel(input_dim=input_dim)
        loss_fn = GaussianNLLLoss(reduction="mean")
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        train_model(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            num_epochs=NUM_EPOCHS // 2,
            device=DEVICE,
        )
        het_ensemble_models.append(model)

    # Evaluate heteroscedastic ensemble
    print("  Evaluating ensemble...")
    het_means = []
    het_vars = []
    het_targets = []
    with torch.no_grad():
        for x, x_err, y, y_err in test_loader:
            x = x.to(DEVICE)
            for m in het_ensemble_models:
                mean, logvar = m(x)
                het_means.append(mean.cpu())
                het_vars.append(torch.exp(logvar).cpu())
            het_targets.append(y)

    # Reshape: [n_members * n_batches, batch_size, 1] -> proper shape
    # n_samples = len(test_loader.dataset)
    het_means = torch.cat(het_means, dim=0).view(het_ensemble_size, -1, 1)
    het_vars = torch.cat(het_vars, dim=0).view(het_ensemble_size, -1, 1)
    het_targets = torch.cat(het_targets, dim=0)

    # Decompose uncertainty
    het_mean = het_means.mean(dim=0)
    het_aleatoric = het_vars.mean(dim=0)  # Mean of predicted variances
    het_epistemic = het_means.var(dim=0)  # Variance of means
    het_total_std = torch.sqrt(het_aleatoric + het_epistemic)

    het_rmse = rmse(het_mean, het_targets)
    het_crps = gaussian_crps(het_mean.squeeze(), het_total_std.squeeze(), het_targets.squeeze())
    het_lower = het_mean - 1.96 * het_total_std
    het_upper = het_mean + 1.96 * het_total_std
    het_picp = prediction_interval_coverage_probability(het_lower, het_upper, het_targets)

    results["HetEnsemble"] = {
        "metrics": {
            "rmse": float(het_rmse),
            "mae": float(mae(het_mean, het_targets)),
            "bias": float(bias(het_mean, het_targets)),
            "nmad": 1.48
            * float(torch.median(torch.abs((het_mean - het_targets) / (1 + het_targets)))),
            "crps": float(het_crps),
            "picp_95": float(het_picp),
            "mpiw_95": float(torch.mean(het_upper - het_lower)),
            "mce": float("nan"),
            "predictions": het_mean.numpy(),
            "pred_stds": het_total_std.numpy(),
            "targets": het_targets.numpy(),
            "target_errors": np.zeros_like(het_targets.numpy()),
            "aleatoric_std": torch.sqrt(het_aleatoric).numpy(),
            "epistemic_std": torch.sqrt(het_epistemic).numpy(),
        },
        "train_losses": [],
        "val_losses": [],
    }
    print(f"  HetEnsemble RMSE: {float(het_rmse):.4f}, PICP@95%: {float(het_picp):.3f}")
    print(f"  Aleatoric std: {float(torch.sqrt(het_aleatoric).mean()):.4f}")
    print(f"  Epistemic std: {float(torch.sqrt(het_epistemic).mean()):.4f}")

    # MDN Ensemble: Train 3 MDN models with different seeds
    print("\n--- Training MDN Ensemble (3 MDN models) ---")
    mdn_ensemble_size = 3
    mdn_ensemble_models = []
    for i in range(mdn_ensemble_size):
        print(f"  Training member {i + 1}/{mdn_ensemble_size}...")
        torch.manual_seed(42 + i)
        model = MDNPhotoZModel(input_dim=input_dim, n_components=3)
        loss_fn = MixtureDensityLoss(n_components=3, n_features=1, reduction="mean")
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        train_model(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            num_epochs=NUM_EPOCHS // 2,
            device=DEVICE,
        )
        mdn_ensemble_models.append(model)

    # Evaluate MDN ensemble
    print("  Evaluating ensemble...")
    mdn_ensemble_loss = MixtureDensityLoss(n_components=3, n_features=1, reduction="mean")
    mdn_means = []
    mdn_vars = []
    mdn_targets = []
    with torch.no_grad():
        for x, x_err, y, y_err in test_loader:
            x = x.to(DEVICE)
            batch_means = []
            batch_vars = []
            for m in mdn_ensemble_models:
                outputs = m(x)
                # Use loss function's method to correctly extract parameters
                # (it applies softplus to the raw log_stds)
                mixture_mean, mixture_std = mdn_ensemble_loss.predict_mean_std(outputs)
                batch_means.append(mixture_mean.cpu())
                batch_vars.append((mixture_std**2).cpu())
            mdn_means.append(torch.stack(batch_means))  # [n_models, batch, 1]
            mdn_vars.append(torch.stack(batch_vars))
            mdn_targets.append(y)

    # Reshape: [n_batches, n_models, batch, 1] -> [n_models, total_samples, 1]
    mdn_means = torch.cat(mdn_means, dim=1)
    mdn_vars = torch.cat(mdn_vars, dim=1)
    mdn_targets = torch.cat(mdn_targets, dim=0)

    # Uncertainty decomposition
    mdn_ens_mean = mdn_means.mean(dim=0)
    mdn_aleatoric = mdn_vars.mean(dim=0)  # Mean of mixture variances
    mdn_epistemic = mdn_means.var(dim=0)  # Variance of mixture means
    mdn_total_std = torch.sqrt(mdn_aleatoric + mdn_epistemic).clamp(min=1e-6)

    mdn_rmse = rmse(mdn_ens_mean, mdn_targets)
    mdn_crps = gaussian_crps(mdn_ens_mean.squeeze(), mdn_total_std.squeeze(), mdn_targets.squeeze())
    mdn_lower = mdn_ens_mean - 1.96 * mdn_total_std
    mdn_upper = mdn_ens_mean + 1.96 * mdn_total_std
    mdn_picp = prediction_interval_coverage_probability(mdn_lower, mdn_upper, mdn_targets)

    results["MDNEnsemble"] = {
        "metrics": {
            "rmse": float(mdn_rmse),
            "mae": float(mae(mdn_ens_mean, mdn_targets)),
            "bias": float(bias(mdn_ens_mean, mdn_targets)),
            "nmad": 1.48
            * float(torch.median(torch.abs((mdn_ens_mean - mdn_targets) / (1 + mdn_targets)))),
            "crps": float(mdn_crps),
            "picp_95": float(mdn_picp),
            "mpiw_95": float(torch.mean(mdn_upper - mdn_lower)),
            "mce": float("nan"),
            "predictions": mdn_ens_mean.numpy(),
            "pred_stds": mdn_total_std.numpy(),
            "targets": mdn_targets.numpy(),
            "target_errors": np.zeros_like(mdn_targets.numpy()),
            "aleatoric_std": torch.sqrt(mdn_aleatoric).numpy(),
            "epistemic_std": torch.sqrt(mdn_epistemic).numpy(),
        },
        "train_losses": [],
        "val_losses": [],
    }
    print(f"  MDNEnsemble RMSE: {float(mdn_rmse):.4f}, PICP@95%: {float(mdn_picp):.3f}")
    print(f"  Aleatoric std: {float(torch.sqrt(mdn_aleatoric).mean()):.4f}")
    print(f"  Epistemic std: {float(torch.sqrt(mdn_epistemic).mean()):.4f}")

    # MultiSWAG: Train multiple SWAG models with different seeds
    print("\n--- Training MultiSWAG (3 SWAG models) ---")
    from torchregress.ensemble import SWAG

    swag_ensemble_size = 3
    swag_models = []
    warmup_epochs = NUM_EPOCHS // 3
    swag_epochs = NUM_EPOCHS // 3

    for i in range(swag_ensemble_size):
        print(f"  Training SWAG member {i + 1}/{swag_ensemble_size}...")
        torch.manual_seed(42 + i)

        # Create model and SWAG wrapper
        base_model = PhotoZModel(input_dim=input_dim)
        swag_model = SWAG(base_model, max_num_models=10)
        loss_fn = nn.MSELoss(reduction="mean")
        optimizer = optim.Adam(base_model.parameters(), lr=0.001, weight_decay=1e-5)

        # Warmup phase
        for epoch in range(warmup_epochs):
            base_model.train()
            for x, x_err, y, y_err in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                pred = base_model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

        # SWAG collection phase
        for epoch in range(swag_epochs):
            base_model.train()
            for x, x_err, y, y_err in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                pred = base_model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
            # Collect model at end of epoch
            swag_model.collect_model(base_model)

        swag_models.append(swag_model)

    # Evaluate MultiSWAG
    print("  Evaluating MultiSWAG...")
    swag_samples_per_model = 10
    all_swag_preds = []
    swag_targets_list = []

    for x, x_err, y, y_err in test_loader:
        x = x.to(DEVICE)
        batch_preds = []
        for swag_model in swag_models:
            for _ in range(swag_samples_per_model):
                swag_model.sample(scale=0.5)
                with torch.no_grad():
                    pred = swag_model(x)
                batch_preds.append(pred.cpu())
        all_swag_preds.append(torch.stack(batch_preds))  # [n_samples, batch, 1]
        swag_targets_list.append(y)

    # Reshape: [n_batches, n_samples, batch, 1] -> [n_samples, total, 1]
    all_swag_preds = torch.cat(all_swag_preds, dim=1)
    swag_targets = torch.cat(swag_targets_list, dim=0)

    # Compute statistics
    swag_mean = all_swag_preds.mean(dim=0)
    swag_std = all_swag_preds.std(dim=0)

    swag_rmse = rmse(swag_mean, swag_targets)
    swag_crps = gaussian_crps(swag_mean.squeeze(), swag_std.squeeze(), swag_targets.squeeze())
    swag_lower = swag_mean - 1.96 * swag_std
    swag_upper = swag_mean + 1.96 * swag_std
    swag_picp = prediction_interval_coverage_probability(swag_lower, swag_upper, swag_targets)

    results["MultiSWAG"] = {
        "metrics": {
            "rmse": float(swag_rmse),
            "mae": float(mae(swag_mean, swag_targets)),
            "bias": float(bias(swag_mean, swag_targets)),
            "nmad": 1.48
            * float(torch.median(torch.abs((swag_mean - swag_targets) / (1 + swag_targets)))),
            "crps": float(swag_crps),
            "picp_95": float(swag_picp),
            "mpiw_95": float(torch.mean(swag_upper - swag_lower)),
            "mce": float("nan"),
            "predictions": swag_mean.numpy(),
            "pred_stds": swag_std.numpy(),
            "targets": swag_targets.numpy(),
            "target_errors": np.zeros_like(swag_targets.numpy()),
        },
        "train_losses": [],
        "val_losses": [],
    }
    print(f"  MultiSWAG RMSE: {float(swag_rmse):.4f}, PICP@95%: {float(swag_picp):.3f}")
    print(f"  Mean std: {float(swag_std.mean()):.4f}")

    # MC-Dropout: Simple uncertainty estimation via dropout at inference
    print("\n--- Training MC-Dropout Model ---")
    from torchregress.ensemble import MCDropoutModel

    torch.manual_seed(42)
    mc_dropout_model = MCDropoutModel(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        output_dim=1,
        dropout_rate=0.2,
        n_samples=30,
    ).to(DEVICE)

    mc_loss_fn = nn.MSELoss(reduction="mean")
    mc_optimizer = optim.Adam(mc_dropout_model.parameters(), lr=0.001, weight_decay=1e-5)

    # Training
    mc_dropout_model.train()
    for epoch in range(NUM_EPOCHS // 2):
        for x, x_err, y, y_err in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            mc_optimizer.zero_grad()
            pred = mc_dropout_model(x)
            loss = mc_loss_fn(pred, y)
            loss.backward()
            mc_optimizer.step()

    # Evaluate MC-Dropout
    print("  Evaluating MC-Dropout...")
    mc_means = []
    mc_stds = []
    mc_targets = []

    for x, x_err, y, y_err in test_loader:
        x = x.to(DEVICE)
        mean, std = mc_dropout_model.predict_with_uncertainty(x)
        mc_means.append(mean.cpu())
        mc_stds.append(std.cpu())
        mc_targets.append(y)

    mc_means = torch.cat(mc_means, dim=0)
    mc_stds = torch.cat(mc_stds, dim=0)
    mc_targets = torch.cat(mc_targets, dim=0)

    mc_rmse = rmse(mc_means, mc_targets)
    mc_crps = gaussian_crps(mc_means.squeeze(), mc_stds.squeeze(), mc_targets.squeeze())
    mc_lower = mc_means - 1.96 * mc_stds
    mc_upper = mc_means + 1.96 * mc_stds
    mc_picp = prediction_interval_coverage_probability(mc_lower, mc_upper, mc_targets)

    results["MCDropout"] = {
        "metrics": {
            "rmse": float(mc_rmse),
            "mae": float(mae(mc_means, mc_targets)),
            "bias": float(bias(mc_means, mc_targets)),
            "nmad": 1.48
            * float(torch.median(torch.abs((mc_means - mc_targets) / (1 + mc_targets)))),
            "crps": float(mc_crps),
            "picp_95": float(mc_picp),
            "mpiw_95": float(torch.mean(mc_upper - mc_lower)),
            "mce": float("nan"),
            "predictions": mc_means.numpy(),
            "pred_stds": mc_stds.numpy(),
            "targets": mc_targets.numpy(),
            "target_errors": np.zeros_like(mc_targets.numpy()),
        },
        "train_losses": [],
        "val_losses": [],
    }
    print(f"  MCDropout RMSE: {float(mc_rmse):.4f}, PICP@95%: {float(mc_picp):.3f}")
    print(f"  Mean std: {float(mc_stds.mean()):.4f}")

    # Bayesian Neural Network: Variational inference with weight uncertainty
    print("\n--- Training Bayesian Neural Network ---")
    from torchregress.ensemble import BayesianNeuralNetwork

    torch.manual_seed(42)
    bnn_model = BayesianNeuralNetwork(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        output_dim=1,
        prior_sigma=1.0,
        n_samples=30,
    ).to(DEVICE)

    bnn_optimizer = optim.Adam(bnn_model.parameters(), lr=0.001)
    n_train = len(train_loader.dataset)

    # Training with ELBO
    for epoch in range(NUM_EPOCHS // 2):
        bnn_model.train()
        for x, x_err, y, y_err in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            bnn_optimizer.zero_grad()
            pred = bnn_model(x)
            loss = bnn_model.elbo_loss(pred, y, n_data=n_train, beta=1.0)
            loss.backward()
            bnn_optimizer.step()

    # Evaluate BNN
    print("  Evaluating BNN...")
    bnn_means = []
    bnn_stds = []
    bnn_targets = []

    for x, x_err, y, y_err in test_loader:
        x = x.to(DEVICE)
        mean, std = bnn_model.predict_with_uncertainty(x)
        bnn_means.append(mean.cpu())
        bnn_stds.append(std.cpu())
        bnn_targets.append(y)

    bnn_means = torch.cat(bnn_means, dim=0)
    bnn_stds = torch.cat(bnn_stds, dim=0)
    bnn_targets = torch.cat(bnn_targets, dim=0)

    bnn_rmse = rmse(bnn_means, bnn_targets)
    bnn_crps = gaussian_crps(bnn_means.squeeze(), bnn_stds.squeeze(), bnn_targets.squeeze())
    bnn_lower = bnn_means - 1.96 * bnn_stds
    bnn_upper = bnn_means + 1.96 * bnn_stds
    bnn_picp = prediction_interval_coverage_probability(bnn_lower, bnn_upper, bnn_targets)

    results["BNN"] = {
        "metrics": {
            "rmse": float(bnn_rmse),
            "mae": float(mae(bnn_means, bnn_targets)),
            "bias": float(bias(bnn_means, bnn_targets)),
            "nmad": 1.48
            * float(torch.median(torch.abs((bnn_means - bnn_targets) / (1 + bnn_targets)))),
            "crps": float(bnn_crps),
            "picp_95": float(bnn_picp),
            "mpiw_95": float(torch.mean(bnn_upper - bnn_lower)),
            "mce": float("nan"),
            "predictions": bnn_means.numpy(),
            "pred_stds": bnn_stds.numpy(),
            "targets": bnn_targets.numpy(),
            "target_errors": np.zeros_like(bnn_targets.numpy()),
        },
        "train_losses": [],
        "val_losses": [],
    }
    print(f"  BNN RMSE: {float(bnn_rmse):.4f}, PICP@95%: {float(bnn_picp):.3f}")
    print(f"  Mean std: {float(bnn_stds.mean()):.4f}")

    # === IMBALANCED REGRESSION METHODS ===
    print("\n" + "=" * 90)
    print("TRAINING IMBALANCED REGRESSION METHODS")
    print("=" * 90)

    # Collect all training targets for fitting density estimators
    train_targets_for_imb = []
    for _, _, y, _ in train_loader:
        train_targets_for_imb.append(y)
    train_targets_tensor = torch.cat(train_targets_for_imb, dim=0)
    print(f"Training targets for imbalanced methods: {len(train_targets_tensor)} samples")

    # LDSLoss: Label Distribution Smoothing
    # WARNING: Can break calibration - validate after training!
    print("\n--- Training with LDSLoss (Label Distribution Smoothing) ---")
    print("  Note: LDS can break calibration - use with post-hoc calibration")
    torch.manual_seed(42)
    lds_model = PhotoZModel(input_dim=input_dim)
    lds_loss_fn = LDSLoss(kernel="gaussian", kernel_width=2.0, reweight_factor=0.8, base_loss="mse")
    lds_loss_fn.fit(train_targets_tensor)  # Fit LDS weights on training data
    lds_optimizer = optim.Adam(lds_model.parameters(), lr=0.001, weight_decay=1e-5)

    lds_train_losses, lds_val_losses = train_model(
        lds_model,
        train_loader,
        val_loader,
        lds_loss_fn,
        lds_optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
    )
    lds_metrics = evaluate_model(lds_model, test_loader, DEVICE, model_type="point")
    results["LDS"] = {
        "model": lds_model,
        "train_losses": lds_train_losses,
        "val_losses": lds_val_losses,
        "metrics": lds_metrics,
    }
    print(f"  LDSLoss RMSE: {lds_metrics['rmse']:.4f}, PICP@95%: {lds_metrics['picp_95']:.3f}")

    # DensityWeightedLoss: Inverse density weighting (calibration-safe)
    print("\n--- Training with DensityWeightedLoss (calibration-safe) ---")
    torch.manual_seed(42)
    dw_model = PhotoZModel(input_dim=input_dim)
    dw_loss_fn = DensityWeightedLoss(kernel_width=0.5, reweight_factor=0.8, base_loss="mse")
    dw_loss_fn.fit_density(train_targets_tensor)  # Fit density on training data
    dw_optimizer = optim.Adam(dw_model.parameters(), lr=0.001, weight_decay=1e-5)

    dw_train_losses, dw_val_losses = train_model(
        dw_model,
        train_loader,
        val_loader,
        dw_loss_fn,
        dw_optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
    )
    dw_metrics = evaluate_model(dw_model, test_loader, DEVICE, model_type="point")
    results["DensityWeighted"] = {
        "model": dw_model,
        "train_losses": dw_train_losses,
        "val_losses": dw_val_losses,
        "metrics": dw_metrics,
    }
    print(
        f"  DensityWeighted RMSE: {dw_metrics['rmse']:.4f}, PICP@95%: {dw_metrics['picp_95']:.3f}"
    )

    # FocalRLoss: Focus on hard samples (larger errors)
    print("\n--- Training with FocalRLoss (focus on hard samples) ---")
    torch.manual_seed(42)
    focal_model = PhotoZModel(input_dim=input_dim)
    focal_loss_fn = FocalRLoss(beta=0.2, gamma=1.0, base_loss="mse")
    focal_optimizer = optim.Adam(focal_model.parameters(), lr=0.001, weight_decay=1e-5)

    focal_train_losses, focal_val_losses = train_model(
        focal_model,
        train_loader,
        val_loader,
        focal_loss_fn,
        focal_optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
    )
    focal_metrics = evaluate_model(focal_model, test_loader, DEVICE, model_type="point")
    results["FocalR"] = {
        "model": focal_model,
        "train_losses": focal_train_losses,
        "val_losses": focal_val_losses,
        "metrics": focal_metrics,
    }
    print(f"  FocalR RMSE: {focal_metrics['rmse']:.4f}, PICP@95%: {focal_metrics['picp_95']:.3f}")

    # Compare tail performance across all imbalanced methods
    print("\n--- Comparing Tail Performance: Imbalanced Methods vs MSE Baseline ---")
    imb_methods = [
        "MSE",
        "LDS",
        "DensityWeighted",
        "FocalR",
    ]
    for name in imb_methods:
        if name not in results:
            continue
        m = results[name]["metrics"]
        preds = np.asarray(m["predictions"]).flatten()
        targets = np.asarray(m["targets"]).flatten()

        # Split into quintiles
        q20 = np.quantile(targets, 0.2)
        q80 = np.quantile(targets, 0.8)

        low_mask = targets <= q20
        high_mask = targets >= q80

        low_rmse = np.sqrt(np.mean((preds[low_mask] - targets[low_mask]) ** 2))
        high_rmse = np.sqrt(np.mean((preds[high_mask] - targets[high_mask]) ** 2))
        ratio = high_rmse / (low_rmse + 1e-8)

        print(
            f"  {name:<15} Low-z RMSE: {low_rmse:.4f}, High-z RMSE: {high_rmse:.4f}, Ratio: {ratio:.2f}x"
        )

    # Plot and compare results
    all_methods = list(loss_configs.keys()) + [
        "DeepEnsemble",
        "HetEnsemble",
        "MDNEnsemble",
        "MultiSWAG",
        "MCDropout",
        "BNN",
        "LDS",
        "DensityWeighted",
        "FocalR",
    ]
    plot_results(results, all_methods)

    # Print results tables using utility function
    print_metrics_table(
        {name: res["metrics"] for name, res in results.items()},
        include_point=True,
        include_prob=True,
    )

    # Print comprehensive metrics table (all categories)
    print_comprehensive_metrics_table({name: res["metrics"] for name, res in results.items()})

    # === ADDITIONAL VISUALIZATIONS ===
    print("\n" + "=" * 90)
    print("GENERATING CALIBRATION DIAGNOSTICS")
    print("=" * 90)

    # Create calibration comparison plot for probabilistic models
    prob_models = ["GaussianNLL", "Quantile", "MDN", "Evidential"]
    prob_results = {}
    for name in prob_models:
        if name in results:
            m = results[name]["metrics"]
            prob_results[name] = {
                "predictions": m["predictions"],
                "pred_stds": m["pred_stds"],
                "targets": m["targets"],
                "crps": m["crps"],
                "picp_95": m["picp_95"],
                "mpiw_95": m["mpiw_95"],
                "mce": m["mce"],
            }

    if prob_results:
        fig = compare_calibration(prob_results)
        fig.savefig("photoz_calibration_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved: photoz_calibration_comparison.png")

    # === TAIL/BINNED ANALYSIS ===
    print("\n" + "=" * 90)
    print("TAIL PERFORMANCE ANALYSIS (by redshift bin)")
    print("=" * 90)

    # Analyze performance in redshift bins for key models
    key_models = ["MSE", "GaussianNLL", "Quantile", "Evidential"]
    for model_name in key_models:
        if model_name not in results:
            continue
        m = results[model_name]["metrics"]
        binned = compute_binned_metrics(m["predictions"], m["pred_stds"], m["targets"], n_bins=5)
        print(f"\n{model_name}:")
        print(f"  {'Bin':<20} {'n':<6} {'RMSE':<8} {'PICP@95%':<10}")
        print("  " + "-" * 50)
        for bin_name, metrics in binned.items():
            print(
                f"  {bin_name:<20} {metrics['n_samples']:<6} "
                f"{metrics['rmse']:<8.4f} {metrics['picp_95']:<10.3f}"
            )

    # Create binned metrics plot for GaussianNLL
    if "GaussianNLL" in results:
        m = results["GaussianNLL"]["metrics"]
        binned = compute_binned_metrics(m["predictions"], m["pred_stds"], m["targets"], n_bins=5)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_binned_metrics(binned, "rmse", ax=axes[0], label="GaussianNLL")
        plot_binned_metrics(binned, "picp_95", ax=axes[1], label="GaussianNLL")
        axes[1].axhline(y=0.95, color="red", linestyle="--", label="Expected")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig("photoz_binned_metrics.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("\nSaved: photoz_binned_metrics.png")

    # === EVIDENTIAL UNCERTAINTY DECOMPOSITION ===
    if "Evidential" in results:
        m = results["Evidential"]["metrics"]
        if "aleatoric_uncertainty" in m:
            print("\n" + "=" * 90)
            print("EVIDENTIAL UNCERTAINTY DECOMPOSITION")
            print("=" * 90)
            ale = np.sqrt(m["aleatoric_uncertainty"].flatten())
            epi = np.sqrt(m["epistemic_uncertainty"].flatten())
            total = np.sqrt(ale**2 + epi**2)
            print(f"Mean aleatoric std:  {np.mean(ale):.4f}")
            print(f"Mean epistemic std:  {np.mean(epi):.4f}")
            print(f"Mean total std:      {np.mean(total):.4f}")
            print(f"Epistemic fraction:  {np.mean(epi**2 / (ale**2 + epi**2)):.2%}")

            # Plot uncertainty decomposition
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Aleatoric vs Epistemic
            axes[0].scatter(ale, epi, alpha=0.3, s=10)
            axes[0].set_xlabel("Aleatoric Uncertainty (σ)")
            axes[0].set_ylabel("Epistemic Uncertainty (σ)")
            axes[0].set_title("Uncertainty Decomposition")
            axes[0].grid(True, alpha=0.3)

            # Uncertainty by redshift
            z = m["targets"].flatten()
            order = np.argsort(z)
            axes[1].plot(z[order], ale[order], "b-", alpha=0.5, label="Aleatoric")
            axes[1].plot(z[order], epi[order], "r-", alpha=0.5, label="Epistemic")
            axes[1].set_xlabel("Spectroscopic Redshift")
            axes[1].set_ylabel("Uncertainty (σ)")
            axes[1].set_title("Uncertainty vs Redshift")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("photoz_evidential_uncertainty.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved: photoz_evidential_uncertainty.png")

    # === EXPECTED METRICS GUIDE ===
    print_expected_metrics_guide()


if __name__ == "__main__":
    main()
