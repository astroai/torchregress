"""
Photometric redshift estimation using SDSS data.

This example demonstrates using torchregress's error-in-variables (EIV) losses
and reliability tools for photometric redshift estimation.

Modes:
    --mode quick: Fast training on simulated data (minimal models)
    --mode full:  Full training on real/simulated data (all models + stress tests)
"""

import argparse
import json
import os
import time
from io import StringIO
from typing import List, Optional

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
    print_comprehensive_metrics_table,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Import torchregress components
from torchregress.losses import (
    EvidentialRegressionLoss,
    GaussianNLLLoss,
    MultiQuantileLoss,
)
from torchregress.losses.eiv import (
    BaseEIVLoss,
    FunctionalEIVLoss,
)
from torchregress.metrics import (
    ExpectedCalibrationError,
    MeanAbsoluteError,
    MeanPredictionIntervalWidth,
    MeanSquaredError,
    MedianAbsoluteDeviation,
    PredictionIntervalCoverageProbability,
)

# Constants
DATA_DIR = os.path.join("data", "sdss")
SDSS_API_URL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# DATA UTILITIES
# ==============================================================================

class SDSSDataset(Dataset):
    """SDSS photometric dataset for redshift estimation."""

    def __init__(
        self, 
        data: pd.DataFrame, 
        feature_cols: List[str], 
        error_cols: List[str], 
        target_col: str = "spec_z", 
        target_error_col: str = "spec_z_err",
        feature_scaler: Optional[StandardScaler] = None
    ):
        self.data = data
        self.feature_cols = feature_cols
        self.error_cols = error_cols
        self.target_col = target_col
        self.target_error_col = target_error_col

        # Scale features
        if feature_scaler is None:
            self.feature_scaler = StandardScaler()
            self.features = self.feature_scaler.fit_transform(data[feature_cols].values)
        else:
            self.feature_scaler = feature_scaler
            self.features = self.feature_scaler.transform(data[feature_cols].values)

        # Scale feature errors (using the same scaling factors as features)
        # Var(ax) = a^2 * Var(x) => Std(ax) = |a| * Std(x)
        # Here scale = 1/std, so we divide errors by original std
        if self.feature_scaler.scale_ is not None:
             self.feature_errors = data[error_cols].values / self.feature_scaler.scale_
        else:
             self.feature_errors = data[error_cols].values

        # Get targets and target errors
        self.targets = data[target_col].values.reshape(-1, 1).astype(np.float32)
        self.target_errors = data[target_error_col].values.reshape(-1, 1).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        x_err = torch.tensor(self.feature_errors[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        y_err = torch.tensor(self.target_errors[idx], dtype=torch.float32)
        return x, x_err, y, y_err


def download_sdss_data(force_download=False, sample_size=10000):
    """Download real SDSS data using SkyServer API."""
    os.makedirs(DATA_DIR, exist_ok=True)
    data_file = os.path.join(DATA_DIR, "sdss_photoz_real.csv")

    if os.path.exists(data_file) and not force_download:
        print("Real SDSS data already exists. Loading from file...")
        return pd.read_csv(data_file)

    print(f"Downloading {sample_size} galaxies from SDSS database...")
    
    # SQL query for galaxies with good photometry and spectroscopy
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
        AND p.modelMag_u BETWEEN 10 AND 25
        AND p.modelMag_g BETWEEN 10 AND 25
        AND p.modelMag_r BETWEEN 10 AND 25
        AND p.modelMag_i BETWEEN 10 AND 25
        AND p.modelMag_z BETWEEN 10 AND 25
    ORDER BY NEWID()
    """

    try:
        response = requests.post(SDSS_API_URL, data={"cmd": sql_query, "format": "csv"}, timeout=120)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), comment="#")
        if len(df) == 0:
            raise ValueError("No data returned")

        # Add colors
        df["u_g"] = df["u"] - df["g"]
        df["g_r"] = df["g"] - df["r"]
        df["r_i"] = df["r"] - df["i"]
        df["i_z"] = df["i"] - df["z_mag"]
        
        # Propagate errors
        df["u_g_err"] = np.sqrt(df["u_err"]**2 + df["g_err"]**2)
        df["g_r_err"] = np.sqrt(df["g_err"]**2 + df["r_err"]**2)
        df["r_i_err"] = np.sqrt(df["r_err"]**2 + df["i_err"]**2)
        df["i_z_err"] = np.sqrt(df["i_err"]**2 + df["z_mag_err"]**2)

        df.to_csv(data_file, index=False)
        print(f"Saved {len(df)} galaxies to {data_file}")
        return df

    except Exception as e:
        print(f"Error downloading stats: {e}")
        return create_simulated_sdss_data(sample_size)

def create_simulated_sdss_data(n_galaxies=5000):
    """Create simulated SDSS-like data as a fallback."""
    print(f"Creating simulated data ({n_galaxies} galaxies)...")
    np.random.seed(42)
    
    # Redshifts
    z_spec = np.random.lognormal(mean=-1.3, sigma=0.5, size=n_galaxies)
    z_spec = np.clip(z_spec, 0.01, 1.0)

    # Intrinsic magnitudes
    u_true = 20.0 + 2.0 * z_spec + np.random.normal(0, 0.10, n_galaxies)
    g_true = 19.0 + 1.8 * z_spec + np.random.normal(0, 0.06, n_galaxies)
    r_true = 18.5 + 1.6 * z_spec + np.random.normal(0, 0.04, n_galaxies)
    i_true = 18.0 + 1.4 * z_spec + np.random.normal(0, 0.03, n_galaxies)
    z_mag_true = 17.5 + 1.2 * z_spec + np.random.normal(0, 0.05, n_galaxies)

    # Errors
    u_err = 0.02 + 0.08 * np.exp((u_true - 18) / 4)
    g_err = 0.015 + 0.05 * np.exp((g_true - 17) / 5)
    r_err = 0.01 + 0.03 * np.exp((r_true - 16) / 6)
    i_err = 0.01 + 0.03 * np.exp((i_true - 16) / 6)
    z_mag_err = 0.015 + 0.04 * np.exp((z_mag_true - 15) / 5)

    # Observed magnitudes
    df = pd.DataFrame({
        "objid": np.arange(n_galaxies),
        "spec_z": z_spec,
        "spec_z_err": 0.0005 + 0.001 * z_spec,
        "u": u_true + np.random.normal(0, 1, n_galaxies) * u_err,
        "g": g_true + np.random.normal(0, 1, n_galaxies) * g_err,
        "r": r_true + np.random.normal(0, 1, n_galaxies) * r_err,
        "i": i_true + np.random.normal(0, 1, n_galaxies) * i_err,
        "z_mag": z_mag_true + np.random.normal(0, 1, n_galaxies) * z_mag_err,
        "u_err": u_err, "g_err": g_err, "r_err": r_err, "i_err": i_err, "z_mag_err": z_mag_err
    })

    # Colors
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z_mag"]
    
    # Propagate errors
    df["u_g_err"] = np.sqrt(df["u_err"]**2 + df["g_err"]**2)
    df["g_r_err"] = np.sqrt(df["g_err"]**2 + df["r_err"]**2)
    df["r_i_err"] = np.sqrt(df["r_err"]**2 + df["i_err"]**2)
    df["i_z_err"] = np.sqrt(df["i_err"]**2 + df["z_mag_err"]**2)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "sdss_photoz_simulated.csv"), index=False)
    return df

# ==============================================================================
# MODELS
# ==============================================================================

class PhotoZModel(nn.Module):
    """Simple MLP."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

class HeteroscedasticPhotoZModel(nn.Module):
    """Outputs mean and log-variance."""
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        )
        self.mean = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.logvar = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    
    def forward(self, x):
        h = self.shared(x)
        return self.mean(h), self.logvar(h)

class QuantilePhotoZModel(nn.Module):
    """Quantile regression."""
    def __init__(self, input_dim, quantiles=(0.025, 0.5, 0.975)):
        super().__init__()
        self.quantiles = quantiles
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, len(quantiles))
        )
    def forward(self, x): return self.net(x)

class MDNPhotoZModel(nn.Module):
    """Mixture Density Network."""
    def __init__(self, input_dim, n_components=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        )
        self.logits = nn.Linear(32, n_components)
        self.means = nn.Linear(32, n_components)
        self.raw_log_stds = nn.Linear(32, n_components)

    def forward(self, x):
        h = self.shared(x)
        return torch.cat([self.logits(h), self.means(h), self.raw_log_stds(h)], dim=-1)

class EvidentialPhotoZModel(nn.Module):
    """Evidential Regression."""
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        )
        self.gamma = nn.Linear(32, 1)
        self.nu = nn.Linear(32, 1)
        self.alpha = nn.Linear(32, 1)
        self.beta = nn.Linear(32, 1)

    def forward(self, x):
        h = self.shared(x)
        gamma = self.gamma(h)
        nu = nn.functional.softplus(self.nu(h)) + 0.01
        alpha = nn.functional.softplus(self.alpha(h)) + 1.01
        beta = nn.functional.softplus(self.beta(h)) + 0.01
        return torch.cat([gamma, nu, alpha, beta], dim=-1)

# ==============================================================================
# HELPER METRICS
# ==============================================================================

def gaussian_crps(mean, std, target):
    """Compute CRPS for Gaussian predictive distribution."""
    # crps = std * [z(2Φ(z) - 1) + 2φ(z) - 1/√π], z = (y - μ) / σ
    z = (target - mean) / (std + 1e-8)
    phi = torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
    crps = std * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return torch.mean(crps)

# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

class PhotoZExperiment:
    def __init__(self, args):
        self.args = args
        self.results = {}
        self.feature_cols = ["u_g", "g_r", "r_i", "i_z"]
        self.error_cols = ["u_g_err", "g_r_err", "r_i_err", "i_z_err"]
        self.input_dim = len(self.feature_cols)
        
        # Setup data
        self.setup_data()
        
        # Configure models
        self.configure_models()

    def setup_data(self):
        """Load and prepare data."""
        if self.args.mode == "quick":
            self.df = create_simulated_sdss_data(n_galaxies=2000)
            self.epochs = 2
            self.batch_size = 32
        else:
            self.df = download_sdss_data(sample_size=20000)
            self.epochs = 50
            self.batch_size = 64

        # Splits
        train_size = int(0.7 * len(self.df))
        val_size = int(0.15 * len(self.df))
        # test_size = len(self.df) - train_size - val_size
        
        # Create full dataset first to fit scaler
        full_ds = SDSSDataset(self.df, self.feature_cols, self.error_cols)
        self.scaler = full_ds.feature_scaler
        
        # Split dataframe indices
        # We manually split so we can create Datasets with the shared scaler
        perm = np.random.RandomState(42).permutation(len(self.df))
        train_idx = perm[:train_size]
        val_idx = perm[train_size:train_size+val_size]
        test_idx = perm[train_size+val_size:]
        
        self.train_ds = SDSSDataset(self.df.iloc[train_idx], self.feature_cols, self.error_cols, feature_scaler=self.scaler)
        self.val_ds = SDSSDataset(self.df.iloc[val_idx], self.feature_cols, self.error_cols, feature_scaler=self.scaler)
        self.test_ds = SDSSDataset(self.df.iloc[test_idx], self.feature_cols, self.error_cols, feature_scaler=self.scaler)

        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_ds, batch_size=self.batch_size)
    
    def configure_models(self):
        """Define models and losses based on mode."""
        self.configs = {}
        
        # Baseline Point
        self.configs["MSE"] = {
            "model_cls": PhotoZModel,
            "loss_fn": lambda m: nn.MSELoss(),
            "type": "point"
        }
        
        # Heteroscedastic
        self.configs["GaussianNLL"] = {
            "model_cls": HeteroscedasticPhotoZModel,
            "loss_fn": lambda m: GaussianNLLLoss(),
            "type": "heteroscedastic"
        }

        if self.args.mode == "full":
            # Add advanced models for full mode
            quantiles = [0.025, 0.5, 0.975]
            self.configs["Quantile"] = {
                "model_cls": lambda dim: QuantilePhotoZModel(dim, quantiles),
                "loss_fn": lambda m: MultiQuantileLoss(quantiles=quantiles),
                "type": "quantile"
            }
            
            self.configs["Evidential"] = {
                "model_cls": EvidentialPhotoZModel,
                "loss_fn": lambda m: EvidentialRegressionLoss(coeff_nig=0.01),
                "type": "evidential"
            }
            
            # EIV
            # Estimate sigmas
            # Use RMS of errors for homoscedastic approximation
            x_mse = (self.train_ds.feature_errors**2).mean(axis=0)
            sigma_x = torch.tensor(np.sqrt(x_mse), dtype=torch.float32)
            sigma_y = 0.05 # Approximate
            
            self.configs["FunctionalEIV"] = {
                "model_cls": PhotoZModel,
                "loss_fn": lambda m: FunctionalEIVLoss(m, sigma_x=sigma_x, sigma_y=sigma_y),
                "type": "point"
            }

    def train_single(self, name, config, scenario_name="baseline"):
        """Train a single model configuration."""
        print(f"Training {name} [{scenario_name}]...")
        model = config["model_cls"](self.input_dim).to(DEVICE)
        loss_fn = config["loss_fn"](model)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for x, x_err, y, y_err in self.train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                
                if isinstance(loss_fn, BaseEIVLoss):
                    loss = loss_fn(x, y)
                else:
                    out = model(x)
                    loss = loss_fn(out, y)
                    
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss {total_loss/len(self.train_loader):.4f}")
                
        return model, loss_fn

    def inject_stress(self, loader, stress_type=None, severity=1.0):
        """Yield batches with injected stress for testing."""
        for x, x_err, y, y_err in loader:
            if stress_type == "noise":
                # Add noise to features
                noise_scale = 0.5 * severity
                noise = torch.randn_like(x) * noise_scale
                x = x + noise
            
            elif stress_type == "outliers":
                # Replace % of inputs with random outliers
                mask = torch.rand(x.shape[0]) < (0.1 * severity)
                if mask.any():
                    x[mask] = torch.randn_like(x[mask]) * 5.0
            
            elif stress_type == "mask":
                # Drop random features
                mask_prob = 0.2 * severity
                mask = torch.rand_like(x) < mask_prob
                x[mask] = 0.0

            elif stress_type == "shift":
                # Covariate shift: Keep only high-z galaxies (z > 0.4)
                mask = y.squeeze() > 0.4
                if mask.sum() == 0:
                    continue
                x, x_err, y, y_err = x[mask], x_err[mask], y[mask], y_err[mask]

            yield x, x_err, y, y_err

    def evaluate(self, model, loss_fn, model_type, stress_type=None):
        """Evaluate model and Compute metrics."""
        model.eval()
        
        # Metrics collection
        preds_list = []
        std_list = []
        targets_list = []
        lower_list = []
        upper_list = []
        
        loader_gen = self.inject_stress(self.test_loader, stress_type, severity=1.0)
        
        with torch.no_grad():
            for x, x_err, y, y_err in loader_gen:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                
                # Logic to extract mean/std/intervals based on model type
                if model_type == "point":
                    mean = out
                    std = torch.full_like(mean, 0.05) # Dummy
                    lower = mean - 1.96 * 0.05
                    upper = mean + 1.96 * 0.05
                    
                elif model_type == "heteroscedastic":
                    mean, logvar = out
                    std = torch.exp(0.5 * logvar)
                    lower = mean - 1.96 * std
                    upper = mean + 1.96 * std
                    
                elif model_type == "quantile":
                    # out is [low, med, high]
                    lower = out[:, 0:1]
                    mean = out[:, 1:2]
                    upper = out[:, 2:3]
                    std = (upper - lower) / 4.0 # Approx
                    
                elif model_type == "evidential":
                    # gamma, nu, alpha, beta
                    mean, ale, epi = loss_fn.predict_with_uncertainty(out)
                    std = torch.sqrt(ale + epi)
                    lower = mean - 1.96 * std
                    upper = mean + 1.96 * std
                
                preds_list.append(mean.cpu())
                std_list.append(std.cpu())
                targets_list.append(y.cpu())
                lower_list.append(lower.cpu())
                upper_list.append(upper.cpu())

        preds = torch.cat(preds_list)
        stds = torch.cat(std_list)
        targets = torch.cat(targets_list)
        lowers = torch.cat(lower_list)
        uppers = torch.cat(upper_list)
        
        # Calculate Metrics using torchregress.metrics
        mse = MeanSquaredError()(preds, targets)
        mae = MeanAbsoluteError()(preds, targets)
        nmad = MedianAbsoluteDeviation()(preds, targets)
        picp = PredictionIntervalCoverageProbability()(lowers, uppers, targets)
        mpiw = MeanPredictionIntervalWidth()(lowers, uppers)
        
        # bias
        bias = torch.mean(preds - targets)
        
        # crps
        crps = gaussian_crps(preds, stds, targets)
        
        # Functional calls for some complex ones
        try:
             # ECE needs quantiles dict, we have point+std or interval
             # We simulate quantiles for ECE from Gaussian assumption
             q_dict = {
                 0.16: preds - stds,
                 0.50: preds,
                 0.84: preds + stds
             }
             ece = ExpectedCalibrationError()(q_dict, targets)["expected_calibration_error"]
        except Exception:
             ece = float("nan")

        return {
            "mae": float(mae),
            "rmse": float(torch.sqrt(mse)),
            "bias": float(bias),
            "nmad": float(nmad),
            "crps": float(crps),
            "picp_95": float(picp),
            "mpiw_95": float(mpiw),
            "mce": float(ece),
            "predictions": preds.numpy(),
            "pred_stds": stds.numpy(),
            "targets": targets.numpy()
        }

    def run(self):
        """Run the full experiment pipeline."""
        print(f"Starting Photo-z Experiment (Mode: {self.args.mode})")
        
        # Define Stress Scenarios
        scenarios = ["baseline"]
        if self.args.mode == "full":
            scenarios += ["shift", "noise", "outliers"]

        # 1. Train all models
        trained_models = {}
        for name, config in self.configs.items():
            model, loss_fn = self.train_single(name, config)
            trained_models[name] = (model, loss_fn, config["type"])
            
        # 2. Evaluate on all scenarios
        start_time = time.time()
        
        for name, (model, loss_fn, mtype) in trained_models.items():
            for scen in scenarios:
                key = f"{name}_{scen}" if scen != "baseline" else name
                print(f"Evaluating {key}...")
                metrics = self.evaluate(model, loss_fn, mtype, stress_type=scen if scen != "baseline" else None)
                self.results[key] = metrics
                
        duration = time.time() - start_time
        print(f"Evaluation complete in {duration:.2f}s")
        
        # 3. Report
        print_comprehensive_metrics_table(self.results)
        
        # 4. Save
        self.save_results()

    def save_results(self):
        """Save metrics to JSON."""
        # Convert numpy to float
        clean_results = {}
        for k, v in self.results.items():
            clean_results[k] = {
                m: float(val) if isinstance(val, (np.float32, np.float64)) else val
                for m, val in v.items() 
                if m not in ["predictions", "pred_stds", "targets"] # Don't dump arrays to JSON
            }
            
        try:
            with open("photoz_metrics.json", "w") as f:
                json.dump(clean_results, f, indent=2)
            print("Saved metrics to photoz_metrics.json")
            
            # Plotting
            compare_calibration(self.results)
            plt.savefig("photoz_calibration.png")
            print("Saved plots to photoz_calibration.png")
            
        except Exception as e:
            print(f"Error saving results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo-z Experiment Runner")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"],
                        help="Experiment mode: quick (sim data, fast) or full (real data, stress tests)")
    args = parser.parse_args()
    
    experiment = PhotoZExperiment(args)
    experiment.run()
