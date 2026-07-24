"""
Stellar Spectra Parameter Estimation with Multi-Target Deep Ensembles

This example demonstrates how to use Deep Ensembles in torchregress for
multi-target heteroscedastic regression on 1D stellar spectra.

Task:
    Given 1D stellar spectra (1000 channels), estimate stellar parameters:
      - Effective Temperature Teff (K)
      - Surface Gravity log(g) (dex)
      - Metallicity [Fe/H] (dex)

Uncertainty Decomposition:
    - Aleatoric Uncertainty: Heteroscedastic noise predicted by each network head
    - Epistemic Uncertainty: Disagreement across ensemble members (model uncertainty)
    - Total Uncertainty: Sigma_total^2 = Sigma_aleatoric^2 + Sigma_epistemic^2
"""

import os
import tempfile
from pathlib import Path

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.comparison import (
    compute_point_metrics,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.ensemble import BaseEnsembleModel
from torchregress.losses import GaussianNLLLoss
from torchregress.metrics import uncertainty_decomposition
from torchregress.viz import plot_corner_plot

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. Synthetic Stellar Spectra Data Generator
# ============================================================================


def generate_synthetic_stellar_spectra(n_samples=2500, n_channels=1000):
    teff = np.random.uniform(4000, 7000, size=n_samples)
    logg = np.random.uniform(1.0, 4.5, size=n_samples)
    feh = np.random.uniform(-2.0, 0.5, size=n_samples)

    wavelengths = np.linspace(4000, 7000, n_channels)
    spectra = np.zeros((n_samples, n_channels), dtype=np.float32)

    for i in range(n_samples):
        t_norm = (teff[i] - 5500) / 1000
        g_norm = logg[i] - 2.75
        z_norm = feh[i]

        cont = 1.0 - 0.05 * ((wavelengths - 5500) / 1500) ** 2
        line1 = 0.3 * np.exp(-((wavelengths - 4861) ** 2) / (2 * (3 + 0.5 * g_norm) ** 2))
        line2 = (0.2 + 0.1 * z_norm) * np.exp(-((wavelengths - 5172) ** 2) / (2 * 2.5**2))
        line3 = 0.25 * (1 + 0.2 * t_norm) * np.exp(-((wavelengths - 6563) ** 2) / (2 * 4.0**2))

        noise = np.random.normal(0, 0.02, size=n_channels)
        spec = cont - line1 - line2 - line3 + noise
        spectra[i] = np.clip(spec, 0.01, 1.5)

    targets = np.column_stack([teff, logg, feh]).astype(np.float32)
    return spectra, targets


# ============================================================================
# 2. Spectrum 1D CNN Member Model
# ============================================================================


class HeteroscedasticSpectrumCNN(nn.Module):
    """1D CNN Backbone + Diagonal Gaussian Head [mean (3), log_var (3)]."""

    def __init__(self, input_len=1000, n_targets=3):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_targets * 2)  # [mean, log_var]

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.conv_net(x).squeeze(-1)
        return self.head(feat)


# ============================================================================
# Main Execution
# ============================================================================


def main():
    print("=" * 70)
    print("Multi-Target Stellar Spectra Parameter Estimation via Deep Ensembles")
    print("=" * 70)

    n_samples = 2500
    n_channels = 1000
    spectra, targets = generate_synthetic_stellar_spectra(n_samples, n_channels)

    n_train = 2000
    train_spectra = torch.tensor(spectra[:n_train])
    train_targets = torch.tensor(targets[:n_train])
    test_spectra = torch.tensor(spectra[n_train:])
    test_targets = torch.tensor(targets[n_train:])

    target_mean = train_targets.mean(dim=0, keepdim=True)
    target_std = train_targets.std(dim=0, keepdim=True)
    train_targets_norm = (train_targets - target_mean) / target_std

    # Create Deep Ensemble of 5 CNN models
    n_ensemble_members = 5
    base_cnn = HeteroscedasticSpectrumCNN(input_len=n_channels, n_targets=3)
    ensemble = BaseEnsembleModel(base_model=base_cnn, ensemble_size=n_ensemble_members)
    loss_fn = GaussianNLLLoss(reduction="mean")

    print(f"\n1. Training Deep Ensemble ({n_ensemble_members} members)...")

    def train_ensemble():
        for m_idx, member in enumerate(ensemble.models):
            torch.manual_seed(42 + m_idx)
            optimizer = torch.optim.AdamW(member.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
            dataset = TensorDataset(train_spectra, train_targets_norm)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            member.train()
            for epoch in range(40):
                for spec_b, target_b in loader:
                    optimizer.zero_grad()
                    out = member(spec_b)
                    loss = loss_fn(out, target_b)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

    _, fit_seconds = timed_call(train_ensemble)
    print(f"   Ensemble training completed in {fit_seconds:.2f} seconds.")

    print("\n2. Evaluating uncertainty decomposition on test set...")
    ensemble.eval()
    with torch.no_grad():
        member_preds = [member(test_spectra) for member in ensemble.models]

    # Process member Gaussian outputs
    member_means = torch.stack([out[:, :3] for out in member_preds])  # [M, N_test, 3]
    member_logvars = torch.stack([out[:, 3:] for out in member_preds])  # [M, N_test, 3]
    member_vars = torch.exp(member_logvars)

    # Deconstruct Aleatoric & Epistemic variances via torchregress.metrics
    ens_mean_norm = member_means.mean(dim=0)  # [N_test, 3]
    epi_var_norm, alea_var_norm = uncertainty_decomposition(member_means, member_vars)
    tot_var_norm = alea_var_norm + epi_var_norm

    # Convert back to unnormalized physical units
    ens_mean_unnorm = ens_mean_norm * target_std + target_mean
    tot_std_unnorm = torch.sqrt(tot_var_norm) * target_std
    alea_std_unnorm = torch.sqrt(alea_var_norm) * target_std
    epi_std_unnorm = torch.sqrt(epi_var_norm) * target_std

    param_names = ["Teff (K)", "logg (dex)", "[Fe/H] (dex)"]
    metrics_summary = {}

    for dim_idx, p_name in enumerate(param_names):
        y_t = test_targets[:, dim_idx]
        y_p = ens_mean_unnorm[:, dim_idx]
        m = compute_point_metrics(y_p, y_t)
        rmse = float(np.sqrt(m["MSE"]))
        metrics_summary[p_name] = {
            "rmse": rmse,
            "mae": m["MAE"],
            "r2": m["R2"],
            "mean_alea_std": float(alea_std_unnorm[:, dim_idx].mean()),
            "mean_epi_std": float(epi_std_unnorm[:, dim_idx].mean()),
            "mean_tot_std": float(tot_std_unnorm[:, dim_idx].mean()),
        }
        print(
            f"   - {p_name:12s} | RMSE: {rmse:.4f} | R2: {m['R2']:.4f} | "
            f"Alea std: {metrics_summary[p_name]['mean_alea_std']:.4f} | "
            f"Epi std: {metrics_summary[p_name]['mean_epi_std']:.4f}"
        )

    print("\n3. Generating Ensemble Corner Plot for test spectrum...")
    test_idx = 0
    true_params_single = test_targets[test_idx].numpy()

    # Draw Monte Carlo samples from each member Gaussian distribution
    n_draws_per_member = 1000
    member_samples_list = []
    for m in range(n_ensemble_members):
        m_mu = member_means[m, test_idx] * target_std.squeeze(0) + target_mean.squeeze(0)
        m_std = torch.sqrt(member_vars[m, test_idx]) * target_std.squeeze(0)
        dist = torch.distributions.Normal(m_mu, m_std)
        draws = dist.sample((n_draws_per_member,)).cpu().numpy()
        member_samples_list.append(draws)

    total_samples = np.vstack(member_samples_list)

    plot_corner_plot(
        samples=total_samples,
        member_samples=member_samples_list,
        true_vals=true_params_single,
        param_names=[
            r"$T_{\mathrm{eff}}$ (K)",
            r"$\log g$ (dex)",
            r"$[\mathrm{Fe/H}]$ (dex)",
        ],
        title=r"Deep Ensemble Gaussian Posterior $p(T_{\mathrm{eff}}, \log g, [\mathrm{Fe/H}] \mid \mathrm{Spectrum})$",
        save_path="figures/stellar_spectra_ensemble.png",
    )

    output_json = Path("reports/comparison_summaries/stellar_spectra_ensemble.json")
    write_comparison_summary_json(
        output_json,
        example="stellar_spectra_ensemble",
        task="multi_target_stellar_spectra_ensemble_regression",
        config={
            "n_train": n_train,
            "n_test": len(test_spectra),
            "ensemble_size": n_ensemble_members,
            "fit_time_seconds": round(fit_seconds, 3),
        },
        rows=[
            {
                "method": f"Deep Ensemble ({n_ensemble_members} Heteroscedastic CNNs)",
                "Teff_RMSE": float(metrics_summary["Teff (K)"]["rmse"]),
                "logg_RMSE": float(metrics_summary["logg (dex)"]["rmse"]),
                "FeH_RMSE": float(metrics_summary["[Fe/H] (dex)"]["rmse"]),
                "Teff_R2": float(metrics_summary["Teff (K)"]["r2"]),
                "logg_R2": float(metrics_summary["logg (dex)"]["r2"]),
                "FeH_R2": float(metrics_summary["[Fe/H] (dex)"]["r2"]),
            }
        ],
        notes=[
            "Multi-target Gaussian Deep Ensemble for stellar spectra parameter estimation with aleatoric/epistemic decomposition."
        ],
    )
    print(f"\nSummary JSON exported to: {output_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
