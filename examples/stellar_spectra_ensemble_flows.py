"""
Stellar Spectra Parameter Estimation with Deep Ensembles of Normalizing Flows

This example demonstrates how to combine Deep Ensembling with Conditional Normalizing Flows
in torchregress for state-of-the-art multi-target regression on 1D stellar spectra.

Key Concepts:
    - Normalizing Flow Head (NSF): Captures non-Gaussian aleatoric noise (degeneracies, heavy tails)
    - Deep Ensemble (M=4 Flow Models): Quantifies epistemic model uncertainty across member flow density estimates
    - Uncertainty Decomposition on Corner Plot:
        - Aleatoric Uncertainty (sigma_al): Average width/spread of individual flow distributions
        - Epistemic Uncertainty (sigma_ep): Disagreement between predicted member means
        - Total Posterior (sigma_tot): Combined mixture distribution p_ensemble(theta | spectrum)
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
from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model
from torchregress.viz import plot_corner_plot

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. Synthetic Stellar Spectra Generator
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
# 2. 1D CNN Context Backbone & Flow Module
# ============================================================================


class SpectrumCNNBackbone(nn.Module):
    def __init__(self, input_len=1000, context_dim=32):
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
        self.fc_context = nn.Linear(128, context_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.conv_net(x).squeeze(-1)
        return self.fc_context(feat)


class FlowEnsembleMember(nn.Module):
    def __init__(self, input_len=1000, n_targets=3, context_dim=32):
        super().__init__()
        self.backbone = SpectrumCNNBackbone(input_len=input_len, context_dim=context_dim)
        self.flow = create_flow_model(
            n_features=n_targets,
            context_dim=context_dim,
            flow_type="nsf",
            n_transforms=3,
            hidden_features=[64, 64],
        )
        self.loss_wrapper = NormalizingFlowLoss(flow=self.flow, reduction="mean")

    def forward(self, x, y):
        ctx = self.backbone(x)
        return self.loss_wrapper(ctx, y)

    def sample_posterior(self, x, n_samples=1000):
        ctx = self.backbone(x)
        return self.loss_wrapper.sample(ctx, n_samples=n_samples).squeeze(0)


# ============================================================================
# Main Execution
# ============================================================================


def main():
    print("=" * 70)
    print("Stellar Spectra Parameter Estimation via Ensemble of Normalizing Flows")
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

    n_ensemble_members = 4
    base_flow_member = FlowEnsembleMember(input_len=n_channels, n_targets=3, context_dim=32)
    ensemble = BaseEnsembleModel(base_model=base_flow_member, ensemble_size=n_ensemble_members)

    print(f"\n1. Training Ensemble of Normalizing Flows ({n_ensemble_members} NSF members)...")

    def train_flow_ensemble():
        for m_idx, member in enumerate(ensemble.models):
            torch.manual_seed(42 + m_idx * 10)
            optimizer = torch.optim.AdamW(member.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)
            dataset = TensorDataset(train_spectra, train_targets_norm)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            member.train()
            for epoch in range(35):
                for spec_b, target_b in loader:
                    optimizer.zero_grad()
                    loss = member(spec_b, target_b)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

    _, fit_seconds = timed_call(train_flow_ensemble()) if False else timed_call(train_flow_ensemble)
    print(f"   Flow Ensemble training completed in {fit_seconds:.2f} seconds.")

    print("\n2. Evaluating test set predictions across Flow Ensemble...")
    ensemble.eval()

    n_samples_per_member = 100
    member_point_preds = []

    with torch.no_grad():
        for member in ensemble.models:
            m_preds = []
            test_loader = DataLoader(TensorDataset(test_spectra), batch_size=50)
            for (spec_b,) in test_loader:
                ctx_b = member.backbone(spec_b)
                draws_norm = member.loss_wrapper.sample(
                    ctx_b, n_samples=n_samples_per_member
                )  # [50, 100, 3]
                draws_unnorm = draws_norm * target_std.unsqueeze(0) + target_mean.unsqueeze(0)
                m_preds.append(draws_unnorm.median(dim=1).values)
            member_point_preds.append(torch.cat(m_preds, dim=0))

    ens_point_preds = torch.stack(member_point_preds).mean(dim=0)  # [N_test, 3]

    param_names = ["Teff (K)", "logg (dex)", "[Fe/H] (dex)"]
    metrics_summary = {}

    for dim_idx, p_name in enumerate(param_names):
        y_t = test_targets[:, dim_idx]
        y_p = ens_point_preds[:, dim_idx]
        m = compute_point_metrics(y_p, y_t)
        rmse = float(np.sqrt(m["MSE"]))
        metrics_summary[p_name] = {"rmse": rmse, "mae": m["MAE"], "r2": m["R2"]}
        print(f"   - {p_name:12s} | RMSE: {rmse:.4f} | MAE: {m['MAE']:.4f} | R2: {m['R2']:.4f}")

    print("\n3. Generating Flow Ensemble Corner Plot with Aleatoric + Epistemic Decomposition...")
    test_idx = 0
    test_spec_single = test_spectra[test_idx : test_idx + 1]
    true_params_single = test_targets[test_idx].numpy()

    member_draws_unnorm = []
    with torch.no_grad():
        for member in ensemble.models:
            draws_norm = member.sample_posterior(test_spec_single, n_samples=1500)
            draws_unnorm = (draws_norm * target_std + target_mean).cpu().numpy()
            member_draws_unnorm.append(draws_unnorm)

    total_ensemble_draws = np.vstack(member_draws_unnorm)

    plot_corner_plot(
        samples=total_ensemble_draws,
        member_samples=member_draws_unnorm,
        true_vals=true_params_single,
        param_names=[
            r"$T_{\mathrm{eff}}$ (K)",
            r"$\log g$ (dex)",
            r"$[\mathrm{Fe/H}]$ (dex)",
        ],
        show_uncertainty_decomposition=True,
        title=r"Flow Ensemble Posterior $p(T_{\mathrm{eff}}, \log g, [\mathrm{Fe/H}] \mid \mathrm{Spectrum})$ (Aleatoric vs Epistemic)",
        save_path="figures/stellar_spectra_ensemble_flows.png",
    )

    output_json = Path("reports/comparison_summaries/stellar_spectra_ensemble_flows.json")
    write_comparison_summary_json(
        output_json,
        example="stellar_spectra_ensemble_flows",
        task="multi_target_stellar_spectra_ensemble_flows_regression",
        config={
            "n_train": n_train,
            "n_test": len(test_spectra),
            "flow_type": "nsf",
            "ensemble_size": n_ensemble_members,
            "fit_time_seconds": round(fit_seconds, 3),
        },
        rows=[
            {
                "method": f"Deep Ensemble ({n_ensemble_members} NSF Flow Models)",
                "Teff_RMSE": float(metrics_summary["Teff (K)"]["rmse"]),
                "logg_RMSE": float(metrics_summary["logg (dex)"]["rmse"]),
                "FeH_RMSE": float(metrics_summary["[Fe/H] (dex)"]["rmse"]),
                "Teff_R2": float(metrics_summary["Teff (K)"]["r2"]),
                "logg_R2": float(metrics_summary["logg (dex)"]["r2"]),
                "FeH_R2": float(metrics_summary["[Fe/H] (dex)"]["r2"]),
            }
        ],
        notes=[
            "Deep Ensemble of Normalizing Flows for joint stellar parameter estimation with aleatoric + epistemic corner plot visualization."
        ],
    )
    print(f"\nSummary JSON exported to: {output_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
