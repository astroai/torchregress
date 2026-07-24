"""
Stellar Spectra Parameter Estimation with Conditional Normalizing Flows

This example demonstrates how to use conditional normalizing flows in torchregress
for standard supervised regression on high-dimensional physical observables.

Task:
    Given 1D stellar spectra (1000 wavelength channels), estimate joint posterior
    distributions p(theta | spectrum) for stellar parameters:
      - Effective Temperature Teff (K)
      - Surface Gravity log(g) (dex)
      - Metallicity [Fe/H] (dex)

Why Normalizing Flows for Spectroscopy?
    1. Spectroscopy exhibits non-Gaussian parameter degeneracies (e.g. Teff vs log(g)).
    2. Flows model arbitrary non-linear joint densities without forcing symmetric Gaussian shapes.
    3. Exact sampling enables rendering full 1D and 2D corner plots for individual test spectra.
"""

import os
import tempfile
from pathlib import Path

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.comparison import (
    compute_point_metrics,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. Synthetic Stellar Spectra Data Generator
# ============================================================================


def generate_stellar_spectra_dataset(n_samples=2500, n_pixels=1000, noise_std=0.03):
    """
    Generate synthetic 1D stellar spectra and associated physical parameters.

    Target Parameters:
      - Teff:  [4000, 7000] K  (Effective temperature)
      - logg:  [1.0, 5.0] dex  (Surface gravity)
      - [Fe/H]: [-2.5, +0.5] dex (Metallicity)
    """
    # Uniform sampling across physical parameter ranges
    teff = torch.empty(n_samples, 1).uniform_(4000, 7000)
    logg = torch.empty(n_samples, 1).uniform_(1.0, 5.0)
    feh = torch.empty(n_samples, 1).uniform_(-2.5, 0.5)

    targets = torch.cat([teff, logg, feh], dim=1)  # [n_samples, 3]

    # Wavelength grid (normalized [0, 1])
    wave = torch.linspace(0, 1, n_pixels).unsqueeze(0)  # [1, 1000]

    # Normalized parameters for spectral synthesis
    t_norm = (teff - 4000.0) / 3000.0
    g_norm = (logg - 1.0) / 4.0
    m_norm = (feh + 2.5) / 3.0

    # Continuum baseline (Blackbody approximation / continuum curvature)
    continuum = 1.0 - 0.3 * (wave - 0.5) ** 2 - 0.1 * (1.0 - t_norm) * wave

    # Absorption line centers (synthetic spectral absorption lines)
    line_centers = torch.tensor([0.12, 0.18, 0.25, 0.33, 0.42, 0.51, 0.58, 0.67, 0.76, 0.85, 0.92])

    # Line strengths depend non-linearly on Teff, logg, and [Fe/H]
    # Balmer lines sensitive to Teff, Fe lines to [Fe/H], gravity lines to logg
    spectra = continuum.clone()

    for i, c in enumerate(line_centers):
        if i % 3 == 0:
            depth = 0.4 * (1.0 - t_norm * 0.7) + 0.1 * m_norm
            width = 0.015 + 0.008 * (1.0 - g_norm)
        elif i % 3 == 1:
            depth = 0.35 * m_norm * (1.0 + 0.2 * g_norm)
            width = 0.01 + 0.005 * m_norm
        else:
            depth = 0.3 * torch.sin(t_norm * np.pi) + 0.2 * g_norm
            width = 0.012 + 0.005 * (1.0 - t_norm)

        profile = depth * torch.exp(-0.5 * ((wave - c) / width) ** 2)
        spectra = spectra - profile

    # Add Gaussian observational noise (spectral shot noise)
    noise = torch.randn_like(spectra) * noise_std
    spectra = (spectra + noise).clamp(0.0, 1.5)

    return spectra, targets


# ============================================================================
# 2. 1D CNN Spectrum Feature Extractor (Backbone)
# ============================================================================


class SpectrumCNNBackbone(nn.Module):
    """
    1D CNN architecture for extracting context representations from 1D spectra.

    Maps input spectrum [B, 1000] -> context vector [B, context_dim].
    """

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


# ============================================================================
# 3. Built-in Corner Plot Renderer
# ============================================================================


def plot_stellar_corner_plot(
    samples,
    true_vals,
    param_names,
    title="Stellar Parameter Posterior Distribution $p(\\theta \\mid x_{\\mathrm{obs}})$",
    save_path="figures/stellar_spectra_flow_corner.png",
):
    """
    Generate a 3x3 publication-quality corner plot for posterior samples.

    Shows:
      - 1D marginal histograms with 16th, 50th (median), and 84th percentiles.
      - 2D joint density contours for parameter pairs.
      - True parameter values marked with red dashed crosshairs.
    """
    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(9, 8.5), dpi=150, sharex="col")

    color_hist = "#2b5c8f"
    color_contour = "#34495e"
    color_true = "#e74c3c"

    percentiles = np.percentile(samples, [16, 50, 84], axis=0)

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
                continue

            if i == j:
                p16, p50, p84 = percentiles[:, i]
                err_low = p50 - p16
                err_high = p84 - p50

                ax.hist(
                    samples[:, i],
                    bins=35,
                    density=True,
                    color=color_hist,
                    alpha=0.65,
                    edgecolor="none",
                )
                ax.axvline(p50, color=color_hist, linestyle="-", linewidth=1.5, label="Median")
                ax.axvline(p16, color=color_hist, linestyle="--", linewidth=1.0)
                ax.axvline(p84, color=color_hist, linestyle="--", linewidth=1.0)
                ax.axvline(
                    true_vals[i],
                    color=color_true,
                    linestyle="-",
                    linewidth=1.8,
                    label="Truth",
                )

                title_str = f"{param_names[i]}\n${p50:.2f}_{{-{err_low:.2f}}}^{{+{err_high:.2f}}}$"
                ax.set_title(title_str, fontsize=10, pad=4)
                ax.set_yticks([])

            else:
                ax.scatter(
                    samples[::3, j],
                    samples[::3, i],
                    s=2,
                    color=color_hist,
                    alpha=0.15,
                    rasterized=True,
                )

                try:
                    from scipy.stats import gaussian_kde

                    kde = gaussian_kde(np.vstack([samples[:, j], samples[:, i]]))
                    x_grid = np.linspace(samples[:, j].min(), samples[:, j].max(), 50)
                    y_grid = np.linspace(samples[:, i].min(), samples[:, i].max(), 50)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                    ax.contour(
                        X,
                        Y,
                        Z,
                        levels=4,
                        colors=color_contour,
                        linewidths=1.0,
                        alpha=0.8,
                    )
                except Exception:
                    pass

                ax.axvline(true_vals[j], color=color_true, linestyle="--", linewidth=1.2)
                ax.axhline(true_vals[i], color=color_true, linestyle="--", linewidth=1.2)
                ax.plot(
                    true_vals[j],
                    true_vals[i],
                    "s",
                    color=color_true,
                    markersize=6,
                    label="Truth",
                )

            if i == n_params - 1:
                ax.set_xlabel(param_names[j], fontsize=10)
            if j == 0 and i > 0:
                ax.set_ylabel(param_names[i], fontsize=10)

            ax.tick_params(labelsize=8, direction="in")
            ax.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle(title, fontsize=13, y=0.99, fontweight="bold")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    print(f"Saved corner plot artifact to: {save_path}")
    return fig


# ============================================================================
# 4. Main Training and Evaluation Pipeline
# ============================================================================


def main():
    print("=" * 70)
    print("Stellar Spectra Parameter Estimation with Conditional Normalizing Flows")
    print("=" * 70)

    print("\n1. Generating synthetic stellar spectra (1000 channels)...")
    spectra, targets = generate_stellar_spectra_dataset(n_samples=2500, n_pixels=1000)

    n_train = 2000
    train_spectra, test_spectra = spectra[:n_train], spectra[n_train:]
    train_targets, test_targets = targets[:n_train], targets[n_train:]

    target_mean = train_targets.mean(dim=0, keepdim=True)
    target_std = train_targets.std(dim=0, keepdim=True) + 1e-6

    train_targets_norm = (train_targets - target_mean) / target_std

    loader = DataLoader(
        TensorDataset(train_spectra, train_targets_norm), batch_size=64, shuffle=True
    )

    context_dim = 32
    n_features = 3

    cnn = SpectrumCNNBackbone(input_len=1000, context_dim=context_dim)

    flow = create_flow_model(
        n_features=n_features,
        context_dim=context_dim,
        flow_type="nsf",
        n_transforms=3,
        hidden_features=[64, 64],
    )

    loss_fn = NormalizingFlowLoss(flow=flow, reduction="mean")

    optimizer = torch.optim.AdamW(
        list(cnn.parameters()) + list(flow.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    print("\n2. Training CNN backbone + NSF Normalizing Flow head...")

    def train_loop():
        cnn.train()
        flow.train()
        for epoch in range(30):
            total_nll = 0.0
            for spec_b, target_b in loader:
                optimizer.zero_grad()
                context = cnn(spec_b)
                loss = loss_fn(context, target_b)
                loss.backward()
                optimizer.step()
                total_nll += loss.item() * len(spec_b)
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                mean_nll = total_nll / len(train_spectra)
                print(f"   Epoch {epoch + 1:02d}/30 | NLL Loss: {mean_nll:.4f}")

    _, fit_seconds = timed_call(train_loop)
    print(f"   Training completed in {fit_seconds:.2f} seconds.")

    print("\n3. Evaluating test set posterior distributions...")
    cnn.eval()
    flow.eval()

    n_samples_per_spectrum = 1000
    point_preds_median = []

    with torch.no_grad():
        for i in range(len(test_spectra)):
            context = cnn(test_spectra[i : i + 1])
            samples_norm = loss_fn.sample(context, n_samples=n_samples_per_spectrum).squeeze(0)
            samples_unnorm = samples_norm * target_std + target_mean

            median_pred = samples_unnorm.median(dim=0).values
            point_preds_median.append(median_pred)

    point_preds_median = torch.stack(point_preds_median)

    param_names = ["Teff (K)", "logg (dex)", "[Fe/H] (dex)"]
    metrics_summary = {}

    for dim_idx, p_name in enumerate(param_names):
        y_t = test_targets[:, dim_idx]
        y_p = point_preds_median[:, dim_idx]
        m = compute_point_metrics(y_p, y_t)
        rmse = float(np.sqrt(m["MSE"]))
        metrics_summary[p_name] = {"rmse": rmse, "mae": m["MAE"], "r2": m["R2"]}
        print(f"   - {p_name:12s} | RMSE: {rmse:.4f} | MAE: {m['MAE']:.4f} | R2: {m['R2']:.4f}")

    print("\n4. Generating 3x3 corner plot for test spectrum...")
    test_idx = 0
    test_spec_single = test_spectra[test_idx : test_idx + 1]
    true_params_single = test_targets[test_idx].numpy()

    with torch.no_grad():
        ctx_single = cnn(test_spec_single)
        draws_norm = loss_fn.sample(ctx_single, n_samples=3000).squeeze(0)
        draws_unnorm = (draws_norm * target_std + target_mean).cpu().numpy()

    from torchregress.viz import plot_corner_plot

    plot_corner_plot(
        samples=draws_unnorm,
        true_vals=true_params_single,
        param_names=[
            r"$T_{\mathrm{eff}}$ (K)",
            r"$\log g$ (dex)",
            r"$[\mathrm{Fe/H}]$ (dex)",
        ],
        title=r"Stellar Parameter Posterior $p(T_{\mathrm{eff}}, \log g, [\mathrm{Fe/H}] \mid \mathrm{Spectrum})$",
        save_path="figures/stellar_spectra_flow_corner.png",
    )

    output_json = Path("reports/comparison_summaries/stellar_spectra_flow_corner.json")
    write_comparison_summary_json(
        output_json,
        example="stellar_spectra_flow_corner",
        task="multi_target_stellar_spectra_regression",
        config={
            "n_train": n_train,
            "n_test": len(test_spectra),
            "flow_type": "nsf",
            "context_dim": context_dim,
            "fit_time_seconds": round(fit_seconds, 3),
        },
        rows=[
            {
                "method": "SpectrumCNN + NSF",
                "Teff_RMSE": float(metrics_summary["Teff (K)"]["rmse"]),
                "logg_RMSE": float(metrics_summary["logg (dex)"]["rmse"]),
                "FeH_RMSE": float(metrics_summary["[Fe/H] (dex)"]["rmse"]),
                "Teff_R2": float(metrics_summary["Teff (K)"]["r2"]),
                "logg_R2": float(metrics_summary["logg (dex)"]["r2"]),
                "FeH_R2": float(metrics_summary["[Fe/H] (dex)"]["r2"]),
            }
        ],
        notes=[
            "Joint stellar parameter posterior estimation from 1D spectra via normalizing flows."
        ],
    )
    print(f"\nSummary JSON exported to: {output_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
