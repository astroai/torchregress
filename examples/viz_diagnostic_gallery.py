"""
Visualization Diagnostic Gallery.

This script demonstrates all 26 visualization and plotting functions in the
`torchregress.viz` package. It generates synthetic data, creates a multi-panel
diagnostic report, and saves it to a file.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchregress.viz import (
    create_color_palette,
    create_grid_figure,
    enable_latex_rendering,
    format_metric_label,
    plot_binned_metrics,
    plot_calibration_curve,
    plot_distribution_comparison,
    plot_early_stopping,
    plot_feature_importance,
    plot_gaussian_reliability_diagram,
    plot_learning_curves,
    plot_lr_find_results,
    plot_model_ensemble_contributions,
    plot_parameter_sensitivity,
    plot_performance_comparison,
    plot_pit_histogram,
    plot_prediction_intervals,
    plot_qq_plot,
    plot_residual_histogram,
    plot_residuals,
    plot_uncertainty_vs_error,
    plot_validation_metrics,
    save_figure,
    set_style,
)


def main():
    print("Initializing Visualization Diagnostic Gallery...")
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Set standard premium styling
    set_style()
    enable_latex_rendering(False)  # Keep disabled for speed and local compatibility

    # Generate synthetic regression data
    n_samples = 200
    x = np.linspace(-3, 3, n_samples)
    y_true = 2.0 * x + np.sin(3.0 * x) + np.random.normal(0.0, 0.5, n_samples)

    # Simulate predictions: mean, std, and interval bounds
    y_pred = 2.0 * x + np.sin(3.0 * x) + np.random.normal(0.0, 0.1, n_samples)
    y_pred_std = 0.4 + 0.1 * np.abs(x)
    y_lower = y_pred - 1.96 * y_pred_std
    y_upper = y_pred + 1.96 * y_pred_std

    # Simulate quantile predictions for reliability diagram
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    {q: y_pred + np.percentile(np.random.normal(0, 0.5, 1000), q * 100) for q in quantiles}

    # Simulate ensemble / posterior samples
    n_ensemble = 20
    predicted_samples = np.random.normal(
        loc=y_pred[None, :], scale=y_pred_std[None, :], size=(n_ensemble, n_samples)
    )

    # Create directories if not exist
    output_dir = "examples/outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "diagnostic_gallery.png")

    # 2. Demonstrate Diagnostic plots using a grid figure
    print("Generating diagnostic plots...")
    fig, axes = create_grid_figure(n_plots=6, n_cols=3, figsize=(18, 12))
    axes = axes.flatten()

    plot_residuals(y_pred, y_true, ax=axes[0], title="1. Residual Plot")
    plot_prediction_intervals(
        y_pred, y_lower, y_upper, y_true, ax=axes[1], title="2. Prediction Intervals"
    )
    plot_qq_plot(y_pred, y_true, ax=axes[2], title="3. Q-Q Plot")
    plot_residual_histogram(y_pred, y_true, ax=axes[3], title="4. Residual Histogram")
    plot_pit_histogram(y_pred, y_pred_std, y_true, ax=axes[4], title="5. PIT Histogram")
    plot_uncertainty_vs_error(
        y_pred, y_pred_std, y_true, ax=axes[5], title="6. Uncertainty vs Error"
    )

    plt.suptitle("Regression Diagnostic Plot Gallery", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_path)
    print(f"Saved diagnostic grid plot to {output_path}")

    # 3. Demonstrate remaining diagnostic plots
    fig_cal = plot_gaussian_reliability_diagram(
        y_pred, y_pred_std, y_true, return_figure=True, title="Gaussian Reliability Diagram"
    )
    save_figure(fig_cal, os.path.join(output_dir, "gaussian_reliability.png"))

    fig_bin = plot_binned_metrics(
        y_pred, y_pred_std, y_true, metric="rmse", return_figure=True, title="Binned RMSE Plot"
    )
    save_figure(fig_bin, os.path.join(output_dir, "binned_rmse.png"))

    fig_dist = plot_distribution_comparison(
        predicted_samples,
        y_true,
        n_samples_to_show=4,
        return_figure=True,
        title="Distribution Comparison",
    )
    save_figure(fig_dist, os.path.join(output_dir, "distribution_comparison.png"))

    # Simulate binary probs for calibration curve
    y_true_binary = (np.random.normal(0, 1, 500) > 0).astype(int)
    y_pred_probs = 1.0 / (1.0 + np.exp(-np.random.normal(0, 1.5, 500)))
    fig_curve = plot_calibration_curve(
        y_pred_probs, y_true_binary, return_figure=True, title="Probability Calibration Curve"
    )
    if isinstance(fig_curve, tuple):
        fig_curve = fig_curve[0]
    save_figure(fig_curve, os.path.join(output_dir, "probability_calibration.png"))

    # 4. Demonstrate Monitoring plots
    print("Generating training monitoring plots...")
    epochs = list(range(1, 51))
    train_history = {
        "loss": [1.0 / (e**0.5) + np.random.normal(0, 0.02) for e in epochs],
        "rmse": [0.8 / (e**0.4) + np.random.normal(0, 0.015) for e in epochs],
    }
    val_history = {
        "loss": [1.1 / (e**0.45) + np.random.normal(0, 0.02) for e in epochs],
        "rmse": [0.9 / (e**0.38) + np.random.normal(0, 0.015) for e in epochs],
    }
    # Early stopping simulation: min validation loss around epoch 40, rising after
    early_val_losses = [
        1.1 / (e**0.45) if e <= 40 else 1.1 / (40**0.45) + 0.005 * (e - 40) for e in epochs
    ]
    early_train_losses = [1.0 / (e**0.5) for e in epochs]

    fig_learning = plot_learning_curves(
        train_history, val_history, return_figure=True, title="Training Learning Curves"
    )
    save_figure(fig_learning, os.path.join(output_dir, "learning_curves.png"))

    fig_val = plot_validation_metrics(
        epochs, val_history, return_figure=True, title="Validation Metrics History"
    )
    save_figure(fig_val, os.path.join(output_dir, "validation_metrics.png"))

    fig_stopping = plot_early_stopping(
        early_train_losses,
        early_val_losses,
        patience=5,
        return_figure=True,
        title="Early Stopping Analysis",
    )
    save_figure(fig_stopping, os.path.join(output_dir, "early_stopping_analysis.png"))

    # LR Find simulation
    lrs = np.logspace(-6, -1, 100).tolist()
    lr_losses = [
        2.0 - 0.5 * np.log10(lr + 1e-7)
        if lr < 1e-3
        else 2.0 - 0.5 * np.log10(1e-3) + 50.0 * (lr - 1e-3) ** 2
        for lr in lrs
    ]
    fig_lr = plot_lr_find_results(
        lrs, lr_losses, return_figure=True, title="Learning Rate Finder Curve"
    )
    if isinstance(fig_lr, tuple):
        fig_lr = fig_lr[0]
    save_figure(fig_lr, os.path.join(output_dir, "lr_finder.png"))

    # 5. Demonstrate Results plots
    print("Generating results plots...")
    metrics_comp = {
        "Gaussian NLL": {"RMSE": 0.45, "MAE": 0.35, "ECE": 0.02, "NLL": 0.55},
        "Evidential": {"RMSE": 0.48, "MAE": 0.37, "ECE": 0.03, "NLL": 0.60},
        "Quantile Reg": {"RMSE": 0.50, "MAE": 0.32, "ECE": 0.05, "NLL": 0.68},
    }
    fig_comp = plot_performance_comparison(
        metrics_comp, return_figure=True, title="Performance Comparison (Bar)"
    )
    save_figure(fig_comp, os.path.join(output_dir, "performance_comparison_bar.png"))

    fig_radar = plot_performance_comparison(
        metrics_comp, plot_type="radar", return_figure=True, title="Performance Comparison (Radar)"
    )
    save_figure(fig_radar, os.path.join(output_dir, "performance_comparison_radar.png"))

    # Parameter sensitivity simulation
    param_vals = {"hidden_dim": [32, 64, 128, 256]}
    sens_metrics = {"RMSE": [0.55, 0.48, 0.45, 0.46]}
    fig_sens = plot_parameter_sensitivity(
        param_vals, sens_metrics, return_figure=True, title="Hidden Dimension Sensitivity"
    )
    save_figure(fig_sens, os.path.join(output_dir, "parameter_sensitivity.png"))

    # Feature importance simulation
    feat_names = [f"Feature {i}" for i in range(1, 11)]
    feat_importances = np.random.dirichlet(np.ones(10))
    fig_feat = plot_feature_importance(
        feat_importances, feat_names, return_figure=True, title="Feature Importance"
    )
    save_figure(fig_feat, os.path.join(output_dir, "feature_importance.png"))

    # Model ensemble contributions
    member_contributions = np.array([0.15, 0.25, 0.35, 0.10, 0.15])
    fig_contrib = plot_model_ensemble_contributions(
        member_contributions, return_figure=True, title="Ensemble Member Contributions"
    )
    save_figure(fig_contrib, os.path.join(output_dir, "ensemble_contributions.png"))

    # 6. Demonstrate helper utilities
    print("Formatting metric label:", format_metric_label("expected_calibration_error"))
    palette = create_color_palette("viridis", n_colors=5)
    print("Created color palette:", palette)

    print("\nVisualization Diagnostic Gallery created successfully!")
    print(f"All outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
