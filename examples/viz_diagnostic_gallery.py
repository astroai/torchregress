"""
Visualization Diagnostic Gallery.

This script demonstrates all 31 visualization and plotting functions in the
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
    plot_causal_uplift_qini,
    plot_censored_survival_curves,
    plot_conditional_density_slices,
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
    plot_reliability_diagram,
    plot_residual_histogram,
    plot_residuals,
    plot_risk_coverage_curve,
    plot_simex_extrapolation,
    plot_target_density_error_overlap,
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
    q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    y_pred_quantiles = {
        q: y_pred + np.percentile(np.random.normal(0, 0.5, 1000), int(q * 100)) for q in q_levels
    }

    # Simulate ensemble / posterior samples
    n_ensemble = 20
    predicted_samples = np.random.normal(
        loc=y_pred[None, :], scale=y_pred_std[None, :], size=(n_ensemble, n_samples)
    )

    # Create directories if not exist
    output_dir = "examples/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Demonstrate Diagnostic plots using a grid figure
    print("Generating diagnostic plots...")
    fig, axes = create_grid_figure(n_plots=6, ncols=3, figsize=(18, 12))

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
    save_figure(fig, "diagnostic_gallery", directory=output_dir, formats=["png"])
    print(f"Saved diagnostic grid plot to {output_dir}")

    # 3. Demonstrate remaining diagnostic plots
    fig_cal = plot_gaussian_reliability_diagram(
        y_pred, y_pred_std, y_true, return_figure=True, title="Gaussian Reliability Diagram"
    )
    save_figure(fig_cal, "gaussian_reliability", directory=output_dir, formats=["png"])

    # Quantile reliability diagram
    fig_rel = plot_reliability_diagram(
        y_pred_quantiles, y_true, return_figure=True, title="Quantile Reliability Diagram"
    )
    save_figure(fig_rel, "quantile_reliability", directory=output_dir, formats=["png"])

    fig_bin = plot_binned_metrics(
        y_pred, y_pred_std, y_true, metric="rmse", return_figure=True, title="Binned RMSE Plot"
    )
    save_figure(fig_bin, "binned_rmse", directory=output_dir, formats=["png"])

    # Target density vs error overlap — checks rare-target performance
    fig_overlap = plot_target_density_error_overlap(
        y_true, y_pred, return_figure=True, title="Target Density vs Error Overlap"
    )
    save_figure(fig_overlap, "density_error_overlap", directory=output_dir, formats=["png"])

    fig_dist = plot_distribution_comparison(
        predicted_samples,
        y_true,
        n_samples_to_show=4,
        return_figure=True,
        title="Distribution Comparison",
    )
    save_figure(fig_dist, "distribution_comparison", directory=output_dir, formats=["png"])

    # Conditional density slices — for mixture / flow models
    x_slices = x[:5].reshape(-1, 1).astype(np.float32)
    y_grid = np.linspace(-10, 10, 200).astype(np.float32)

    def _demo_density_fn(x_slice, y_grid_vals):
        # Use predicted mean as the Gaussian centre for realistic demo
        mu = float(2.0 * x_slice[0] + np.sin(3.0 * x_slice[0]))
        sigma = 0.5
        return np.exp(-0.5 * ((y_grid_vals - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    y_true_slices = y_true[:5]
    fig_cond = plot_conditional_density_slices(
        _demo_density_fn,
        x_slices,
        y_grid,
        y_true_slices=y_true_slices,
        return_figure=True,
        title="Conditional Density Slices",
    )
    save_figure(fig_cond, "conditional_density_slices", directory=output_dir, formats=["png"])

    # Censored survival curves — for survival / AFT regression
    n_surv = 100
    true_times = np.random.exponential(scale=50, size=n_surv)
    censor_times = np.random.exponential(scale=30, size=n_surv)
    observed_times = np.minimum(true_times, censor_times)
    censoring_indicators = (true_times <= censor_times).astype(int)
    time_grid = np.linspace(0, 80, 50)
    predicted_survival = np.exp(-0.5 * time_grid[None, :] / 25.0)
    fig_surv = plot_censored_survival_curves(
        predicted_survival,
        time_grid,
        observed_times,
        censoring_indicators,
        return_figure=True,
        title="Predicted vs Empirical Survival",
    )
    save_figure(fig_surv, "censored_survival", directory=output_dir, formats=["png"])

    # Simulate binary probs for calibration curve
    y_true_binary = (np.random.normal(0, 1, 500) > 0).astype(int)
    y_pred_probs = 1.0 / (1.0 + np.exp(-np.random.normal(0, 1.5, 500)))
    fig_curve = plot_calibration_curve(
        y_pred_probs, y_true_binary, return_figure=True, title="Probability Calibration Curve"
    )
    if isinstance(fig_curve, tuple):
        fig_curve = fig_curve[0]
    save_figure(fig_curve, "probability_calibration", directory=output_dir, formats=["png"])

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
    save_figure(fig_learning, "learning_curves", directory=output_dir, formats=["png"])

    fig_val = plot_validation_metrics(
        epochs, val_history, return_figure=True, title="Validation Metrics History"
    )
    save_figure(fig_val, "validation_metrics", directory=output_dir, formats=["png"])

    fig_stopping = plot_early_stopping(
        early_train_losses,
        early_val_losses,
        patience=5,
        return_figure=True,
        title="Early Stopping Analysis",
    )
    save_figure(fig_stopping, "early_stopping_analysis", directory=output_dir, formats=["png"])

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
    save_figure(fig_lr, "lr_finder", directory=output_dir, formats=["png"])

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
    save_figure(fig_comp, "performance_comparison_bar", directory=output_dir, formats=["png"])

    fig_radar = plot_performance_comparison(
        metrics_comp, plot_type="radar", return_figure=True, title="Performance Comparison (Radar)"
    )
    save_figure(fig_radar, "performance_comparison_radar", directory=output_dir, formats=["png"])

    # Parameter sensitivity simulation
    param_vals = {"hidden_dim": [32, 64, 128, 256]}
    sens_metrics = {"RMSE": [0.55, 0.48, 0.45, 0.46]}
    fig_sens = plot_parameter_sensitivity(
        param_vals, sens_metrics, return_figure=True, title="Hidden Dimension Sensitivity"
    )
    save_figure(fig_sens, "parameter_sensitivity", directory=output_dir, formats=["png"])

    # Feature importance simulation
    feat_names = [f"Feature {i}" for i in range(1, 11)]
    feat_importances = np.random.dirichlet(np.ones(10))
    fig_feat = plot_feature_importance(
        feat_importances, feat_names, return_figure=True, title="Feature Importance"
    )
    save_figure(fig_feat, "feature_importance", directory=output_dir, formats=["png"])

    # Model ensemble contributions
    member_preds = {
        f"Member {i + 1}": y_pred + np.random.normal(0, 0.1, n_samples) for i in range(5)
    }
    ensemble_prediction = np.mean(list(member_preds.values()), axis=0)
    fig_contrib = plot_model_ensemble_contributions(
        member_preds, ensemble_prediction, return_figure=True, title="Ensemble Member Contributions"
    )
    save_figure(fig_contrib, "ensemble_contributions", directory=output_dir, formats=["png"])

    # Risk-coverage curve — for selective prediction / OOD evaluation
    rejection_scores = y_pred_std + np.random.normal(0, 0.05, n_samples)
    fig_rc = plot_risk_coverage_curve(
        y_true,
        y_pred,
        rejection_scores,
        return_figure=True,
        title="Risk-Coverage Selective Prediction Curve",
    )
    save_figure(fig_rc, "risk_coverage", directory=output_dir, formats=["png"])

    # Causal uplift Qini curve
    n_causal = 200
    uplift_scores = np.random.normal(0.5, 1.0, n_causal)
    treatment = np.random.binomial(1, 0.5, n_causal)
    y_obs = 1.0 * treatment + 0.5 * uplift_scores + np.random.normal(0, 0.3, n_causal)
    fig_qini = plot_causal_uplift_qini(
        uplift_scores, treatment, y_obs, return_figure=True, title="Causal Uplift Qini Curve"
    )
    save_figure(fig_qini, "causal_uplift_qini", directory=output_dir, formats=["png"])

    # SIMEX extrapolation — for measurement error correction
    # (uses plain ASCII labels to avoid LaTeX/Unicode issues with Agg backend)
    try:
        plt.rcParams["text.usetex"] = False
        lambda_vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        sim_vals = np.array([1.0, 0.85, 0.72, 0.62, 0.55])
        simex_extrapolator = np.polynomial.Polynomial.fit(lambda_vals, sim_vals, 2)
        fig_simex = plot_simex_extrapolation(
            lambda_vals,
            sim_vals,
            simex_extrapolator,
            return_figure=True,
            title="SIMEX Extrapolation Diagnostics",
        )
        save_figure(fig_simex, "simex_extrapolation", directory=output_dir, formats=["png"])
    except Exception as exc:
        print(f"SIMEX plot skipped (backend limitation): {exc}")

    # 6. Demonstrate helper utilities
    print("Formatting metric label:", format_metric_label("expected_calibration_error"))
    palette = create_color_palette(5, palette_name="viridis")
    print("Created color palette:", palette)

    print("\nVisualization Diagnostic Gallery created successfully!")
    print(f"All outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
