"""
Metrics Suite Showcase.

This script demonstrates all 55 evaluation metrics (point, distribution,
interval, multivariate, OOD, decision, ensemble, and weak ground truth)
available in the `torchregress.metrics` module.
"""

import numpy as np
import torch

from torchregress.metrics import (
    GaussianNLLEnsemble,
    MeanPredictionIntervalWidth,
    # Multivariate
    MultivariateMAE,
    MultivariateRMSE,
    NormalizedRMSE,
    PredictionIntervalCoverageProbability,
    RejectionPolicy,
    # Decision
    RiskCoverageCurve,
    TrimmedMeanSquaredError,
    attenuation_factor,
    continuous_ranked_probability_score,
    # Distribution
    distribution_metrics_report,
    ensemble_statistics,
    highest_posterior_density_coverage,
    highest_posterior_density_level,
    # Interval
    interval_metrics_report,
    kolmogorov_smirnov_uniform_statistic,
    ood_metrics_report,
    probability_integral_transform,
    # Point
    regression_metrics_report,
    task_agnostic_correlations,
    uncertain_gt_metrics_report,
    uncertainty_decomposition,
)


def main():
    print("================================================================================")
    print("                     torchregress Metrics Suite Showcase                        ")
    print("================================================================================")

    # Setup reproducible random data
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 150
    y_true = torch.randn(n_samples)
    y_pred = y_true + torch.randn(n_samples) * 0.3
    y_pred_std = torch.ones(n_samples) * 0.3 + torch.abs(y_true) * 0.1
    y_lower = y_pred - 1.96 * y_pred_std
    y_upper = y_pred + 1.96 * y_pred_std

    # 1. Point Regression Metrics
    print("\n--- 1. Point Regression Metrics ---")
    point_report = regression_metrics_report(y_pred, y_true, as_numpy=True)
    for k, v in point_report.items():
        print(f"  {k:20s}: {v:.6f}")

    nrmse_metric = NormalizedRMSE()
    nrmse_metric.update(y_pred, y_true)
    print(f"  Normalized RMSE     : {nrmse_metric.compute().item():.6f}")

    trimmed_mse = TrimmedMeanSquaredError(trim_fraction=0.1)
    trimmed_mse.update(y_pred, y_true)
    print(f"  Trimmed MSE (10%)   : {trimmed_mse.compute().item():.6f}")

    # Attenuation factor (signal degradation correction factor)
    atten_factor = attenuation_factor(y_pred, y_true)
    print(f"  Attenuation Factor  : {atten_factor.item():.6f}")

    # 2. Distributional Metrics
    print("\n--- 2. Distributional Metrics ---")
    # Wrap in a PyTorch normal distribution
    dist = torch.distributions.Normal(y_pred, y_pred_std)

    # We can also generate sample predictions from normal distribution
    samples = dist.sample((100,))  # [n_samples, batch]

    dist_report = distribution_metrics_report(
        dist=dist,
        y_true=y_true,
        samples=samples,
    )
    for k, v in dist_report.items():
        val = v.item() if isinstance(v, torch.Tensor) else v
        print(f"  {k:20s}: {val:.6f}")

    # Manual check of details
    crps_val = continuous_ranked_probability_score(y_pred, y_pred_std, y_true)
    print(f"  Manual CRPS         : {crps_val.mean().item():.6f}")

    # Highest Posterior Density (HPD) coverage
    hpd_level = highest_posterior_density_level(samples, credible_interval=0.90)
    hpd_cov = highest_posterior_density_coverage(y_true, samples, credible_interval=0.90)
    print(f"  HPD 90% Level Threshold: {hpd_level.mean().item():.6f}")
    print(f"  HPD 90% Coverage       : {hpd_cov.item():.6f}")

    pit_vals = probability_integral_transform(y_pred, y_pred_std, y_true)
    ks_stat = kolmogorov_smirnov_uniform_statistic(pit_vals)
    print(f"  PIT KS Uniformity Stat: {ks_stat.item():.6f}")

    # 3. Interval Metrics
    print("\n--- 3. Interval & Coverage Metrics ---")
    predictions = {"Baseline Model": {"lower": y_lower, "upper": y_upper}}
    int_report = interval_metrics_report(predictions, y_true, alpha=0.05)
    for model_name, metrics in int_report.items():
        print(f"  Model: {model_name}")
        for k, v in metrics.items():
            print(f"    {k:18s}: {v:.6f}")

    mpiw_metric = MeanPredictionIntervalWidth()
    mpiw_metric.update(y_lower, y_upper)
    picp_metric = PredictionIntervalCoverageProbability(alpha=0.05)
    picp_metric.update(y_lower, y_upper, y_true)
    print(f"  MPIW Class Metric   : {mpiw_metric.compute().item():.6f}")
    print(f"  PICP Class Metric   : {picp_metric.compute().item():.6f}")

    # 4. Out-of-Distribution (OOD) Metrics
    print("\n--- 4. Out-of-Distribution (OOD) Metrics ---")
    # Simulate features and prediction outputs
    x_test = torch.randn(50, 4)
    x_ref = torch.randn(200, 4)

    # Mahalanobis requires mean and covariance of train features
    mean_feat = x_ref.mean(dim=0)
    cov_feat = torch.cov(x_ref.T)

    model_output = {"predictions": torch.randn(50, 2)}  # typicality expects logits or probs

    ood_report = ood_metrics_report(
        model_output=model_output,
        x_test=x_test,
        x_reference=x_ref,
        mean=mean_feat,
        cov=cov_feat,
        samples=samples,  # predictive samples to compute entropy
    )
    for k, v in ood_report.items():
        print(f"  {k:20s}: {v.item():.6f}")

    # 5. Decision & Selective Prediction Metrics
    print("\n--- 5. Selective Prediction & Decision Metrics ---")
    # Simulate uncertainty-based rejection policy
    uncertainty_scores = y_pred_std.numpy()
    errors = torch.abs(y_pred - y_true).numpy()

    rcc = RiskCoverageCurve()
    rcc.update(errors, uncertainty_scores)
    coverage, risk = rcc.compute()
    print(
        f"  Risk-Coverage Curve (50% coverage): Coverage={coverage[len(coverage) // 2]:.2f}, Risk={risk[len(risk) // 2]:.6f}"
    )

    policy = RejectionPolicy(rejection_fraction=0.20)
    rejection_mask = policy(uncertainty_scores)
    print(
        f"  Selective Rejection Policy : Rejected {rejection_mask.sum()} of {len(uncertainty_scores)} samples ({rejection_mask.mean() * 100:.1f}%)"
    )

    # 6. Ensemble Metrics
    print("\n--- 6. Ensemble Metrics & Uncertainty Decomposition ---")
    # Simulate an ensemble of 5 models predicting mean and std
    n_members = 5
    ensemble_means = torch.randn(n_members, n_samples) * 0.5 + y_true[None, :]
    ensemble_stds = torch.ones(n_members, n_samples) * 0.3

    decomp = uncertainty_decomposition(ensemble_means, ensemble_stds)
    for k, v in decomp.items():
        print(f"  {k:25s}: {v.mean().item():.6f}")

    stats = ensemble_statistics(ensemble_means, ensemble_stds)
    print(f"  Ensemble Mean Stdev    : {stats['mean'].std().item():.6f}")
    print(f"  Ensemble Total Stdev   : {stats['total_std'].mean().item():.6f}")

    ens_nll = GaussianNLLEnsemble()
    ens_nll.update(ensemble_means, ensemble_stds, y_true)
    print(f"  Ensemble Gaussian NLL  : {ens_nll.compute().item():.6f}")

    # 7. Multivariate & Correlation Metrics
    print("\n--- 7. Multivariate & Correlation Metrics ---")
    # Simulate multi-target predictions
    y_true_mv = torch.randn(n_samples, 3)
    y_pred_mv = y_true_mv + torch.randn(n_samples, 3) * 0.2

    mv_mae = MultivariateMAE()
    mv_mae.update(y_pred_mv, y_true_mv)
    mv_rmse = MultivariateRMSE()
    mv_rmse.update(y_pred_mv, y_true_mv)
    print(f"  Multivariate MAE    : {mv_mae.compute().item():.6f}")
    print(f"  Multivariate RMSE   : {mv_rmse.compute().item():.6f}")

    # Task Agnostic Correlations
    corrs = task_agnostic_correlations(y_pred_mv, y_true_mv)
    print(f"  Task Agnostic Corr  : {corrs.mean().item():.6f}")

    # 8. Weak Ground Truth / Uncertain GT Metrics
    print("\n--- 8. Weak/Uncertain Ground Truth Metrics ---")
    # Target has variance (uncertain ground truth)
    target_variance = torch.ones(n_samples) * 0.1
    teacher_preds = y_pred + torch.randn(n_samples) * 0.05
    pseudo_conf = torch.rand(n_samples)  # Simulated confidence in [0, 1]

    wgt_report = uncertain_gt_metrics_report(
        pred_mean=y_pred,
        pred_variance=y_pred_std**2,
        target=y_true,
        target_variance=target_variance,
        teacher_pred=teacher_preds,
        pseudo_confidence=pseudo_conf,
    )
    for k, v in wgt_report.items():
        print(f"  {k:20s}: {v.item():.6f}")

    print("================================================================================")
    print("                     Metrics Suite Showcase completed!                          ")
    print("================================================================================")


if __name__ == "__main__":
    main()
