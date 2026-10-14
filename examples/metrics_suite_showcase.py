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
    crps_gaussian,
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

    trimmed_mse = TrimmedMeanSquaredError(proportion=0.1)
    trimmed_mse.update(y_pred, y_true)
    print(f"  Trimmed MSE (10%)   : {trimmed_mse.compute().item():.6f}")

    # Attenuation factor (signal degradation correction factor)
    atten_factor = attenuation_factor(y_pred, y_true)
    print(f"  Attenuation Factor  : {float(atten_factor):.6f}")

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

    # Analytic Gaussian CRPS (per sample, then averaged)
    crps_val = crps_gaussian(y_pred, y_true, y_pred_std, reduction="none")
    print(f"  Manual CRPS         : {crps_val.mean().item():.6f}")

    # Highest Posterior Density (HPD) calibration on a 1D density grid
    support = torch.linspace(-5.0, 5.0, 501)
    density = (
        torch.distributions.Normal(y_pred[:, None], y_pred_std[:, None]).log_prob(support).exp()
    )
    hpd_level = highest_posterior_density_level(support, density, y_true)
    hpd_cov = highest_posterior_density_coverage(support, density, y_true, alpha=0.90)
    print(f"  HPD Level (mean)       : {hpd_level.mean().item():.6f}")
    print(f"  HPD 90% Coverage       : {hpd_cov:.6f}")

    pit_dist = torch.distributions.Normal(y_pred, y_pred_std)
    pit_vals = probability_integral_transform(pit_dist.cdf, y_true)
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
    picp_metric = PredictionIntervalCoverageProbability()
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

    # Typicality expects a Gaussian (mean, variance) matching the shape of x_test.
    model_output = {"mean": torch.zeros(50, 4), "variance": torch.ones(50, 4)}

    ood_report = ood_metrics_report(
        model_output=model_output,
        x_test=x_test,
        x_reference=x_ref,
        mean=mean_feat,
        cov=cov_feat,
        samples=samples.unsqueeze(-1),  # [n_samples, batch, output_dim] for entropy
    )
    for k, v in ood_report.items():
        print(f"  {k:20s}: {v.item():.6f}")

    # 5. Decision & Selective Prediction Metrics
    print("\n--- 5. Selective Prediction & Decision Metrics ---")
    # Reject the most uncertain samples first; risk defaults to squared error.
    rcc = RiskCoverageCurve()
    rcc.update(y_pred, y_true, y_pred_std)
    curve = rcc.compute()
    mid = len(curve["coverage"]) // 2
    print(
        f"  Risk-Coverage Curve (~50% coverage): "
        f"Coverage={curve['coverage'][mid].item():.2f}, Risk={curve['risk'][mid].item():.6f}"
    )
    print(f"  Area Under Risk-Coverage Curve (AURC): {curve['aurc'].item():.6f}")

    policy = RejectionPolicy(fraction=0.20)
    policy.update(y_pred, y_true, y_pred_std)
    outcome = policy.compute()
    print(
        f"  Selective Rejection Policy : Rejected {outcome['n_rejected'].item():.0f} of "
        f"{n_samples} samples, kept-sample risk={outcome['mean_risk'].item():.6f}, "
        f"coverage={outcome['coverage'].item():.2f}"
    )

    # 6. Ensemble Metrics
    print("\n--- 6. Ensemble Metrics & Uncertainty Decomposition ---")
    # Simulate an ensemble of 5 models predicting mean and variance
    n_members = 5
    ensemble_means = torch.randn(n_members, n_samples) * 0.5 + y_true[None, :]
    ensemble_vars = torch.full((n_members, n_samples), 0.3**2)

    decomp = uncertainty_decomposition(ensemble_means, ensemble_vars)
    for k, v in decomp.items():
        print(f"  {k:25s}: {v.mean().item():.6f}")

    ens_mean, ens_var = ensemble_statistics(ensemble_means)
    print(f"  Ensemble Mean Stdev    : {ens_mean.std().item():.6f}")
    print(f"  Ensemble Member Spread : {ens_var.sqrt().mean().item():.6f}")
    print(f"  Ensemble Total Stdev   : {decomp['total_uncertainty'].sqrt().mean().item():.6f}")

    ens_nll = GaussianNLLEnsemble()
    ens_nll.update(ensemble_means, ensemble_vars, y_true)
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
    # TAC scores a predicted covariance [batch, dim, dim] against the residuals.
    cov_mv = (0.2**2 * torch.eye(3, dtype=y_pred_mv.dtype, device=y_pred_mv.device)).expand(
        n_samples, 3, 3
    )
    tac = task_agnostic_correlations(y_pred_mv, y_true_mv, cov_mv)
    print(f"  Task Agnostic Corr  : {tac.item():.6f}")

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
