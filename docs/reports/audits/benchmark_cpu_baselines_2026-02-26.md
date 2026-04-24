# Benchmark CPU Baselines (2026-02-26)

<!-- BENCHMARK-BASELINE-GENERATED:START -->

## Smoke Baseline Summary

# Benchmark Summary (benchmark_smoke)

- Device: `cpu`
- Cases: `22`
- OK/Skipped/Error: `22/0/0`

| Case | Status | Mean (ms) | Std (ms) | Params |
|---|---:|---:|---:|---|
| gaussian_diag_nll_forward | ok | 0.099 | 0.020 |  |
| gaussian_multivariate_full_forward | ok | 0.207 | 0.033 |  |
| gaussian_low_rank_forward | ok | 0.717 | 0.611 |  |
| mdn_diagonal_forward | ok | 0.198 | 0.020 |  |
| mdn_full_forward | ok | 0.261 | 0.040 |  |
| functional_eiv_forward | ok | 0.600 | 0.132 |  |
| ensemble_variance_decomposition | ok | 0.734 | 1.034 |  |
| calibration_score_gaussian | ok | 1.268 | 0.404 |  |
| ood_metrics_report_combo | ok | 0.664 | 0.105 |  |
| ordinal_cross_entropy_forward | ok | 0.071 | 0.013 |  |
| ordinal_cumulative_link_forward | ok | 0.099 | 0.022 |  |
| censored_gaussian_nll_forward | ok | 0.830 | 0.766 |  |
| censored_aft_forward | ok | 0.272 | 0.024 |  |
| uncertain_gt_noisy_target_gaussian_nll_forward | ok | 0.069 | 0.004 |  |
| uncertain_gt_pseudo_label_nll_forward | ok | 0.159 | 0.013 |  |
| conformal_density_calibrate_predict | ok | 0.761 | 0.235 |  |
| conformal_monte_carlo_calibrate_predict | ok | 0.711 | 0.070 |  |
| ppi_mean_ci | ok | 0.541 | 0.098 |  |
| ppi_quantile_ci | ok | 1.279 | 0.098 |  |
| ppi_ols_ci | ok | 16.170 | 1.482 |  |
| causal_dr_ate_crossfit | ok | 4.329 | 1.406 |  |
| normalizing_flow_forward_optional | ok | 5.358 | 0.318 |  |

## Sweep Baseline Summary (Grouped)

# Benchmark Summary (benchmark_sweep)

- Device: `cpu`
- Cases: `38`
- OK/Skipped/Error: `38/0/0`

| Case | Rows | OK | Mean of Means (ms) | Max Mean (ms) |
|---|---:|---:|---:|---:|
| sweep_causal_dr_ate_crossfit | 2 | 2 | 2.543 | 3.144 |
| sweep_censored_gaussian_nll_forward | 4 | 4 | 0.286 | 0.421 |
| sweep_ensemble_variance_decomposition | 4 | 4 | 0.188 | 0.291 |
| sweep_gaussian_low_rank_forward | 4 | 4 | 0.369 | 0.492 |
| sweep_gaussian_multivariate_full_forward | 4 | 4 | 0.256 | 0.448 |
| sweep_mdn_diagonal_forward | 4 | 4 | 0.215 | 0.274 |
| sweep_mdn_full_forward | 4 | 4 | 0.578 | 0.985 |
| sweep_ordinal_cross_entropy_forward | 4 | 4 | 0.103 | 0.192 |
| sweep_ppi_ols_ci | 4 | 4 | 6.237 | 8.427 |
| sweep_uncertain_gt_noisy_target_gaussian_nll_forward | 4 | 4 | 0.092 | 0.143 |
<!-- BENCHMARK-BASELINE-GENERATED:END -->
