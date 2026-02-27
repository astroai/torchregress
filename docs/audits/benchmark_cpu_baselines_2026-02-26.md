# Benchmark CPU Baselines (2026-02-26)

<!-- BENCHMARK-BASELINE-GENERATED:START -->

## Smoke Baseline Summary

# Benchmark Summary (benchmark_smoke)

- Device: `cpu`
- Cases: `10`
- OK/Skipped/Error: `10/0/0`

| Case | Status | Mean (ms) | Std (ms) | Params |
|---|---:|---:|---:|---|
| gaussian_diag_nll_forward | ok | 0.151 | 0.072 |  |
| gaussian_multivariate_full_forward | ok | 0.192 | 0.031 |  |
| gaussian_low_rank_forward | ok | 0.435 | 0.120 |  |
| mdn_diagonal_forward | ok | 0.371 | 0.165 |  |
| mdn_full_forward | ok | 0.365 | 0.050 |  |
| functional_eiv_forward | ok | 1.630 | 1.169 |  |
| ensemble_variance_decomposition | ok | 0.131 | 0.012 |  |
| calibration_score_gaussian | ok | 1.606 | 0.707 |  |
| ood_metrics_report_combo | ok | 0.773 | 0.114 |  |
| normalizing_flow_forward_optional | ok | 8.015 | 0.209 |  |

## Sweep Baseline Summary (Grouped)

# Benchmark Summary (benchmark_sweep)

- Device: `cpu`
- Cases: `20`
- OK/Skipped/Error: `20/0/0`

| Case | Rows | OK | Mean of Means (ms) | Max Mean (ms) |
|---|---:|---:|---:|---:|
| sweep_ensemble_variance_decomposition | 4 | 4 | 0.908 | 2.048 |
| sweep_gaussian_low_rank_forward | 4 | 4 | 0.651 | 1.102 |
| sweep_gaussian_multivariate_full_forward | 4 | 4 | 0.549 | 1.143 |
| sweep_mdn_diagonal_forward | 4 | 4 | 0.409 | 0.753 |
| sweep_mdn_full_forward | 4 | 4 | 0.476 | 0.891 |
<!-- BENCHMARK-BASELINE-GENERATED:END -->
