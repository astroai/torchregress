# Benchmark CPU Baselines (2026-02-26)

<!-- BENCHMARK-BASELINE-GENERATED:START -->

## Smoke Baseline Summary

# Benchmark Summary (benchmark_smoke)

- Device: `cpu`
- Cases: `10`
- OK/Skipped/Error: `10/0/0`

| Case | Status | Mean (ms) | Std (ms) | Params |
|---|---:|---:|---:|---|
| gaussian_diag_nll_forward | ok | 0.132 | 0.077 |  |
| gaussian_multivariate_full_forward | ok | 0.210 | 0.096 |  |
| gaussian_low_rank_forward | ok | 0.383 | 0.174 |  |
| mdn_diagonal_forward | ok | 0.117 | 0.033 |  |
| mdn_full_forward | ok | 0.284 | 0.153 |  |
| functional_eiv_forward | ok | 98.607 | 98.189 |  |
| ensemble_variance_decomposition | ok | 0.079 | 0.020 |  |
| calibration_score_gaussian | ok | 0.476 | 0.084 |  |
| ood_metrics_report_combo | ok | 1.073 | 0.705 |  |
| normalizing_flow_forward_optional | ok | 2.060 | 0.117 |  |

## Sweep Baseline Summary (Grouped)

# Benchmark Summary (benchmark_sweep)

- Device: `cpu`
- Cases: `20`
- OK/Skipped/Error: `20/0/0`

| Case | Rows | OK | Mean of Means (ms) | Max Mean (ms) |
|---|---:|---:|---:|---:|
| sweep_ensemble_variance_decomposition | 4 | 4 | 0.070 | 0.105 |
| sweep_gaussian_low_rank_forward | 4 | 4 | 0.192 | 0.281 |
| sweep_gaussian_multivariate_full_forward | 4 | 4 | 0.197 | 0.445 |
| sweep_mdn_diagonal_forward | 4 | 4 | 0.088 | 0.135 |
| sweep_mdn_full_forward | 4 | 4 | 0.120 | 0.168 |
<!-- BENCHMARK-BASELINE-GENERATED:END -->
