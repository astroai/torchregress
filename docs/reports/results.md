# Benchmark results

> ← [Reports](index.md) | [Home](../index.md) →

This page points to the reproducible evidence behind the 0.3.0 release and the
post-release research benchmarks. All numbers come from the
[`torchregress-harness`](https://github.com/astroai/torchregress-harness)
repository; each suite uses fixed seeds, shared splits and comparable training
budgets.

## 0.3.0 parity report

The full external parity report (MAPIE, crepes, torchCP, torch-uncertainty,
Lightning-UQ-Box, sbi/sbibm, TabPFN/TabICL, foundation models, EIV, image
regression and imbalanced baselines) lives in the harness repository:

- [PARITY_REPORT.md](https://github.com/astroai/torchregress-harness/blob/main/PARITY_REPORT.md)
- [BENCHMARK_GPU_RESULTS.md](https://github.com/astroai/torchregress-harness/blob/main/BENCHMARK_GPU_RESULTS.md)

The parity report records every external library version/commit and marks
rows where the comparison is not decision-grade. The harness's
[seed-spread report](https://github.com/astroai/torchregress-harness/blob/main/reports/seed_spread_run6.md)
lists the rows whose ranking is not stable across seeds; rank flaky rows only
with error bars.

## Selective highlights

These are directional, single-seed or small-seed measurements; see the linked
reports for caveats and the exact budgets.

### Errors in variables

On the synthetic EIV designs, `TLS`/`ODR` win low-noise linear problems,
`EnsembleEIV` and `RC` are safer at high noise, and input-noise marginalization

On California housing, `BMC` gives the best tail/worst-region tradeoff among
loss-level methods, `GroupDRO` wins worst-group MAE, and `VIR` wins NLL/CRPS.
On the synthetic stress target, simpler methods and representation-level losses
remain competitive; `RankSim` is neutral under the shared budget. See the same
summary for the measured table.

### LSST DESC PZ Data Challenge

The harness now benchmarks the official challenge task-set-1 files with
torchregress heads and TabM. On a held-out 20% split of the 100k training
catalog, the NSF flow gave the best 10yr point/p(z) metrics
(RMSE ≈ 0.020, σ_NMAD ≈ 0.010, NLL ≈ −2.70) while TabM was nearly as accurate
in ~30 s instead of ~45 min. These are strong internal numbers, **not** an
official leaderboard result: the challenge test labels are withheld.

## Reproducing

```bash
# torchregress-harness checkout
pixi run python tools/run_all.py --suites eiv --datasets synthetic_eiv --epochs 60
pixi run python tools/run_all.py --suites imbalanced_representation \
  --datasets stress_imbalanced,california_housing --epochs 60
pixi run python tools/run_all.py --suites pz_challenge \
  --datasets pz_challenge_cardinal_10yr --epochs 60
```

The harness is not part of the torchregress wheel; it is the external evidence
layer used for the parity and research reports.
