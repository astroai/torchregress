# EIV Method Comparison

This example compares multiple Error-in-Variables (EIV) loss variants on a shared measurement-error regression task, on either a synthetic dataset or a real tabular dataset with injected measurement error.

→ API: [`FunctionalEIVLoss`](../api/losses.md), [`OrthogonalDistanceRegressionLoss`](../api/losses.md). Guide: [EIV losses](../losses/eiv.md).

Script: `examples/eiv_method_comparison.py`

## Datasets

Select the dataset with `--dataset`. Each dataset keeps its own fixed seed, split sizes, and model width.

| `--dataset` | Data | Model | Defaults |
|:--|:--|:--|:--|
| `synthetic` (default) | 2-feature synthetic regression with injected feature/label measurement error | 1-hidden-layer Tanh MLP | `seed=321`, `n_train=96`, `n_test=96`, `epochs=20`, `lr=0.03`, $\sigma_x=0.20$, $\sigma_y=0.10$ |
| `diabetes` | Real `sklearn.datasets.load_diabetes` features/targets (targets standardized on clean train labels) with **synthetic** feature/label measurement-error injection | 2-hidden-layer Tanh MLP | `seed=4321`, `n_train=256`, `n_test=120`, `epochs=15`, `lr=0.02`, $\sigma_x=\sigma_y=0.08$ |

## What It Compares

- Baseline observed-data MSE training (context baseline)
- `FunctionalEIVLoss` (analytic)
- `FunctionalEIVLoss` (Monte Carlo)
- `StructuralEIVLoss`
- `OrthogonalDistanceRegressionLoss`
- `EnsembleEIVLoss`

## Metrics Reported

- Clean test MSE (true inputs / true targets)
- Observed test MSE (noisy inputs / noisy targets)
- Stress MSE (noisy inputs / clean targets)
- Runtime (`train_s`, `eval_s`)

## Fairness / Comparability Notes

- Fixed seed and shared measurement-error splits (synthetic generation, or a shared Diabetes split plus a shared injection policy)
- Shared model architecture and optimizer budget within each dataset
- Shared summary table formatting and runtime tracking
- ODR uses a reduced inner-optimization budget for tractable example runtime

## When This Fails

- The synthetic low-dimensional setup is not a substitute for a real benchmark.
- The `diabetes` run is real data plus **synthetic** measurement error, not a domain benchmark; Diabetes is a small tabular dataset, so rankings may not transfer to high-dimensional settings.
- ODR runtime can dominate on larger problems; compare quality/runtime tradeoffs on your target data.
- Structural EIV with zero cross-covariance in this example does not exercise all structural-error regimes.

## Run

```bash
pixi run python examples/eiv_method_comparison.py
pixi run python examples/eiv_method_comparison.py --dataset diabetes
```

## Summary Artifact

```bash
pixi run python examples/eiv_method_comparison.py --dataset diabetes \
  --summary-json-path reports/example_summaries/eiv_method_comparison_diabetes.json
```
