# OOD / Selective Prediction Comparison

This example compares uncertainty-driven selective prediction and OOD signals on a shared synthetic task.

Script: `examples/ood_selective_prediction_comparison.py`

## What It Compares

- `DeepEnsemble`
- `HeteroscedasticEnsembleModel`-style ensemble uncertainty decomposition (example uses a small heteroscedastic ensemble)
- `MCDropoutWrapper`-style uncertainty (example uses MC dropout sampling proxy)
- `SWAG`
- `BayesianNeuralNetwork` (variational BNN)

## Metrics Reported

- ID / OOD MSE
- Risk-Coverage Curve (`AURC`)
- Rejection policy risk / coverage at fixed rejection fraction
- OOD uncertainty gap (ID vs OOD average uncertainty)
- Runtime (`train_s`, `eval_s`)

## Fairness / Comparability Notes

- Fixed seed and shared synthetic train/ID/OOD splits
- Shared model width/depth and epoch budget
- Shared summary table output with runtime tracking

## When This Fails

- Synthetic OOD shift is simple and may overstate separation quality.
- SWAG / BNN training here uses lightweight budgets for comparability, not tuned performance.
- OOD uncertainty-gap alone is not sufficient; combine with task metrics and risk-coverage analysis.

## Run

```bash
uv run python examples/ood_selective_prediction_comparison.py
```
