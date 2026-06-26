# Debugging Guide

This guide provides a structured approach to diagnosing and resolving common issues encountered when training regression models with `torchregress`.

## Bug Triage Matrix

Use this matrix to quickly identify potential causes based on observed symptoms.

| Symptom | Likely Root Cause | Metric/Diagnostic to Inspect | Fix Path |
|:---|:---|:---|:---|
| **Loss is NaN** | Exploding gradients or division by zero (e.g. variance -> 0) | `grad_norm`, `target.min/max`, `prediction.min/max` | 1. Enable gradient clipping.<br>2. Check for `NaN` in inputs/targets.<br>3. Enforce minimum variance in loss (e.g. `eps` in `GaussianNLL`). |
| **Loss does not decrease** | Learning rate too high/low, or broken data pipeline | `train_loss` vs `val_loss`, `learning_rate` | 1. Sweep learning rates.<br>2. Overfit on a single batch (should reach ~0 loss).<br>3. Verify input normalization. |
| **Poor Calibration (Expected Calibration Error / ECE high)** | Model is overconfident (variance too low) | [`ExpectedCalibrationError`](../api/metrics.md), `prediction_std` distribution | 1. Use `temperature_scaling`.<br>2. Add regularization (e.g. weight decay).<br>3. Check if targets have noise unseen by model. |
| **Intervals too wide (high width)** | Model is underconfident, or data is extremely noisy | `MeanPredictionIntervalWidth` (MPIW), `PredictionIntervalCoverageProbability` (PICP) | 1. If PICP >> 0.9, increase penalty for width in loss.<br>2. Check for outliers driving width up.<br>3. Verify `alpha` setting matches target coverage. |
| **Metrics mismatch (Func vs Class)** | Incorrect reduction or state accumulation | Difference between `Metric.compute()` and functional call | 1. Ensure `reduction` arg matches.<br>2. Call `metric.reset()` between epochs.<br>3. Check for `NaN` propagation differences. |
| **Crashes during evaluation** | Input shape mismatch or incompatible device | Tensor shapes, `.device` attribute | 1. Ensure `y_pred` and `target` have same shape (samples, dim).<br>2. Move all tensors to same device. |
| **"RuntimeError: element 0 of tensors does not require grad"** | Detached tensor used in loss calculation | Code inspection | 1. Ensure `y_pred` retains gradient history.<br>2. Don't use `torch.no_grad()` during training forward pass. |

## Common Failure Patterns

### 1. Numerical Instability in Probabilistic Losses
Losses involving logarithms (like [`GaussianNLLLoss`](../api/losses.md), `Dirichlet`) are prone to numerical instability if the predicted parameters (like variance) approach zero.
**Fix:** Always enforce a minimum value (epsilon) or use `softplus` + `epsilon` for positive constraints.

### 2. Masking Issues
When using weighted losses with masks, ensure the mask is correctly broadcasted and applied. A common bug is masking the loss but not normalizing by the *valid* number of samples, leading to tiny loss values.
**Fix:** Use [`WeightedLossWrapper`](../api/losses.md) or built-in weighted losses which handle reduction correctly.

### 3. Metric State Accumulation
`torchmetrics` accumulate state. If you reuse a metric object across validation batches without resetting, it will consume excessive memory or produce incorrect "rolling" averages.
**Fix:** Call `.reset()` at the start/end of each epoch.

## Requesting Help
If you encounter a bug not listed here, please open an issue with:
1. A minimal reproduction script.
2. The output of `torchregress-health`.
3. Versions of `torch` and `torchregress`.
