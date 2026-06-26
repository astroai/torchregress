# Comparison API

Complete reference for `torchregress.comparison`. Every exported function is
listed here. For the full catalogue of comparison examples, see
[Examples overview](../examples/index.md).

→ **Related:** [Examples](../examples/index.md) · [Metrics API](metrics.md) · [Reporting](../reports/index.md)

---

## Module overview

The `torchregress.comparison` module provides **reproducibility helpers**
and **structured reporting** for comparison examples. It is designed to
ensure that method-to-method comparisons are fair, repeatable, and
machine-readable.

All comparison examples follow the same contract:

1. **Fixed seeds** via `set_comparison_seed()`.
2. **Common splits** and comparable training budgets across methods.
3. **Timed runs** via `timed_call()` to report training/inference cost.
4. **Aggregated outputs** as human-readable tables and JSON artifacts.

---

## Symbols

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `set_comparison_seed` | `set_comparison_seed(seed: int) → None` | Set Python, NumPy, and PyTorch random number generators for reproducible comparisons. Wraps `torchregress.utils.pytorch_compat.set_all_seeds`. |
| `timed_call` | `timed_call(fn, *args, **kwargs) → tuple[result, elapsed_seconds]` | Run a callable and return `(result, elapsed_seconds)`. Uses `time.perf_counter` for wall-clock timing. |
| `compute_point_metrics` | `compute_point_metrics(y_pred, y_true) → dict[str, float]` | Compute common point-prediction regression metrics: returns `{"MSE": …, "MAE": …, "R2": …}`. |
| `print_comparison_summary` | `print_comparison_summary(title, rows, *, metric_order=None) → None` | Print an aligned summary table for comparison examples. Default metric order: `["MSE", "MAE", "R2", "CRPS", "ECE", "train_s", "eval_s"]`. Includes a `"Notes"` column when any row has non-empty notes. |
| `print_fairness_notes` | `print_fairness_notes(*, title, seed_policy, train_budget, metric_policy) → None` | Print a compact comparability statement for example outputs. Emits a block with seed policy, training budget, and metric reporting policy under a titled header. |
| `write_comparison_summary_json` | `write_comparison_summary_json(path, *, example, task, config, rows, notes=None) → Path` | Write a machine-readable summary artifact (`artifact: "comparison_example_summary"`, `version: 1`) to a JSON file. Accepts data-classes or objects with `__dict__` as `config`. Creates parent directories as needed. Returns the output path. |

---

## JSON artifact schema

Every comparison example emits a standard JSON artifact:

```json
{
  "artifact": "comparison_example_summary",
  "version": 1,
  "example": "loss_comparison",
  "task": "regression",
  "config": { /* training hyperparams */ },
  "rows": [
    { "Method": "WeightedMSE", "MSE": 0.1234, "MAE": 0.2345, "R2": 0.89, "train_s": 1.2, "eval_s": 0.05, "Notes": "" }
  ],
  "notes": ["All methods use 5 seeds, 100 epochs, Adam lr=1e-3."]
}
```

These artifacts power the [Comparative Evidence Matrix](../reports/comparative_evidence_matrix.md)
and the [Real Data Recommendation Guide](../reports/real_data_recommendation_guide.md).

---

## Next steps

- [Examples overview](../examples/index.md) — browse all comparison examples
- [Metrics API](metrics.md) — the full evaluation metrics surface
- [Reports & evidence](../reports/index.md) — machine-readable benchmark matrices
- [Method selection guide](../guide/method-selection.md) — choose methods by problem type
