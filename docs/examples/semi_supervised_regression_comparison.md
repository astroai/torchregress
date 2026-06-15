# Semi-Supervised Regression Comparison

Script: `examples/semi_supervised_regression_comparison.py`

→ API: [`PseudoLabelConsistencyLoss`](../api/losses.md#pseudolabelconsistencyloss), [`PseudoLabelNLL`](../api/losses.md#pseudolabelnll). Guide: [Uncertain ground truth](../losses/uncertain_ground_truth.md).

Compares the pseudo-label + teacher-consistency branch of semi-supervised regression on a real-data proxy (`sklearn` Diabetes) using:

- `SupervisedMSE`
- `PseudoLabelConsistencyLoss`
- `PseudoLabelNLL`

Workflow:

- label only part of the train split
- fit a Gaussian teacher on labeled data
- generate confidence-weighted pseudo labels for unlabeled data
- train SSL students with shared architecture and budget

Reported metrics:

- `MSE`
- `MAE`
- `R2`
- `PseudoAcceptRate`
- `PseudoMeanConf`
- `train_s`, `eval_s`

## Run

```bash
uv run python examples/semi_supervised_regression_comparison.py
```

## Summary Artifact

```bash
uv run python examples/semi_supervised_regression_comparison.py \
  --summary-json-path reports/example_summaries/semi_supervised_regression_comparison_full.json
```
