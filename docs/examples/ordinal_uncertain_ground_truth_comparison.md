# Ordinal Uncertain Ground Truth Comparison

Script: `examples/ordinal_uncertain_ground_truth_comparison.py`

→ API: [`OrdinalCrossEntropyLoss`](../api/losses.md), [`CumulativeLinkLoss`](../api/losses.md), [`PseudoLabelConsistencyLoss`](../api/losses.md).

Compares regression-as-classification methods when ordered labels are ambiguous and represented as soft bin probabilities.

Methods:

- `HardOrdinalCE`
- `SoftOrdinalCE`
- `SoftOrdinalCE+Pseudo`
- `SoftCumulativeLink`

What it demonstrates:

- plausibility-style soft ordered-bin targets on the labeled subset
- confidence-gated soft pseudo labels from a teacher model
- direct use of the existing ordinal losses with soft targets

Reported metrics:

- `Accuracy`
- `OrdinalMAE`
- `QWK`
- `TrueNLL`
- `PlausibilityCE`
- `PseudoAcceptRate`
- `train_s`, `eval_s`

## Run

```bash
pixi run python examples/ordinal_uncertain_ground_truth_comparison.py
```

## Summary Artifact

```bash
pixi run python examples/ordinal_uncertain_ground_truth_comparison.py \
  --summary-json-path reports/example_summaries/ordinal_uncertain_ground_truth_comparison_full.json
```
