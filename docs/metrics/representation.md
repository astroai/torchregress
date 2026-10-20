# Representation Geometry and Frozen Readouts

A learned encoder can rank differently under different regression readouts.  The
companion methods paper therefore reports three descriptive summaries on frozen
features before fitting a readout:

| Diagnostic | Definition | Interpretation |
|:--|:--|:--|
| `neighbour_target_gap` | median absolute target difference among each point's $k$ nearest neighbours (standardized features), divided by target std | smaller = more target-homogeneous neighbourhoods |
| `target_ordering` | Spearman correlation between pairwise feature distance and pairwise target distance | larger = feature distances order targets better |
| `frozen_readout_metrics` | OLS and $k$-NN readouts on a deterministic frozen-feature split: RMSE, MAE, NMAD | shows whether a ranking survives a change of readout |

These capture different geometry: neighbourhood homogeneity and distance
ordering can move in opposite directions.  Report both rather than a single
"representation quality" score.

```python
from torchregress.metrics import (
    frozen_readout_metrics,
    neighbour_target_gap,
    representation_geometry_report,
    target_ordering,
)

features = encoder(x)
report = representation_geometry_report(features, z, k=10)
# {'neighbour_target_gap': ..., 'target_ordering': ...,
#  'ols_RMSE': ..., 'knn_NMAD': ...}
```

`nmad(y_pred, y_true)` is the normalized median absolute deviation
$1.4826\,\mathrm{median}|r|$ and is used in the readout metrics.

!!! warning "Descriptor, not a benchmark"
    The readout split here is a quick deterministic diagnostic.  A manuscript
    comparison should use a task-specific validation protocol, paired seeds, and
    test-sample uncertainty intervals; see
    [`paired_bootstrap_ci`](photo_z.md#pdf-width-and-paired-uncertainty).

## References

| # | Reference |
|:-:|:----------|
| 1 | Chang et al. (2026), companion methods paper; neighbourhood gap, target ordering and frozen readouts. |
| 2 | Zha, K. et al. ["Rank-N-Contrast."](https://arxiv.org/abs/2301.06658) NeurIPS 2023. |
