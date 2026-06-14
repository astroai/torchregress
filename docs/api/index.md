# API Reference

The API reference is generated from the installed `torchregress` package.

- [Losses](losses.md)
- [Ensemble](ensemble.md)
- [Conformal](conformal.md)
- [Algorithms](algorithms.md)
- [Metrics](metrics.md)
- [Calibration](calibration.md)
- [Constraints](constraints.md)
- [Inference](inference.md)
- [Causal](causal.md)
- [Visualization](viz.md)
- [Utilities](utils.md)

## Package structure

```
torchregress/
├── losses/              # Regression and distributional losses
├── metrics/             # Point, interval, distribution, calibration, OOD metrics
├── calibration/         # Post-hoc transforms, calibration metrics, shift calibrator
├── ensemble/            # Deep / heteroscedastic ensembles
├── algorithms/          # IRLS, SIMEX, regression calibration (RC)
├── test_time/           # Label shift, transport, subspace alignment
├── inference/           # PPI and related inference helpers
├── constraints/         # Output-constrained heads
├── comparison.py        # Comparison-example helpers
├── prediction.py        # Predictive batch containers
├── viz/                 # Diagnostic and monitoring plots
└── utils/               # gaussian_output, validation, tensor_ops, reduction, transforms
```

## Core module: losses

```python
import torchregress.losses as losses

loss_fn = losses.WeightedHuberLoss(delta=1.0)
loss = loss_fn(y_pred, y_true)
```

[Learn more about loss functions →](losses.md)

## Core module: metrics

```python
from torchregress.metrics import mse, expected_calibration_error

point_error = mse(y_pred, y_true)
calibration = expected_calibration_error(quantiles, y_true)
```

[Learn more about metrics →](metrics.md)

## Comparison examples

Reproducible example scripts use `torchregress.comparison` for seeds, timing, and summary JSON:

```python
from torchregress.comparison import set_comparison_seed, write_comparison_summary_json
```
