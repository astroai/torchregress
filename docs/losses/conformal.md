# Conformal Prediction

`torchregress.losses.conformal` provides conformal predictors for calibrated interval
regression.

## Core APIs

- `ConformalLoss(method="split" | "cqr")`
- `SplitConformal`
- `CQR`
- `MultiTargetConformal`
- `DensityConformal`
- `PrevalenceAdjustedCP`
- `MonteCarloConformal`
- `DistributionalConformal`
- `CTI`
- `R2CConformal`

## Quick Start

```python
import torch
from torchregress.losses import DensityConformal

cp = DensityConformal(alpha=0.1)
cp.calibrate(y_pred_cal, y_cal)
lower, upper = cp.predict_interval(y_pred_test)
```

## Method Notes

- `SplitConformal`: residual-based baseline.
- `CQR`: conformalized quantile heads (`[q_lo, q_hi]` predictions).
- `DensityConformal`: widens intervals in low-density target regions.
- `PrevalenceAdjustedCP`: group-prevalence aware thresholds for rare regimes.
- `MonteCarloConformal`: uses MC predictive samples (e.g., dropout/ensembles).
- `DistributionalConformal` and `CTI`: for CDF/density-driven workflows.

## Calibration Contract

- Calibrate on held-out data only.
- Use shared train/cal/test splits for fair method comparisons.
- Coverage guarantees apply under exchangeability assumptions.
