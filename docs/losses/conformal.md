# Conformal Prediction

Conformal prediction produces **prediction intervals with finite-sample coverage guarantees** — no distributional assumptions required. Unlike parametric uncertainty methods (Gaussian NLL, ensembles) which estimate density, conformal prediction directly controls the *frequency* of coverage errors via a held-out calibration set.

!!! info "Conformal prediction is a methodology, not a loss function"
    The `ConformalLoss` wrapper and individual predictors (`SplitConformal`, `CQR`, `UACQR`, etc.) are exported from `torchregress.losses`, but the methodology spans calibration, predictors, and distributional variants. Full documentation is in the methods section:

    - **[Conformal Prediction Overview](../methods/conformal/index.md)** — methodology intro, comparison table, decision tree
    - **[Predictors](../methods/conformal/predictors.md)** — `SplitConformal`, `CQR`, `DensityConformal`, `MonteCarloConformal`, etc.
    - **[Distributional Conformal](../methods/conformal/distributional.md)** — `DistributionalConformal`, `CTI`

See the [Predictors page](../methods/conformal/predictors.md) for the recommended standalone API. For a quickstart, see [§4 of the Quickstart](../getting-started/quickstart.md#4-conformal-prediction-for-guaranteed-coverage).

## How the guarantee works

Given a calibration set $\{(x_i, y_i)\}_{i=1}^n$ and a nonconformity score $s(x, y)$ (e.g. the absolute residual for split conformal, or a quantile-residual score for CQR), compute the level

$$
\hat q = \left\lceil (n+1)(1-\alpha) \right\rceil\text{-th smallest of } \{s_1, \dots, s_n\},
$$

which is $\,+ \infty\,$ (infinite intervals, i.e. no finite guarantee is achievable) when $\lceil(n+1)(1-\alpha)\rceil > n$. Interval $\hat q$ then satisfies

$$
\Pr\big(y_{n+1} \in \mathcal{C}(x_{n+1})\big) \ge 1 - \alpha
$$

for **any** exchangeable $(x_{n+1}, y_{n+1})$ — no density model, no asymptotics. Violations happen only under exchangeability failure (distribution shift); weighted and transport variants in [Distributional Conformal](../methods/conformal/distributional.md) compensate for a known shift.

## Minimal example

```python
import torch
from torchregress.losses import SplitConformal

cal_pred, cal_true = torch.randn(500), torch.randn(500)
cp = SplitConformal(alpha=0.1)
cp.calibrate(cal_pred, cal_true)                # held-out calibration set

test_pred = torch.randn(200)
lower, upper = cp.predict_interval(test_pred)   # >= 90% coverage under exchangeability
```

!!! note
    The single-number `ConformalLoss` wrapper exists for loss-style integration; most users are better served by calling a predictor's `fit`/`predict` directly.

## References

| # | Reference |
|:-:|:----------|
| 1 | Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. |
| 2 | Romano, Y., Patterson, E. & Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS*. |
| 3 | Angelopoulos, A. N. & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. *arXiv:2107.07511*. |
