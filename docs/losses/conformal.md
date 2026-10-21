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
| 4 | Lei, L., G'Sell, M., Rinaldo, A., Ryan, T. J. & Wasserman, L. (2018). Distribution-Free Predictive Inference for Regression. *JASA*, 113(523):1094–1111. |

## Density level sets and Wald regions

Two newer calibration layers are available for conditional density models:

- `ConformalizedDensity` / `ContrastiveFlowConformal` — split-conformal
  level sets using `-log p(y|x)` as the nonconformity score.  Scalar targets get
  intervals through `CTI`; multi-target targets get exact level-set membership
  through `contains`.  `ContrastiveFlowConformal` wraps a fitted
  `ContrastiveFlowLoss`'s positive-context density.
- `WaldoConformal` / `WaldoPosteriorCalibrator` / `WaldoDensityConformal` — WALDO's Wald statistic
  `(mean - y)^T Cov^{-1}(mean - y)` with Neyman inversion and optionally binned
  critical values.  The posterior calibrator estimates moments from predictive
  samples (flow/ensemble) and supports joint multi-target regions.

```python
from torchregress.losses import ConformalizedDensity, WaldoPosteriorCalibrator

cp = ConformalizedDensity(alpha=0.1)
cp.calibrate(lambda c, y: -flow_loss.log_prob(c, y), ctx_cal, y_cal)
lower, upper = cp.predict_interval(
    lambda c, y: -flow_loss.log_prob(c, y), ctx_test, -3.0, 3.0
)

waldo = WaldoPosteriorCalibrator(alpha=0.1)
waldo.calibrate(samples_cal, y_cal)
inside = waldo.contains(samples_test, y_test)
```

These are **marginal** split-conformal guarantees.  WALDO's original
conditional guarantee uses simulator draws at each parameter and quantile
regression for critical values; the binned critical-value option here is an
approximation and must be validated by parameter-region coverage.
