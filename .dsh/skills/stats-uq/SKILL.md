---
name: stats-uq
description: Statistics-principled uncertainty quantification — conformal coverage checks, proper scoring rules, calibration plots, PIT diagnostics, post-hoc calibration
metadata:
  domains: statistics, uncertainty-quantification, astronomy
---

# Stats-principled UQ

The ecosystem's UQ standard (extends science-core): **calibration over
confidence**, proper scoring, honest coverage, reproducibility. torchregress
is the shared implementation (losses, metrics, calibration, viz); torchz and
uspm reuse it. API names below are the current public surface — verify
signatures with `help()` / docstrings against the installed torchregress
version before writing code.

## 1. Report calibrated intervals with conformal prediction

Split conformal (CQR-style), calibration on a held-out split ONLY:

```python
from torchregress.losses.conformal import SplitConformal

cal = SplitConformal(alpha=0.1)                # 90% intervals
cal.calibrate(y_pred_cal, y_cal)               # calibration split — never train
lower, upper = cal.predict_interval(y_pred_test)   # (Tensor, Tensor)
```

Rules:

- The calibration split is held out; tuning `alpha` on the test set is
  leakage and must never happen.
- Use `normalize_fn` (per-sample difficulty) for adaptive intervals when
  heteroscedasticity is real — and document why.
- For multimodal predictive densities, `LevelSetConformalPredictor` beats
  interval CQR; read its docstring before defaulting to interval form.

## 2. Coverage checks (non-negotiable)

Verify on the TEST set, never the calibration set:

```python
from torchregress.metrics.interval import (
    interval_score,                              # Winkler score (sharpness + coverage penalty)
    prediction_interval_coverage_probability,
    interval_metrics_report,
)
from torchregress.metrics.distribution import (
    probability_integral_transform,
    kolmogorov_smirnov_uniform_statistic,
)
```

- Empirical coverage must match the stated alpha within sampling error.
- Report `interval_score` AND mean width — coverage alone is meaningless (a
  trivially wide interval hits any coverage).
- PIT check: for a calibrated predictive distribution, PIT values are
  ~ Uniform(0,1); test with the KS statistic. U-shaped PIT = overconfidence,
  ∩-shaped = underconfidence.

## 3. Proper scoring rules

- Full predictive distribution: CRPS (`ContinuousRankedProbabilityScore`
  metric, or closed-form `crps_gaussian(mean, y_true, std)`) and NLL
  (`gaussian_nll(mean, y_true, var)`).
- Quantile heads: `MultiQuantileLoss` for training, paired with
  `QuantileCrossoverLoss` to prevent quantile crossing.
- A point metric (MSE/MAE) is never the headline for a science claim.

## 4. Calibration diagnostics and plots

- Metrics: `ExpectedCalibrationError(n_bins=10)` (update with
  `y_pred_quantiles: Dict[float, Tensor]`, `y_true`) and
  `MarginalCalibrationError(n_bins=20)` (update with samples).
- Plots (`torchregress.viz.diagnostic`): `plot_reliability_diagram`,
  `plot_qq_plot`, `plot_residuals`, `plot_prediction_intervals`,
  `plot_residual_histogram` — check docstrings for expected argument shapes.
- Post-hoc fixes (fit on the CALIBRATION split, apply to test — never refit
  on test): `VarianceTemperatureScaler`, `IsotonicMeanCalibrator`,
  `PITCalibrator`.

## 5. Semi-supervised and shift-aware calibration

- `SemiConformalCalibrator` (calibration/semicp.py) uses unlabeled data to
  tighten intervals — use when unlabeled data is plentiful and cheap.
- `RepresentationShiftInflator` / `BinnedLabelShiftEstimator`
  (calibration/shift.py) handle distribution shift at test time — report both
  in-distribution and shifted coverage.

## Application per repo

- **torchregress** — the canonical implementations; every new loss/metric
  ships with a test that verifies calibration empirically (simulated ground
  truth where possible).
- **torchz** — photo-z heads (gaussian / quantile / binned_pdf) must report
  sigma_nmad and outlier fraction AND interval coverage + PIT calibration;
  n(z) work uses proper density scoring, not histogram eyeballing.
- **uspm** — latent representations are not point estimates: report
  predictive spread and marginal calibration on held-out spectra/photometry
  before any science claim (rest-frame SEDs, population studies).

## Guardrails

- Never report bare point estimates for science claims.
- Never tune on the test set; state the split scheme exactly.
- Report all trials and seeds; no p-hacking or cherry-picked bins.
- Every number in a report must be reproducible from the session log +
  provenance manifests.
