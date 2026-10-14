# Estimators API

Reference for `torchregress.estimators`: model-agnostic conformal and calibration
wrappers and a calibrated deep-ensemble recipe. **Experimental API, may change in
0.4.** Background, equations and limitations are in
[Model-Agnostic UQ Wrappers](../methods/estimators.md).

```python
from torchregress.estimators import (
    ConformalRegressor, CalibratedRegressor, calibrated_deep_ensemble,
)
```

---

## ConformalRegressor

`ConformalRegressor(base, method="split", alpha=0.1, n_folds=5, prefit=False, *, fit_fn=None, spread_model=None)`

| Argument | Description |
|:---------|:------------|
| `base` | Estimator with `fit`/`predict`; for `"cqr"` a pair `(lo, hi)` or triple `(lo, mid, hi)` of estimators or an object with `predict_quantiles(X)` (lowest quantile first, highest last); an `nn.Module` (one column: point, several: quantiles); a fitted `TabularFit` (implies `prefit=True`) |
| `method` | `"split"`, `"normalized"`, `"cqr"`, `"cv+"` or `"jackknife+"`; computed by `SplitConformal`, `CQR` and `CVPlus` |
| `alpha` | Miscoverage level; overridable per call |
| `n_folds` | Folds for `"cv+"` |
| `prefit` | The base is trained; `fit` only calibrates (not allowed for `"cv+"` / `"jackknife+"`) |
| `fit_fn` | `fit_fn(model, X, y)` training an `nn.Module` in place (float tensors, `y` of shape `(n, 1)`) |
| `spread_model` | Regressor fitted on training absolute residuals for `"normalized"` |

| Method | Description |
|:-------|:------------|
| `fit(X, y, X_cal=None, y_cal=None, calib_fraction=0.25, seed=0, *, cal_weights=None)` | Trains the base on the non-calibration rows (random seeded carve, or the explicit `X_cal`, `y_cal`; with `prefit=True` and no `X_cal` the data is the calibration set) and calibrates. `"cv+"` / `"jackknife+"` pool `X` with `X_cal` and refit per fold. Returns `self` |
| `predict(X)` | Base point prediction (mean over refits for `"cv+"` / `"jackknife+"`) |
| `predict_interval(X, alpha=None, test_weights=None)` | `(lower, upper)`; `test_weights` give weighted conformal thresholds (split, normalized, cqr) |

Attributes: `base_` (fitted base; list of refits for the cross methods), `n_calibration_`.

## CalibratedRegressor

`CalibratedRegressor(base, method="vts", *, alpha=0.1, conformal=False, prefit=True, fit_fn=None)`

| Argument | Description |
|:---------|:------------|
| `base` | `predict(X) -> (mean, variance)`, `predict_dist(X) -> (mean, std)`, a Gaussian `TabularFit` / `TabularEnsembleFit` (mixture moments), or an `nn.Module` with `[mean, log_variance]` outputs |
| `method` | `"vts"` (`VarianceTemperatureScaler`) or `"isotonic+vts"` (`IsotonicMeanCalibrator` on half of the calibration set, temperature on the other half) |
| `alpha` | Default interval level |
| `conformal` | Normalized split-conformal width instead of Gaussian quantiles |
| `prefit` | Base already trained (default); with `False` it is fitted on the non-calibration rows |

| Method | Description |
|:-------|:------------|
| `fit(X, y, X_cal=None, y_cal=None, calib_fraction=0.25, seed=0)` | Fits the calibrator on `(X_cal, y_cal)`, on `(X, y)` when prefit, or on a seeded carve |
| `predict(X)` | Calibrated mean |
| `predict_dist(X)` | Calibrated `(mean, std)` |
| `predict_interval(X, alpha=None, test_weights=None)` | Central interval; `test_weights` need `conformal=True` |

Attributes: `temperature_`, `isotonic_`, `base_`, `n_calibration_`.

## calibrated_deep_ensemble

`calibrated_deep_ensemble(X, y, *, n_members=5, val_fraction=0.2, loss="gaussian", alpha=0.1, conformal=True, seed=0, member_val_fraction=0.15, model_kwargs=None, **fit_kwargs) -> CalibratedRegressor`

Trains `n_members` Gaussian `TabularMLP` members with `fit_tabular_ensemble` on the
rows outside a seeded `val_fraction` carve, combines them as a Gaussian mixture
(total variance = mean member variance + variance of member means), fits a
variance temperature on the carve and, with `conformal=True`, sets interval widths
by normalized split conformal on the same carve. `loss` is `"gaussian"` or
`"beta_nll"` (`beta=0.5`); `model_kwargs` go to `TabularMLP` and `**fit_kwargs` to
`fit_tabular`. The result is a fitted `CalibratedRegressor` whose `base_` is the
`TabularEnsembleFit`.
