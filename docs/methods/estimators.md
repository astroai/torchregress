# Model-Agnostic UQ Wrappers

`torchregress.estimators` puts conformal intervals and post-hoc calibration on top
of **any** base model, plus a one-call calibrated deep-ensemble recipe. The
wrappers are small, sklearn-style and delegate all the maths to the library's
existing classes ([`SplitConformal`](../api/conformal.md), `CQR`, `CVPlus`,
[`VarianceTemperatureScaler`](calibration.md), `IsotonicMeanCalibrator`).

!!! warning "Experimental API, may change in 0.4"
    The names, signatures and defaults on this page are new in 0.3.0 and are not
    covered by the stability promise of the loss and metric APIs. The underlying
    conformal and calibration classes are unchanged.

!!! abstract "What it is"
    - [`ConformalRegressor`](../api/estimators.md#conformalregressor): split, normalized, CQR, CV+ or Jackknife+ intervals for a scikit-learn or LightGBM model, a quantile model, a torch module with a training callable, or a fitted [`TabularFit`](../api/models.md#tabularfit).
    - [`CalibratedRegressor`](../api/estimators.md#calibratedregressor): variance temperature (optionally preceded by an isotonic mean map) for any model with a mean and a variance output.
    - [`calibrated_deep_ensemble`](../api/estimators.md#calibrated_deep_ensemble): Gaussian [`TabularMLP`](tabular_mlp.md) ensemble, mixture moments, variance temperature on a held-out carve and conformal intervals on top.

---

## When to use it

| Situation | Use |
|:----------|:----|
| Strong point model (gradient boosting, a network) and you want distribution-free intervals | `ConformalRegressor(method="split")` |
| Intervals should widen where the model is uncertain | `method="normalized"` (needs a spread) or `method="cqr"` (needs quantiles) |
| Little data, no room for a calibration split | `method="cv+"` (guarantee $1-2\alpha$, $K$ refits) |
| Model already outputs a mean and a variance but they are over- or under-confident | `CalibratedRegressor` |
| Tabular data and you want epistemic spread, calibration and intervals in one call | `calibrated_deep_ensemble` |
| Weighted conformal under covariate shift | `ConformalRegressor` with `fit(..., cal_weights=w)` and `predict_interval(X, test_weights=w_test)` |
| Non-exchangeable data, multi-target, level sets or custom scores | The classes in [Conformal prediction](conformal/index.md) directly |

---

## Mathematical background

Let $\hat\mu$ be the base prediction and $(x_i, y_i)_{i\le n}$ a calibration set
that the base did not see.

**Split.** Scores $s_i = |y_i - \hat\mu(x_i)|$; with
$\hat q$ the $\lceil (n+1)(1-\alpha)\rceil$-th smallest score,

$$C(x) = \bigl[\hat\mu(x) - \hat q,\ \hat\mu(x) + \hat q\bigr],
\qquad \Pr(y \in C(x)) \ge 1-\alpha .$$

**Normalized.** With a spread $\hat\sigma(x) > 0$, $s_i = |y_i - \hat\mu(x_i)| / \hat\sigma(x_i)$ and
$C(x) = \hat\mu(x) \pm \hat q\,\hat\sigma(x)$.

**CQR.** With base quantiles $\hat q_{lo}, \hat q_{hi}$,
$s_i = \max\{\hat q_{lo}(x_i) - y_i,\ y_i - \hat q_{hi}(x_i)\}$ and
$C(x) = [\hat q_{lo}(x) - \hat q,\ \hat q_{hi}(x) + \hat q]$.

**CV+ / Jackknife+.** For fold models $\hat\mu_{-k(i)}$ trained without the fold of $i$ and
residuals $R_i = |y_i - \hat\mu_{-k(i)}(x_i)|$,

$$C(x) = \Bigl[\ q^-_{\alpha}\{\hat\mu_{-k(i)}(x) - R_i\},\ q^+_{1-\alpha}\{\hat\mu_{-k(i)}(x) + R_i\}\Bigr],$$

with the finite-sample order statistics of Barber et al. (2021). The guarantee is
$1 - 2\alpha$ (typically close to $1 - \alpha$ in practice); Jackknife+ is the
$K = n$ case.

**Calibrated Gaussian.** `CalibratedRegressor` returns
$\mathcal N\bigl(g(\hat\mu(x)),\ T\,\hat\sigma^2(x)\bigr)$, where $T$ minimises the
Gaussian NLL on the calibration set ($T = \overline{(y-\hat\mu)^2/\hat\sigma^2}$ in
closed form) and $g$ is an optional isotonic map.

**Ensemble mixture.** For $M$ Gaussian members $\mathcal N(\mu_m, \sigma_m^2)$,

$$\mu = \tfrac1M\sum_m \mu_m, \qquad
\sigma^2 = \underbrace{\tfrac1M\sum_m \sigma_m^2}_{\text{aleatoric}}
+ \underbrace{\tfrac1M\sum_m (\mu_m-\mu)^2}_{\text{epistemic}},$$

and the calibrated variance is $T\sigma^2$.

---

## Example

```python
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from torchregress.estimators import (
    CalibratedRegressor, ConformalRegressor, calibrated_deep_ensemble,
)

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(3000, 4))
y = np.sin(2 * X[:, 0]) + (0.15 + 0.4 * np.abs(X[:, 1])) * rng.normal(size=3000)
X_train, y_train, X_test, y_test = X[:2400], y[:2400], X[2400:], y[2400:]

# 1. Conformal intervals around a LightGBM point model (25% of the rows calibrate).
cr = ConformalRegressor(LGBMRegressor(n_estimators=200, verbose=-1), method="split", alpha=0.1)
cr.fit(X_train, y_train, seed=0)
lo, hi = cr.predict_interval(X_test)
print("split coverage", np.mean((y_test >= lo) & (y_test <= hi)))

# 2. Conformalized quantile regression around two quantile models.
qmodels = tuple(GradientBoostingRegressor(loss="quantile", alpha=a) for a in (0.05, 0.95))
cqr = ConformalRegressor(qmodels, method="cqr", alpha=0.1).fit(X_train, y_train)
lo, hi = cqr.predict_interval(X_test)

# 3. Calibrate a Gaussian model: any predict(X) -> (mean, variance) will do.
class Overconfident:
    def predict(self, X):
        return np.sin(2 * X[:, 0]), np.full(len(X), 0.01)

cal = CalibratedRegressor(Overconfident(), method="vts").fit(X_train, y_train)
mean, std = cal.predict_dist(X_test)
print("temperature", cal.temperature_)

# 4. One call: Gaussian TabularMLP ensemble + variance temperature + conformal intervals.
model = calibrated_deep_ensemble(X_train, y_train, n_members=5, epochs=60, alpha=0.1)
mean, std = model.predict_dist(X_test)
lo, hi = model.predict_interval(X_test)
```

On the synthetic design used by the test suite (three features, $y=\sin 2x_1$ plus
noise scaled by $0.15 + 0.4|x_2|$; 3 members, 40 epochs) the calibrated ensemble
reaches a mean absolute quantile-calibration error of 0.036 and 88% coverage at
$\alpha=0.1$. Over five seeds the split, normalized, CQR and CV+ wrappers cover
89.2%, 88.5%, 88.8% and 90.0% at $\alpha=0.1$ (600 training rows, 2000 test rows);
the first three calibrate on 150 rows, so a 1-2 point shortfall is within the
finite-sample fluctuation of the calibration quantile.

---

## Input and output types

- NumPy in, NumPy out; a tensor passed to `predict`, `predict_interval` or `predict_dist` gives tensors back (on its device, in its dtype). Training data and calibration targets may be either.
- Single-target regression only. Base predictions are made in float64 and handed to the conformal classes as tensors.
- `predict_interval(X, alpha=...)` re-thresholds the stored calibration scores, so changing $\alpha$ costs nothing.
- Base models are deep-copied before fitting (the object you pass stays unfitted); with `prefit=True` the base is used as is.

## Supported bases

| Base | `split` / `normalized` | `cqr` | `cv+` / `jackknife+` | `CalibratedRegressor` |
|:-----|:----------------------:|:-----:|:--------------------:|:---------------------:|
| `fit`/`predict` estimator (scikit-learn, LightGBM) | yes (`normalized` needs `predict_std` or `spread_model`) | no | yes | `predict` returns `(mean, var)` or `predict_dist` returns `(mean, std)` |
| Pair/triple of quantile estimators, or object with `predict_quantiles` | yes (centre as point) | yes | refit via the estimator protocol | no |
| `nn.Module` + `fit_fn(model, X, y)` | yes | yes (quantile columns) | yes | `[mean, log_variance]` outputs |
| Fitted `TabularFit` / `TabularEnsembleFit` | yes (prefit) | yes (quantile head) | no | Gaussian head |

---

## Limitations

- **Experimental API, may change in 0.4.**
- CV+ and Jackknife+ refit the base $K$ or $n$ times and guarantee $1-2\alpha$; Jackknife+ is only practical for small $n$ or cheap bases. `test_weights` are not supported for them.
- `normalized` with `spread_model` fits the spread on in-sample residuals of the training rows, which are optimistic for flexible bases; prefer a base that exposes its own spread, or CQR.
- `CalibratedRegressor` fits the temperature on the same rows that set the conformal threshold (`conformal=True`), so that guarantee is approximate (one fitted scalar). The same holds for `calibrated_deep_ensemble`.
- The recipe uses one random calibration carve and equal-weight mixture moments; the temperature is a single scalar, so it cannot fix heteroscedastic miscalibration.
- Evidence so far is synthetic (and parity against MAPIE on shared bases); no real medium-size tabular suites yet.

---

## Next steps

- [Estimators API](../api/estimators.md)
- [Conformal prediction](conformal/index.md) and [Post-training calibration](calibration.md)
- [Tabular MLP](tabular_mlp.md), the backbone of the recipe

---

## References

| # | Reference |
|:-:|:----------|
| 1 | J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman. ["Distribution-Free Predictive Inference for Regression."](https://arxiv.org/abs/1604.04173) *JASA*, **2018**. |
| 2 | Y. Romano, E. Patterson, E. J. Candes. ["Conformalized Quantile Regression."](https://arxiv.org/abs/1905.03222) *NeurIPS*, **2019**. |
| 3 | R. F. Barber, E. J. Candes, A. Ramdas, R. J. Tibshirani. ["Predictive inference with the jackknife+."](https://arxiv.org/abs/1905.02928) *Annals of Statistics*, **2021**. |
| 4 | R. J. Tibshirani, R. F. Barber, E. J. Candes, A. Ramdas. ["Conformal Prediction Under Covariate Shift."](https://arxiv.org/abs/1904.06019) *NeurIPS*, **2019**. |
| 5 | C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. ["On Calibration of Modern Neural Networks."](https://arxiv.org/abs/1706.04599) *ICML*, **2017**. |
| 6 | B. Lakshminarayanan, A. Pritzel, C. Blundell. ["Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles."](https://arxiv.org/abs/1612.01474) *NeurIPS*, **2017**. |
