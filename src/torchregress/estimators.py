"""Model-agnostic uncertainty wrappers (experimental).

.. warning::
   **Experimental API, may change in 0.4.** The names, signatures and defaults in
   this module are small, sklearn-style conveniences over the conformal and
   post-hoc calibration classes; they are not covered by the stability promise of
   the loss and metric APIs.

``ConformalRegressor`` puts split, normalized, CQR, CV+ or Jackknife+ conformal
intervals on top of *any* point (or quantile) predictor: scikit-learn, LightGBM, a
torch module with a training callable, or a fitted
:class:`~torchregress.models.TabularFit`. ``CalibratedRegressor`` recalibrates the
mean and variance of any Gaussian-output model with
:class:`~torchregress.calibration.VarianceTemperatureScaler` (and optionally
:class:`~torchregress.calibration.IsotonicMeanCalibrator`).
``calibrated_deep_ensemble`` is the one-call recipe: a Gaussian
:class:`~torchregress.models.TabularMLP` ensemble with mixture moments, a variance
temperature fitted on a held-out carve and optional conformal intervals on top.

The maths is not re-implemented here. Intervals come from
:class:`~torchregress.losses.conformal.SplitConformal`,
:class:`~torchregress.losses.conformal.CQR` and
:class:`~torchregress.losses.conformal.CVPlus`; this module only prepares their
inputs (base predictions, spreads, folds) and converts between NumPy and tensors.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import torch
from torch import nn

from .calibration.posthoc import IsotonicMeanCalibrator, VarianceTemperatureScaler
from .losses.beta_nll import BetaNLLLoss
from .losses.conformal import CQR, ConformalPredictor, CVPlus, SplitConformal
from .losses.gaussian import GaussianNLLLoss
from .models.tabular_mlp import TabularMLP
from .models.training import (
    TabularEnsembleFit,
    TabularFit,
    fit_tabular_ensemble,
)

__all__ = [
    "ConformalRegressor",
    "CalibratedRegressor",
    "calibrated_deep_ensemble",
]

ArrayLike = Union[np.ndarray, torch.Tensor]
ConformalMethod = Literal["split", "normalized", "cqr", "cv+", "jackknife+"]
CalibrationMethod = Literal["vts", "isotonic+vts"]
FitFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], Any]

_METHODS = ("split", "normalized", "cqr", "cv+", "jackknife+")
_CALIBRATIONS = ("vts", "isotonic+vts")


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def _as_data(X: Any) -> ArrayLike:
    """Keep tensors as they are; everything else becomes a NumPy array."""
    if isinstance(X, torch.Tensor):
        return X
    arr = np.asarray(X)
    if arr.dtype == object:
        arr = arr.astype(np.float64)
    return arr


def _features_np(X: ArrayLike) -> np.ndarray:
    """Features as a NumPy array (dtype kept)."""
    return X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)


def _np2(a: Any) -> np.ndarray:
    """Tensor or array to a float64 NumPy array of shape ``(n, k)``."""
    arr = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    arr = arr.astype(np.float64, copy=False)
    return arr.reshape(len(arr), -1)


def _np1(a: Any) -> np.ndarray:
    """Tensor or array to a flat float64 NumPy vector."""
    arr = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    return arr.astype(np.float64, copy=False).reshape(-1)


def _finite_y(y: Any, name: str = "y") -> np.ndarray:
    """Flat float64 target vector; NaN or infinite values raise ``ValueError``."""
    arr = _np1(y)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite (found NaN or infinite values)")
    return arr


def _t64(a: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


def _cast_out(ref: Any, arr: Any) -> Any:
    """``arr`` as NumPy, or as a tensor on ``ref``'s device and dtype if ``ref`` is one."""
    out = np.asarray(arr, dtype=np.float64)
    if isinstance(ref, torch.Tensor):
        dtype = ref.dtype if ref.is_floating_point() else torch.float32
        return torch.as_tensor(out).to(device=ref.device, dtype=dtype)
    return out


def _take(a: ArrayLike, idx: np.ndarray) -> ArrayLike:
    return a[torch.as_tensor(idx)] if isinstance(a, torch.Tensor) else a[idx]


def _concat(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.cat([a, b.to(a.dtype)])
    return np.concatenate([_features_np(a), _features_np(b)])


def _split_idx(n: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded random split into (train, calibration) row indices."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"calib_fraction must be in (0, 1), got {fraction}")
    n_cal = max(1, int(round(n * fraction)))
    if n_cal >= n:
        raise ValueError("calib_fraction leaves no training rows")
    perm = np.random.default_rng(seed).permutation(n)
    return perm[n_cal:], perm[:n_cal]


def _fold_ids(n: int, k: int, seed: int) -> np.ndarray:
    """Balanced random fold id (0..k-1) for every row."""
    perm = np.random.default_rng(seed).permutation(n)
    fold = np.empty(n, dtype=np.int64)
    fold[perm] = np.arange(n) % k
    return fold


def _module_outputs(module: nn.Module, X: ArrayLike) -> np.ndarray:
    """Forward ``X`` through ``module`` in eval mode; returns ``(n, k)`` float64."""
    param = next(module.parameters(), None)
    dtype = param.dtype if param is not None else torch.float32
    device = param.device if param is not None else torch.device("cpu")
    Xt = torch.as_tensor(_features_np(X)).to(dtype=dtype, device=device)
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            out = module(Xt)
    finally:
        module.train(was_training)
    if isinstance(out, (tuple, list)):
        out = torch.cat([o.reshape(len(Xt), -1) for o in out], dim=1)
    return _np2(out)


def _centre(arr: np.ndarray) -> np.ndarray:
    """Central column of a quantile block: the median column, else the lo/hi midpoint."""
    k = arr.shape[1]
    if k == 1:
        return arr[:, 0]
    if k % 2 == 1:
        return arr[:, k // 2]
    return 0.5 * (arr[:, 0] + arr[:, -1])


def _mixture_moments(members: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Moments of an equal-weight mixture of Gaussians.

    ``members`` has shape ``(M, n, 2)`` with columns ``[mean, log_variance]``. The
    mixture mean is the mean of the member means and the total variance is the mean
    of the member variances (aleatoric) plus the variance of the member means
    (epistemic, population variance across members).
    """
    means = members[..., 0]
    variances = np.exp(members[..., 1])
    return means.mean(axis=0), variances.mean(axis=0) + means.var(axis=0)


# ---------------------------------------------------------------------------
# Base-model adapter
# ---------------------------------------------------------------------------


class _Adapter:
    """Uniform view of the supported base models.

    Supported ``base`` objects: a fitted :class:`TabularFit` or
    :class:`TabularEnsembleFit`, an ``nn.Module`` (with ``fit_fn`` when it must be
    trained), a pair or triple of quantile models ``(lo, hi)`` / ``(lo, mid, hi)``,
    an object with ``predict_quantiles``, and anything with ``fit``/``predict``.
    """

    def __init__(self, base: Any, *, fit_fn: Optional[FitFn] = None) -> None:
        self.base = base
        self.fit_fn = fit_fn
        if isinstance(base, TabularFit):
            self.kind = "tabular_fit"
        elif isinstance(base, TabularEnsembleFit):
            self.kind = "tabular_ensemble"
        elif isinstance(base, nn.Module):
            self.kind = "module"
        elif isinstance(base, (list, tuple)):
            if len(base) not in (2, 3):
                raise ValueError(
                    "a quantile base must be a pair (lo, hi) or a triple (lo, mid, hi) of models"
                )
            self.kind = "quantile_models"
        else:
            self.kind = "estimator"

    @property
    def always_fitted(self) -> bool:
        """Torchregress fit objects carry trained weights and cannot be refitted here."""
        return self.kind in ("tabular_fit", "tabular_ensemble")

    def fresh_copy(self) -> "_Adapter":
        return _Adapter(copy.deepcopy(self.base), fit_fn=self.fit_fn)

    # -- fitting -------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        if self.always_fitted:
            raise ValueError(
                "a fitted TabularFit / TabularEnsembleFit cannot be refitted; use prefit=True"
            )
        y1 = _np1(y)
        if self.kind == "module":
            if self.fit_fn is None:
                raise ValueError(
                    "an nn.Module base needs fit_fn=callable(model, X, y) to be trained; "
                    "pass prefit=True for an already trained module"
                )
            module: nn.Module = self.base
            param = next(module.parameters(), None)
            dtype = param.dtype if param is not None else torch.float32
            Xt = torch.as_tensor(_features_np(X)).to(dtype)
            self.fit_fn(module, Xt, torch.as_tensor(y1).to(dtype).reshape(-1, 1))
            return
        Xn = _features_np(X)
        models = self.base if self.kind == "quantile_models" else [self.base]
        for m in models:
            if not hasattr(m, "fit"):
                raise ValueError(f"{type(m).__name__} has no fit(); pass prefit=True")
            m.fit(Xn, y1)

    # -- predictions ---------------------------------------------------------

    def _raw(self, X: ArrayLike) -> np.ndarray:
        """``(n, k)`` raw head outputs of a TabularFit or nn.Module."""
        if isinstance(self.base, TabularFit):
            return _np2(self.base.predict(X))
        return _module_outputs(self.base, X)

    def point(self, X: ArrayLike, *, want_quantiles: bool = False) -> np.ndarray:
        """Point prediction ``(n,)``."""
        if self.kind == "tabular_fit":
            raw = self._raw(X)
            if self.base.output_layout == "gaussian":
                return raw[:, 0]
            return _centre(raw)
        if self.kind == "module":
            raw = self._raw(X)
            return _centre(raw) if want_quantiles else raw[:, 0]
        if self.kind == "tabular_ensemble":
            if self._ensemble_is_gaussian():
                return self.gauss(X)[0]
            return _centre(_np2(self.base.predict(X)))
        Xn = _features_np(X)
        if self.kind == "quantile_models":
            models = self.base
            if len(models) == 3:
                return _np1(models[1].predict(Xn))
            q = self.quantiles(X)
            return 0.5 * (q[:, 0] + q[:, 1])
        est = self.base
        if hasattr(est, "predict"):
            out = est.predict(Xn)
            if isinstance(out, tuple):
                out = out[0]
            return _np1(out)
        if hasattr(est, "predict_quantiles"):
            return _centre(_np2(est.predict_quantiles(Xn)))
        raise TypeError(f"{type(est).__name__} has neither predict nor predict_quantiles")

    def quantiles(self, X: ArrayLike) -> np.ndarray:
        """Lower and upper quantile predictions, shape ``(n, 2)``."""
        if self.kind in ("tabular_fit", "module"):
            raw = self._raw(X)
            if raw.shape[1] < 2:
                raise ValueError(
                    "method='cqr' needs a base with at least two quantile outputs "
                    f"(lowest first, highest last); got {raw.shape[1]}"
                )
            return raw[:, [0, -1]]
        if self.kind == "quantile_models":
            Xn = _features_np(X)
            return np.stack(
                [_np1(self.base[0].predict(Xn)), _np1(self.base[-1].predict(Xn))], axis=1
            )
        if self.kind == "estimator" and hasattr(self.base, "predict_quantiles"):
            q = _np2(self.base.predict_quantiles(_features_np(X)))
            if q.shape[1] < 2:
                raise ValueError("predict_quantiles must return at least two columns")
            return q[:, [0, -1]]
        raise ValueError(
            "method='cqr' needs a quantile base: a pair/triple of models, an object with "
            "predict_quantiles, or a TabularFit / nn.Module with quantile outputs"
        )

    def _ensemble_is_gaussian(self) -> bool:
        return all(m.output_layout == "gaussian" for m in self.base.members)

    def gauss(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance of a Gaussian-output base."""
        if self.kind == "tabular_fit":
            if self.base.output_layout != "gaussian":
                raise ValueError(
                    "this TabularFit has no Gaussian head (output_layout != 'gaussian')"
                )
            raw = self._raw(X)
            return raw[:, 0], np.exp(raw[:, 1])
        if self.kind == "tabular_ensemble":
            if not self._ensemble_is_gaussian():
                raise ValueError("the ensemble members have no Gaussian head")
            members = self.base.predict_members(X).detach().cpu().numpy().astype(np.float64)
            return _mixture_moments(members)
        if self.kind == "module":
            raw = self._raw(X)
            if raw.shape[1] != 2:
                raise ValueError(
                    "a Gaussian nn.Module base must output [mean, log_variance] (2 columns); "
                    f"got {raw.shape[1]}"
                )
            return raw[:, 0], np.exp(raw[:, 1])
        if self.kind == "estimator":
            Xn = _features_np(X)
            if hasattr(self.base, "predict_dist"):
                mean, std = self.base.predict_dist(Xn)
                return _np1(mean), _np1(std) ** 2
            out = self.base.predict(Xn)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                return _np1(out[0]), _np1(out[1])
        raise ValueError(
            "a Gaussian base must return (mean, variance) from predict(), expose "
            "predict_dist() -> (mean, std), or be a Gaussian TabularFit / ensemble / nn.Module"
        )

    def spread(self, X: ArrayLike) -> np.ndarray:
        """Per-row spread used by ``method='normalized'``."""
        if self.kind == "estimator" and hasattr(self.base, "predict_std"):
            return _np1(self.base.predict_std(_features_np(X)))
        try:
            return np.sqrt(self.gauss(X)[1])
        except (ValueError, TypeError, AttributeError):
            pass
        try:
            q = self.quantiles(X)
        except ValueError:
            q = None
        if q is not None:
            return 0.5 * (q[:, 1] - q[:, 0])
        raise ValueError(
            "method='normalized' needs a spread: a base with predict_std / a Gaussian output "
            "/ quantile outputs, or pass spread_model=<regressor> to ConformalRegressor"
        )


class _CPCache:
    """Builds and caches calibrated conformal predictors per ``alpha``."""

    def __init__(self, make: Callable[[float], ConformalPredictor]) -> None:
        self._make = make
        self._cache: dict[float, ConformalPredictor] = {}

    def get(self, alpha: float) -> ConformalPredictor:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if alpha not in self._cache:
            self._cache[alpha] = self._make(alpha)
        return self._cache[alpha]


def _weights_t(test_weights: Any) -> Optional[torch.Tensor]:
    return None if test_weights is None else _t64(_np1(test_weights))


# ---------------------------------------------------------------------------
# ConformalRegressor
# ---------------------------------------------------------------------------


class ConformalRegressor:
    """Conformal prediction intervals on top of any regression model (experimental).

    .. warning::
       Experimental API, may change in 0.4.

    A thin sklearn-style wrapper: it trains (or reads) the base model, collects
    calibration predictions and hands them to the library's conformal classes
    (:class:`~torchregress.losses.conformal.SplitConformal`,
    :class:`~torchregress.losses.conformal.CQR`,
    :class:`~torchregress.losses.conformal.CVPlus`), so the intervals are exactly
    those of direct use.

    Parameters
    ----------
    base : object
        One of

        * an estimator with ``fit(X, y)`` / ``predict(X)`` (scikit-learn, LightGBM, ...);
        * for ``method="cqr"``: a pair ``(lo_model, hi_model)`` or triple
          ``(lo_model, mid_model, hi_model)`` of such estimators, or an object with
          ``predict_quantiles(X) -> (n, k)`` (lowest quantile in the first column,
          highest in the last);
        * a torch ``nn.Module`` (with ``fit_fn`` unless ``prefit=True``): one output
          column is a point prediction, two or more are read as quantiles (lowest
          first, highest last) for ``"cqr"``; outputs must be in target units;
        * a fitted :class:`~torchregress.models.TabularFit` (implies ``prefit=True``):
          a point or quantile head, or a Gaussian head whose mean is the point
          prediction.
    method : {"split", "normalized", "cqr", "cv+", "jackknife+"}
        ``"split"``: absolute-residual split conformal. ``"normalized"``: residuals
        divided by a per-row spread (base ``predict_std``, a Gaussian head's std, the
        half quantile width, or ``spread_model``). ``"cqr"``: conformalized quantile
        regression. ``"cv+"`` / ``"jackknife+"``: Barber et al. (2021) intervals from
        ``n_folds`` / ``n`` refits of the base on the pooled data (marginal
        guarantee ``1 - 2 alpha``, not ``1 - alpha``; Jackknife+ costs ``n`` refits).
    alpha : float, default 0.1
        Miscoverage level; :meth:`predict_interval` can override it.
    n_folds : int, default 5
        Folds for ``"cv+"``.
    prefit : bool, default False
        The base is already trained: :meth:`fit` only calibrates. Not available for
        ``"cv+"`` / ``"jackknife+"``, which refit.
    fit_fn : callable, optional
        ``fit_fn(model, X, y)`` that trains an ``nn.Module`` in place, called with
        float tensors ``X`` of shape ``(n, F)`` and ``y`` of shape ``(n, 1)``.
    spread_model : regressor, optional
        For ``"normalized"``: estimator fitted on the absolute residuals of the base
        on its training rows (must already be fitted when ``prefit=True``).

    Attributes
    ----------
    base_ : object
        The fitted base (a deep copy of ``base`` unless ``prefit=True``; a list of
        the refitted copies for ``"cv+"`` / ``"jackknife+"``).
    n_calibration_ : int
        Number of calibration points (pooled rows for ``"cv+"`` / ``"jackknife+"``).

    Notes
    -----
    Single-target only. NumPy in gives NumPy out and tensors in give tensors out
    (the type of the ``X`` passed to the method). Calibration ``y`` may be either.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from torchregress.estimators import ConformalRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 3)); y = X[:, 0] + rng.normal(size=400)
    >>> cr = ConformalRegressor(LinearRegression(), method="split", alpha=0.1).fit(X, y)
    >>> lo, hi = cr.predict_interval(X[:5])
    >>> bool((lo < hi).all())
    True
    """

    def __init__(
        self,
        base: Any,
        method: ConformalMethod = "split",
        alpha: float = 0.1,
        n_folds: int = 5,
        prefit: bool = False,
        *,
        fit_fn: Optional[FitFn] = None,
        spread_model: Any = None,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        self.base = base
        self.method = method
        self.alpha = alpha
        self.n_folds = n_folds
        self.fit_fn = fit_fn
        self.spread_model = spread_model
        adapter = _Adapter(base, fit_fn=fit_fn)
        self.prefit = bool(prefit) or adapter.always_fitted
        if self.prefit and method in ("cv+", "jackknife+"):
            raise ValueError(
                f"method={method!r} refits the base and needs a refittable, not a prefit, base"
            )
        self._adapter = adapter
        self._active = adapter
        self._members: list[_Adapter] = []
        self._cache: Optional[_CPCache] = None
        self._spread_fit: Any = None
        self._spread_floor = 1e-12
        self.base_: Any = None
        self.n_calibration_ = 0

    # -- fit -----------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_cal: Optional[ArrayLike] = None,
        y_cal: Optional[ArrayLike] = None,
        calib_fraction: float = 0.25,
        seed: int = 0,
        *,
        cal_weights: Optional[ArrayLike] = None,
    ) -> "ConformalRegressor":
        """Train the base (unless prefit) and calibrate the intervals.

        Parameters
        ----------
        X, y : array-like
            Training data. With ``prefit=True`` and no ``X_cal`` they are used as the
            calibration set.
        X_cal, y_cal : array-like, optional
            Explicit calibration set. If omitted, ``calib_fraction`` of the rows is
            carved out at random (seeded) and the base trains on the rest. For
            ``"cv+"`` / ``"jackknife+"`` they are pooled with ``X, y``.
        calib_fraction : float, default 0.25
            Calibration share of an automatic split.
        seed : int, default 0
            Seed of the split / folds.
        cal_weights : array-like, optional
            Importance weights of the calibration rows (weighted conformal; split,
            normalized and cqr). Aligned with ``X_cal`` if given, else with ``X``.
        """
        X = _as_data(X)
        y1 = _finite_y(y)
        if len(y1) != len(X):
            raise ValueError("X and y have different lengths")
        if (X_cal is None) != (y_cal is None):
            raise ValueError("pass both X_cal and y_cal, or neither")
        if y_cal is not None:
            _finite_y(y_cal, "y_cal")
        w_all = None if cal_weights is None else _np1(cal_weights)

        if self.method in ("cv+", "jackknife+"):
            if cal_weights is not None:
                raise ValueError("CV+ / Jackknife+ do not support cal_weights")
            if X_cal is not None and y_cal is not None:
                X = _concat(X, _as_data(X_cal))
                y1 = np.concatenate([y1, _np1(y_cal)])
            self._fit_cross(X, y1, seed)
            return self

        Xtr: Optional[ArrayLike] = None
        ytr: Optional[np.ndarray] = None
        if self.prefit:
            adapter = self._adapter
            self.base_ = self.base
            if X_cal is not None and y_cal is not None:
                Xc, yc = _as_data(X_cal), _np1(y_cal)
            else:
                Xc, yc = X, y1
            wc = w_all
        else:
            adapter = self._adapter.fresh_copy()
            if X_cal is not None and y_cal is not None:
                Xtr, ytr = X, y1
                Xc, yc, wc = _as_data(X_cal), _np1(y_cal), w_all
            else:
                tr, ca = _split_idx(len(X), calib_fraction, seed)
                Xtr, ytr = _take(X, tr), y1[tr]
                Xc, yc = _take(X, ca), y1[ca]
                wc = None if w_all is None else w_all[ca]
            adapter.fit(Xtr, ytr)
            self.base_ = adapter.base
        if wc is not None and len(wc) != len(yc):
            raise ValueError("cal_weights must match the calibration rows")
        self._active = adapter
        self._calibrate_single(adapter, Xtr, ytr, Xc, yc, wc)
        return self

    def _spread(self, adapter: _Adapter, X: ArrayLike) -> np.ndarray:
        if self.spread_model is not None:
            model = self._spread_fit if self._spread_fit is not None else self.spread_model
            return np.maximum(_np1(model.predict(_features_np(X))), self._spread_floor)
        return adapter.spread(X)

    def _calibrate_single(
        self,
        adapter: _Adapter,
        Xtr: Optional[ArrayLike],
        ytr: Optional[np.ndarray],
        Xc: ArrayLike,
        yc: np.ndarray,
        wc: Optional[np.ndarray],
    ) -> None:
        method = self.method
        y_t = _t64(yc)
        w_t = None if wc is None else _t64(wc)
        self.n_calibration_ = len(yc)
        make: Callable[[float], ConformalPredictor]
        if method == "cqr":
            q_t = _t64(adapter.quantiles(Xc))

            def make(a: float) -> ConformalPredictor:
                cp = CQR(alpha=a)
                cp.calibrate(q_t, y_t, weights=w_t)
                return cp

        elif method == "normalized":
            if self.spread_model is not None and Xtr is not None and ytr is not None:
                resid = np.abs(ytr - adapter.point(Xtr))
                sm = copy.deepcopy(self.spread_model)
                sm.fit(_features_np(Xtr), resid)
                self._spread_fit = sm
                self._spread_floor = float(max(1e-3 * np.median(resid), 1e-12))
            pred_t = _t64(adapter.point(Xc))
            spread_t = _t64(self._spread(adapter, Xc))

            def make(a: float) -> ConformalPredictor:
                cp = SplitConformal(alpha=a, normalize_fn=lambda _pred, spread: spread)
                cp.calibrate(pred_t, y_t, weights=w_t, x=spread_t)
                return cp

        else:
            pred_split_t = _t64(adapter.point(Xc))

            def make(a: float) -> ConformalPredictor:
                cp = SplitConformal(alpha=a)
                cp.calibrate(pred_split_t, y_t, weights=w_t)
                return cp

        self._cache = _CPCache(make)
        self._cache.get(self.alpha)  # fail early on bad calibration data

    def _fit_cross(self, X: ArrayLike, y: np.ndarray, seed: int) -> None:
        n = len(y)
        k = n if self.method == "jackknife+" else self.n_folds
        if k > n:
            raise ValueError("more folds than rows")
        fold = np.arange(n) if self.method == "jackknife+" else _fold_ids(n, k, seed)
        oof = np.empty(n)
        members: list[_Adapter] = []
        for j in range(k):
            held = fold == j
            member = self._adapter.fresh_copy()
            member.fit(_take(X, np.flatnonzero(~held)), y[~held])
            oof[held] = member.point(_take(X, np.flatnonzero(held)))
            members.append(member)
        self._members = members
        self.base_ = [m.base for m in members]
        self.n_calibration_ = n
        oof_t = _t64(oof)[:, None]
        y_t = _t64(y)[:, None]
        fold_t = torch.as_tensor(fold)

        def make(a: float) -> ConformalPredictor:
            cp = CVPlus(alpha=a)
            cp.calibrate_ensemble(oof_t, y_t, fold_t)
            return cp

        self._cache = _CPCache(make)
        self._cache.get(self.alpha)

    # -- predict -------------------------------------------------------------

    def _check_fitted(self) -> _CPCache:
        if self._cache is None:
            raise RuntimeError("call fit() before predicting")
        return self._cache

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Point prediction of the base (mean over refits for ``"cv+"`` / ``"jackknife+"``)."""
        self._check_fitted()
        Xd = _as_data(X)
        if self._members:
            point = np.mean([m.point(Xd) for m in self._members], axis=0)
        else:
            point = self._active.point(Xd, want_quantiles=self.method == "cqr")
        return _cast_out(X, point)

    def predict_interval(
        self,
        X: ArrayLike,
        alpha: Optional[float] = None,
        test_weights: Optional[ArrayLike] = None,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Prediction interval ``(lower, upper)`` with marginal coverage ``1 - alpha``.

        Parameters
        ----------
        X : array-like
            Features; the output type follows this input.
        alpha : float, optional
            Miscoverage level; defaults to the constructor's. Cheap to change: the
            calibration scores are kept and re-thresholded.
        test_weights : array-like, optional
            Importance weights ``w(x)`` of the test rows for weighted conformal
            (covariate shift); on the scale of ``cal_weights`` (unit scale if the
            calibration was unweighted). Not supported by ``"cv+"`` / ``"jackknife+"``.
        """
        cache = self._check_fitted()
        a = self.alpha if alpha is None else float(alpha)
        cp = cache.get(a)
        Xd = _as_data(X)
        tw = _weights_t(test_weights)
        if self._members:
            members = np.stack([m.point(Xd) for m in self._members])  # (K, n)
            lo, hi = cp.predict_interval(_t64(members)[:, :, None], test_weights=tw)
        elif self.method == "cqr":
            lo, hi = cp.predict_interval(_t64(self._active.quantiles(Xd)), test_weights=tw)
        elif self.method == "normalized":
            pred = _t64(self._active.point(Xd))
            spread = _t64(self._spread(self._active, Xd))
            lo, hi = cp.predict_interval(pred, x=spread, test_weights=tw)
        else:
            lo, hi = cp.predict_interval(_t64(self._active.point(Xd)), test_weights=tw)
        return _cast_out(X, _np1(lo)), _cast_out(X, _np1(hi))


# ---------------------------------------------------------------------------
# CalibratedRegressor
# ---------------------------------------------------------------------------


class CalibratedRegressor:
    """Post-hoc mean/variance calibration of a Gaussian-output model (experimental).

    .. warning::
       Experimental API, may change in 0.4.

    Fits :class:`~torchregress.calibration.VarianceTemperatureScaler` (and, for
    ``"isotonic+vts"``, :class:`~torchregress.calibration.IsotonicMeanCalibrator`
    on the mean first) on a held-out calibration set. The calibrated predictive
    distribution is :math:`\\mathcal{N}(g(\\mu), T\\sigma^2)`.

    Parameters
    ----------
    base : object
        A model with a Gaussian output: ``predict(X)`` returning ``(mean, variance)``,
        an object with ``predict_dist(X) -> (mean, std)``, a Gaussian
        :class:`~torchregress.models.TabularFit` or
        :class:`~torchregress.models.TabularEnsembleFit` (equal-weight mixture
        moments: mean of the member variances plus the variance of the member means),
        or an ``nn.Module`` with ``[mean, log_variance]`` outputs in target units.
    method : {"vts", "isotonic+vts"}
        ``"isotonic+vts"`` fits the isotonic mean map on a random half of the
        calibration set and the variance temperature on the other half.
    alpha : float, default 0.1
        Default level of :meth:`predict_interval`.
    conformal : bool, default False
        Use split-conformal intervals (scores :math:`|y-\\mu|/\\sigma` on the
        calibration set, :class:`~torchregress.losses.conformal.SplitConformal` with
        normalisation) instead of Gaussian quantiles. The temperature and the
        conformal threshold are fitted on the same rows, so the finite-sample
        guarantee is approximate (one fitted scalar).
    prefit : bool, default True
        The base is already trained and :meth:`fit` only calibrates. With ``False``
        the base is trained (deep copy) on the rows left after the calibration carve.
    fit_fn : callable, optional
        ``fit_fn(model, X, y)`` for training an ``nn.Module`` base (``prefit=False``).

    Attributes
    ----------
    temperature_ : float
        Fitted variance temperature :math:`T`.
    isotonic_ : IsotonicMeanCalibrator or None
        Fitted mean calibrator (``"isotonic+vts"``).
    base_ : object
        The trained base.

    Examples
    --------
    >>> import numpy as np
    >>> from torchregress.estimators import CalibratedRegressor
    >>> class Overconfident:
    ...     def predict(self, X):
    ...         return X[:, 0], np.full(len(X), 0.01)
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(500, 2)); y = X[:, 0] + rng.normal(size=500)
    >>> cal = CalibratedRegressor(Overconfident()).fit(X, y)
    >>> bool(cal.temperature_ > 10)
    True
    """

    def __init__(
        self,
        base: Any,
        method: CalibrationMethod = "vts",
        *,
        alpha: float = 0.1,
        conformal: bool = False,
        prefit: bool = True,
        fit_fn: Optional[FitFn] = None,
    ) -> None:
        if method not in _CALIBRATIONS:
            raise ValueError(f"method must be one of {_CALIBRATIONS}, got {method!r}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.base = base
        self.method = method
        self.alpha = alpha
        self.conformal = conformal
        self.prefit = prefit
        self.fit_fn = fit_fn
        self._adapter = _Adapter(base, fit_fn=fit_fn)
        if not prefit and self._adapter.always_fitted:
            raise ValueError(
                "a TabularFit / TabularEnsembleFit is already trained; use prefit=True"
            )
        self._active = self._adapter
        self.temperature_ = float("nan")
        self.isotonic_: Optional[IsotonicMeanCalibrator] = None
        self.base_: Any = None
        self._scaler: Optional[VarianceTemperatureScaler] = None
        self._cache: Optional[_CPCache] = None
        self.n_calibration_ = 0

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_cal: Optional[ArrayLike] = None,
        y_cal: Optional[ArrayLike] = None,
        calib_fraction: float = 0.25,
        seed: int = 0,
    ) -> "CalibratedRegressor":
        """Fit the calibrator on a held-out split.

        With ``prefit=True`` and no ``X_cal``, ``(X, y)`` is the calibration set.
        Otherwise ``X_cal`` is used, or ``calib_fraction`` of the rows is carved out
        (seeded) and the base trains on the rest. ``seed`` also drives the half-split
        of ``"isotonic+vts"``.
        """
        X = _as_data(X)
        y1 = _finite_y(y)
        if len(y1) != len(X):
            raise ValueError("X and y have different lengths")
        if (X_cal is None) != (y_cal is None):
            raise ValueError("pass both X_cal and y_cal, or neither")
        if y_cal is not None:
            _finite_y(y_cal, "y_cal")
        Xtr: Optional[ArrayLike] = None
        ytr: Optional[np.ndarray] = None
        if X_cal is not None and y_cal is not None:
            Xc, yc = _as_data(X_cal), _np1(y_cal)
            Xtr, ytr = X, y1
        elif self.prefit:
            Xc, yc = X, y1
        else:
            tr, ca = _split_idx(len(X), calib_fraction, seed)
            Xtr, ytr, Xc, yc = _take(X, tr), y1[tr], _take(X, ca), y1[ca]

        if self.prefit:
            adapter = self._adapter
            self.base_ = self.base
        else:
            adapter = self._adapter.fresh_copy()
            assert Xtr is not None and ytr is not None
            adapter.fit(Xtr, ytr)
            self.base_ = adapter.base
        self._active = adapter

        mean_cal, var_cal = adapter.gauss(Xc)
        n = len(yc)
        if n < 2:
            raise ValueError("need at least two calibration rows")
        idx_vts = np.arange(n)
        self.isotonic_ = None
        if self.method == "isotonic+vts":
            perm = np.random.default_rng(seed).permutation(n)
            idx_iso, idx_vts = perm[: n // 2], perm[n // 2 :]
            if len(idx_iso) < 2 or len(idx_vts) < 2:
                raise ValueError("isotonic+vts needs at least four calibration rows")
            self.isotonic_ = IsotonicMeanCalibrator().fit(
                _t64(mean_cal[idx_iso]), _t64(yc[idx_iso])
            )
            mean_cal = _np1(self.isotonic_.transform(_t64(mean_cal)))
        scaler = VarianceTemperatureScaler().fit(
            _t64(mean_cal[idx_vts]), _t64(var_cal[idx_vts]), _t64(yc[idx_vts])
        )
        self._scaler = scaler
        self.temperature_ = float(scaler.temperature)
        self.n_calibration_ = len(idx_vts)

        if self.conformal:
            std_cal = np.sqrt(_np1(scaler.transform(_t64(var_cal[idx_vts]))))
            pred_t, y_t, std_t = _t64(mean_cal[idx_vts]), _t64(yc[idx_vts]), _t64(std_cal)

            def make(a: float) -> ConformalPredictor:
                cp = SplitConformal(alpha=a, normalize_fn=lambda _pred, spread: spread)
                cp.calibrate(pred_t, y_t, x=std_t)
                return cp

            self._cache = _CPCache(make)
            self._cache.get(self.alpha)
        return self

    def _dist(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        if self._scaler is None:
            raise RuntimeError("call fit() before predicting")
        mean, var = self._active.gauss(_as_data(X))
        if self.isotonic_ is not None:
            mean = _np1(self.isotonic_.transform(_t64(mean)))
        std = np.sqrt(_np1(self._scaler.transform(_t64(var))))
        return mean, std

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Calibrated predictive mean."""
        return _cast_out(X, self._dist(X)[0])

    def predict_dist(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Calibrated Gaussian ``(mean, std)``."""
        mean, std = self._dist(X)
        return _cast_out(X, mean), _cast_out(X, std)

    def predict_interval(
        self,
        X: ArrayLike,
        alpha: Optional[float] = None,
        test_weights: Optional[ArrayLike] = None,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Central interval with nominal coverage ``1 - alpha``.

        Gaussian quantiles :math:`\\mu \\pm z_{1-\\alpha/2}\\sigma` by default,
        :math:`\\mu \\pm \\hat q\\,\\sigma` with a conformal :math:`\\hat q` if
        ``conformal=True`` (only then ``test_weights`` are accepted).
        """
        a = self.alpha if alpha is None else float(alpha)
        if not 0.0 < a < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {a}")
        mean, std = self._dist(X)
        if self.conformal:
            assert self._cache is not None
            cp = self._cache.get(a)
            lo, hi = cp.predict_interval(
                _t64(mean), x=_t64(std), test_weights=_weights_t(test_weights)
            )
            return _cast_out(X, _np1(lo)), _cast_out(X, _np1(hi))
        if test_weights is not None:
            raise ValueError("test_weights need conformal=True")
        level = torch.tensor(1.0 - a / 2.0, dtype=torch.float64)
        z = float(torch.distributions.Normal(0.0, 1.0).icdf(level))
        return _cast_out(X, mean - z * std), _cast_out(X, mean + z * std)


# ---------------------------------------------------------------------------
# Calibrated deep ensemble recipe
# ---------------------------------------------------------------------------


def calibrated_deep_ensemble(
    X: ArrayLike,
    y: ArrayLike,
    *,
    n_members: int = 5,
    val_fraction: float = 0.2,
    loss: Literal["gaussian", "beta_nll"] = "gaussian",
    alpha: float = 0.1,
    conformal: bool = True,
    seed: int = 0,
    member_val_fraction: float = 0.15,
    model_kwargs: Optional[dict[str, Any]] = None,
    **fit_kwargs: Any,
) -> CalibratedRegressor:
    """Train a calibrated Gaussian deep ensemble in one call (experimental).

    .. warning::
       Experimental API, may change in 0.4.

    Recipe: carve ``val_fraction`` of the rows (seeded) as the calibration set;
    train ``n_members`` Gaussian :class:`~torchregress.models.TabularMLP` members
    (seeds ``seed, seed + 1, ...``, each early-stopping on its own
    ``member_val_fraction`` split of the remaining rows) with
    :func:`~torchregress.models.fit_tabular_ensemble`; combine them as an
    equal-weight Gaussian mixture, whose total variance is the mean of the member
    variances plus the variance of the member means; fit a
    :class:`~torchregress.calibration.VarianceTemperatureScaler` on the carve; and,
    with ``conformal=True``, set the interval width by normalized split conformal on
    the same carve (a guarantee that is approximate because the temperature is
    fitted on those rows too).

    Parameters
    ----------
    X, y : array-like
        Training data (single target).
    n_members : int, default 5
        Ensemble size.
    val_fraction : float, default 0.2
        Share of rows held out for calibration (in ``(0, 1)``).
    loss : {"gaussian", "beta_nll"}
        Member loss: :class:`~torchregress.losses.GaussianNLLLoss`, or
        :class:`~torchregress.losses.BetaNLLLoss` with ``beta=0.5``.
    alpha : float, default 0.1
        Default interval miscoverage.
    conformal : bool, default True
        Conformal instead of Gaussian-quantile intervals.
    seed : int, default 0
        Seed of the carve and of the first member.
    member_val_fraction : float, default 0.15
        Early-stopping share of each member (taken from the non-calibration rows).
    model_kwargs : dict, optional
        Extra arguments of :class:`~torchregress.models.TabularMLP` (for example
        ``hidden=(64, 64)``).
    **fit_kwargs
        Forwarded to :func:`~torchregress.models.fit_tabular` (``epochs``, ``lr``, ...).

    Returns
    -------
    CalibratedRegressor
        With ``predict_dist(X) -> (mean, std)``, ``predict_interval(X, alpha=None)``
        and ``predict(X)``; the trained ensemble is ``result.base_`` and the fitted
        temperature ``result.temperature_``.

    Examples
    --------
    >>> import numpy as np
    >>> from torchregress.estimators import calibrated_deep_ensemble
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 3)); y = X[:, 0] + 0.3 * rng.normal(size=300)
    >>> model = calibrated_deep_ensemble(
    ...     X, y, n_members=2, epochs=12, batch_size=64, model_kwargs={"hidden": (16,)}
    ... )
    >>> mean, std = model.predict_dist(X[:4])
    >>> mean.shape
    (4,)
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    loss_fn: Callable[..., torch.Tensor]
    if loss == "gaussian":
        loss_fn = GaussianNLLLoss()
    elif loss == "beta_nll":
        loss_fn = BetaNLLLoss(beta=0.5)
    else:
        raise ValueError(f"loss must be 'gaussian' or 'beta_nll', got {loss!r}")
    for reserved in ("output_layout", "X_val", "y_val"):
        if reserved in fit_kwargs:
            raise ValueError(f"{reserved!r} is set by the recipe and cannot be passed")
    X = _as_data(X)
    y1 = _finite_y(y)
    if len(y1) != len(X):
        raise ValueError("X and y have different lengths")
    tr, ca = _split_idx(len(X), val_fraction, seed)
    kwargs = dict(model_kwargs or {})

    def make_model(in_features: int) -> nn.Module:
        return TabularMLP(in_features, 2, **kwargs)

    ensemble = fit_tabular_ensemble(
        make_model,
        loss_fn,
        _take(X, tr),
        y1[tr],
        n_members=n_members,
        seed=seed,
        val_fraction=member_val_fraction,
        output_layout="gaussian",
        **fit_kwargs,
    )
    return CalibratedRegressor(ensemble, "vts", alpha=alpha, conformal=conformal).fit(
        _take(X, ca), y1[ca]
    )
