"""Tests for torchregress.estimators: ConformalRegressor, CalibratedRegressor, recipe."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from torch import nn

import torchregress as tr
from torchregress.calibration import calibration_score
from torchregress.estimators import (
    CalibratedRegressor,
    ConformalRegressor,
    _fold_ids,
    calibrated_deep_ensemble,
)
from torchregress.losses import GaussianNLLLoss, MultiQuantileLoss, WeightedMSELoss
from torchregress.losses.conformal import CQR, CVPlus, JackknifePlus, SplitConformal
from torchregress.metrics import gaussian_nll
from torchregress.models import TabularMLP, fit_tabular


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _hetero(n: int, seed: int, d: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Linear mean, noise scale growing with |x_1|."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, d))
    y = X[:, 0] + (0.2 + 0.6 * np.abs(X[:, 1])) * rng.normal(size=n)
    return X, y


def _t(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


def _coverage(lo, hi, y) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def _tol(alpha: float, n_cal: int, n_test: int, n_rep: int) -> float:
    """Three standard errors of an average coverage over ``n_rep`` replications."""
    var = alpha * (1 - alpha) * (1.0 / n_cal + 1.0 / n_test) / n_rep
    return 3.0 * float(np.sqrt(var))


# ---------------------------------------------------------------------------
# Parity with direct use of the conformal classes
# ---------------------------------------------------------------------------


class TestParityWithConformalClasses:
    def setup_method(self):
        self.X, self.y = _hetero(900, 0)
        self.Xtr, self.ytr = self.X[:400], self.y[:400]
        self.Xc, self.yc = self.X[400:700], self.y[400:700]
        self.Xt = self.X[700:]

    def test_split_matches_split_conformal(self):
        base = LinearRegression().fit(self.Xtr, self.ytr)
        cr = ConformalRegressor(base, "split", alpha=0.1, prefit=True).fit(self.Xc, self.yc)
        lo, hi = cr.predict_interval(self.Xt)
        cp = SplitConformal(alpha=0.1)
        cp.calibrate(_t(base.predict(self.Xc)), _t(self.yc))
        e_lo, e_hi = cp.predict_interval(_t(base.predict(self.Xt)))
        np.testing.assert_allclose(lo, e_lo.numpy(), atol=1e-12)
        np.testing.assert_allclose(hi, e_hi.numpy(), atol=1e-12)

    def test_split_matches_mapie(self):
        mapie = pytest.importorskip("mapie.regression")
        base = LinearRegression().fit(self.Xtr, self.ytr)
        cr = ConformalRegressor(base, "split", alpha=0.1, prefit=True).fit(self.Xc, self.yc)
        lo, hi = cr.predict_interval(self.Xt)
        scr = mapie.SplitConformalRegressor(
            estimator=base, confidence_level=0.9, conformity_score="absolute", prefit=True
        )
        scr.conformalize(self.Xc, self.yc)
        _, iv = scr.predict_interval(self.Xt)
        np.testing.assert_allclose(lo, iv[:, 0, 0], atol=1e-6)
        np.testing.assert_allclose(hi, iv[:, 1, 0], atol=1e-6)

    def test_cqr_matches_cqr_class_and_mapie(self):
        mapie = pytest.importorskip("mapie.regression")
        models = [
            GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=25, random_state=0)
            for a in (0.05, 0.95, 0.5)
        ]
        for m in models:
            m.fit(self.Xtr, self.ytr)
        lo_m, hi_m, mid_m = models
        cr = ConformalRegressor((lo_m, mid_m, hi_m), "cqr", alpha=0.1, prefit=True)
        cr.fit(self.Xc, self.yc)
        lo, hi = cr.predict_interval(self.Xt)

        cqr = CQR(alpha=0.1)
        cqr.calibrate(_t(np.c_[lo_m.predict(self.Xc), hi_m.predict(self.Xc)]), _t(self.yc))
        e_lo, e_hi = cqr.predict_interval(_t(np.c_[lo_m.predict(self.Xt), hi_m.predict(self.Xt)]))
        np.testing.assert_allclose(lo, e_lo.numpy().ravel(), atol=1e-12)
        np.testing.assert_allclose(hi, e_hi.numpy().ravel(), atol=1e-12)

        # MAPIE order is (lower, upper, median); the symmetric correction is the CQR one.
        cqr_m = mapie.ConformalizedQuantileRegressor(
            estimator=[lo_m, hi_m, mid_m], confidence_level=0.9, prefit=True
        )
        cqr_m.conformalize(self.Xc, self.yc)
        _, iv = cqr_m.predict_interval(self.Xt, symmetric_correction=True)
        np.testing.assert_allclose(lo, iv[:, 0, 0], atol=1e-6)
        np.testing.assert_allclose(hi, iv[:, 1, 0], atol=1e-6)

    def test_cqr_pair_and_predict_quantiles_object(self):
        lo_m = GradientBoostingRegressor(
            loss="quantile", alpha=0.05, n_estimators=20, random_state=0
        )
        hi_m = GradientBoostingRegressor(
            loss="quantile", alpha=0.95, n_estimators=20, random_state=0
        )
        lo_m.fit(self.Xtr, self.ytr)
        hi_m.fit(self.Xtr, self.ytr)

        class QObject:
            def predict_quantiles(self, X):
                return np.c_[lo_m.predict(X), hi_m.predict(X)]

        a = ConformalRegressor((lo_m, hi_m), "cqr", prefit=True).fit(self.Xc, self.yc)
        b = ConformalRegressor(QObject(), "cqr", prefit=True).fit(self.Xc, self.yc)
        for u, v in zip(a.predict_interval(self.Xt), b.predict_interval(self.Xt)):
            np.testing.assert_allclose(u, v, atol=1e-12)
        point = a.predict(self.Xt)
        np.testing.assert_allclose(
            point, 0.5 * (lo_m.predict(self.Xt) + hi_m.predict(self.Xt)), atol=1e-12
        )

    def test_normalized_matches_split_conformal_with_difficulty(self):
        class WithStd:
            def __init__(self):
                self.m = LinearRegression()

            def fit(self, X, y):
                self.m.fit(X, y)
                return self

            def predict(self, X):
                return self.m.predict(X)

            def predict_std(self, X):
                return 0.2 + 0.6 * np.abs(X[:, 1])

        base = WithStd().fit(self.Xtr, self.ytr)
        cr = ConformalRegressor(base, "normalized", prefit=True).fit(self.Xc, self.yc)
        lo, hi = cr.predict_interval(self.Xt)
        cp = SplitConformal(alpha=0.1, normalize_fn=lambda _p, s: s)
        cp.calibrate(_t(base.predict(self.Xc)), _t(self.yc), x=_t(base.predict_std(self.Xc)))
        e_lo, e_hi = cp.predict_interval(_t(base.predict(self.Xt)), x=_t(base.predict_std(self.Xt)))
        np.testing.assert_allclose(lo, e_lo.numpy(), atol=1e-12)
        np.testing.assert_allclose(hi, e_hi.numpy(), atol=1e-12)
        # Adaptive: wider where the noise is larger.
        width = hi - lo
        assert np.corrcoef(width, np.abs(self.Xt[:, 1]))[0, 1] > 0.9

    def test_normalized_with_spread_model(self):
        X, y = self.X[:600], self.y[:600]
        cr = ConformalRegressor(
            LinearRegression(),
            "normalized",
            spread_model=GradientBoostingRegressor(n_estimators=20),
        ).fit(X, y, seed=3)
        lo, hi = cr.predict_interval(self.Xt)
        assert np.all(hi > lo)
        assert 0.8 < _coverage(lo, hi, self.y[700:]) < 0.97

    @pytest.mark.parametrize("method", ["cv+", "jackknife+"])
    def test_cross_methods_match_cvplus(self, method):
        n = 60 if method == "jackknife+" else 300
        X, y = self.X[:n], self.y[:n]
        n_folds = 5
        cr = ConformalRegressor(LinearRegression(), method, alpha=0.2, n_folds=n_folds)
        cr.fit(X, y, seed=7)
        lo, hi = cr.predict_interval(self.Xt)

        fold = np.arange(n) if method == "jackknife+" else _fold_ids(n, n_folds, 7)
        k = fold.max() + 1
        oof = np.empty(n)
        members = np.empty((k, len(self.Xt)))
        for j in range(k):
            m = LinearRegression().fit(X[fold != j], y[fold != j])
            oof[fold == j] = m.predict(X[fold == j])
            members[j] = m.predict(self.Xt)
        cls = JackknifePlus if method == "jackknife+" else CVPlus
        cp = cls(alpha=0.2)
        cp.calibrate_ensemble(_t(oof)[:, None], _t(y)[:, None], torch.as_tensor(fold))
        e_lo, e_hi = cp.predict_interval(_t(members)[:, :, None])
        np.testing.assert_allclose(lo, e_lo.numpy().ravel(), atol=1e-12)
        np.testing.assert_allclose(hi, e_hi.numpy().ravel(), atol=1e-12)
        np.testing.assert_allclose(cr.predict(self.Xt), members.mean(0), atol=1e-12)

    def test_alpha_override_equals_fresh_instance(self):
        base = LinearRegression().fit(self.Xtr, self.ytr)
        a = ConformalRegressor(base, alpha=0.1, prefit=True).fit(self.Xc, self.yc)
        b = ConformalRegressor(base, alpha=0.3, prefit=True).fit(self.Xc, self.yc)
        for u, v in zip(a.predict_interval(self.Xt, alpha=0.3), b.predict_interval(self.Xt)):
            np.testing.assert_allclose(u, v, atol=1e-12)

    def test_test_weights_match_class(self):
        base = LinearRegression().fit(self.Xtr, self.ytr)
        rng = np.random.default_rng(1)
        w_cal = rng.uniform(0.5, 2.0, size=len(self.yc))
        w_test = rng.uniform(0.5, 2.0, size=len(self.Xt))
        cr = ConformalRegressor(base, prefit=True).fit(self.Xc, self.yc, cal_weights=w_cal)
        lo, hi = cr.predict_interval(self.Xt, test_weights=w_test)
        cp = SplitConformal(alpha=0.1)
        cp.calibrate(_t(base.predict(self.Xc)), _t(self.yc), weights=_t(w_cal))
        e_lo, e_hi = cp.predict_interval(_t(base.predict(self.Xt)), test_weights=_t(w_test))
        np.testing.assert_allclose(lo, e_lo.numpy(), atol=1e-12)
        np.testing.assert_allclose(hi, e_hi.numpy(), atol=1e-12)
        # Unit weights reproduce the unweighted interval.
        u_lo, u_hi = cr.predict_interval(self.Xt, test_weights=np.ones(len(self.Xt)))
        assert np.all(u_hi > u_lo)

    def test_cv_rejects_test_weights(self):
        cr = ConformalRegressor(LinearRegression(), "cv+").fit(self.X[:100], self.y[:100])
        with pytest.raises(ValueError, match="test_weights"):
            cr.predict_interval(self.Xt, test_weights=np.ones(len(self.Xt)))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


class TestCoverage:
    @pytest.mark.parametrize("method", ["split", "normalized", "cqr", "cv+"])
    def test_marginal_coverage_over_seeds(self, method):
        alpha, n_train, n_test, seeds = 0.1, 600, 2000, (0, 1, 2, 3, 4)
        covs = []
        for seed in seeds:
            X, y = _hetero(n_train, seed)
            Xt, yt = _hetero(n_test, 1000 + seed)
            if method == "cqr":
                base = tuple(
                    GradientBoostingRegressor(
                        loss="quantile", alpha=a, n_estimators=30, max_depth=2, random_state=seed
                    )
                    for a in (0.05, 0.95)
                )
                cr = ConformalRegressor(base, "cqr", alpha=alpha)
            elif method == "normalized":
                cr = ConformalRegressor(
                    LinearRegression(),
                    "normalized",
                    alpha=alpha,
                    spread_model=GradientBoostingRegressor(n_estimators=30, max_depth=2),
                )
            else:
                cr = ConformalRegressor(Ridge(), method, alpha=alpha)
            cr.fit(X, y, seed=seed)
            lo, hi = cr.predict_interval(Xt)
            covs.append(_coverage(lo, hi, yt))
        n_cal = n_train if method == "cv+" else int(0.25 * n_train)
        tol = _tol(alpha, n_cal, n_test, len(seeds))
        assert np.mean(covs) >= 1 - alpha - tol, (method, covs)
        # Not absurdly conservative either.
        assert np.mean(covs) <= 1 - alpha + 0.06, (method, covs)


# ---------------------------------------------------------------------------
# Input / output types and base kinds
# ---------------------------------------------------------------------------


class TestIOAndBases:
    def setup_method(self):
        self.X, self.y = _hetero(500, 0)
        self.Xt = _hetero(100, 5)[0]

    def test_numpy_in_numpy_out_and_tensor_in_tensor_out(self):
        cr = ConformalRegressor(Ridge(), "split").fit(self.X, self.y, seed=2)
        lo_n, hi_n = cr.predict_interval(self.Xt)
        assert isinstance(lo_n, np.ndarray) and isinstance(cr.predict(self.Xt), np.ndarray)
        Xt64 = torch.as_tensor(self.Xt)
        lo_t, hi_t = cr.predict_interval(Xt64)
        assert isinstance(lo_t, torch.Tensor) and lo_t.dtype == torch.float64
        np.testing.assert_allclose(lo_t.numpy(), lo_n, atol=1e-12)
        np.testing.assert_allclose(hi_t.numpy(), hi_n, atol=1e-12)
        pred_t = cr.predict(Xt64)
        assert isinstance(pred_t, torch.Tensor)
        np.testing.assert_allclose(pred_t.numpy(), cr.predict(self.Xt), atol=1e-12)
        lo32, _ = cr.predict_interval(Xt64.float())
        assert lo32.dtype == torch.float32

    def test_tensor_training_data(self):
        a = ConformalRegressor(Ridge(), "split").fit(self.X, self.y, seed=2)
        b = ConformalRegressor(Ridge(), "split").fit(
            torch.as_tensor(self.X), torch.as_tensor(self.y), seed=2
        )
        for u, v in zip(a.predict_interval(self.Xt), b.predict_interval(self.Xt)):
            np.testing.assert_allclose(u, v, atol=1e-10)

    def test_explicit_calibration_set(self):
        cr = ConformalRegressor(Ridge(), "split").fit(
            self.X[:300], self.y[:300], X_cal=self.X[300:], y_cal=self.y[300:]
        )
        assert cr.n_calibration_ == 200
        base = Ridge().fit(self.X[:300], self.y[:300])
        cp = SplitConformal(alpha=0.1)
        cp.calibrate(_t(base.predict(self.X[300:])), _t(self.y[300:]))
        lo, _ = cr.predict_interval(self.Xt)
        e_lo, _ = cp.predict_interval(_t(base.predict(self.Xt)))
        np.testing.assert_allclose(lo, e_lo.numpy(), atol=1e-10)

    def test_user_base_is_not_mutated(self):
        base = Ridge()
        ConformalRegressor(base, "split").fit(self.X, self.y)
        assert not hasattr(base, "coef_")

    def test_lightgbm_base(self):
        lgb = pytest.importorskip("lightgbm")
        for method in ("split", "cv+"):
            cr = ConformalRegressor(
                lgb.LGBMRegressor(n_estimators=20, num_leaves=7, verbose=-1, n_jobs=1), method
            )
            cr.fit(self.X, self.y, seed=1)
            lo, hi = cr.predict_interval(self.Xt)
            assert lo.shape == (100,) and np.all(hi > lo)

    def test_tabular_fit_point_and_gaussian_heads(self):
        torch.manual_seed(0)
        point = fit_tabular(
            TabularMLP(self.X.shape[1], 1, hidden=(16,)),
            WeightedMSELoss(),
            self.X[:300],
            self.y[:300],
            epochs=15,
            batch_size=64,
        )
        cr = ConformalRegressor(point, "split").fit(self.X[300:], self.y[300:])
        assert cr.prefit and cr.n_calibration_ == 200
        lo, hi = cr.predict_interval(self.Xt)
        assert np.all(hi > lo)
        np.testing.assert_allclose(
            cr.predict(self.Xt), point.predict(self.Xt).numpy().ravel(), atol=1e-5
        )

        gauss = fit_tabular(
            TabularMLP(self.X.shape[1], 2, hidden=(16,)),
            GaussianNLLLoss(),
            self.X[:300],
            self.y[:300],
            epochs=15,
            batch_size=64,
        )
        crn = ConformalRegressor(gauss, "normalized").fit(self.X[300:], self.y[300:])
        lo, hi = crn.predict_interval(self.Xt)
        assert np.all(hi > lo)
        with pytest.raises(ValueError, match="prefit"):
            ConformalRegressor(gauss, "cv+")

    def test_tabular_fit_quantile_head_cqr(self):
        torch.manual_seed(0)
        fit = fit_tabular(
            TabularMLP(self.X.shape[1], 3, hidden=(16,)),
            MultiQuantileLoss([0.05, 0.5, 0.95]),
            self.X[:300],
            self.y[:300],
            epochs=15,
            batch_size=64,
        )
        cr = ConformalRegressor(fit, "cqr").fit(self.X[300:], self.y[300:])
        lo, hi = cr.predict_interval(self.Xt)
        assert lo.shape == (100,)
        raw = fit.predict(self.Xt).numpy()
        np.testing.assert_allclose(cr.predict(self.Xt), raw[:, 1], atol=1e-5)

    def test_nn_module_with_training_callable(self):
        def fit_fn(model, X, y):
            opt = torch.optim.LBFGS(model.parameters(), max_iter=50)

            def closure():
                opt.zero_grad()
                loss = ((model(X) - y) ** 2).mean()
                loss.backward()
                return loss

            opt.step(closure)

        module = nn.Linear(self.X.shape[1], 1)
        cr = ConformalRegressor(module, "split", fit_fn=fit_fn).fit(self.X, self.y, seed=0)
        assert cr.base_ is not module  # deep-copied
        lo, hi = cr.predict_interval(self.Xt)
        assert np.all(hi > lo)
        assert 0.75 < _coverage(lo, hi, _hetero(100, 5)[1]) <= 1.0
        # Tensor in, tensor out; prefit reuses the trained copy as is.
        cr2 = ConformalRegressor(cr.base_, "split", prefit=True).fit(self.X, self.y)
        lo_t, _ = cr2.predict_interval(torch.as_tensor(self.Xt, dtype=torch.float32))
        assert isinstance(lo_t, torch.Tensor) and lo_t.dtype == torch.float32

    def test_error_paths(self):
        with pytest.raises(ValueError, match="method"):
            ConformalRegressor(Ridge(), "bogus")
        with pytest.raises(ValueError, match="alpha"):
            ConformalRegressor(Ridge(), alpha=1.5)
        with pytest.raises(ValueError, match="prefit"):
            ConformalRegressor(Ridge(), "cv+", prefit=True)
        with pytest.raises(ValueError, match="fit_fn"):
            ConformalRegressor(nn.Linear(3, 1)).fit(self.X, self.y)
        with pytest.raises(ValueError, match="quantile"):
            ConformalRegressor(Ridge(), "cqr").fit(self.X, self.y)
        with pytest.raises(ValueError, match="spread"):
            ConformalRegressor(Ridge(), "normalized").fit(self.X, self.y)
        with pytest.raises(RuntimeError, match="fit"):
            ConformalRegressor(Ridge()).predict_interval(self.Xt)
        with pytest.raises(ValueError, match="both X_cal"):
            ConformalRegressor(Ridge()).fit(self.X, self.y, X_cal=self.X)


# ---------------------------------------------------------------------------
# CalibratedRegressor
# ---------------------------------------------------------------------------


class _MisCalibrated:
    """Gaussian base with the right mean (or a biased one) and a variance that is far too small."""

    def __init__(self, var_scale: float = 0.05, mean_map=lambda m: m):
        self.var_scale = var_scale
        self.mean_map = mean_map

    def predict(self, X):
        mean = self.mean_map(X[:, 0])
        var = self.var_scale * (0.2 + 0.6 * np.abs(X[:, 1])) ** 2
        return mean, var


class TestCalibratedRegressor:
    def setup_method(self):
        self.Xc, self.yc = _hetero(600, 0)
        self.Xt, self.yt = _hetero(3000, 1)

    def _nll(self, base_or_cal, calibrated: bool) -> float:
        if calibrated:
            mean, std = base_or_cal.predict_dist(self.Xt)
            var = std**2
        else:
            mean, var = base_or_cal.predict(self.Xt)
        return float(gaussian_nll(_t(mean), _t(self.yt), _t(var)))

    def test_vts_improves_nll_and_coverage(self):
        base = _MisCalibrated(var_scale=0.05)
        cal = CalibratedRegressor(base, "vts", alpha=0.1).fit(self.Xc, self.yc)
        assert cal.temperature_ > 10
        assert self._nll(cal, True) < self._nll(base, False) - 1.0
        lo, hi = cal.predict_interval(self.Xt)
        assert abs(_coverage(lo, hi, self.yt) - 0.9) < 0.03
        lo99, hi99 = cal.predict_interval(self.Xt, alpha=0.01)
        assert _coverage(lo99, hi99, self.yt) > 0.97

    def test_vts_matches_direct_scaler(self):
        from torchregress.calibration import VarianceTemperatureScaler

        base = _MisCalibrated(var_scale=0.3)
        cal = CalibratedRegressor(base).fit(self.Xc, self.yc)
        mean, var = base.predict(self.Xc)
        vts = VarianceTemperatureScaler().fit(_t(mean), _t(var), _t(self.yc))
        assert cal.temperature_ == pytest.approx(vts.temperature, rel=1e-12)
        _, std = cal.predict_dist(self.Xt)
        np.testing.assert_allclose(
            std, np.sqrt(vts.transform(_t(base.predict(self.Xt)[1])).numpy())
        )

    def test_isotonic_fixes_a_distorted_mean(self):
        # Monotone distortion of the mean that the isotonic map can undo.
        base_good = _MisCalibrated(var_scale=0.5)

        class Stretched:
            def predict(self, X):
                m, v = base_good.predict(X)
                return np.exp(0.6 * m), v  # monotone in the true mean

        stretched = Stretched()
        vts = CalibratedRegressor(stretched, "vts").fit(self.Xc, self.yc)
        iso = CalibratedRegressor(stretched, "isotonic+vts").fit(self.Xc, self.yc)
        assert iso.isotonic_ is not None
        assert self._nll(iso, True) < self._nll(vts, True)
        # Mean error shrinks too.
        rmse = lambda c: float(np.sqrt(np.mean((c.predict(self.Xt) - self.yt) ** 2)))  # noqa: E731
        assert rmse(iso) < rmse(vts)

    def test_conformal_intervals(self):
        base = _MisCalibrated(var_scale=0.05)
        cal = CalibratedRegressor(base, conformal=True, alpha=0.1).fit(self.Xc, self.yc)
        lo, hi = cal.predict_interval(self.Xt)
        assert abs(_coverage(lo, hi, self.yt) - 0.9) < 0.03
        mean, std = cal.predict_dist(self.Xt)
        cp = SplitConformal(alpha=0.1, normalize_fn=lambda _p, s: s)
        cmean, cvar = base.predict(self.Xc)
        from torchregress.calibration import VarianceTemperatureScaler

        vts = VarianceTemperatureScaler().fit(_t(cmean), _t(cvar), _t(self.yc))
        cstd = vts.transform(_t(cvar)).sqrt()
        cp.calibrate(_t(cmean), _t(self.yc), x=cstd)
        e_lo, e_hi = cp.predict_interval(_t(mean), x=_t(std))
        np.testing.assert_allclose(lo, e_lo.numpy(), atol=1e-10)
        np.testing.assert_allclose(hi, e_hi.numpy(), atol=1e-10)
        with pytest.raises(ValueError, match="conformal"):
            CalibratedRegressor(base).fit(self.Xc, self.yc).predict_interval(
                self.Xt, test_weights=np.ones(len(self.Xt))
            )

    def test_tensor_io(self):
        cal = CalibratedRegressor(_MisCalibrated()).fit(self.Xc, self.yc)
        Xt = torch.as_tensor(self.Xt[:50], dtype=torch.float32)
        mean, std = cal.predict_dist(Xt)
        assert isinstance(mean, torch.Tensor) and mean.dtype == torch.float32
        lo, hi = cal.predict_interval(Xt)
        assert isinstance(lo, torch.Tensor) and bool((hi > lo).all())
        assert isinstance(cal.predict(Xt), torch.Tensor)

    def test_gaussian_tabular_fit_and_module_bases(self):
        X, y = _hetero(800, 3)
        torch.manual_seed(0)
        fit = fit_tabular(
            TabularMLP(3, 2, hidden=(16,)),
            GaussianNLLLoss(),
            X[:500],
            y[:500],
            epochs=20,
            batch_size=64,
        )
        cal = CalibratedRegressor(fit).fit(X[500:], y[500:])
        mean, std = cal.predict_dist(X[500:])
        raw = fit.predict(X[500:]).numpy()
        np.testing.assert_allclose(mean, raw[:, 0], atol=1e-5)
        np.testing.assert_allclose(std**2, cal.temperature_ * np.exp(raw[:, 1]), rtol=1e-4)

        # Raw module outputs [mean, log_variance] in target units.
        module = nn.Linear(3, 2)
        with torch.no_grad():
            module.weight.zero_()
            module.bias.copy_(torch.tensor([0.0, -2.0]))
        calm = CalibratedRegressor(module).fit(X, y)
        assert calm.temperature_ > 1.0
        with pytest.raises(ValueError, match="Gaussian"):
            CalibratedRegressor(Ridge().fit(X, y)).fit(X, y)

    def test_unfitted_base_trained_on_split(self):
        class MeanVar:
            def fit(self, X, y):
                self.lr = LinearRegression().fit(X, y)
                self.v = float(np.var(y - self.lr.predict(X))) / 20.0
                return self

            def predict(self, X):
                return self.lr.predict(X), np.full(len(X), self.v)

        cal = CalibratedRegressor(MeanVar(), prefit=False).fit(self.Xc, self.yc, seed=1)
        assert cal.n_calibration_ == 150 and cal.temperature_ > 5

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="method"):
            CalibratedRegressor(_MisCalibrated(), "nope")
        with pytest.raises(RuntimeError, match="fit"):
            CalibratedRegressor(_MisCalibrated()).predict_dist(self.Xt)


# ---------------------------------------------------------------------------
# calibrated_deep_ensemble
# ---------------------------------------------------------------------------


class TestCalibratedDeepEnsemble:
    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.default_rng(0)

        def make(n):
            X = rng.uniform(-2, 2, size=(n, 3))
            y = np.sin(2 * X[:, 0]) + (0.15 + 0.4 * np.abs(X[:, 1])) * rng.normal(size=n)
            return X, y

        return make(1200), make(1500)

    @pytest.fixture(scope="class")
    def model(self, data):
        (X, y), _ = data
        torch.manual_seed(0)
        return calibrated_deep_ensemble(
            X,
            y,
            n_members=3,
            epochs=40,
            batch_size=64,
            patience=40,
            model_kwargs={"hidden": (32, 32)},
            seed=0,
        )

    def test_runs_and_is_sane(self, model, data):
        _, (Xt, yt) = data
        mean, std = model.predict_dist(Xt)
        assert mean.shape == (len(yt),) and bool((std > 0).all())
        rmse = float(np.sqrt(np.mean((mean - yt) ** 2)))
        assert rmse < float(np.std(yt))  # beats the constant predictor
        nll = float(gaussian_nll(_t(mean), _t(yt), _t(std**2)))
        assert np.isfinite(nll) and nll < 1.6
        ece = float(
            calibration_score(_t(yt), _t(mean), _t(std), as_numpy=True)[
                "mean_absolute_calibration_error"
            ]
        )
        assert ece < 0.06
        assert model.base_.members[0].output_layout == "gaussian" and len(model.base_.members) == 3

    def test_conformal_intervals_cover(self, model, data):
        _, (Xt, yt) = data
        lo, hi = model.predict_interval(Xt)
        assert abs(_coverage(lo, hi, yt) - 0.9) < 0.04
        lo5, hi5 = model.predict_interval(Xt, alpha=0.5)
        assert np.mean(hi5 - lo5) < np.mean(hi - lo)

    def test_mixture_variance_is_aleatoric_plus_epistemic(self, model, data):
        _, (Xt, _) = data
        members = model.base_.predict_members(torch.as_tensor(Xt[:200])).numpy().astype(np.float64)
        mean_m, var_m = members[..., 0], np.exp(members[..., 1])
        total = var_m.mean(0) + mean_m.var(0)
        assert float(model.temperature_) > 0
        _, std = model.predict_dist(Xt[:200])
        np.testing.assert_allclose(std**2, model.temperature_ * total, rtol=1e-4)
        assert np.all(mean_m.var(0) > 0)

    def test_gaussian_interval_variant_and_tensor_io(self, data):
        (X, y), (Xt, yt) = data
        torch.manual_seed(0)
        m = calibrated_deep_ensemble(
            torch.as_tensor(X[:600]),
            torch.as_tensor(y[:600]),
            n_members=2,
            conformal=False,
            loss="beta_nll",
            alpha=0.2,
            epochs=20,
            batch_size=64,
            model_kwargs={"hidden": (16,)},
        )
        lo, hi = m.predict_interval(torch.as_tensor(Xt[:500]))
        assert isinstance(lo, torch.Tensor)
        assert 0.65 < _coverage(lo.numpy(), hi.numpy(), yt[:500]) < 0.95

    def test_invalid_arguments(self, data):
        (X, y), _ = data
        with pytest.raises(ValueError, match="loss"):
            calibrated_deep_ensemble(X, y, loss="nope")
        with pytest.raises(ValueError, match="val_fraction"):
            calibrated_deep_ensemble(X, y, val_fraction=1.0)
        with pytest.raises(ValueError, match="output_layout"):
            calibrated_deep_ensemble(X, y, output_layout="none")


def test_lazy_submodule_and_exports():
    assert tr.estimators.ConformalRegressor is ConformalRegressor
    assert "estimators" in tr.__all__
    assert set(tr.estimators.__all__) == {
        "ConformalRegressor",
        "CalibratedRegressor",
        "calibrated_deep_ensemble",
    }
    assert "experimental" in (tr.estimators.__doc__ or "").lower()
