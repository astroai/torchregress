"""Regression tests for the 0.3.0 metrics / calibration audit (M-MET-*, M-CAL-*).

Each test reproduces one audit finding and pins the fixed behaviour. References
come from closed forms, :mod:`scipy` (core) or :mod:`sklearn` (``test`` extra).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats as st
import torch
from scipy.spatial.distance import mahalanobis
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import median_absolute_error as sk_medae

import torchregress.metrics as M
from torchregress.calibration import IsotonicMeanCalibrator, VarianceTemperatureScaler
from torchregress.metrics.distribution import _pit_from_quantiles


def _crps_normal_np(y: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (y - mu) / sigma
    return sigma * (z * (2 * st.norm.cdf(z) - 1) + 2 * st.norm.pdf(z) - 1 / math.sqrt(math.pi))


# --------------------------------------------------------------------------- M-MET-001


def _normal_samples(n: int = 5000, s: int = 200):
    g = torch.Generator().manual_seed(0)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    samples = torch.randn(s, n, generator=g, dtype=torch.float64)
    return y, samples


def test_M_MET_001_report_crps_uses_sample_estimator():
    y, samples = _normal_samples()
    rep = M.distribution_metrics_report(y_true=y, samples=samples)
    assert rep["crps"] == pytest.approx(M.crps_from_samples(samples, y), rel=1e-12)
    analytic = float(_crps_normal_np(y.numpy(), 0.0, 1.0).mean())
    assert rep["crps"] == pytest.approx(analytic, rel=1e-2)  # 7-quantile version: -3%
    assert {"coverage_90", "interval_width_90"} <= set(rep)


def test_M_MET_001_report_crps_closed_form_for_normal_dist():
    y, _ = _normal_samples()
    rep = M.distribution_metrics_report(
        dist={"loc": torch.zeros_like(y), "scale": torch.ones_like(y)}, y_true=y
    )
    analytic = float(_crps_normal_np(y.numpy(), 0.0, 1.0).mean())
    assert rep["crps"] == pytest.approx(analytic, rel=1e-10)


def test_M_MET_001_report_crps_quantile_inputs_unchanged():
    y, _ = _normal_samples(n=200)
    q = {lv: torch.full_like(y, float(st.norm.ppf(lv))) for lv in (0.1, 0.5, 0.9)}
    rep = M.distribution_metrics_report(y_true=y, y_pred_quantiles=q)
    assert rep["crps"] == pytest.approx(M.continuous_ranked_probability_score(q, y))


# --------------------------------------------------------------------------- M-MET-002


def test_M_MET_002_quantile_pit_ks_for_calibrated_forecast():
    levels = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
    n = 20000
    y = torch.randn(n, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    q = {lv: torch.full((n,), float(st.norm.ppf(lv)), dtype=torch.float64) for lv in levels}
    rep = M.distribution_metrics_report(y_true=y, y_pred_quantiles=q)
    # 1% KS critical value for n=20000 is ~0.0115; end-point PIT gave ~0.05 (= tau_min).
    assert float(rep["pit_ks"]) < 0.0115
    pit = _pit_from_quantiles(q, y)
    assert float(pit.min()) > 0.0 and float(pit.max()) < 1.0
    torch.testing.assert_close(pit, _pit_from_quantiles(q, y))  # deterministic default


# --------------------------------------------------------------------------- M-MET-003


def test_M_MET_003_energy_and_kernel_float32_offset():
    rng = np.random.default_rng(3)
    ens = torch.tensor((20 + rng.normal(size=(200, 50, 2)) * 0.01).astype(np.float32))
    y = torch.tensor((20 + rng.normal(size=(50, 2)) * 0.01).astype(np.float32))
    ref = M.energy_score(ens.double(), y.double())
    assert M.energy_score(ens, y) == pytest.approx(ref, rel=1e-4)
    metric = M.EnergyScore()
    metric.update(ens, y)
    assert float(metric.compute()) == pytest.approx(ref, rel=1e-4)

    x_test, x_ref = ens[0], ens[1]
    kd_ref = M.kernel_density_score(x_test.double(), x_ref.double(), bandwidth=0.01)
    kd = M.kernel_density_score(x_test, x_ref, bandwidth=0.01)
    torch.testing.assert_close(kd.double(), kd_ref, rtol=1e-4, atol=1e-6)


# --------------------------------------------------------------------------- M-MET-004


def test_M_MET_004_single_sample_is_dirac():
    got = M.crps_from_samples(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]))
    assert got == pytest.approx(1.5)  # CRPS of a Dirac = absolute error
    assert M.vario_score(torch.tensor([[1.0, 2.0]]), torch.tensor([0.0, 0.0])) == pytest.approx(
        -1.5
    )
    assert M.vario_score(
        torch.tensor([[1.0, 2.0]]), torch.tensor([0.0, 0.0]), rho=0.5
    ) == pytest.approx(-(1.0 + math.sqrt(2.0)) / 2)


# --------------------------------------------------------------------------- M-MET-005


def test_M_MET_005_vario_higher_is_better():
    y = torch.zeros(10)
    good = torch.randn(50, 10, generator=torch.Generator().manual_seed(0)) * 0.1
    assert M.vario_score(good, y) > M.vario_score(good + 5.0, y)
    assert M.VarioScore.higher_is_better is True


# --------------------------------------------------------------------------- M-MET-006/007


def test_M_MET_006_medae_multioutput_matches_sklearn():
    rng = np.random.default_rng(1)
    yt = rng.normal(size=(9, 3))
    yp = yt + rng.normal(size=(9, 3)) * np.array([1, 5, 0.1])
    ref = sk_medae(yt, yp)
    assert float(M.median_absolute_error(torch.tensor(yp), torch.tensor(yt))) == pytest.approx(
        ref, rel=1e-12
    )
    m = M.MedianAbsoluteError()
    m.update(torch.tensor(yp), torch.tensor(yt))
    assert float(m.compute()) == pytest.approx(ref, rel=1e-12)
    raw = M.median_absolute_error(torch.tensor(yp), torch.tensor(yt), multioutput="raw_values")
    np.testing.assert_allclose(np.asarray(raw), sk_medae(yt, yp, multioutput="raw_values"))


def test_M_MET_007_even_n_median():
    yt = np.zeros(4)
    yp = np.array([0.0, 1.0, 5.0, 6.0])
    assert float(M.median_absolute_error(torch.tensor(yp), torch.tensor(yt))) == pytest.approx(
        sk_medae(yt, yp)
    )
    mad_ref = 1.4826 * st.median_abs_deviation(yt - yp)
    assert float(M.median_absolute_deviation(torch.tensor(yp), torch.tensor(yt))) == pytest.approx(
        mad_ref, rel=1e-12
    )
    nmad = M.NormalizedMedianAbsoluteDeviation()
    nmad.update(torch.tensor(yp), torch.tensor(yt))
    assert float(nmad.compute()) == pytest.approx(
        1.4826 * st.median_abs_deviation(yp - yt), rel=1e-12
    )


# --------------------------------------------------------------------------- M-MET-008


@pytest.mark.parametrize("n", [1, 2, 5, 9, 10, 33])
@pytest.mark.parametrize("p", [0.0, 0.1, 0.25, 0.3, 0.45])
def test_M_MET_008_trimmed_mse_matches_scipy(n, p):
    rng = np.random.default_rng(n)
    yt = rng.normal(size=n)
    yp = yt + rng.normal(size=n)
    ref = st.trim_mean((yt - yp) ** 2, p)
    got = M.trimmed_mean_squared_error(torch.tensor(yp), torch.tensor(yt), proportion=p)
    assert float(got) == pytest.approx(ref, rel=1e-12)
    m = M.TrimmedMeanSquaredError(proportion=p)
    m.update(torch.tensor(yp), torch.tensor(yt))
    assert float(m.compute()) == pytest.approx(ref, rel=1e-12)


# --------------------------------------------------------------------------- M-MET-009


def test_M_MET_009_ensemble_interval_reset():
    g = torch.Generator().manual_seed(0)
    a = (
        torch.randn(5, 30, generator=g),
        torch.rand(5, 30, generator=g),
        torch.randn(30, generator=g),
    )
    b = (
        torch.randn(5, 35, generator=g),
        torch.rand(5, 35, generator=g),
        torch.randn(35, generator=g),
    )
    m = M.EnsembleIntervalMetrics()
    m.update(*a)
    m.compute()
    m.reset()
    m.update(*b)
    fresh = M.EnsembleIntervalMetrics()
    fresh.update(*b)
    got, exp = m.compute(), fresh.compute()
    assert float(got["interval_score"]) == pytest.approx(float(exp["interval_score"]))
    assert float(got["picp"]) == pytest.approx(float(exp["picp"]))


# --------------------------------------------------------------------------- M-MET-010


def test_M_MET_010_small_scale_variances_not_floored():
    s = 1e-4
    y = torch.randn(100, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * s
    ref = float(-st.norm.logpdf(y.numpy(), 0, s).mean())
    mu = torch.zeros(5, 100, dtype=torch.float64)
    v = torch.full((5, 100), s**2, dtype=torch.float64)
    assert float(M.gaussian_nll_ensemble(mu, v, y)) == pytest.approx(ref, rel=1e-5)
    lo, hi = M.ensemble_interval_bounds(mu, v)
    assert float((hi - lo).mean()) == pytest.approx(2 * st.norm.ppf(0.95) * s, rel=1e-6)

    s = 1e-5
    y = y * 0.1
    ref = float(-st.norm.logpdf(y.numpy(), 0, s).mean())
    var = torch.full((100,), s**2, dtype=torch.float64)
    assert M.gaussian_nll(torch.zeros(100, dtype=torch.float64), y, var) == pytest.approx(
        ref, rel=1e-10
    )
    # the explicit argument restores a regularising floor
    floored = M.gaussian_nll(torch.zeros(100, dtype=torch.float64), y, var, min_variance=1e-8)
    assert floored > ref + 1.0


# --------------------------------------------------------------------------- M-MET-011


def test_M_MET_011_relative_jitter():
    x = np.array([[1e-4, 0.0]])
    cov = np.eye(2) * 1e-8
    ref = mahalanobis(x[0], np.zeros(2), np.linalg.inv(cov))  # 1.0
    got = M.mahalanobis_distance(
        torch.tensor(x), torch.zeros(2, dtype=torch.float64), torch.tensor(cov)
    )
    assert float(got[0]) == pytest.approx(ref, rel=1e-5)
    m = M.MahalanobisDistance()
    m.update(torch.tensor(x), torch.zeros(2, dtype=torch.float64), torch.tensor(cov))
    assert float(m.compute()) == pytest.approx(ref, rel=1e-5)

    cov_t = torch.tensor([[1.0, 0.8], [0.8, 1.0]], dtype=torch.float64)
    p = torch.zeros(1, 2, dtype=torch.float64)
    y = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    a = M.task_agnostic_correlations(p, y, cov_t[None])
    b = M.task_agnostic_correlations(p * 1e-4, y * 1e-4, cov_t[None] * 1e-8)
    assert float(b) * 1e4 == pytest.approx(float(a), rel=1e-5)


# --------------------------------------------------------------------------- M-CAL-001/002


def _vts_data(t_true: float, n: int = 4000, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    mean = torch.randn(n, generator=g)
    var = torch.rand(n, generator=g) * 0.5 + 0.1
    y = mean + torch.randn(n, generator=g) * torch.sqrt(var * t_true)
    return mean, var, y


@pytest.mark.parametrize("t_true", [0.01, 1.0, 15.0, 50.0])
def test_M_CAL_001_002_vts_reaches_closed_form_mle_unclamped(t_true):
    mean, var, y = _vts_data(t_true)
    mle = float(((y.double() - mean.double()) ** 2 / var.double()).mean())
    s = VarianceTemperatureScaler().fit(mean, var, y)
    assert s.temperature == pytest.approx(mle, rel=1e-6)
    mt, vt, yt = _vts_data(t_true, seed=1)
    cover = ((yt - mt).abs() <= 1.6448536 * s.transform(vt).sqrt()).float().mean()
    assert float(cover) == pytest.approx(0.90, abs=0.02)


def test_M_CAL_001_vts_optional_bounds():
    mean, var, y = _vts_data(50.0)
    s = VarianceTemperatureScaler().fit(mean, var, y, temperature_bounds=(0.05, 20.0))
    assert s.temperature == pytest.approx(20.0)
    s = VarianceTemperatureScaler().fit(
        mean, var, y, fit_floor=True, temperature_bounds=(0.05, 20.0), max_iter=50
    )
    assert s.temperature <= 20.0 + 1e-9
    with pytest.raises(ValueError, match="temperature_bounds"):
        VarianceTemperatureScaler().fit(mean, var, y, temperature_bounds=(2.0, 1.0))


def test_M_CAL_002_vts_floor_path_starts_from_closed_form():
    mean, var, y = _vts_data(15.0)
    s = VarianceTemperatureScaler().fit(mean, var, y, fit_floor=True, max_iter=500)
    assert s.temperature > 1.0


# --------------------------------------------------------------------------- M-CAL-003


def test_M_CAL_003_isotonic_ties_and_training_points():
    got = IsotonicMeanCalibrator().fit(torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 2.0, 1.0]))
    np.testing.assert_allclose(got.transform(torch.tensor([0.0, 1.0])).numpy(), [1.0, 1.0])
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    cal = IsotonicMeanCalibrator().fit(x, torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64))
    np.testing.assert_allclose(cal.transform(x).numpy(), [0.5, 0.5, 2.0], atol=1e-12)


@pytest.mark.parametrize("ties", [False, True])
def test_M_CAL_003_isotonic_matches_sklearn(ties):
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    if ties:
        x = np.round(x, 1)
    y = x + rng.normal(size=300)
    xt = rng.normal(size=1000) * 1.5
    got = IsotonicMeanCalibrator().fit(torch.tensor(x), torch.tensor(y)).transform(torch.tensor(xt))
    ref = IsotonicRegression(out_of_bounds="clip").fit(x, y).predict(xt)
    np.testing.assert_allclose(got.numpy(), ref, atol=1e-10)
