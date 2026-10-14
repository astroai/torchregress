"""Reference parity for every public export of :mod:`torchregress.metrics`.

Each export is compared, on random float64 data with a fixed seed, against an
independent reference: ``scoringrules``, ``properscoring``, ``scipy`` or
``scikit-learn`` where one exists, otherwise a numpy closed form. Exports
without any meaningful reference are listed in :data:`NO_REFERENCE` with a
one-line reason, and :func:`test_every_metrics_export_is_covered` enforces that
nothing in ``torchregress.metrics.__all__`` is left out.

Conventions (where torchregress deliberately differs from a reference default)
-----------------------------------------------------------------------------
* Ensemble CRPS (:func:`crps_from_samples`), :func:`energy_score` and
  :func:`vario_score` use the *fair* estimator (``1 / (M (M - 1))`` spread
  normaliser): ``scoringrules`` ``estimator="fair"``. ``properscoring`` and the
  ``scoringrules`` default (``"nrg"``) use the biased ``1 / M^2`` version.
* :func:`variogram_score` sums over ``i < j`` only, i.e. one half of
  ``scoringrules.variogram_score`` (which sums over all ``i != j``).
* :func:`continuous_ranked_probability_score` integrates ``2 QS_tau`` with
  trapezoid weights ``(tau_{i+1} - tau_{i-1}) / 2`` (``tau_0 = 0``,
  ``tau_{K+1} = 1``), not the equal weights ``2 / K`` of
  ``scoringrules.crps_quantile``.
* :func:`vario_score` is positively oriented (``-CRPS`` at ``rho = 1``).
* Median-based point metrics average the two middle values (numpy/scipy).
* :func:`rmse` of a 2-D target is ``sqrt`` of the MSE over all outputs, not the
  mean of per-output RMSEs (sklearn ``multioutput="uniform_average"``).
* Stateful metrics accumulate in float32 states (``torch.tensor(0.0)``), so
  class results (and functionals routed through a class) are compared at
  ``rel=1e-6``; pure functionals keep float64 and are compared at ~1e-10.

``scoringrules`` and ``properscoring`` live in the ``test`` extra; tests that
need them are skipped when they are missing (``pytest.importorskip``), but the
coverage-enforcement test always runs.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.special as ssp
import scipy.stats as st
import sklearn.metrics as skm
import torch
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import KernelDensity

import torchregress.metrics as M

RTOL = 1e-10

COVERED: dict[str, str] = {}

NO_REFERENCE: dict[str, str] = {
    "MarginalCalibrationError": "torchregress-specific binned marginal-CDF gap; no library equivalent",
    "marginal_calibration_error": "torchregress-specific binned marginal-CDF gap; no library equivalent",
    "calibration_metrics_report": "composition of ECE/MCE/bias, whose components are parity-tested",
    "censoring_rate": "trivial mean of a censoring mask; no library equivalent",
    "observed_mae": "trivial MAE over the observed subset; torchregress censoring codes",
    "interval_overlap_rate": "torchregress-specific overlap of predicted and censoring intervals",
    "RiskCoverageCurve": "torchregress-specific risk-coverage curve / AURC (trapezoid over coverage)",
    "RejectionPolicy": "torchregress-specific selective-prediction policy",
    "risk_coverage_curve": "torchregress-specific risk-coverage curve / AURC (trapezoid over coverage)",
    "OutlierFraction": "torchregress-specific |e|/(1+y) or |e| > k*std outlier rule (photo-z style)",
    "ood_metrics_report": "composition of OOD scores, whose components are parity-tested",
    "consistency_error": "trivial student/teacher L_p discrepancy; no library equivalent",
    "pseudo_label_acceptance_rate": "trivial thresholded mean; no library equivalent",
    "uncertain_gt_metrics_report": "composition of uncertain-target metrics; NLL component parity-tested",
}


def covers(*names: str, ref: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Register the exports a parity test covers and the reference it uses."""

    def deco(fn: Callable[..., None]) -> Callable[..., None]:
        for name in names:
            COVERED[name] = ref
        return fn

    return deco


def _sr():
    return pytest.importorskip("scoringrules", minversion="0.11")


def _ps():
    return pytest.importorskip("properscoring")


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


def _f(x) -> float:
    return float(np.asarray(x.detach() if torch.is_tensor(x) else x))


# =========================================================================== distribution


@covers("crps_gaussian", ref="scoringrules.crps_normal, properscoring.crps_gaussian")
def test_crps_gaussian():
    sr, ps = _sr(), _ps()
    rng = _rng(1)
    mu, sig, y = rng.normal(size=200), rng.uniform(0.1, 3, 200), rng.normal(size=200) * 2
    got = M.crps_gaussian(_t(mu), _t(y), _t(sig), reduction="none").numpy()
    np.testing.assert_allclose(got, sr.crps_normal(y, mu, sig), rtol=RTOL)
    np.testing.assert_allclose(got, ps.crps_gaussian(y, mu, sig), rtol=RTOL)


@covers("gaussian_nll", ref="-scipy.stats.norm.logpdf, scoringrules.logs_normal")
def test_gaussian_nll():
    rng = _rng(2)
    mu, var, y = rng.normal(size=200), rng.uniform(1e-3, 3, 200), rng.normal(size=200)
    got = M.gaussian_nll(_t(mu), _t(y), _t(var), reduction="none").numpy()
    ref = -st.norm.logpdf(y, mu, np.sqrt(var))
    np.testing.assert_allclose(got, ref, rtol=RTOL)
    assert M.gaussian_nll(_t(mu), _t(y), _t(var)) == pytest.approx(ref.mean(), rel=RTOL)
    np.testing.assert_allclose(got, _sr().logs_normal(y, mu, np.sqrt(var)), rtol=RTOL)


@covers("crps_from_samples", ref="scoringrules.crps_ensemble(estimator='fair')")
def test_crps_from_samples_fair_and_conventions():
    sr, ps = _sr(), _ps()
    rng = _rng(3)
    n_mem, n = 40, 300
    ens, y = rng.normal(size=(n_mem, n)), rng.normal(size=n) * 1.5
    got = M.crps_from_samples(_t(ens), _t(y), reduction="none").numpy()
    np.testing.assert_allclose(got, sr.crps_ensemble(y, ens.T, estimator="fair"), rtol=RTOL)
    np.testing.assert_allclose(got, sr.crps_ensemble(y, ens.T, estimator="pwm"), rtol=1e-8)
    # Convention: the biased 1/M^2 ("nrg") estimator, which properscoring uses, has a
    # spread term (M - 1) / M times the fair one.
    nrg = ps.crps_ensemble(y, ens.T)
    np.testing.assert_allclose(nrg, sr.crps_ensemble(y, ens.T, estimator="nrg"), rtol=1e-8)
    term1 = np.abs(ens - y).mean(0)
    np.testing.assert_allclose(term1 - got, (term1 - nrg) * n_mem / (n_mem - 1), rtol=1e-8)


def _trapezoid_crps_ref(levels: np.ndarray, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    sr = _sr()
    edges = np.concatenate([[0.0], levels, [1.0]])
    w = 0.5 * (edges[2:] - edges[:-2])
    qs = np.stack([sr.quantile_score(y, q[i], levels[i]) for i in range(len(levels))])
    return 2.0 * (w[:, None] * qs).sum(0)


@covers(
    "continuous_ranked_probability_score",
    "ContinuousRankedProbabilityScore",
    ref="2 * sum_i w_i scoringrules.quantile_score, trapezoid w_i",
)
def test_quantile_crps_trapezoid_convention():
    sr = _sr()
    rng = _rng(4)
    levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    n = 250
    mu, y = rng.normal(size=n), rng.normal(size=n)
    q = mu[None, :] + st.norm.ppf(levels)[:, None]
    qd = {float(lv): _t(q[i]) for i, lv in enumerate(levels)}
    ref = _trapezoid_crps_ref(levels, q, y)
    got = M.continuous_ranked_probability_score(qd, _t(y), reduction="none").numpy()
    np.testing.assert_allclose(got, ref, rtol=RTOL)
    # Not the equal-weight scoringrules.crps_quantile (documented convention).
    assert not np.allclose(got, sr.crps_quantile(y, q.T, levels), rtol=1e-3)
    m = M.ContinuousRankedProbabilityScore()
    m.update({k: v[:100] for k, v in qd.items()}, _t(y[:100]))
    m.update({k: v[100:] for k, v in qd.items()}, _t(y[100:]))
    assert _f(m.compute()) == pytest.approx(ref.mean(), rel=1e-6)  # float32 state


@covers("energy_score", "EnergyScore", ref="scoringrules.energy_score(estimator='fair')")
def test_energy_score_fair():
    sr = _sr()
    rng = _rng(5)
    ens, y = rng.normal(size=(30, 60, 3)), rng.normal(size=(60, 3))
    ref = sr.energy_score(y, np.swapaxes(ens, 0, 1), estimator="fair")
    got = M.energy_score(_t(ens), _t(y), reduction="none").numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-9)
    m = M.EnergyScore()
    m.update(_t(ens[:, :25]), _t(y[:25]))
    m.update(_t(ens[:, 25:]), _t(y[25:]))
    assert _f(m.compute()) == pytest.approx(ref.mean(), rel=1e-6)


@covers("variogram_score", "VariogramScore", ref="scoringrules.variogram_score(p=0.5, 'nrg') / 2")
def test_variogram_score_half_of_scoringrules():
    sr = _sr()
    rng = _rng(6)
    ens, y = rng.normal(size=(30, 40, 4)), rng.normal(size=(40, 4))
    ref = sr.variogram_score(y, np.swapaxes(ens, 0, 1), p=0.5, estimator="nrg") / 2.0
    got = M.variogram_score(_t(ens), _t(y), p=0.5, reduction="none").numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-9)
    m = M.VariogramScore(p=0.5)
    m.update(_t(ens[:, :15]), _t(y[:15]))
    m.update(_t(ens[:, 15:]), _t(y[15:]))
    assert _f(m.compute()) == pytest.approx(ref.mean(), rel=1e-6)


@covers("vario_score", "VarioScore", ref="-scoringrules.crps_ensemble(fair) at rho=1; numpy U-stat")
@pytest.mark.parametrize("rho", [1.0, 0.5, 2.0])
def test_vario_score(rho):
    sr = _sr()
    rng = _rng(7)
    ens, y = rng.normal(size=(25, 50)), rng.normal(size=50)
    n = ens.shape[0]
    pair = np.abs(ens[:, None, :] - ens[None, :, :]) ** rho
    half_spread = pair.sum((0, 1)) / (n * (n - 1)) / 2.0
    ref = half_spread - (np.abs(ens - y) ** rho).mean(0)
    if rho == 1.0:
        np.testing.assert_allclose(ref, -sr.crps_ensemble(y, ens.T, estimator="fair"), rtol=1e-9)
    # vario_score reduces through the metric's float32 state
    assert M.vario_score(_t(ens), _t(y), rho=rho) == pytest.approx(ref.mean(), rel=1e-6)
    m = M.VarioScore(rho=rho)
    m.update(_t(ens[:, :20]), _t(y[:20]))
    m.update(_t(ens[:, 20:]), _t(y[20:]))
    assert _f(m.compute()) == pytest.approx(ref.mean(), rel=1e-6)
    assert M.VarioScore.higher_is_better is True


@covers("dss_score", "DawidSebastianiScore", ref="2 * -scipy.stats.norm.logpdf - log(2 pi)")
def test_dawid_sebastiani():
    rng = _rng(8)
    mu, sig, y = rng.normal(size=100), rng.uniform(0.2, 2, 100), rng.normal(size=100)
    ref = 2.0 * -st.norm.logpdf(y, mu, sig) - math.log(2 * math.pi)
    assert M.dss_score(_t(mu), _t(sig), _t(y)) == pytest.approx(ref.mean(), rel=1e-6)
    m = M.DawidSebastianiScore()
    m.update(_t(mu[:40]), _t(sig[:40]), _t(y[:40]))
    m.update(_t(mu[40:]), _t(sig[40:]), _t(y[40:]))
    assert _f(m.compute()) == pytest.approx(ref.mean(), rel=1e-6)


@covers(
    "pinball_loss",
    "pinball_metric",
    "PinballMetric",
    ref="sklearn.metrics.mean_pinball_loss, scoringrules.quantile_score",
)
def test_pinball():
    sr = _sr()
    rng = _rng(9)
    y = rng.normal(size=120)
    levels = [0.1, 0.5, 0.8]
    qd = {lv: y + rng.normal(size=120) * 0.5 for lv in levels}
    np.testing.assert_allclose(
        M.pinball_loss(0.8, _t(qd[0.8]), _t(y)).numpy(),
        sr.quantile_score(y, qd[0.8], 0.8),
        rtol=RTOL,
    )
    ref = np.mean([skm.mean_pinball_loss(y, qd[lv], alpha=lv) for lv in levels])
    assert M.pinball_metric({k: _t(v) for k, v in qd.items()}, _t(y)) == pytest.approx(
        ref, rel=1e-6
    )
    m = M.PinballMetric()
    m.update({k: _t(v[:50]) for k, v in qd.items()}, _t(y[:50]))
    m.update({k: _t(v[50:]) for k, v in qd.items()}, _t(y[50:]))
    assert _f(m.compute()) == pytest.approx(ref, rel=1e-6)


def _bures_w2(m1, c1, m2, c2) -> float:
    r1 = sla.sqrtm(c1)
    cross = sla.sqrtm(r1 @ c2 @ r1)
    w2 = np.sum((m1 - m2) ** 2) + np.trace(c1 + c2 - 2 * np.real(cross))
    return math.sqrt(max(w2, 0.0))


@covers("wasserstein_gaussian_p2", ref="Bures formula with scipy.linalg.sqrtm")
def test_wasserstein_gaussian_p2():
    rng = _rng(10)
    a, b = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    c1, c2 = a @ a.T + 0.1 * np.eye(3), b @ b.T + 0.1 * np.eye(3)
    m1, m2 = rng.normal(size=3), rng.normal(size=3)
    assert M.wasserstein_gaussian_p2(_t(m1), _t(c1), _t(m2), _t(c2)) == pytest.approx(
        _bures_w2(m1, c1, m2, c2), rel=1e-8
    )
    s1, s2 = rng.uniform(0.1, 2, 3), rng.uniform(0.1, 2, 3)
    assert M.wasserstein_gaussian_p2(_t(m1), _t(s1), _t(m2), _t(s2)) == pytest.approx(
        _bures_w2(m1, np.diag(s1**2), m2, np.diag(s2**2)), rel=1e-8
    )


@covers("WassersteinGaussian", ref="Bures formula (scipy.linalg.sqrtm) to a Dirac")
def test_wasserstein_gaussian_metric():
    rng = _rng(11)
    mu, sig, y = rng.normal(size=20), rng.uniform(0.1, 2, 20), rng.normal(size=20)
    ref = np.mean(
        [
            _bures_w2(
                np.array([mu[i]]), np.array([[sig[i] ** 2]]), np.array([y[i]]), np.zeros((1, 1))
            )
            for i in range(20)
        ]
    )
    m = M.WassersteinGaussian()
    m.update(_t(mu), _t(sig), _t(y))
    assert _f(m.compute()) == pytest.approx(ref, rel=1e-6)


@covers(
    "probability_integral_transform",
    "kolmogorov_smirnov_uniform_statistic",
    ref="scipy.stats.norm.cdf, numpy.histogram, scipy.stats.kstest",
)
def test_pit_and_ks():
    rng = _rng(12)
    mu, sig = rng.normal(size=500), rng.uniform(0.5, 2, 500)
    y = mu + sig * rng.normal(size=500) * 1.3
    dist = torch.distributions.Normal(_t(mu), _t(sig))
    out = M.probability_integral_transform(dist.cdf, _t(y), n_bins=10, return_histogram=True)
    assert isinstance(out, dict)
    pit_ref = st.norm.cdf(y, mu, sig)
    np.testing.assert_allclose(np.asarray(out["pit_values"]), pit_ref, rtol=1e-12, atol=1e-15)
    counts, _ = np.histogram(pit_ref, bins=10, range=(0.0, 1.0))
    np.testing.assert_array_equal(np.asarray(out["histogram_counts"]), counts)
    ks_ref = st.kstest(pit_ref, "uniform").statistic
    assert _f(out["uniformity_ks"]) == pytest.approx(ks_ref, rel=1e-12)
    assert _f(M.kolmogorov_smirnov_uniform_statistic(_t(pit_ref))) == pytest.approx(
        ks_ref, rel=1e-12
    )


def _gaussian_grid(n: int = 200, seed: int = 13):
    rng = _rng(seed)
    mu, sig = rng.normal(size=n) * 0.5, rng.uniform(0.5, 1.5, n)
    y = mu + sig * rng.normal(size=n)
    support = np.linspace(-10, 10, 4001)
    dens = st.norm.pdf(support[None, :], mu[:, None], sig[:, None])
    return mu, sig, y, support, dens


@covers(
    "highest_posterior_density_level",
    "highest_posterior_density_coverage",
    ref="2 * scipy.stats.norm.cdf(|z|) - 1 for Gaussian densities (grid accuracy)",
)
def test_hpd_level_and_coverage():
    mu, sig, y, support, dens = _gaussian_grid()
    ref = 2 * st.norm.cdf(np.abs(y - mu) / sig) - 1
    got = M.highest_posterior_density_level(_t(support), _t(dens), _t(y)).numpy()
    np.testing.assert_allclose(got, ref, atol=5e-3)  # endpoint-width grid error
    cov = M.highest_posterior_density_coverage(_t(support), _t(dens), _t(y), alpha=0.5)
    # +/- a few samples whose level lies within the grid error of alpha
    assert cov == pytest.approx(np.mean(ref <= 0.5), abs=3 / len(y))


@covers(
    "conditional_density_estimation_loss",
    ref="Gaussian closed form 1/(2 sqrt(pi) sigma) - 2 scipy.stats.norm.pdf (grid accuracy)",
)
def test_cde_loss():
    mu, sig, y, support, dens = _gaussian_grid()
    ref = 1 / (2 * math.sqrt(math.pi) * sig) - 2 * st.norm.pdf(y, mu, sig)
    got = M.conditional_density_estimation_loss(_t(support), _t(dens), _t(y), reduction="none")
    np.testing.assert_allclose(np.asarray(got), ref, atol=1e-4)  # linear interpolation on grid


@covers(
    "distribution_metrics_report",
    ref="scoringrules.crps_normal / crps_ensemble(fair), scipy.stats.norm.logpdf, kstest",
)
def test_distribution_metrics_report():
    sr = _sr()
    rng = _rng(14)
    mu, sig = rng.normal(size=400), rng.uniform(0.5, 2, 400)
    y = mu + sig * rng.normal(size=400)
    rep = M.distribution_metrics_report(dist={"loc": _t(mu), "scale": _t(sig)}, y_true=_t(y))
    assert rep["crps"] == pytest.approx(sr.crps_normal(y, mu, sig).mean(), rel=1e-10)
    assert rep["log_prob"] == pytest.approx(st.norm.logpdf(y, mu, sig).mean(), rel=1e-10)
    ks = st.kstest(st.norm.cdf(y, mu, sig), "uniform").statistic
    assert _f(rep["pit_ks"]) == pytest.approx(ks, rel=1e-10)
    ens = mu[None, :] + sig[None, :] * rng.normal(size=(50, 400))
    rep = M.distribution_metrics_report(samples=_t(ens), y_true=_t(y))
    ref = sr.crps_ensemble(y, ens.T, estimator="fair").mean()
    assert rep["crps"] == pytest.approx(ref, rel=1e-10)


# =========================================================================== interval


def _intervals(seed: int = 15, n: int = 300):
    rng = _rng(seed)
    y = rng.normal(size=n)
    centre = y + rng.normal(size=n) * 0.8
    half = rng.uniform(0.2, 2.0, n)
    return centre - half, centre + half, y


@covers("interval_score", "IntervalScore", ref="scoringrules.interval_score")
def test_interval_score():
    sr = _sr()
    lo, hi, y = _intervals()
    ref = sr.interval_score(y, lo, hi, 0.1)
    np.testing.assert_allclose(
        np.asarray(M.interval_score(_t(lo), _t(hi), _t(y), alpha=0.1, reduction="none")),
        ref,
        rtol=RTOL,
    )
    assert _f(M.interval_score(_t(lo), _t(hi), _t(y), alpha=0.1)) == pytest.approx(ref.mean())
    m = M.IntervalScore(alpha=0.1)
    m.update(_t(lo[:100]), _t(hi[:100]), _t(y[:100]))
    m.update(_t(lo[100:]), _t(hi[100:]), _t(y[100:]))
    assert _f(m.compute()) == pytest.approx(ref.mean(), rel=1e-6)


@covers(
    "prediction_interval_coverage_probability",
    "prediction_interval_coverage",
    "PredictionIntervalCoverageProbability",
    "MeanPredictionIntervalWidth",
    "Sharpness",
    "sharpness",
    ref="numpy closed form: mean(l <= y <= u), mean(u - l)",
)
def test_coverage_and_width():
    lo, hi, y = _intervals()
    lo[0], y[0] = y[0], y[0]  # closed interval boundary
    cov_ref = np.mean((lo <= y) & (y <= hi))
    width_ref = np.mean(hi - lo)
    assert _f(M.prediction_interval_coverage_probability(_t(lo), _t(hi), _t(y))) == pytest.approx(
        cov_ref
    )
    assert _f(M.prediction_interval_coverage(_t(lo), _t(hi), _t(y))) == pytest.approx(cov_ref)
    m = M.PredictionIntervalCoverageProbability()
    m.update(_t(lo), _t(hi), _t(y))
    assert _f(m.compute()) == pytest.approx(cov_ref)
    for cls in (M.MeanPredictionIntervalWidth, M.Sharpness):
        w = cls()
        w.update(_t(lo), _t(hi))
        assert _f(w.compute()) == pytest.approx(width_ref, rel=1e-6)
    assert M.sharpness(_t(np.stack([lo, hi], axis=-1))) == pytest.approx(width_ref, rel=1e-12)


@covers("interval_metrics_report", ref="scoringrules.interval_score + numpy coverage/width")
def test_interval_metrics_report():
    sr = _sr()
    lo, hi, y = _intervals()
    rep = M.interval_metrics_report({"m": {"lower": _t(lo), "upper": _t(hi)}}, _t(y), alpha=0.2)
    assert _f(rep["m"]["score"]) == pytest.approx(sr.interval_score(y, lo, hi, 0.2).mean())
    assert _f(rep["m"]["picp"]) == pytest.approx(np.mean((lo <= y) & (y <= hi)))
    assert _f(rep["m"]["mpiw"]) == pytest.approx(np.mean(hi - lo))


# =========================================================================== ensemble


def _ensemble(seed: int = 16):
    rng = _rng(seed)
    means = rng.normal(size=(5, 200))
    variances = rng.uniform(0.05, 1.0, (5, 200))
    y = means.mean(0) + rng.normal(size=200)
    return means, variances, y


@covers(
    "ensemble_statistics",
    "ensemble_mean",
    "ensemble_std",
    "uncertainty_decomposition",
    "ensemble_variance_decomposition",
    ref="numpy mean / var(ddof=0) (law of total variance)",
)
def test_ensemble_moments():
    means, variances, _ = _ensemble()
    m, v = M.ensemble_statistics(_t(means))
    np.testing.assert_allclose(m.numpy(), means.mean(0), rtol=RTOL)
    np.testing.assert_allclose(v.numpy(), means.var(0), rtol=RTOL)
    np.testing.assert_allclose(M.ensemble_mean(_t(means)).numpy(), means.mean(0), rtol=RTOL)
    np.testing.assert_allclose(M.ensemble_std(_t(means)).numpy(), means.std(0), rtol=RTOL)
    dec = M.uncertainty_decomposition(_t(means), _t(variances))
    np.testing.assert_allclose(dec["epistemic_uncertainty"].numpy(), means.var(0), rtol=RTOL)
    np.testing.assert_allclose(dec["aleatoric_uncertainty"].numpy(), variances.mean(0), rtol=RTOL)
    np.testing.assert_allclose(
        dec["total_uncertainty"].numpy(), means.var(0) + variances.mean(0), rtol=RTOL
    )
    epi, ale = M.ensemble_variance_decomposition(_t(means), _t(variances))
    np.testing.assert_allclose(epi.numpy(), means.var(0), rtol=RTOL)
    np.testing.assert_allclose(ale.numpy(), variances.mean(0), rtol=RTOL)


@covers(
    "gaussian_nll_ensemble",
    "GaussianNLLEnsemble",
    "ensemble_interval_bounds",
    "ensemble_interval_metrics",
    "EnsembleIntervalMetrics",
    ref="scipy.stats.norm.logpdf / norm.ppf on the moment-matched Gaussian; scoringrules.interval_score",
)
def test_ensemble_gaussian_metrics():
    sr = _sr()
    means, variances, y = _ensemble()
    mu, sd = means.mean(0), np.sqrt(means.var(0) + variances.mean(0))
    nll_ref = -st.norm.logpdf(y, mu, sd).mean()
    assert _f(M.gaussian_nll_ensemble(_t(means), _t(variances), _t(y))) == pytest.approx(
        nll_ref, rel=1e-6
    )
    m = M.GaussianNLLEnsemble()
    m.update(_t(means[:, :80]), _t(variances[:, :80]), _t(y[:80]))
    m.update(_t(means[:, 80:]), _t(variances[:, 80:]), _t(y[80:]))
    assert _f(m.compute()) == pytest.approx(nll_ref, rel=1e-6)
    lo, hi = M.ensemble_interval_bounds(_t(means), _t(variances), alpha=0.1)
    z = st.norm.ppf(0.95)
    np.testing.assert_allclose(lo.numpy(), mu - z * sd, rtol=1e-6)
    np.testing.assert_allclose(hi.numpy(), mu + z * sd, rtol=1e-6)
    lo_r, hi_r = mu - z * sd, mu + z * sd
    out = M.ensemble_interval_metrics(_t(means), _t(variances), _t(y), alpha=0.1)
    assert _f(out["interval_score"]) == pytest.approx(
        sr.interval_score(y, lo_r, hi_r, 0.1).mean(), rel=1e-5
    )
    assert _f(out["picp"]) == pytest.approx(np.mean((lo_r <= y) & (y <= hi_r)))
    em = M.EnsembleIntervalMetrics(alpha=0.1)
    em.update(_t(means), _t(variances), _t(y))
    assert _f(em.compute()["interval_score"]) == pytest.approx(_f(out["interval_score"]))


# =========================================================================== point


def _point(seed: int = 17, shape=(101,)):
    rng = _rng(seed)
    y = rng.normal(size=shape) * 2 + 1
    return y + rng.standard_t(3, size=shape), y


@covers(
    "mean_squared_error",
    "rmse",
    "mean_absolute_error",
    "r2_score",
    ref="sklearn.metrics.{mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score}",
)
@pytest.mark.parametrize("shape", [(101,), (60, 3)])
def test_basic_point_metrics(shape):
    yp, yt = _point(shape=shape)
    w = _rng(1).uniform(0.1, 2.0, shape[0])

    def rmse_ref(yt, yp, sample_weight=None):
        # Convention: torchregress RMSE of a 2-D target is sqrt of the MSE over all
        # outputs, not sklearn's mean of per-output RMSEs (identical for 1-D).
        return math.sqrt(skm.mean_squared_error(yt, yp, sample_weight=sample_weight))

    for fn, ref in [
        (M.mean_squared_error, skm.mean_squared_error),
        (M.mean_absolute_error, skm.mean_absolute_error),
        (M.rmse, rmse_ref),
    ]:
        assert _f(fn(_t(yp), _t(yt))) == pytest.approx(ref(yt, yp), rel=1e-12)
        assert _f(fn(_t(yp), _t(yt), sample_weight=_t(w))) == pytest.approx(
            ref(yt, yp, sample_weight=w), rel=1e-12
        )
    assert _f(M.r2_score(_t(yp), _t(yt))) == pytest.approx(skm.r2_score(yt, yp), rel=1e-12)


@covers("median_absolute_error", "MedianAbsoluteError", ref="sklearn.metrics.median_absolute_error")
@pytest.mark.parametrize("shape", [(101,), (100,), (60, 3), (61, 4)])
def test_median_absolute_error(shape):
    yp, yt = _point(shape=shape)
    ref = skm.median_absolute_error(yt, yp)
    assert _f(M.median_absolute_error(_t(yp), _t(yt))) == pytest.approx(ref, rel=1e-12)
    m = M.MedianAbsoluteError()
    m.update(_t(yp[:30]), _t(yt[:30]))
    m.update(_t(yp[30:]), _t(yt[30:]))
    assert _f(m.compute()) == pytest.approx(ref, rel=1e-12)
    if len(shape) == 2:
        raw = M.median_absolute_error(_t(yp), _t(yt), multioutput="raw_values")
        np.testing.assert_allclose(
            np.asarray(raw), skm.median_absolute_error(yt, yp, multioutput="raw_values")
        )


@covers(
    "median_absolute_deviation",
    "MedianAbsoluteDeviation",
    "NormalizedMedianAbsoluteDeviation",
    ref="1.4826 * scipy.stats.median_abs_deviation",
)
@pytest.mark.parametrize("n", [100, 101])
def test_mad_and_nmad(n):
    yp, yt = _point(shape=(n,))
    yt = np.abs(yt)  # keep 1 + y away from 0 for the relative NMAD
    ref = 1.4826 * st.median_abs_deviation(yt - yp)
    assert _f(M.median_absolute_deviation(_t(yp), _t(yt))) == pytest.approx(ref, rel=1e-12)
    m = M.MedianAbsoluteDeviation()
    m.update(_t(yp), _t(yt))
    assert _f(m.compute()) == pytest.approx(ref, rel=1e-12)
    nmad = M.NormalizedMedianAbsoluteDeviation()
    nmad.update(_t(yp), _t(yt))
    assert _f(nmad.compute()) == pytest.approx(1.4826 * st.median_abs_deviation(yp - yt), rel=1e-12)
    rel = M.NormalizedMedianAbsoluteDeviation(normalization="relative")
    rel.update(_t(yp), _t(yt))
    ref_rel = 1.4826 * st.median_abs_deviation((yp - yt) / (1 + yt))
    assert _f(rel.compute()) == pytest.approx(ref_rel, rel=1e-12)


@covers(
    "trimmed_mean_squared_error", "TrimmedMeanSquaredError", ref="scipy.stats.trim_mean(e^2, p)"
)
@pytest.mark.parametrize("p", [0.0, 0.1, 0.23, 0.4])
def test_trimmed_mse(p):
    yp, yt = _point(shape=(57,))
    ref = st.trim_mean((yt - yp) ** 2, p)
    assert _f(M.trimmed_mean_squared_error(_t(yp), _t(yt), proportion=p)) == pytest.approx(
        ref, rel=1e-12
    )
    m = M.TrimmedMeanSquaredError(proportion=p)
    m.update(_t(yp[:20]), _t(yt[:20]))
    m.update(_t(yp[20:]), _t(yt[20:]))
    assert _f(m.compute()) == pytest.approx(ref, rel=1e-12)


@covers("huber_loss", "HuberMetric", ref="scipy.special.huber(delta, r)")
def test_huber():
    yp, yt = _point()
    ref = ssp.huber(1.5, yt - yp).mean()
    assert _f(M.huber_loss(_t(yp), _t(yt), delta=1.5)) == pytest.approx(ref, rel=1e-12)
    m = M.HuberMetric(delta=1.5)
    m.update(_t(yp), _t(yt))
    assert _f(m.compute()) == pytest.approx(ref, rel=1e-6)


@covers(
    "normalized_rmse",
    "NormalizedRMSE",
    ref="sklearn root_mean_squared_error / (std ddof=1 | ptp | mean|y| | scipy.stats.iqr)",
)
@pytest.mark.parametrize("norm", ["std", "range", "mean", "iqr"])
def test_normalized_rmse(norm):
    yp, yt = _point()
    scale = {
        "std": np.std(yt, ddof=1),
        "range": np.ptp(yt),
        "mean": np.mean(np.abs(yt)),
        "iqr": st.iqr(yt),
    }[norm]
    ref = skm.root_mean_squared_error(yt, yp) / scale
    assert _f(M.normalized_rmse(_t(yp), _t(yt), normalization=norm)) == pytest.approx(
        ref, rel=1e-12
    )
    m = M.NormalizedRMSE(normalization=norm)
    m.update(_t(yp), _t(yt))
    assert _f(m.compute()) == pytest.approx(ref, rel=1e-12)


@covers("attenuation_factor", ref="scipy.stats.linregress(y_true, y_pred).slope")
def test_attenuation_factor():
    yp, yt = _point()
    assert _f(M.attenuation_factor(_t(0.7 * yp), _t(yt))) == pytest.approx(
        st.linregress(yt, 0.7 * yp).slope, rel=1e-12
    )


@covers("tail_mae", "tail_rmse", ref="sklearn MAE / RMSE on the numpy.quantile tail subset")
@pytest.mark.parametrize("tail", ["upper", "lower", "both"])
def test_tail_metrics(tail):
    yp, yt = _point(shape=(203,))
    if tail == "upper":
        mask = yt >= np.quantile(yt, 0.9)
    elif tail == "lower":
        mask = yt <= np.quantile(yt, 0.1)
    else:
        dev = np.abs(yt - np.quantile(yt, 0.5))
        mask = dev >= np.quantile(dev, 0.9)
    assert _f(M.tail_mae(_t(yp), _t(yt), tail=tail)) == pytest.approx(
        skm.mean_absolute_error(yt[mask], yp[mask]), rel=1e-12
    )
    assert _f(M.tail_rmse(_t(yp), _t(yt), tail=tail)) == pytest.approx(
        skm.root_mean_squared_error(yt[mask], yp[mask]), rel=1e-12
    )


@covers("regression_metrics_report", ref="sklearn.metrics / scipy (per key)")
def test_regression_metrics_report():
    yp, yt = _point()
    rep = M.regression_metrics_report(_t(yp), _t(yt))
    e = yt - yp
    assert _f(rep["mse"]) == pytest.approx(skm.mean_squared_error(yt, yp), rel=1e-12)
    assert _f(rep["rmse"]) == pytest.approx(skm.root_mean_squared_error(yt, yp), rel=1e-12)
    assert _f(rep["mae"]) == pytest.approx(skm.mean_absolute_error(yt, yp), rel=1e-12)
    assert _f(rep["r2"]) == pytest.approx(skm.r2_score(yt, yp), rel=1e-12)
    assert _f(rep["huber_loss"]) == pytest.approx(ssp.huber(1.0, e).mean(), rel=1e-12)
    assert _f(rep["mad"]) == pytest.approx(1.4826 * st.median_abs_deviation(e), rel=1e-12)
    assert _f(rep["nmad"]) == pytest.approx(1.4826 * st.median_abs_deviation(-e), rel=1e-12)


# =========================================================================== calibration


def _quantile_forecast(seed: int = 18, n: int = 400):
    rng = _rng(seed)
    mu, sig = rng.normal(size=n), rng.uniform(0.5, 1.5, n)
    y = mu + 1.3 * sig * rng.normal(size=n)
    return mu, sig, y


def _qce(levels: np.ndarray, q: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    err = np.array([np.mean(y <= q[i]) for i in range(len(levels))]) - levels
    return np.abs(err).mean(), math.sqrt(np.mean(err**2)), np.abs(err).max()


@covers(
    "expected_calibration_error",
    "ExpectedCalibrationError",
    "calibration_score",
    ref="numpy quantile calibration error (Kuleshov et al. 2018) on scipy.stats.norm.ppf quantiles",
)
def test_quantile_calibration_error():
    mu, sig, y = _quantile_forecast()
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    q = mu[None, :] + sig[None, :] * st.norm.ppf(levels)[:, None]
    mace, rmsce, maxce = _qce(levels, q, y)
    qd = {float(lv): _t(q[i]) for i, lv in enumerate(levels)}
    out = M.expected_calibration_error(qd, _t(y))
    assert _f(out["mean_absolute_calibration_error"]) == pytest.approx(mace, rel=1e-6)
    assert _f(out["root_mean_squared_calibration_error"]) == pytest.approx(rmsce, rel=1e-6)
    assert _f(out["maximum_calibration_error"]) == pytest.approx(maxce, rel=1e-6)
    m = M.ExpectedCalibrationError()
    m.update({k: v[:150] for k, v in qd.items()}, _t(y[:150]))
    m.update({k: v[150:] for k, v in qd.items()}, _t(y[150:]))
    assert _f(m.compute()["mean_absolute_calibration_error"]) == pytest.approx(mace, rel=1e-6)

    levels = np.linspace(0.05, 0.95, 19)
    q = mu[None, :] + sig[None, :] * st.norm.ppf(levels)[:, None]
    out = M.calibration_score(_t(y), _t(mu), _t(sig))
    assert _f(out["mean_absolute_calibration_error"]) == pytest.approx(
        _qce(levels, q, y)[0], rel=1e-6
    )


@covers("bias", ref="numpy mean(y_pred - y_true)")
def test_bias():
    yp, yt = _point()
    assert _f(M.bias(_t(yp), _t(yt))) == pytest.approx(np.mean(yp - yt), rel=1e-12)


# =========================================================================== censored / ordinal


@covers(
    "concordance_index", ref="(scipy.stats.kendalltau + 1) / 2 uncensored; brute-force Harrell C"
)
def test_concordance_index():
    rng = _rng(19)
    y = rng.permutation(80).astype(float)
    pred = y + rng.normal(size=80) * 20
    tau = st.kendalltau(pred, y).statistic
    # concordance_index returns a float32 tensor
    assert _f(M.concordance_index(_t(pred), _t(y))) == pytest.approx((tau + 1) / 2, rel=1e-6)
    cens = rng.choice([0, 1], size=80, p=[0.7, 0.3])
    num = den = 0.0
    for i in range(80):
        if cens[i] != 0:
            continue
        for j in range(80):
            if (cens[j] == 0 and y[i] < y[j]) or (cens[j] == 1 and y[j] >= y[i] and i != j):
                den += 1
                num += 1.0 if pred[i] < pred[j] else (0.5 if pred[i] == pred[j] else 0.0)
    got = M.concordance_index(_t(pred), _t(y), torch.as_tensor(cens))
    assert _f(got) == pytest.approx(num / den, rel=1e-6)


@covers(
    "ordinal_accuracy",
    "mean_absolute_class_error",
    "quadratic_weighted_kappa",
    ref="sklearn.metrics.{accuracy_score, mean_absolute_error, cohen_kappa_score(quadratic)}",
)
def test_ordinal_metrics():
    rng = _rng(20)
    yt = rng.integers(0, 5, 300)
    yp = np.clip(yt + rng.integers(-2, 3, 300), 0, 4)
    tp, tt = torch.as_tensor(yp), torch.as_tensor(yt)
    assert _f(M.ordinal_accuracy(tp, tt)) == pytest.approx(skm.accuracy_score(yt, yp))
    assert _f(M.mean_absolute_class_error(tp, tt)) == pytest.approx(skm.mean_absolute_error(yt, yp))
    assert _f(M.quadratic_weighted_kappa(tp, tt, num_classes=5)) == pytest.approx(
        skm.cohen_kappa_score(yt, yp, weights="quadratic"), rel=1e-6
    )


# =========================================================================== multivariate / TAC


@covers(
    "MultivariateRMSE",
    "MultivariateMAE",
    ref="sqrt(D * sklearn MSE), D * sklearn MAE (per-sample vector norms, not divided by D)",
)
def test_multivariate_point():
    yp, yt = _point(shape=(70, 3))
    r = M.MultivariateRMSE()
    r.update(_t(yp[:30]), _t(yt[:30]))
    r.update(_t(yp[30:]), _t(yt[30:]))
    assert _f(r.compute()) == pytest.approx(math.sqrt(3 * skm.mean_squared_error(yt, yp)), rel=1e-6)
    a = M.MultivariateMAE()
    a.update(_t(yp), _t(yt))
    assert _f(a.compute()) == pytest.approx(3 * skm.mean_absolute_error(yt, yp), rel=1e-6)


@covers(
    "task_agnostic_correlations",
    "TaskAgnosticCorrelations",
    ref="numpy conditional-Gaussian residual |y_j - E[y_j | y_-j]| (Shukla et al. 2024)",
)
def test_tac():
    rng = _rng(21)
    n, d = 40, 3
    a = rng.normal(size=(n, d, d))
    cov = a @ np.swapaxes(a, 1, 2) + 0.5 * np.eye(d)
    mu = rng.normal(size=(n, d))
    y = mu + rng.normal(size=(n, d))
    errs = []
    for b in range(n):
        for j in range(d):
            o = [k for k in range(d) if k != j]
            cond = mu[b, j] + cov[b, j, o] @ np.linalg.solve(
                cov[b][np.ix_(o, o)], y[b, o] - mu[b, o]
            )
            errs.append(abs(y[b, j] - cond))
    got = M.task_agnostic_correlations(_t(mu), _t(y), _t(cov))
    assert _f(got) == pytest.approx(np.mean(errs), rel=1e-5)  # relative jitter 1e-6
    m = M.TaskAgnosticCorrelations()
    m.update(_t(mu), _t(y), _t(cov))
    assert _f(m.compute()) == pytest.approx(np.mean(errs), rel=1e-5)


# =========================================================================== OOD


@covers("mahalanobis_distance", "MahalanobisDistance", ref="scipy.spatial.distance.mahalanobis")
@pytest.mark.parametrize("scale", [1.0, 1e-8])
def test_mahalanobis(scale):
    rng = _rng(22)
    a = rng.normal(size=(3, 3))
    cov = (a @ a.T + 0.3 * np.eye(3)) * scale
    mean = rng.normal(size=3) * math.sqrt(scale)
    x = mean + rng.normal(size=(50, 3)) * math.sqrt(scale)
    vi = np.linalg.inv(cov)
    ref = np.array([mahalanobis(xi, mean, vi) for xi in x])
    np.testing.assert_allclose(
        M.mahalanobis_distance(_t(x), _t(mean), _t(cov)).numpy(), ref, rtol=1e-5
    )
    m = M.MahalanobisDistance()
    m.update(_t(x), _t(mean), _t(cov))
    assert _f(m.compute()) == pytest.approx(ref.mean(), rel=1e-5)


@covers(
    "typicality_score",
    "TypicalityScore",
    ref="scipy.stats.norm.logpdf (x given); -scipy.stats.norm.entropy (Monte-Carlo class)",
)
def test_typicality():
    rng = _rng(23)
    mu, var = rng.normal(size=(30, 2)), rng.uniform(0.2, 2, (30, 2))
    x = mu + rng.normal(size=(30, 2))
    got = M.typicality_score((_t(mu), _t(var)), _t(x))
    np.testing.assert_allclose(got.numpy(), st.norm.logpdf(x, mu, np.sqrt(var)).sum(-1), rtol=1e-12)
    torch.manual_seed(0)
    m = M.TypicalityScore(n_samples=4000)
    m.update((_t(mu[:, 0]), _t(var[:, 0])))
    ref = -st.norm(mu[:, 0], np.sqrt(var[:, 0])).entropy().mean()
    assert _f(m.compute()) == pytest.approx(ref, abs=0.02)  # Monte-Carlo


@covers(
    "entropy_score", "EntropyScore", ref="scipy.stats.entropy(numpy.histogram(samples, n_bins))"
)
def test_entropy_score():
    rng = _rng(24)
    samples = rng.normal(size=(500, 10, 2)) * rng.uniform(0.5, 2, (1, 10, 2))
    ref = np.zeros(10)
    for b in range(10):
        for d in range(2):
            counts, _ = np.histogram(samples[:, b, d], bins=12)
            ref[b] += st.entropy(counts)
    np.testing.assert_allclose(M.entropy_score(_t(samples), n_bins=12).numpy(), ref, rtol=1e-6)
    m = M.EntropyScore(n_bins=12)
    m.update(_t(samples))
    assert _f(m.compute()) == pytest.approx(ref.mean(), rel=1e-6)


@covers(
    "kernel_density_score",
    "KernelDensityScore",
    ref="sklearn.neighbors.KernelDensity (exp(score_samples) * (2 pi h^2)^(D/2))",
)
def test_kernel_density_score():
    rng = _rng(25)
    ref_pts, test_pts, h = rng.normal(size=(200, 3)), rng.normal(size=(40, 3)) * 1.5, 0.7
    kde = KernelDensity(bandwidth=h, kernel="gaussian").fit(ref_pts)
    ref = np.exp(kde.score_samples(test_pts)) * (2 * math.pi * h**2) ** 1.5
    got = M.kernel_density_score(_t(test_pts), _t(ref_pts), bandwidth=h)
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-9)
    m = M.KernelDensityScore(bandwidth=h)
    m.update(_t(test_pts), _t(ref_pts))
    assert _f(m.compute()) == pytest.approx(ref.mean(), rel=1e-9)


# =========================================================================== uncertain targets


@covers("noisy_target_gaussian_nll", ref="-scipy.stats.norm.logpdf(y, mu, sqrt(var_p + var_t))")
def test_noisy_target_gaussian_nll():
    rng = _rng(26)
    mu, vp, vt = rng.normal(size=100), rng.uniform(0.1, 1, 100), rng.uniform(0.1, 1, 100)
    y = mu + rng.normal(size=100)
    ref = -st.norm.logpdf(y, mu, np.sqrt(vp + vt)).mean()
    assert _f(M.noisy_target_gaussian_nll(_t(mu), _t(vp), _t(y), _t(vt))) == pytest.approx(
        ref, rel=1e-12
    )


# =========================================================================== coverage


def test_every_metrics_export_is_covered():
    """Every ``torchregress.metrics.__all__`` name has a parity test or a reason."""
    exported = set(M.__all__)
    covered, unexplained = set(COVERED), set(NO_REFERENCE)
    assert not covered & unexplained, f"both covered and NO_REFERENCE: {covered & unexplained}"
    missing = exported - covered - unexplained
    assert not missing, f"exports without parity test or NO_REFERENCE reason: {sorted(missing)}"
    stale = (covered | unexplained) - exported
    assert not stale, f"registry names not exported by torchregress.metrics: {sorted(stale)}"
    assert all(reason.strip() for reason in NO_REFERENCE.values())
