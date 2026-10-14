"""Regression tests for the 0.3.0 audit, batch B: conformal + losses (B-CONF-*, B-LOSS-*).

Each test reproduces one audit finding and pins the fixed behaviour. Coverage
simulations use fewer replications than the original reproductions; each one
states the binomial tolerance it relies on. Where the fixed procedure returns an
infinite interval the coverage is exactly 1 (deterministic), so those checks need
only a handful of replications. All randomness is seeded.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from torchregress.calibration.semicp import SemiConformalCalibrator
from torchregress.losses import (
    CQR,
    BalancedMSELoss,
    BinReweightedMSELoss,
    DensityWeightedLoss,
    FocalRLoss,
    FunctionalEIVLoss,
    GaussianNLLLoss,
    InputNoiseAugmentationLoss,
    LDSLoss,
    PropensityWeightedLoss,
    PseudoLabelConsistencyLoss,
    PseudoLabelNLL,
    SLSLoss,
    SplitConformal,
    StructuralEIVLoss,
    WeightedMSELoss,
)
from torchregress.losses import OrthogonalDistanceRegressionLoss as ODR
from torchregress.losses.conformal import (
    CTI,
    CVPlus,
    DensityConformal,
    DistributionalConformal,
    JackknifePlus,
    MultiTargetConformal,
    MultivariateScoreConformal,
    NonExchangeableConformalRegressor,
    R2CConformal,
    SLSConformal,
    _weighted_quantile,
    finite_sample_quantile,
)
from torchregress.losses.eiv import LatentMarginalizationLoss
from torchregress.utils.tensor_ops import compute_model_gradients


def _binom_lower(p: float, reps: int, z: float = 3.0) -> float:
    """Lower ``z``-sigma binomial bound around a nominal coverage ``p``."""
    return p - z * math.sqrt(p * (1.0 - p) / reps)


# ═══════════════════════════════════════════════════════════════════════════════
# B-CONF-001 — finite-sample quantile is +inf when ceil((n+1)(1-alpha)) > n
# ═══════════════════════════════════════════════════════════════════════════════


def test_B_CONF_001_finite_sample_quantile_inf_when_rank_exceeds_n() -> None:
    scores = torch.arange(1.0, 11.0)  # n = 10
    assert float(finite_sample_quantile(scores, 0.05)) == math.inf  # k = 11 > 10
    assert float(finite_sample_quantile(scores, 0.1)) == 10.0  # k = 10: max score
    assert float(_weighted_quantile(scores, 0.95)) == math.inf  # unweighted path


def test_B_CONF_001_small_n_predictors_return_infinite_intervals() -> None:
    torch.manual_seed(0)
    # Split CP, n = 10, alpha = 0.05.
    cp = SplitConformal(alpha=0.05)
    cp.calibrate(torch.zeros(10), torch.randn(10))
    lo, hi = cp.predict_interval(torch.zeros(3))
    assert torch.isneginf(lo).all() and torch.isposinf(hi).all()
    # CQR, n = 12, alpha = 0.05 (ceil(13 * 0.95) = 13 > 12).
    band = torch.tensor([[-0.5, 0.5]]).repeat(12, 1)
    cqr = CQR(alpha=0.05)
    cqr.calibrate(band, torch.randn(12, 1))
    lo, hi = cqr.predict_interval(band[:2])
    assert torch.isneginf(lo).all() and torch.isposinf(hi).all()
    # Mondrian: group 1 has 15 points (ceil(16 * 0.95) = 16 > 15), group 0 has 200.
    groups = torch.tensor([0] * 200 + [1] * 15)
    cp = SplitConformal(alpha=0.05)
    cp.calibrate(torch.zeros(215), torch.randn(215), groups=groups)
    lo, hi = cp.predict_interval(torch.zeros(2), groups=torch.tensor([0, 1]))
    assert torch.isfinite(hi[0]) and torch.isposinf(hi[1]) and torch.isneginf(lo[1])
    # MultiTargetConformal, n = 5, alpha = 0.1 (ceil(6 * 0.9) = 6 > 5).
    mtc = MultiTargetConformal(alpha=0.1)
    mtc.calibrate(torch.zeros(5, 2), torch.randn(5, 2))
    lo, hi = mtc.predict_interval(torch.zeros(1, 2))
    assert torch.isposinf(hi).all() and torch.isneginf(lo).all()


def test_B_CONF_001_split_conformal_coverage_small_n() -> None:
    """n=10, alpha=0.05: the clamp covered exactly 10/11 = 0.909 < 0.95.

    With the fix the threshold is +inf, so coverage is exactly 1 for every draw
    (deterministic; 200 replications suffice).
    """
    rng = np.random.default_rng(0)
    n, alpha, reps = 10, 0.05, 200
    covered = 0
    for _ in range(reps):
        y = torch.from_numpy(rng.standard_normal(n + 1))
        cp = SplitConformal(alpha=alpha)
        cp.calibrate(torch.zeros(n, dtype=torch.float64), y[:n])
        lo, hi = cp.predict_interval(torch.zeros(1, dtype=torch.float64))
        covered += int(lo.item() <= y[n].item() <= hi.item())
    assert covered == reps


# ═══════════════════════════════════════════════════════════════════════════════
# B-CONF-002 — weighted quantile: scale invariance, +inf test atom, test weights
# ═══════════════════════════════════════════════════════════════════════════════


def _tibshirani_threshold(scores: np.ndarray, w: np.ndarray, w_test: float, alpha: float) -> float:
    """Independent numpy reference (Tibshirani et al. 2019)."""
    o = np.argsort(scores)
    s, p = scores[o], w[o] / (w.sum() + w_test)
    idx = np.searchsorted(np.cumsum(p), 1 - alpha - 1e-12)
    return float(s[idx]) if idx < len(s) else math.inf


def test_B_CONF_002_weighted_quantile_scale_invariant() -> None:
    torch.manual_seed(0)
    scores = torch.rand(25, dtype=torch.float64)
    ref = _weighted_quantile(scores, 0.9, torch.ones(25, dtype=torch.float64))
    assert float(ref) == float(finite_sample_quantile(scores, 0.1))
    for c in (0.05, 10.0):
        got = _weighted_quantile(scores, 0.9, torch.full((25,), c, dtype=torch.float64))
        assert float(got) == float(ref)
    w = torch.rand(25, dtype=torch.float64) + 0.1
    assert float(_weighted_quantile(scores, 0.9, w)) == float(
        _weighted_quantile(scores, 0.9, 7.0 * w)
    )


def test_B_CONF_002_weighted_threshold_inf_on_test_atom() -> None:
    scores = torch.arange(1.0, 11.0, dtype=torch.float64)
    # Uniform weights, n = 10, alpha = 0.05: calibration mass tops out at 10/11.
    q = _weighted_quantile(scores, 0.95, torch.ones(10, dtype=torch.float64))
    assert float(q) == math.inf
    # Non-uniform weights: the heavy test point (raw weight 20) needs the atom.
    w = torch.linspace(0.5, 1.5, 10, dtype=torch.float64)
    assert float(_weighted_quantile(scores, 0.9, w, test_weight=20.0)) == math.inf


def test_B_CONF_002_constant_weights_coverage() -> None:
    """Constant weights w=10, n=25, alpha=0.1: exact split CP, coverage 24/26=0.923.

    The old raw-scale test weight picked k=23 (23/26 = 0.885 < 0.9).  4000 reps,
    3-sigma binomial bound around 0.9 (0.886).
    """
    rng = np.random.default_rng(0)
    n, alpha, reps = 25, 0.1, 4000
    w = torch.full((n,), 10.0, dtype=torch.float64)
    covered = 0
    for _ in range(reps):
        y = torch.from_numpy(rng.standard_normal(n + 1))
        cp = SplitConformal(alpha=alpha)
        cp.calibrate(torch.zeros(n, dtype=torch.float64), y[:n], weights=w)
        lo, hi = cp.predict_interval(torch.zeros(1, dtype=torch.float64))
        covered += int(lo.item() <= y[n].item() <= hi.item())
    assert covered / reps >= _binom_lower(1 - alpha, reps)


def test_B_CONF_002_nexcp_threshold_scale_invariant() -> None:
    scores = torch.arange(1.0, 101.0, dtype=torch.float64)
    a = NonExchangeableConformalRegressor(alpha=0.1).calibrate(
        scores, torch.full((100,), 0.01, dtype=torch.float64)
    )
    b = NonExchangeableConformalRegressor(alpha=0.1).calibrate(
        scores, torch.ones(100, dtype=torch.float64)
    )
    c = NonExchangeableConformalRegressor(alpha=0.1, normalize_weights=False).calibrate(
        scores, torch.ones(100, dtype=torch.float64)
    )
    assert float(a.threshold_) == float(b.threshold_) == float(c.threshold_) == 91.0


def test_B_CONF_002_covariate_shift_test_weights_match_reference() -> None:
    """Covariate shift x_src~N(0,1), x_tgt~N(1.5,1), y|x~N(0,e^x), true density
    ratios, n=20, alpha=0.1, 1000 reps.

    With the test point's own weight (``test_weights=``) the library threshold equals
    the independent Tibshirani reference in every replication, and covers >= 0.9
    (observed ~0.97; 3-sigma bound 0.872).  The old raw-scale ``w_{n+1}=1`` gave ~0.77.
    """
    rng = np.random.default_rng(0)
    n, alpha, shift, reps = 20, 0.1, 1.5, 1000
    covered = 0
    for _ in range(reps):
        xc = rng.standard_normal(n)
        yc = rng.standard_normal(n) * np.exp(xc / 2)
        xt = rng.standard_normal() + shift
        yt = rng.standard_normal() * np.exp(xt / 2)
        w = np.exp(shift * xc - shift**2 / 2)
        wt = float(np.exp(shift * xt - shift**2 / 2))
        cp = SplitConformal(alpha=alpha)
        cp.calibrate(
            torch.zeros(n, dtype=torch.float64), torch.from_numpy(yc), weights=torch.from_numpy(w)
        )
        lo, hi = cp.predict_interval(
            torch.zeros(1, dtype=torch.float64), test_weights=torch.tensor([wt])
        )
        assert float(hi) == _tibshirani_threshold(np.abs(yc), w, wt, alpha)
        covered += int(lo.item() <= yt <= hi.item())
    assert covered / reps >= _binom_lower(1 - alpha, reps)


def test_B_CONF_002_test_weights_other_entry_points() -> None:
    rng = np.random.default_rng(3)
    n, alpha = 30, 0.1
    scores = np.abs(rng.standard_normal(n))
    w = rng.uniform(0.2, 2.0, n)
    wt = np.array([0.1, 1.0, 5.0, 50.0])
    expected = [_tibshirani_threshold(scores, w, float(v), alpha) for v in wt]
    assert math.isinf(expected[-1])
    s_t, w_t, wt_t = (torch.from_numpy(a) for a in (scores, w, wt))
    nex = NonExchangeableConformalRegressor(alpha=alpha).calibrate(s_t, w_t)
    assert nex.thresholds(wt_t).tolist() == expected
    # Mondrian + test weights: each group uses its own calibration scores/weights.
    groups = torch.tensor([0] * 15 + [1] * 15)
    cp = SplitConformal(alpha=alpha)
    cp.calibrate(torch.zeros(n, dtype=torch.float64), s_t, groups=groups, weights=w_t)
    _, hi = cp.predict_interval(
        torch.zeros(2, dtype=torch.float64),
        groups=torch.tensor([0, 1]),
        test_weights=torch.tensor([1.0, 1.0], dtype=torch.float64),
    )
    for g, row in ((0, slice(0, 15)), (1, slice(15, 30))):
        assert float(hi[g]) == _tibshirani_threshold(scores[row], w[row], 1.0, alpha)
    # MultivariateScoreConformal: per-point radii from test weights.
    mu = torch.zeros(n, 2, dtype=torch.float64)
    y = torch.from_numpy(rng.standard_normal((n, 2)))
    msc = MultivariateScoreConformal(alpha=alpha).calibrate(
        mu, torch.ones(2, dtype=torch.float64), y, w_t
    )
    maha = ((y - mu) ** 2).sum(dim=-1).numpy()
    radii = [_tibshirani_threshold(maha, w, float(v), alpha) for v in wt]
    assert math.isinf(radii[-1]) and not math.isinf(radii[0])
    far = torch.full((4, 2), 1e3, dtype=torch.float64)
    inside = msc.covers(
        torch.zeros(4, 2, dtype=torch.float64), torch.ones(2), far, test_weights=wt_t
    )
    assert inside.tolist() == [math.isinf(r) for r in radii]  # only +inf radii cover


# ═══════════════════════════════════════════════════════════════════════════════
# B-CONF-003 — SemiConformalCalibrator thresholds
# ═══════════════════════════════════════════════════════════════════════════════


def test_B_CONF_003_semicp_inf_when_test_atom_needed() -> None:
    cal = SemiConformalCalibrator().fit(torch.arange(1.0, 11.0), torch.ones(10))
    assert cal.compute_thresholds(1.0, alpha=0.05) == math.inf
    batch = cal.compute_thresholds(torch.tensor([1.0, 1.0]), alpha=0.05)
    assert torch.isposinf(batch).all()


def test_B_CONF_003_semicp_exact_order_statistic() -> None:
    """Uniform weights, w_target=1: exactly the ceil((n+1)(1-alpha))-th order statistic
    (the float32 CDF missed it by one in ~1% of (n, alpha) pairs)."""
    bad = []
    for n in range(5, 120):
        s = torch.arange(1.0, n + 1.0, dtype=torch.float64)
        cal = SemiConformalCalibrator().fit(s, torch.ones(n, dtype=torch.float64))
        for alpha in (0.05, 0.1, 0.2, 0.25, 0.3):
            k = math.ceil((n + 1) * (1 - alpha))
            expected = float(k) if k <= n else math.inf
            got = cal.compute_thresholds(1.0, alpha=alpha)
            if got != expected:
                bad.append((n, alpha, got, expected))
    assert not bad, bad[:4]


def test_B_CONF_003_semicp_coverage_small_n() -> None:
    """n=10, alpha=0.05: the max-score fallback covered 10/11; +inf covers always."""
    rng = np.random.default_rng(0)
    n, reps = 10, 200
    covered = 0
    for _ in range(reps):
        s = np.abs(rng.standard_normal(n + 1))
        cal = SemiConformalCalibrator().fit(torch.from_numpy(s[:n]), torch.ones(n))
        covered += int(s[n] <= cal.compute_thresholds(1.0, alpha=0.05))
    assert covered == reps


def test_B_CONF_003_semicp_covariate_shift_coverage() -> None:
    """True density-ratio weights for calibration and target points, n=20,
    alpha=0.1, 1000 reps.  Old behaviour ~0.78; fixed >= 0.9 (3-sigma bound 0.872)."""
    rng = np.random.default_rng(0)
    n, alpha, shift, reps = 20, 0.1, 1.5, 1000
    covered = 0
    for _ in range(reps):
        xc = rng.standard_normal(n)
        yc = rng.standard_normal(n) * np.exp(xc / 2)
        xt = rng.standard_normal() + shift
        yt = rng.standard_normal() * np.exp(xt / 2)
        w = np.exp(shift * xc - shift**2 / 2)
        wt = float(np.exp(shift * xt - shift**2 / 2))
        cal = SemiConformalCalibrator().fit(torch.from_numpy(np.abs(yc)), torch.from_numpy(w))
        covered += int(abs(yt) <= cal.compute_thresholds(wt, alpha=alpha))
    assert covered / reps >= _binom_lower(1 - alpha, reps)


# ═══════════════════════════════════════════════════════════════════════════════
# B-CONF-004 — CV+ / Jackknife+ ranks (Barber et al. 2021, eq. 6)
# ═══════════════════════════════════════════════════════════════════════════════


def _jackknife_mean_inputs(y_cal: np.ndarray):
    """Leave-one-out sample-mean 'models' (K = n): oob preds and member preds."""
    n = y_cal.shape[0]
    loo = (y_cal.sum() - y_cal) / (n - 1)
    oob = torch.from_numpy(loo).reshape(n, 1)
    members = torch.from_numpy(loo).reshape(n, 1, 1)  # [K=n, n_test=1, 1]
    return oob, members, torch.arange(n)


def _reference_jackknife_plus(y_cal: np.ndarray, alpha: float):
    n = y_cal.shape[0]
    loo = (y_cal.sum() - y_cal) / (n - 1)
    r = np.abs(y_cal - loo)
    lo_c, hi_c = np.sort(loo - r), np.sort(loo + r)
    kl = math.floor(alpha * (n + 1))
    ku = math.ceil((1 - alpha) * (n + 1))
    lo = -math.inf if kl < 1 else float(lo_c[kl - 1])
    hi = math.inf if ku > n else float(hi_c[ku - 1])
    return lo, hi


def test_B_CONF_004_jackknife_plus_matches_barber_reference() -> None:
    rng = np.random.default_rng(0)
    y_cal = rng.standard_normal(10)
    alpha = 0.1  # (n+1) alpha = 1.1 -> floor 1 (the old code used ceil = 2)
    oob, members, folds = _jackknife_mean_inputs(y_cal)
    jp = JackknifePlus(alpha=alpha)
    jp.calibrate_ensemble(oob, torch.from_numpy(y_cal).reshape(-1, 1), folds)
    lo, hi = jp.predict_interval(members)
    assert (float(lo), float(hi)) == _reference_jackknife_plus(y_cal, alpha)


def test_B_CONF_004_cvplus_infinite_endpoints_small_n() -> None:
    """n=8, alpha=0.1: floor(0.9)=0 and ceil(8.1)=9>8 -> (-inf, inf)."""
    y_cal = np.random.default_rng(2).standard_normal(8)
    oob, members, folds = _jackknife_mean_inputs(y_cal)
    cp = CVPlus(alpha=0.1)
    cp.calibrate_ensemble(oob, torch.from_numpy(y_cal).reshape(-1, 1), folds)
    lo, hi = cp.predict_interval(members)
    assert float(lo) == -math.inf and float(hi) == math.inf


def test_B_CONF_004_cvplus_coverage_matches_reference() -> None:
    """n=10, alpha=0.1, t_2 data, 2000 reps, paired with the reference on the same data.

    The library interval must coincide with the reference in every replication
    (old: lower endpoint one rank too high, coverage ~0.865 vs ~0.906); the
    reference attains Thm 1's >= 1 - 2 alpha = 0.8.
    """
    rng = np.random.default_rng(1)
    n, alpha, reps = 10, 0.1, 2000
    cov = 0
    for _ in range(reps):
        y = rng.standard_t(2, size=n + 1)
        oob, members, folds = _jackknife_mean_inputs(y[:n])
        cp = CVPlus(alpha=alpha)
        cp.calibrate_ensemble(oob, torch.from_numpy(y[:n]).reshape(-1, 1), folds)
        lo, hi = cp.predict_interval(members)
        assert (float(lo), float(hi)) == _reference_jackknife_plus(y[:n], alpha)
        cov += int(float(lo) <= y[n] <= float(hi))
    assert cov / reps >= 1 - 2 * alpha


# ═══════════════════════════════════════════════════════════════════════════════
# B-CONF-005 — DensityConformal: same density function at calibration/prediction
# ═══════════════════════════════════════════════════════════════════════════════


def test_B_CONF_005_density_conformal_coverage_lognormal() -> None:
    """y = exp(x + 0.5 e), y_hat = E[y|x]; n=50, alpha=0.1, 1000 reps.

    Old (density at y at calibration, at y_hat at prediction) ~0.86; fixed ~0.90.
    3-sigma binomial bound around 0.9 (0.872).
    """
    rng = np.random.default_rng(0)
    n, alpha, reps = 50, 0.1, 1000
    cov = 0
    for _ in range(reps):
        x = rng.standard_normal(n + 1)
        y = np.exp(x + 0.5 * rng.standard_normal(n + 1))
        yp = np.exp(x + 0.125)
        cp = DensityConformal(alpha=alpha)
        cp.calibrate(torch.from_numpy(yp[:n]), torch.from_numpy(y[:n]))
        lo, hi = cp.predict_interval(torch.from_numpy(yp[n:]))
        cov += int(lo.item() <= y[n] <= hi.item())
    assert cov / reps >= _binom_lower(1 - alpha, reps)


def test_B_CONF_005_density_conformal_coverage_bimodal() -> None:
    """y = +-2 + 0.3 e, y_hat = 0; n=100, alpha=0.1, 400 reps.  Old coverage: 0.0.
    3-sigma bound around 0.9 (0.855)."""
    rng = np.random.default_rng(1)
    n, alpha, reps = 100, 0.1, 400
    cov = 0
    for _ in range(reps):
        y = 2 * rng.choice([-1.0, 1.0], size=n + 1) + 0.3 * rng.standard_normal(n + 1)
        yp = np.zeros(n + 1)
        cp = DensityConformal(alpha=alpha)
        cp.calibrate(torch.from_numpy(yp[:n]), torch.from_numpy(y[:n]))
        lo, hi = cp.predict_interval(torch.from_numpy(yp[n:]))
        cov += int(lo.item() <= y[n] <= hi.item())
    assert cov / reps >= _binom_lower(1 - alpha, reps)


def test_B_CONF_005_density_conformal_calibrate_with_mask() -> None:
    torch.manual_seed(0)
    y = torch.randn(40)
    yp = y + 0.1 * torch.randn(40)
    mask = torch.ones(40, dtype=torch.bool)
    mask[::4] = False
    y_nan = y.clone()
    y_nan[~mask] = float("nan")  # masked targets may be missing
    cp = DensityConformal(alpha=0.1)
    cp.calibrate(yp, y_nan, mask=mask)  # used to raise IndexError (double masking)
    ref = DensityConformal(alpha=0.1)
    ref.calibrate(yp[mask], y[mask])
    assert torch.equal(cp.q_hat, ref.q_hat)


# ═══════════════════════════════════════════════════════════════════════════════
# B-CONF-006 — Mondrian: negative ids, per-sample thresholds for level-set methods
# ═══════════════════════════════════════════════════════════════════════════════


def test_B_CONF_006_mondrian_negative_group_ids() -> None:
    torch.manual_seed(0)
    g = torch.tensor([-1, 0, 1] * 40)
    y = torch.randn(120) * torch.tensor([1.0, 10.0, 100.0]).repeat(40)
    cp = SplitConformal(alpha=0.1)
    cp.calibrate(torch.zeros(120), y, groups=g)
    _, hi = cp.predict_interval(torch.zeros(4), groups=torch.tensor([-1, 0, 1, -1]))
    expected = torch.stack([cp.q_hat[-1], cp.q_hat[0], cp.q_hat[1], cp.q_hat[-1]])
    assert torch.equal(hi, expected)
    with pytest.raises(ValueError, match="unseen group"):
        cp.predict_interval(torch.zeros(1), groups=torch.tensor([-2]))


def test_B_CONF_006_mondrian_negative_group_coverage() -> None:
    """Group -1 (sd 10) used to receive group 1's (sd 1) quantile: coverage ~0.14.
    n=60 per group, alpha=0.1, 800 reps, 3-sigma bound around 0.9 (0.868)."""
    rng = np.random.default_rng(0)
    alpha, reps, n = 0.1, 800, 60
    covered = 0
    for _ in range(reps):
        y_wide = 10 * rng.standard_normal(n + 1)
        y = torch.from_numpy(np.concatenate([y_wide[:n], rng.standard_normal(n)]))
        cp = SplitConformal(alpha=alpha)
        cp.calibrate(
            torch.zeros(2 * n, dtype=torch.float64), y, groups=torch.tensor([-1] * n + [1] * n)
        )
        lo, hi = cp.predict_interval(torch.zeros(1, dtype=torch.float64), groups=torch.tensor([-1]))
        covered += int(lo.item() <= y_wide[n] <= hi.item())
    assert covered / reps >= _binom_lower(1 - alpha, reps)


def _r2c_probs(y_center: torch.Tensor, edges: torch.Tensor, width: float) -> torch.Tensor:
    centers = 0.5 * (edges[1:] + edges[:-1])
    logits = -0.5 * ((centers.unsqueeze(0) - y_center.unsqueeze(1)) / width) ** 2
    return torch.softmax(logits, dim=-1)


def test_B_CONF_006_r2c_uses_per_sample_group_threshold() -> None:
    torch.manual_seed(0)
    edges = torch.linspace(-10, 10, 201)
    n = 400
    g = torch.tensor([0] * (n // 2) + [1] * (n // 2))
    y = torch.randn(n) * torch.where(g == 0, 1.0, 4.0)  # group 1: model 4x over-confident
    r2c = R2CConformal(alpha=0.1, bin_edges=edges)
    r2c.calibrate(_r2c_probs(torch.zeros(n), edges, 1.0), y, groups=g)
    test_probs = _r2c_probs(torch.zeros(2), edges, 1.0)
    lo, hi = r2c.predict_interval(test_probs, groups=torch.tensor([0, 1]))
    width = (hi - lo).squeeze(-1)
    assert width[1] > width[0]
    # Each point matches a predictor calibrated on its own group only.
    for k in (0, 1):
        alone = R2CConformal(alpha=0.1, bin_edges=edges)
        alone.calibrate(_r2c_probs(torch.zeros(n // 2), edges, 1.0), y[g == k])
        lo_k, hi_k = alone.predict_interval(test_probs[k : k + 1])
        assert torch.equal(lo[k], lo_k[0]) and torch.equal(hi[k], hi_k[0])


def test_B_CONF_006_dcp_groups_at_prediction() -> None:
    torch.manual_seed(0)
    g = torch.tensor([0] * 200 + [1] * 200)
    pit0 = torch.rand(200)
    pit1 = torch.where(torch.rand(200) < 0.5, torch.rand(200) * 0.02, 1 - torch.rand(200) * 0.02)
    dcp = DistributionalConformal(alpha=0.1)
    dcp.calibrate(torch.cat([pit0, pit1]), torch.zeros(400), groups=g)
    q0, q1 = float(dcp.q_hat[0]), float(dcp.q_hat[1])
    assert q1 > q0

    def icdf(levels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return levels.unsqueeze(0).expand(x.shape[0], 2)

    lo, hi = dcp.predict_intervals_from_cdf(icdf, torch.zeros(3, 1), groups=torch.tensor([1, 0, 1]))
    width = (hi - lo).squeeze(-1)
    assert torch.allclose(width, torch.tensor([q1, q0, q1]), atol=1e-6)


def test_B_CONF_006_cti_groups_at_prediction() -> None:
    torch.manual_seed(0)
    n = 300
    g = torch.tensor([0] * n + [1] * n)
    # Model density N(0, 1) for everyone; group 1 targets are 3x wider.
    y = torch.randn(2 * n) * torch.where(g == 0, 1.0, 3.0)
    log_dens_cal = -0.5 * y**2 - 0.5 * math.log(2 * math.pi)

    def density_fn(y_grid: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (-0.5 * y_grid**2 - 0.5 * math.log(2 * math.pi)).expand(x.shape[0], -1)

    cti = CTI(alpha=0.1, grid_size=801)
    cti.calibrate(log_dens_cal, y, groups=g)
    lo, hi = cti.predict_intervals_from_density(
        density_fn, torch.zeros(2, 1), -20.0, 20.0, groups=torch.tensor([0, 1])
    )
    for k in (0, 1):
        alone = CTI(alpha=0.1, grid_size=801)
        alone.calibrate(log_dens_cal[g == k], y[g == k])
        lo_k, hi_k = alone.predict_intervals_from_density(
            density_fn, torch.zeros(1, 1), -20.0, 20.0
        )
        assert torch.equal(lo[k], lo_k[0]) and torch.equal(hi[k], hi_k[0])
    assert (hi - lo)[1] > (hi - lo)[0]


def test_B_CONF_006_sls_conformal_groups_at_prediction() -> None:
    torch.manual_seed(0)
    sls = SLSLoss(d=1, context_dim=2, K=1, warmup_steps=5, hidden_dim=8, n_transforms=1)
    n = 100
    x = torch.randn(2 * n, 2)
    g = torch.tensor([0] * n + [1] * n)
    y = torch.randn(2 * n, 1) * torch.where(g == 0, 1.0, 3.0).unsqueeze(-1)
    conf = SLSConformal(sls, alpha=0.1, grid_size=200)
    conf.calibrate(x, y, groups=g)
    x_test = torch.randn(2, 2)
    lo, hi = conf.predict_interval_from_grid(x_test, -15.0, 15.0, groups=torch.tensor([0, 1]))
    for k in (0, 1):
        alone = SLSConformal(sls, alpha=0.1, grid_size=200)
        alone.calibrate(x[g == k], y[g == k])
        lo_k, hi_k = alone.predict_interval_from_grid(x_test[k : k + 1], -15.0, 15.0)
        assert torch.equal(lo[k], lo_k[0]) and torch.equal(hi[k], hi_k[0])


# ═══════════════════════════════════════════════════════════════════════════════
# B-LOSS-001 — EIV propagated variance keeps its gradient
# ═══════════════════════════════════════════════════════════════════════════════


def _eiv_data():
    g = torch.Generator().manual_seed(0)
    x_true = torch.randn(64, 1, generator=g, dtype=torch.float64)
    x_obs = x_true + 0.5 * torch.randn(64, 1, generator=g, dtype=torch.float64)
    y_obs = 2.0 * x_true + 0.1 * torch.randn(64, 1, generator=g, dtype=torch.float64)
    return x_obs, y_obs


def _grad_vs_fd(make_loss):
    x_obs, y_obs = _eiv_data()
    w = torch.tensor([[1.3]], dtype=torch.float64, requires_grad=True)
    loss = make_loss(lambda x: x @ w.T)(x_obs, y_obs)
    (g_auto,) = torch.autograd.grad(loss, w)
    h = 1e-6
    w_p, w_m = w.detach() + h, w.detach() - h
    lp = make_loss(lambda x: x @ w_p.T)(x_obs, y_obs)
    lm = make_loss(lambda x: x @ w_m.T)(x_obs, y_obs)
    return float(g_auto), float((lp - lm) / (2 * h))


@pytest.mark.parametrize(
    "make_loss",
    [
        lambda m: FunctionalEIVLoss(m, sigma_x=0.5, sigma_y=0.1),
        lambda m: StructuralEIVLoss(
            m, sigma_x=0.5, sigma_y=0.1, sigma_xy=torch.zeros(1, 1, dtype=torch.float64)
        ),
    ],
    ids=["functional", "structural"],
)
def test_B_LOSS_001_eiv_gradient_matches_finite_difference(make_loss) -> None:
    ga, gf = _grad_vs_fd(make_loss)
    assert abs(ga - gf) <= 1e-5 * max(1.0, abs(gf)), (ga, gf)


def test_B_LOSS_001_jacobian_graph_only_when_grad_enabled() -> None:
    w = torch.tensor([[1.5]], requires_grad=True)
    x = torch.randn(4, 1, requires_grad=True)
    jac = compute_model_gradients(x @ w.T, x, 1)
    assert jac.requires_grad  # differentiable w.r.t. w
    (dw,) = torch.autograd.grad(jac.sum(), w)
    assert torch.allclose(dw, torch.tensor([[4.0]]))
    no_graph = compute_model_gradients(x @ w.T, x, 1, create_graph=False)
    assert not no_graph.requires_grad
    x_obs, y_obs = _eiv_data()
    with torch.no_grad():
        val = FunctionalEIVLoss(nn.Linear(1, 1).double(), sigma_x=0.5, sigma_y=0.1)(x_obs, y_obs)
    assert torch.isfinite(val)


# ═══════════════════════════════════════════════════════════════════════════════
# B-LOSS-002 — marginalisation losses force per-element base losses
# ═══════════════════════════════════════════════════════════════════════════════


class _GaussHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(2, 2).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)  # [mean, log_var]


_EPS = torch.randn(8, 16, 2, generator=torch.Generator().manual_seed(123), dtype=torch.float64)


def _sampler(x_obs: torch.Tensor, n: int) -> torch.Tensor:
    return x_obs.unsqueeze(0) + 0.8 * _EPS[:n, : x_obs.shape[0]]


def _lm_inputs():
    g = torch.Generator().manual_seed(1)
    return (
        torch.randn(16, 2, generator=g, dtype=torch.float64),
        torch.randn(16, 1, generator=g, dtype=torch.float64),
    )


def test_B_LOSS_002_likelihood_mode_is_logmeanexp() -> None:
    model = _GaussHead()
    x, y = _lm_inputs()
    base = GaussianNLLLoss()
    lm = LatentMarginalizationLoss(model, base, posterior_sampler=_sampler, n_samples=8)
    got = lm(x, y)
    assert base.reduction == "mean"  # restored after the call
    out = model(_sampler(x, 8).reshape(-1, 2)).reshape(8, 16, 2)
    mu, lv = out[..., :1], out[..., 1:]
    nll = 0.5 * (lv + (y - mu) ** 2 / lv.exp() + math.log(2 * math.pi))
    ref = (-(torch.logsumexp(-nll, dim=0) - math.log(8))).mean()
    assert torch.allclose(got, ref, rtol=1e-6)
    expectation = LatentMarginalizationLoss(
        model, base, posterior_sampler=_sampler, n_samples=8, marginalization_mode="expectation"
    )(x, y)
    assert float(got) < float(expectation) - 1e-6  # Jensen


def test_B_LOSS_002_weights_applied() -> None:
    model = _GaussHead()
    x, y = _lm_inputs()
    lm = LatentMarginalizationLoss(
        model, GaussianNLLLoss(), posterior_sampler=_sampler, n_samples=8
    )
    w = torch.zeros(16, dtype=torch.float64)
    w[0] = 1.0
    assert torch.allclose(lm(x, y, weights=w), lm(x[:1], y[:1]), rtol=1e-6)
    aug = InputNoiseAugmentationLoss(model, GaussianNLLLoss(), sigma_x=0.3, n_samples=4)
    torch.manual_seed(5)
    full = aug(x, y, weights=w)
    torch.manual_seed(5)
    unweighted = aug(x, y)
    assert not torch.allclose(full, unweighted)


def test_B_LOSS_002_scalar_base_loss_rejected_and_docs_example_runs() -> None:
    x, y = _lm_inputs()
    lm = LatentMarginalizationLoss(
        _GaussHead(), lambda p, t, **kw: ((p - t) ** 2).mean(), posterior_sampler=_sampler
    )
    with pytest.raises(ValueError, match="scalar"):
        lm(x, y)
    # docs/losses/eiv.md example (structural prior + measurement error)
    loss_fn = LatentMarginalizationLoss(
        model=_GaussHead(),
        base_loss=GaussianNLLLoss(),
        prior_mean=x.mean(dim=0),
        prior_cov=torch.cov(x.T),
        sigma_u=0.2,
        n_samples=16,
    )
    loss = loss_fn(x, y)
    loss.backward()
    assert torch.isfinite(loss)


# ═══════════════════════════════════════════════════════════════════════════════
# B-LOSS-003 — ODR: relative jitter, per-sample unrolled steps, masked mean
# ═══════════════════════════════════════════════════════════════════════════════


def test_B_LOSS_003_odr_matches_closed_form_small_sigma() -> None:
    torch.manual_seed(0)
    b, sx, sy = 2.0, 0.03, 0.03
    x = torch.randn(8, 1, dtype=torch.float64)
    y = b * x + 0.1 * torch.randn(8, 1, dtype=torch.float64)
    loss = ODR(
        lambda z: b * z,
        sigma_x=sx,
        sigma_y=sy,
        learning_rate=1e-4,
        max_iterations=4000,
        tolerance=0.0,
        reduction="none",
    )
    got = loss(x, y)
    assert loss.last_optimality_residual_ < 1e-8
    ref = ((y - b * x) ** 2 / (sy**2 + b * b * sx**2)).squeeze(-1)
    assert torch.allclose(got, ref, rtol=1e-3)  # old absolute 1e-3 jitter: off by ~2x


def test_B_LOSS_003_odr_unrolled_independent_of_batch_size() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 1, dtype=torch.float64)
    y = 2 * x + 1.0
    kw = dict(
        sigma_x=1.0,
        sigma_y=1.0,
        learning_rate=0.1,
        max_iterations=50,
        gradient_mode="unrolled",
        reduction="none",
    )
    alone = ODR(lambda z: 2.0 * z, **kw)(x, y)
    batched = ODR(lambda z: 2.0 * z, **kw)(x.repeat(64, 1), y.repeat(64, 1))
    assert torch.allclose(alone, batched[:1], rtol=1e-6)


def test_B_LOSS_003_odr_mask_excluded_from_mean() -> None:
    torch.manual_seed(2)
    x = torch.randn(10, 1, dtype=torch.float64)
    y = 2 * x + torch.randn(10, 1, dtype=torch.float64)
    mask = torch.ones(10, 1, dtype=torch.bool)
    mask[5:] = False
    kw = dict(sigma_x=1.0, sigma_y=1.0, learning_rate=0.01, max_iterations=200)
    masked = ODR(lambda z: 2.0 * z, **kw)(x, y, mask=mask)
    subset = ODR(lambda z: 2.0 * z, **kw)(x[:5], y[:5])
    assert torch.allclose(masked, subset, rtol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# B-LOSS-004 — SLSLoss forward is side-effect free and frontier-consistent
# ═══════════════════════════════════════════════════════════════════════════════


def _sls():
    torch.manual_seed(0)
    return SLSLoss(d=2, context_dim=3, K=2, warmup_steps=5, hidden_dim=16, n_transforms=2)


def _sls_inputs():
    g = torch.Generator().manual_seed(1)
    return torch.randn(32, 3, generator=g), torch.randn(32, 2, generator=g)


def test_B_LOSS_004_forward_does_not_mutate_state() -> None:
    ctx, y = _sls_inputs()
    fresh = _sls()
    used = copy.deepcopy(fresh)
    ref = fresh(ctx, y, step=0)
    used(ctx, y, step=100)  # post-warmup evaluation
    assert torch.allclose(ref, used(ctx, y, step=0))
    assert used.frontier._freeze_weights is True


def test_B_LOSS_004_quantile_pass_uses_frontier_pass_G() -> None:
    ctx, y = _sls_inputs()
    loss = _sls()
    step = 5 + 400
    G, _ = loss.frontier(y, ctx, beta=loss.frontier.beta_at(400), freeze_weights=False)
    G_eval, _ = loss.evaluate_frontier(y, ctx, step=step)
    assert torch.equal(G, G_eval)
    q = loss.quantile_net(ctx)
    phi, psi = loss.get_current_window(step=step)
    levels = torch.tensor([max(0.01, loss.tau - phi), loss.tau, min(0.99, loss.tau + psi)])
    diff = G.unsqueeze(-1) - q
    expected = torch.max(levels * diff, (levels - 1.0) * diff).sum(-1).mean()
    assert torch.allclose(loss.forward_quantiles(ctx, y, step=step), expected, rtol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# B-LOSS-005 — BalancedMSE empty bins
# ═══════════════════════════════════════════════════════════════════════════════


def _per_sample_weights(loss: BalancedMSELoss, y: torch.Tensor) -> torch.Tensor:
    return torch.stack([loss(y[i : i + 1] + 1.0, y[i : i + 1]) for i in range(y.shape[0])])


def test_B_LOSS_005_empty_bin_does_not_rescale_populated_bins() -> None:
    torch.manual_seed(0)
    y = (torch.rand(200, 1) * 6 - 3).clamp(-2.999, 2.999)
    edges = torch.linspace(-3.0, 3.0, 7)
    edges_plus_empty = torch.cat([torch.tensor([-100.0]), edges])
    w_a = _per_sample_weights(BalancedMSELoss(edges).fit(y), y)
    w_b = _per_sample_weights(BalancedMSELoss(edges_plus_empty).fit(y), y)
    assert torch.allclose(w_a, w_b, rtol=1e-4)
    assert torch.allclose(w_a.mean(), torch.tensor(1.0), rtol=1e-5)  # unit mean on train


def test_B_LOSS_005_training_loss_scale_with_empty_bin() -> None:
    torch.manual_seed(0)
    y = torch.randn(1000, 1)
    loss = BalancedMSELoss(torch.linspace(-5, 5, 11)).fit(y)  # bin [-5, -4) is empty
    assert float(loss.bin_weights[0]) == 0.0
    assert float(loss(y + 1.0, y)) == pytest.approx(1.0, rel=1e-5)
    brw = BinReweightedMSELoss(10, noise_sigma=0.0).fit(torch.cat([y, torch.tensor([[40.0]])]))
    assert 0.1 < float(brw(y + 1.0, y)) < 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# B-LOSS-006 — sample weights go through BaseLoss._reduce
# ═══════════════════════════════════════════════════════════════════════════════


def _w_data():
    g = torch.Generator().manual_seed(0)
    return torch.randn(32, 1, generator=g), torch.randn(32, 1, generator=g)


def _make_weighted(name: str, y: torch.Tensor):
    if name == "focal":
        return FocalRLoss(), {}
    if name == "balanced":
        return BalancedMSELoss(torch.linspace(-4, 4, 5), count_smoothing=1.0).fit(y), {}
    if name == "binrew":
        return BinReweightedMSELoss(4).fit(y), {}
    if name == "density":
        loss = DensityWeightedLoss()
        loss.fit_density(y)
        return loss, {"sample_indices": torch.arange(32)}
    if name == "lds":
        loss = LDSLoss()
        loss.fit(y, n_bins=10)
        return loss, {"sample_indices": torch.arange(32)}
    return PropensityWeightedLoss(), {"propensity": torch.full((32, 1), 0.5)}


@pytest.mark.parametrize("name", ["focal", "balanced", "binrew", "density", "lds", "propensity"])
def test_B_LOSS_006_weights_follow_reduce_contract(name: str) -> None:
    yp, y = _w_data()
    loss, kw = _make_weighted(name, y)
    plain = loss(yp, y, **kw)
    assert torch.allclose(plain, loss(yp, y, weights=torch.full((32,), 3.0), **kw), rtol=1e-5)
    # Per-sample weights [B] on a [B, 1] loss must not broadcast to [B, B].
    w = torch.rand(32, generator=torch.Generator().manual_seed(1))
    loss.reduction = "none"
    per = loss(yp, y, **kw)
    weighted_none = loss(yp, y, weights=w, **kw)
    assert weighted_none.shape == per.shape
    loss.reduction = "mean"
    expected = (per.squeeze(-1) * w).sum() / w.sum()
    assert torch.allclose(loss(yp, y, weights=w, **kw), expected, rtol=1e-5)


def test_B_LOSS_006_reference_semantics_weighted_mse() -> None:
    yp, y = _w_data()
    base = WeightedMSELoss()
    assert torch.allclose(base(yp, y), base(yp, y, weights=torch.full((32,), 3.0)))


# ═══════════════════════════════════════════════════════════════════════════════
# B-LOSS-007 — DensityWeightedLoss on-the-fly weights use the training normaliser
# ═══════════════════════════════════════════════════════════════════════════════


def _dw_setup():
    g = torch.Generator().manual_seed(0)
    y = torch.randn(50, 1, generator=g)
    yp = y + torch.randn(50, 1, generator=g)
    loss = DensityWeightedLoss(kernel_width=0.5)
    loss.fit_density(y)
    return loss, yp, y


def test_B_LOSS_007_on_the_fly_matches_precomputed() -> None:
    loss, yp, y = _dw_setup()
    idx = torch.tensor([0, 1, 2])
    assert torch.allclose(
        loss(yp[idx], y[idx], sample_indices=idx), loss(yp[idx], y[idx]), rtol=1e-4
    )
    i = int(torch.argmax(y.abs()))  # rarest target: weight >> 1, even alone in a batch
    expected = loss.density_weights[i] * (yp[i] - y[i]) ** 2
    assert torch.allclose(loss(yp[i : i + 1], y[i : i + 1]), expected.squeeze(), rtol=1e-4)


def test_B_LOSS_007_nan_targets_ignored() -> None:
    loss, yp, y = _dw_setup()
    y_nan = y.clone()
    y_nan[0] = float("nan")
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0] = False
    yp = yp.clone().requires_grad_(True)
    val = loss(yp, y_nan, mask=mask)
    val.backward()
    assert torch.isfinite(val) and torch.isfinite(yp.grad).all()
    # fit_density with a missing target: KDE on finite rows, neutral weight for NaN
    loss2 = DensityWeightedLoss(kernel_width=0.5)
    loss2.fit_density(y_nan)
    assert torch.isfinite(loss2.density_weights).all() and float(loss2.density_weights[0]) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# B-LOSS-008 — pseudo-label placeholders never poison the loss
# ═══════════════════════════════════════════════════════════════════════════════


def _pl_data(placeholder: float):
    g = torch.Generator().manual_seed(0)
    y_pred = torch.randn(8, 1, generator=g, requires_grad=True)
    target = torch.randn(8, 1, generator=g)
    label_mask = torch.ones(8, 1, dtype=torch.bool)
    label_mask[4:] = False
    target_ph = target.clone()
    target_ph[4:] = placeholder
    pseudo = torch.randn(8, 1, generator=g)
    return y_pred, target, target_ph, label_mask, pseudo


def test_B_LOSS_008_consistency_loss_nan_placeholder() -> None:
    y_pred, target, target_nan, label_mask, pseudo = _pl_data(float("nan"))
    pseudo_nan = pseudo.clone()
    pseudo_nan[:4] = float("nan")  # pseudo labels only used on unlabelled entries
    loss = PseudoLabelConsistencyLoss()
    ref = loss(y_pred, target, pseudo_target=pseudo, label_mask=label_mask)
    got = loss(y_pred, target_nan, pseudo_target=pseudo_nan, label_mask=label_mask)
    assert torch.allclose(got, ref)
    got.backward()
    assert torch.isfinite(y_pred.grad).all()


def test_B_LOSS_008_pseudo_label_nll_nan_placeholder() -> None:
    y_pred, target, target_nan, label_mask, pseudo = _pl_data(float("nan"))
    head = torch.cat([y_pred, torch.zeros_like(y_pred)], dim=-1)
    loss = PseudoLabelNLL()
    ref = loss(head, target, pseudo_target=pseudo, label_mask=label_mask)
    got = loss(head, target_nan, pseudo_target=pseudo, label_mask=label_mask)
    assert torch.allclose(got, ref)
    got.backward()
    assert torch.isfinite(y_pred.grad).all()
