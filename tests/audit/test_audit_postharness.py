"""Regression tests for findings from the torchregress-harness run of the 0.3.0 release prep.

* H-001: ``skew_t_nll`` had an exactly zero gradient in the skewness ``alpha`` at
  ``alpha = 0`` (the Student-t CDF went through ``I_x(df/2, 1/2)`` with
  ``x = df / (df + w^2)``, flat at ``w = 0`` and clamped at ``1 - 1e-15``).
* H-002: ``ShiftFactoredPredictiveTransport`` EM collapsed onto empty support
  bins (prior TV ~1 with no shift) because the source prior was an eps-clamped
  label histogram that the predictive probabilities do not marginalise to.
* H-003: EM vs BBSE on binned Gaussian predictions -- not a bug: Saerens EM is
  consistent only for calibrated posteriors (documented); pinned here.

References come from :mod:`scipy` (core) or closed forms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats as st
import torch

from torchregress.losses import SkewTLoss, skew_t_nll
from torchregress.losses.families import _student_t_log_cdf_vec
from torchregress.prediction import PredictiveBatch
from torchregress.test_time import (
    LabelShiftEMConfig,
    ShiftFactoredPredictiveTransport,
    ShiftFactoredTransportConfig,
    estimate_target_prior_em,
    gaussian_bin_edges_from_targets,
    gaussian_bin_probabilities,
)

F64 = torch.float64


def _inv_softplus(v: float) -> float:
    return math.log(math.expm1(v))


# ---------------------------------------------------------------------------
# H-001: skew-t gradient in alpha at alpha = 0
# ---------------------------------------------------------------------------


def _skew_t_raw(xi: float, omega: float, alpha: float, nu: float, n: int) -> torch.Tensor:
    row = [xi, _inv_softplus(omega), alpha, _inv_softplus(nu)]
    return torch.tensor([row] * n, dtype=F64)


@pytest.mark.parametrize("alpha", [0.0, 1e-3, -1e-3, 0.1, -0.1, 2.0])
def test_H_001_skew_t_alpha_gradient_matches_finite_differences(alpha: float) -> None:
    y = torch.tensor([0.7, -1.2, 2.5, 0.05, -4.0], dtype=F64)
    raw = _skew_t_raw(0.1, 0.9, alpha, 2.1, len(y)).requires_grad_(True)
    skew_t_nll(raw, y, eps=0.0, reduction="sum").backward()
    assert raw.grad is not None
    h = 1e-6
    fd = torch.empty(len(y), dtype=F64)
    for i in range(len(y)):
        hi = _skew_t_raw(0.1, 0.9, alpha + h, 2.1, 1)
        lo = _skew_t_raw(0.1, 0.9, alpha - h, 2.1, 1)
        fd[i] = (
            skew_t_nll(hi, y[i : i + 1], eps=0.0, reduction="sum")
            - skew_t_nll(lo, y[i : i + 1], eps=0.0, reduction="sum")
        ) / (2 * h)
    torch.testing.assert_close(raw.grad[:, 2], fd, rtol=1e-6, atol=1e-8)
    if alpha == 0.0:
        # d NLL / d alpha at alpha = 0 is -2 z t_{nu+1}(0) sqrt((nu+1)/(nu+z^2)): nonzero.
        assert float(raw.grad[:, 2].abs().min()) > 1e-3


def test_H_001_harness_design_alpha_gradient_at_zero() -> None:
    """The harness xfail (``test_skew_t_alpha_gradient_at_zero``): about -0.29 by FD."""
    y = torch.tensor([[0.7], [-1.2], [2.5]], dtype=F64)
    raw = torch.tensor([[0.0, 0.5, 0.0, 2.0]] * 3, dtype=F64, requires_grad=True)
    loss = SkewTLoss()(raw, y)
    loss.backward()
    assert raw.grad is not None
    assert float(raw.grad[:, 2].sum()) == pytest.approx(-0.2896, abs=2e-3)


@pytest.mark.parametrize("alpha", [-2.0, -0.1, 0.0, 0.1, 2.0])
@pytest.mark.parametrize("nu", [1.5, 5.0, 30.0])
def test_H_001_skew_t_gradcheck(alpha: float, nu: float) -> None:
    # Includes y == xi (z = 0, so w = 0 for every alpha) and far tails.
    y = torch.tensor([-25.0, -3.0, -0.4, 0.2, 0.2001, 1.1, 6.0, 40.0], dtype=F64)
    raw = _skew_t_raw(0.2, 1.3, alpha, nu, len(y)).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda p: skew_t_nll(p, y, eps=0.0, reduction="none"), (raw,), eps=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("df", [1.0, 2.5, 6.0, 31.0])
def test_H_001_student_t_log_cdf_matches_scipy(df: float) -> None:
    x = torch.tensor(
        [-1e4, -300.0, -40.0, -8.0, -2.0, -0.5, -1e-3, 0.0, 1e-3, 0.5, 2.0, 8.0, 40.0, 1e3],
        dtype=F64,
    )
    got = _student_t_log_cdf_vec(x, torch.full_like(x, df))
    ref = torch.from_numpy(st.t.logcdf(x.numpy(), df))
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-12)
    # d/dx log T(x) = t_df(x) / T(x), including x = 0 (was 0 there).
    xg = x.clone().requires_grad_(True)
    _student_t_log_cdf_vec(xg, torch.full_like(x, df)).sum().backward()
    assert xg.grad is not None
    ref_grad = torch.from_numpy(np.exp(st.t.logpdf(x.numpy(), df) - st.t.logcdf(x.numpy(), df)))
    torch.testing.assert_close(xg.grad, ref_grad, rtol=1e-7, atol=1e-12)


def test_H_001_skew_t_head_learns_skew_from_alpha_zero() -> None:
    """A head started at alpha = 0 moves alpha towards the sign of the skew."""
    rng = np.random.default_rng(0)
    # Right-skewed sample (Azzalini skew-t with alpha = 4, nu = 6).
    delta = 4.0 / math.sqrt(1.0 + 16.0)
    u0, u1 = rng.normal(size=4000), rng.normal(size=4000)
    sn = delta * np.abs(u0) + math.sqrt(1 - delta**2) * u1
    y = torch.from_numpy(sn / np.sqrt(rng.chisquare(6.0, size=4000) / 6.0))
    param = torch.tensor([0.0, 0.5, 0.0, 2.0], dtype=F64, requires_grad=True)
    opt = torch.optim.Adam([param], lr=0.05)
    for _ in range(150):
        opt.zero_grad()
        skew_t_nll(param.expand(len(y), 4), y).backward()
        opt.step()
    assert float(param[2]) > 1.0


def test_H_001_float32_finite() -> None:
    y = torch.tensor([-50.0, -1.0, 0.0, 0.3, 1e3], dtype=torch.float32)
    raw = torch.tensor([[0.0, 0.0, 0.0, -3.0]] * len(y), dtype=torch.float32, requires_grad=True)
    out = skew_t_nll(raw, y, reduction="none")
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert float(raw.grad[:, 2].abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# H-002: ShiftFactoredPredictiveTransport EM on the support grid
# ---------------------------------------------------------------------------

_NOISE = 0.5


def _gaussian_regression(
    rng: np.random.Generator, n: int, shift: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """``x ~ N(0, 1)``, ``y = x + N(0, 0.5^2)``; the target draws ``y`` from
    ``N(shift, 1.25)`` and keeps ``p(x | y)`` (pure label shift). The predictive
    ``N(x, 0.5)`` is the exact (calibrated) source posterior ``p_s(y | x)``."""
    var_y = 1.0 + _NOISE**2
    y = rng.normal(shift, math.sqrt(var_y), size=n)
    x = rng.normal(y / var_y, math.sqrt(_NOISE**2 / var_y))
    return x, y


def _batch(x: np.ndarray) -> PredictiveBatch:
    m = torch.from_numpy(x)
    return PredictiveBatch(point=m, mean=m, std=torch.full_like(m, _NOISE))


def _transport(
    n_support: int,
    shift: float,
    seed: int = 0,
    features: bool = False,
    top_fraction: float = 0.5,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = _gaussian_regression(rng, 3000)
    xt, yt = _gaussian_regression(rng, 2000, shift)
    transport = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(n_support=n_support, top_fraction=top_fraction)
    ).fit_source(_batch(xs), torch.from_numpy(ys), source_inputs=torch.from_numpy(xs)[:, None])
    out = transport.adapt_unlabeled_target(
        target_predictions=_batch(xt),
        target_inputs=torch.from_numpy(xt)[:, None] if features else None,
    )
    assert transport.state_ is not None
    support = transport.state_.source_support.numpy()
    return dict(out.extra or {}), support, ys, yt


@pytest.mark.parametrize("n_support", [16, 64, 256])
def test_H_002_transport_no_shift_keeps_source_prior(n_support: int) -> None:
    meta, support, ys, _ = _transport(n_support, shift=0.0)
    src = np.asarray(meta["source_prior"])
    est = np.asarray(meta["target_prior_raw"])
    assert meta["estimate_converged"] is True
    # Used to be ~0.95-1.0: EM piled the prior onto empty margin bins.
    assert meta["prior_source_target_tv"] < 0.15
    # No mass beyond the source label range (the margins hold no labels).
    outside = (support < ys.min() - 0.25) | (support > ys.max() + 0.25)
    assert est[outside].sum() < 0.01
    assert abs(float(est @ support) - float(src @ support)) < 0.15


@pytest.mark.parametrize("n_support", [16, 64, 256])
@pytest.mark.parametrize("shift", [0.8, -1.0])
def test_H_002_transport_recovers_label_shift(n_support: int, shift: float) -> None:
    meta, support, _, yt = _transport(n_support, shift=shift, features=True)
    src = np.asarray(meta["source_prior"])
    est = np.asarray(meta["target_prior_raw"])
    assert meta["estimate_converged"] is True
    assert meta["transport_applied"] is True
    assert meta["prior_source_target_tv"] > 0.2
    src_mean, est_mean, true_mean = float(src @ support), float(est @ support), float(yt.mean())
    # Estimated target-prior mean moves with the true target label mean (source ~0).
    # The confident-row selection ranks on the unrenormalised in-grid peak, so grid-edge
    # truncation does not bias the estimate (it used to overshoot by up to 0.3 |shift|).
    assert np.sign(est_mean - src_mean) == np.sign(shift)
    assert abs(est_mean - true_mean) < 0.1 * abs(shift)
    # The stabilised (shrunk, ratio-clipped) prior moves the same way.
    adapted = np.asarray(meta["target_prior"])
    assert np.sign(float(adapted @ support) - src_mean) == np.sign(shift)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("shift", [0.8, -0.8])
@pytest.mark.parametrize("n_support", [16, 64, 256])
def test_H_002_default_selection_is_not_biased_by_edge_truncation(
    n_support: int, shift: float, seed: int
) -> None:
    """Default ``top_fraction=0.5``: prior error < 0.1 |shift| on coarse and fine grids.

    Renormalising a predictive that is cut by the grid edge inflates its peak bin
    probability, so ranking confidence on ``probabilities.max()`` favoured edge rows
    on fine grids (prior mean error up to ~0.3 |shift| at 256 bins).
    """
    meta, support, _, yt = _transport(n_support, shift=shift, seed=seed, features=True)
    est = np.asarray(meta["target_prior_raw"])
    assert meta["estimate_converged"] is True
    assert abs(float(est @ support) - float(yt.mean())) < 0.1 * abs(shift)


def test_H_002_select_high_confidence_ranks_on_custom_scores() -> None:
    from torchregress.test_time import select_high_confidence

    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.8, 0.2], [0.55, 0.45]])
    default = select_high_confidence(probs, top_fraction=0.5)
    np.testing.assert_array_equal(default, [True, False, True, False])
    custom = select_high_confidence(probs, top_fraction=0.5, scores=np.array([0.0, 3.0, 1.0, 2.0]))
    np.testing.assert_array_equal(custom, [False, True, False, True])
    # The min_count fallback ranks on the same scores.
    fallback = select_high_confidence(
        probs, top_fraction=0.25, min_count=3, scores=np.array([0.0, 3.0, 1.0, 2.0])
    )
    np.testing.assert_array_equal(fallback, [False, True, True, True])
    with pytest.raises(ValueError, match="scores"):
        select_high_confidence(probs, top_fraction=0.5, scores=np.ones(3))


def test_H_002_in_grid_mass_matches_closed_form_for_every_family() -> None:
    from torchregress.test_time import transport as tt

    support = torch.linspace(-1.0, 1.0, 64, dtype=F64)
    edges = tt._support_edges(support)
    mean = torch.tensor([0.0, 0.9, 1.0, 3.0], dtype=F64)
    std = torch.tensor([0.1, 0.5, 0.5, 0.5], dtype=F64)
    expected = st.norm.cdf(float(edges[-1]), mean.numpy(), std.numpy()) - st.norm.cdf(
        float(edges[0]), mean.numpy(), std.numpy()
    )
    batch = PredictiveBatch(point=mean, mean=mean, std=std)
    _, meta = tt._batch_to_support_density(batch, support, 1e-8)
    np.testing.assert_allclose(meta["in_grid_mass"].numpy(), expected, atol=1e-9)
    assert meta["in_grid_mass"][0] > 0.999 and meta["in_grid_mass"][3] < 0.01

    # A density on a wider grid gives the same mass (trapezoid error only).
    wide = torch.linspace(-6.0, 6.0, 4001, dtype=F64)
    dens = torch.as_tensor(
        st.norm.pdf(wide.numpy()[None], mean.numpy()[:, None], std.numpy()[:, None])
    )
    dbatch = PredictiveBatch(point=mean, support=wide, density=dens)
    _, dmeta = tt._batch_to_support_density(dbatch, support, 1e-8)
    np.testing.assert_allclose(dmeta["in_grid_mass"].numpy(), expected, atol=5e-3)


@pytest.mark.parametrize("n_support", [16, 256])
@pytest.mark.parametrize("shift", [0.8, -1.0])
def test_H_002_transport_em_unbiased_on_all_rows(n_support: int, shift: float) -> None:
    meta, support, _, yt = _transport(n_support, shift=shift, top_fraction=1.0)
    est = np.asarray(meta["target_prior_raw"])
    assert meta["estimate_converged"] is True and meta["transport_applied"] is True
    assert abs(float(est @ support) - float(yt.mean())) < 0.05
    # Coarse-binned target prior matches the empirical target labels.
    coarse = np.linspace(support[0], support[-1], 9)
    est_c = np.histogram(support, bins=coarse, weights=est)[0]
    true_c = np.histogram(yt, bins=coarse)[0] / len(yt)
    assert 0.5 * np.abs(est_c - true_c).sum() < 0.08


def test_H_002_em_ignores_bins_without_source_mass() -> None:
    """Bins with zero source prior get zero target prior (label shift needs
    ``supp p_t(y) <= supp p_s(y)``); they used to absorb the whole prior."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(6), size=400)
    src = np.array([0.0, 0.3, 0.3, 0.2, 0.2, 0.0])
    est = estimate_target_prior_em(probs, source_prior=src)
    assert est.target_prior[[0, 5]].max() < 1e-12
    assert est.target_prior.sum() == pytest.approx(1.0)
    inner = probs[:, 1:5] / probs[:, 1:5].sum(axis=1, keepdims=True)
    ref = estimate_target_prior_em(inner, source_prior=src[1:5] / src[1:5].sum())
    np.testing.assert_allclose(est.target_prior[1:5], ref.target_prior, atol=1e-12)


def test_H_002_em_loglik_tol_stops_on_likelihood_plateau() -> None:
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(32) * 0.5, size=1000)
    src = np.full(32, 1.0 / 32)
    base = estimate_target_prior_em(probs, source_prior=src)
    # Default (loglik_tol=None) keeps the parameter-change criterion only.
    assert base.iterations == 100 and base.converged is False
    est = estimate_target_prior_em(
        probs, source_prior=src, config=LabelShiftEMConfig(loglik_tol=1e-4)
    )
    assert est.converged is True and est.iterations < 100
    with pytest.raises(ValueError, match="loglik_tol"):
        LabelShiftEMConfig(loglik_tol=-1.0)


# ---------------------------------------------------------------------------
# H-003: EM vs BBSE -- EM is consistent for calibrated posteriors
# ---------------------------------------------------------------------------


def _binwise_shift_problem(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin-level label shift: the target reweights source rows by a weight that is
    constant within each bin, so ``p(x | bin)`` is shared."""
    rng = np.random.default_rng(seed)
    var_y = 1.0 + 0.3**2
    y_src = rng.normal(0.0, math.sqrt(var_y), size=40000)
    edges = gaussian_bin_edges_from_targets(y_src, 5)
    bins = np.clip(np.searchsorted(edges, y_src, side="right") - 1, 0, 4)
    wb = np.array([0.3, 0.6, 1.0, 1.5, 1.6])
    idx = rng.choice(y_src.size, size=20000, p=wb[bins] / wb[bins].sum())
    y_tgt = y_src[idx]
    p_tgt = y_tgt + 0.3 * rng.normal(size=y_tgt.size)
    src_prior = np.bincount(bins, minlength=5) / bins.size
    tgt_bins = np.clip(np.searchsorted(edges, y_tgt, side="right") - 1, 0, 4)
    true_prior = np.bincount(tgt_bins, minlength=5) / tgt_bins.size
    return edges, src_prior, true_prior, p_tgt, np.array([var_y])


def test_H_003_em_consistent_with_calibrated_posteriors() -> None:
    edges, src_prior, true_prior, p_tgt, (var_y,) = _binwise_shift_problem()
    # Calibrated source posterior p_s(y | p) = N(p / (1 + s2/var_y) ..), s2 = 0.09.
    shrink = var_y / (var_y + 0.09)
    post = gaussian_bin_probabilities(
        shrink * p_tgt, np.full_like(p_tgt, math.sqrt(0.09 * shrink)), edges
    )
    est = estimate_target_prior_em(
        post, source_prior=src_prior, config=LabelShiftEMConfig(max_iter=5000)
    )
    assert est.converged
    assert 0.5 * np.abs(est.target_prior - true_prior).sum() < 0.02
    assert 0.5 * np.abs(src_prior - true_prior).sum() > 0.2


def test_H_003_em_biased_with_uncalibrated_bin_probabilities() -> None:
    """Documented limitation (the harness design): with ``pred = y + noise``,
    ``N(pred, sigma)`` bin probabilities are the likelihood ``p(pred | y)``, not
    the source posterior; the same EM then stops at a biased fixed point."""
    edges, src_prior, true_prior, p_tgt, _ = _binwise_shift_problem()
    lik = gaussian_bin_probabilities(p_tgt, np.full_like(p_tgt, 0.3), edges)
    est = estimate_target_prior_em(
        lik, source_prior=src_prior, config=LabelShiftEMConfig(max_iter=5000)
    )
    assert 0.5 * np.abs(est.target_prior - true_prior).sum() > 0.05
