"""Regression tests for the 0.3.0 losses audit, batch A (A-LOSS-001 .. A-LOSS-020).

Each test reproduces one audit finding and pins the fixed behaviour. Reference
values come from closed forms or :mod:`scipy` (a core dependency).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.special as ss
import scipy.stats as st
import torch

import torchregress.losses as L
from torchregress.losses import (
    AFTLoss,
    BetaNLLLoss,
    CensoredGaussianNLLLoss,
    CVaRLoss,
    EvidentialRegressionLoss,
    ExpectileCrossoverLoss,
    ExpectileLoss,
    FaithfulGaussianLoss,
    GaussianNLLLoss,
    GaussianWassersteinBoundLoss,
    MultiExpectileLoss,
    MultiQuantileLoss,
    MultivariateGaussianLoss,
    NegativeBinomialNLLLoss,
    WeightedMSELoss,
    ZeroInflatedPoissonNLLLoss,
    symmetric_spd_matrix_sqrt,
)
from torchregress.losses.families import (
    AsymmetricLaplaceNLLLoss,
    BetaRegressionNLLLoss,
    GEVNLLLoss,
    JohnsonSUNLLLoss,
    SinhArcsinhNLLLoss,
    SkewNormalNLLLoss,
    SQRLoss,
    _reg_inc_beta,
    gev_nll,
    skew_normal_nll,
    skew_t_nll,
)

F64 = torch.float64


def _inv_softplus(v: float) -> float:
    return math.log(math.expm1(v))


# ---------------------------------------------------------------------------
# A-LOSS-001 / 019: BaseLoss._reduce mask / weight broadcasting
# ---------------------------------------------------------------------------


def test_A_LOSS_001_reduce_mask2d_weights1d() -> None:
    torch.manual_seed(0)
    p, t = torch.randn(4, 3), torch.randn(4, 3)
    m = torch.ones(4, 3, dtype=torch.bool)
    m[0, 1] = False
    w = torch.tensor([1.0, 2.0, 0.5, 1.0])

    se = (p - t) ** 2
    big_w = w[:, None].expand_as(se) * m
    torch.testing.assert_close(
        WeightedMSELoss()(p, t, mask=m, weights=w), (se * big_w).sum() / big_w.sum()
    )
    torch.testing.assert_close(
        WeightedMSELoss(reduction="sum")(p, t, mask=m, weights=w), (se * big_w).sum()
    )
    none = WeightedMSELoss(reduction="none")(p, t, mask=m, weights=w)
    torch.testing.assert_close(none, se * big_w)

    mean, lv, y = torch.randn(4, 3), torch.zeros(4, 3), torch.randn(4, 3)
    per = GaussianNLLLoss(reduction="none")((mean, lv), y)
    big_w = w[:, None].expand_as(per) * m
    torch.testing.assert_close(
        GaussianNLLLoss()((mean, lv), y, mask=m, weights=w), (per * big_w).sum() / big_w.sum()
    )


@pytest.mark.parametrize("reduction", ["sum", "none", "mean"])
def test_A_LOSS_019_reduce_weight_collapse_averages(reduction: str) -> None:
    torch.manual_seed(0)
    m, t = torch.randn(4, 3), torch.randn(4, 3)
    cov = torch.eye(3, dtype=m.dtype)
    fn = MultivariateGaussianLoss(reduction=reduction)
    ones_w = torch.ones(4, 3)
    all_true = torch.ones(4, 3, dtype=torch.bool)
    # An all-True mask is a no-op and [B, D] all-ones weights are a no-op.
    torch.testing.assert_close(fn(m, t, cov, mask=all_true, weights=ones_w), fn(m, t, cov))

    # [B, D] weights on a per-sample loss are AVERAGED over D (not summed);
    # a partially masked row is dropped (row-collapse policy).
    w = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.5, 0.5, 2.0], [4.0, 0.0, 2.0]])
    mask = all_true.clone()
    mask[1, 0] = False
    per = MultivariateGaussianLoss(reduction="none")(m, t, cov)
    w_row = w.mean(dim=-1) * torch.tensor([1.0, 0.0, 1.0, 1.0])
    expected = {
        "none": per * w_row,
        "sum": (per * w_row).sum(),
        "mean": (per * w_row).sum() / w_row.sum(),
    }[reduction]
    torch.testing.assert_close(fn(m, t, cov, mask=mask, weights=w), expected)


# ---------------------------------------------------------------------------
# A-LOSS-002: BetaNLLLoss with 1-D targets
# ---------------------------------------------------------------------------


def _beta_nll_ref(mean: torch.Tensor, lv: torch.Tensor, y: torch.Tensor, beta: float):
    var = lv.exp()
    nll = 0.5 * (math.log(2 * math.pi) + lv + (y - mean) ** 2 / var)
    return nll * var.pow(beta)


def test_A_LOSS_002_beta_nll_1d_targets() -> None:
    torch.manual_seed(0)
    m = torch.randn(6, dtype=F64)
    lv = torch.randn(6, dtype=F64) * 0.3
    y = torch.randn(6, dtype=F64)
    ref = _beta_nll_ref(m, lv, y, 0.5)
    tol = {"rtol": 1e-6, "atol": 1e-6}

    none = BetaNLLLoss(beta=0.5, reduction="none")((m, lv), y)
    assert none.shape == (6,)
    torch.testing.assert_close(none, ref, **tol)
    torch.testing.assert_close(BetaNLLLoss(beta=0.5)((m, lv), y), ref.mean(), **tol)
    torch.testing.assert_close(L.beta_nll_loss((m, lv), y, beta=0.5), ref.mean(), **tol)

    mask = torch.ones(6, dtype=torch.bool)
    mask[0] = False
    torch.testing.assert_close(BetaNLLLoss(beta=0.5)((m, lv), y, mask=mask), ref[1:].mean(), **tol)

    w = torch.tensor([1.0, 0, 0, 0, 0, 0], dtype=F64)
    torch.testing.assert_close(BetaNLLLoss(beta=0.5)((m, lv), y, weights=w), ref[0], **tol)

    # 2-D [B, 1] inputs give the same per-sample values.
    col = BetaNLLLoss(beta=0.5, reduction="none")((m[:, None], lv[:, None]), y[:, None])
    torch.testing.assert_close(col, ref, **tol)


# ---------------------------------------------------------------------------
# A-LOSS-003 / 004: skew-t incomplete beta, skew-normal log-CDF
# ---------------------------------------------------------------------------


def test_A_LOSS_003_skew_t_incomplete_beta() -> None:
    a = torch.full((5,), 2.25, dtype=F64)
    b = torch.full((5,), 0.5, dtype=F64)
    x = torch.tensor([0.1, 0.5, 0.8, 0.95, 0.99], dtype=F64)
    ref = torch.from_numpy(ss.betainc(2.25, 0.5, x.numpy()))
    torch.testing.assert_close(_reg_inc_beta(a, b, x), ref, rtol=1e-8, atol=1e-10)

    xi, omega, alpha, nu = 0.2, 1.3, 2.0, 5.0
    y = torch.tensor([-1.0, -0.3, 0.0, 0.5, 1.0, 2.5], dtype=F64)
    y_pred = torch.tensor([[xi, _inv_softplus(omega), alpha, _inv_softplus(nu)]], dtype=F64).expand(
        len(y), 4
    )
    got = skew_t_nll(y_pred, y, eps=0.0, reduction="none")
    z = (y.numpy() - xi) / omega
    w = alpha * z * np.sqrt((nu + 1) / (nu + z * z))
    ref_nll = -(math.log(2) + st.t.logpdf(z, nu) - math.log(omega) + st.t.logcdf(w, nu + 1))
    torch.testing.assert_close(got, torch.from_numpy(ref_nll), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_A_LOSS_004_log_normal_cdf_saturates(dtype: torch.dtype) -> None:
    p = torch.tensor([[0.0, _inv_softplus(1.0), 30.0]], dtype=dtype, requires_grad=True)
    y = torch.tensor([-1.0], dtype=dtype)  # alpha * z = -30
    got = skew_normal_nll(p, y, eps=0.0, reduction="none")
    ref = -st.skewnorm.logpdf(-1.0, 30.0, loc=0.0, scale=1.0)  # ~455.06
    assert abs(got.item() - ref) < 1e-3 * abs(ref), (got.item(), ref)
    got.sum().backward()
    assert p.grad is not None
    # d NLL / d alpha = -z * phi(alpha z) / Phi(alpha z) ~ +30
    assert p.grad[0, 2].item() > 1.0, p.grad


# ---------------------------------------------------------------------------
# A-LOSS-005 / 006: GEV near the Gumbel limit and torch.where NaN gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("xi", [-2e-6, 2e-6, 5e-6, 1e-5, 0.3, -0.2])
@pytest.mark.parametrize("z", [3.0, -2.0])
def test_A_LOSS_005_gev_float32_near_gumbel(xi: float, z: float) -> None:
    p = torch.tensor([[0.0, _inv_softplus(1.0), xi]], dtype=torch.float32)
    got = gev_nll(p, torch.tensor([z]), eps=0.0, reduction="none").item()
    ref = -st.genextreme.logpdf(z, -xi)  # scipy shape c = -xi
    assert abs(got - ref) < 1e-3, (got, ref)


def test_A_LOSS_006_gev_where_nan_grad() -> None:
    p = torch.tensor([[0.0, _inv_softplus(1.0), -0.02]], requires_grad=True)
    y = torch.tensor([-95.0])  # 1 + xi*z = 2.9 > 0: inside support, exp(-z) overflows
    loss = gev_nll(p, y, eps=0.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all(), p.grad

    # Out-of-support rows are +inf but do not poison other rows' gradients.
    p2 = torch.tensor([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]], requires_grad=True)
    nll = gev_nll(p2, torch.tensor([-50.0, 1.0]), eps=0.0, reduction="none")
    assert math.isinf(nll[0].item()) and math.isfinite(nll[1].item())
    nll[1].backward()
    assert p2.grad is not None and torch.isfinite(p2.grad).all()


# ---------------------------------------------------------------------------
# A-LOSS-007: [B, 1] targets in BetaRegressionNLLLoss / SQRLoss
# ---------------------------------------------------------------------------


def test_A_LOSS_007_families_target_B1_broadcast() -> None:
    torch.manual_seed(0)
    p, t = torch.randn(4, 2), torch.rand(4, 1)
    for red in ("mean", "none"):
        torch.testing.assert_close(
            BetaRegressionNLLLoss(reduction=red)(p, t),
            BetaRegressionNLLLoss(reduction=red)(p, t[:, 0]),
        )
    assert BetaRegressionNLLLoss(reduction="none")(p, t).shape == (4,)

    q, y = torch.randn(4, 5), torch.randn(4, 1)
    for red in ("mean", "none"):
        torch.testing.assert_close(
            SQRLoss(n_levels=5, reduction=red)(q, y), SQRLoss(n_levels=5, reduction=red)(q, y[:, 0])
        )
    assert SQRLoss(n_levels=5, reduction="none")(q, y).shape == (4,)
    with pytest.raises(ValueError, match="sqr_loss expects targets"):
        SQRLoss(n_levels=5)(q, torch.randn(4, 5))

    sp, st_ = torch.randn(4, 3), torch.randn(4, 1)
    torch.testing.assert_close(SkewNormalNLLLoss()(sp, st_), SkewNormalNLLLoss()(sp, st_[:, 0]))


# ---------------------------------------------------------------------------
# A-LOSS-008: _inverse_softplus overflow for large physical scales
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,row",
    [
        (JohnsonSUNLLLoss, [0.0, 100.0, 0.0, 1.0]),
        (SinhArcsinhNLLLoss, [0.0, 100.0, 0.0, 1.0]),
        (GEVNLLLoss, [0.0, 100.0, 0.1]),
        (AsymmetricLaplaceNLLLoss, [0.0, 100.0, 1.0]),
    ],
)
def test_A_LOSS_008_inverse_softplus_overflow(cls: type, row: list[float]) -> None:
    p = torch.tensor([row], requires_grad=True)
    loss = cls(unconstrained_inputs=False)(p, torch.tensor([50.0]))
    assert torch.isfinite(loss)
    loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_A_LOSS_008_inverse_softplus_skew_normal_value_and_grad() -> None:
    s = 100.0
    p = torch.tensor([[0.0, s, 0.0]], requires_grad=True)
    loss = SkewNormalNLLLoss(unconstrained_inputs=False)(p, torch.tensor([2 * s]))
    ref = -st.norm.logpdf(2 * s, 0, s)
    assert abs(loss.item() - ref) < 1e-4, (loss.item(), ref)
    loss.backward()
    assert p.grad is not None
    torch.testing.assert_close(p.grad[0, 1], torch.tensor(1 / s - 4 / s), rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# A-LOSS-009: flow losses follow the BaseLoss reduction policy
# ---------------------------------------------------------------------------


def test_A_LOSS_009_nflow_reduction_policy() -> None:
    pytest.importorskip("zuko")
    torch.manual_seed(0)
    flow = L.create_flow_model(n_features=1, context_dim=3, flow_type="maf", n_transforms=1)
    c, y = torch.randn(5, 3), torch.randn(5, 1)
    loss = L.NormalizingFlowLoss(flow)
    per = L.NormalizingFlowLoss(flow, reduction="none")(c, y)

    torch.testing.assert_close(loss(c, y, weights=torch.full((5,), 2.0)), loss(c, y))
    w = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(loss(c, y, weights=w), per[0])

    m = torch.tensor([True, True, False, True, True])
    out = L.NormalizingFlowLoss(flow, reduction="none")(c, y, mask=m)
    assert out[2].item() == 0.0
    torch.testing.assert_close(loss(c, y, mask=m), per[m].mean())


# ---------------------------------------------------------------------------
# A-LOSS-010 / 011: expectile multi-level losses
# ---------------------------------------------------------------------------


def test_A_LOSS_010_multi_expectile_reduction() -> None:
    torch.manual_seed(0)
    yp, y = torch.randn(4, 3, 2), torch.randn(4, 2)
    two = torch.full((4,), 2.0)

    fn = MultiExpectileLoss([0.2, 0.5, 0.8])
    torch.testing.assert_close(fn(yp, y, weights=two), fn(yp, y))
    torch.testing.assert_close(
        MultiQuantileLoss([0.2, 0.5, 0.8])(yp, y, weights=two),
        MultiQuantileLoss([0.2, 0.5, 0.8])(yp, y),
    )
    mask = torch.ones(4, 2, dtype=torch.bool)
    mask[0] = False
    torch.testing.assert_close(fn(yp, y, mask=mask), fn(yp[1:], y[1:]))
    none = MultiExpectileLoss([0.2, 0.5, 0.8], reduction="none")(yp, y, mask=mask)
    assert none.shape == (4,) and none[0].item() == 0.0

    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    per = MultiExpectileLoss([0.2, 0.5, 0.8], reduction="none")(yp, y)
    torch.testing.assert_close(fn(yp, y, weights=w), per[0])

    xfn = ExpectileCrossoverLoss([0.2, 0.5, 0.8])
    torch.testing.assert_close(xfn(yp, y, weights=two), xfn(yp, y))
    torch.testing.assert_close(xfn(yp, y, mask=mask), xfn(yp[1:], y[1:]))


def test_A_LOSS_011_expectile_crossover_silent_sort() -> None:
    with pytest.raises(ValueError, match="expectiles must be ascending"):
        ExpectileCrossoverLoss([0.9, 0.1])
    with pytest.raises(ValueError, match="expectiles must be ascending"):
        ExpectileCrossoverLoss(torch.tensor([0.5, 0.5]))

    torch.manual_seed(0)
    yp, y = torch.randn(5, 2, 1), torch.randn(5, 1)
    got = ExpectileCrossoverLoss([0.1, 0.9], crossover_penalty=0.0, reduction="none")(yp, y)
    l0 = ExpectileLoss(0.1, reduction="none")(yp[:, 0], y)[:, 0]
    l1 = ExpectileLoss(0.9, reduction="none")(yp[:, 1], y)[:, 0]
    torch.testing.assert_close(got, (l0 + l1) / 2)


# ---------------------------------------------------------------------------
# A-LOSS-012: CVaR with 1-D inputs
# ---------------------------------------------------------------------------


def test_A_LOSS_012_cvar_1d_inputs() -> None:
    p = torch.zeros(4)
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(CVaRLoss(alpha=0.25, reduction="none")(p, t), t**2)
    torch.testing.assert_close(CVaRLoss(alpha=0.25)(p, t), torch.tensor(16.0))
    torch.testing.assert_close(
        CVaRLoss(alpha=0.5)(p, t), CVaRLoss(alpha=0.5)(p[:, None], t[:, None])
    )
    mask = torch.tensor([True, True, True, False])
    torch.testing.assert_close(
        CVaRLoss(alpha=0.25, reduction="none")(p, t, mask=mask), torch.tensor([1.0, 4.0, 9.0, 0.0])
    )


# ---------------------------------------------------------------------------
# A-LOSS-013 / 014: count losses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_A_LOSS_013_zip_saturation(dtype: torch.dtype) -> None:
    # y > 0 under confident (wrong) zero-inflation: -log(1 - pi) = softplus(logit)
    pi_logit = torch.tensor([25.0], dtype=dtype, requires_grad=True)
    loss = ZeroInflatedPoissonNLLLoss(reduction="none")(
        torch.tensor([math.log(3.0)], dtype=dtype),
        torch.tensor([3.0], dtype=dtype),
        pi_logits=pi_logit,
    )
    ref = 25.0 + math.log1p(math.exp(-25.0)) - st.poisson.logpmf(3, 3.0)
    assert abs(loss.item() - ref) < 1e-3, (loss.item(), ref)
    loss.sum().backward()
    assert pi_logit.grad is not None and pi_logit.grad.item() > 0.9

    # y = 0 with confident "no inflation" and a large rate
    loss0 = ZeroInflatedPoissonNLLLoss(reduction="none")(
        torch.tensor([5.0], dtype=dtype),
        torch.tensor([0.0], dtype=dtype),
        pi_logits=torch.tensor([-30.0], dtype=dtype),
    )
    lam = math.exp(5.0)
    ref0 = -np.logaddexp(-math.log1p(math.exp(30.0)), -math.log1p(math.exp(-30.0)) - lam)
    assert abs(loss0.item() - ref0) < 1e-3, (loss0.item(), ref0)

    # Moderate values match the ZIP pmf exactly.
    rate = torch.tensor([0.7, 2.0, 5.0], dtype=F64)
    logit = torch.tensor([-1.0, 0.0, 1.5], dtype=F64)
    y = torch.tensor([0.0, 2.0, 0.0], dtype=F64)
    got = ZeroInflatedPoissonNLLLoss(log_input=False, eps=0.0, reduction="none")(
        rate, y, pi_logits=logit
    )
    pi = 1 / (1 + np.exp(-logit.numpy()))
    pmf = (1 - pi) * st.poisson.pmf(y.numpy(), rate.numpy()) + pi * (y.numpy() == 0)
    np.testing.assert_allclose(got.numpy(), -np.log(pmf), rtol=1e-10)


def test_A_LOSS_014_nbinom_float32_theta() -> None:
    mu = torch.tensor([0.5, 2.0, 10.0, 100.0], dtype=F64)
    y = torch.tensor([0.0, 3.0, 7.0, 150.0], dtype=F64)
    th = 20.0
    for theta in (th, torch.tensor(th, dtype=torch.float32)):
        got = NegativeBinomialNLLLoss(reduction="none", eps=0.0)(mu, y, theta=theta)
        assert got.dtype == F64
        ref = -st.nbinom.logpmf(y.numpy(), th, th / (th + mu.numpy()))
        np.testing.assert_allclose(got.numpy(), ref, rtol=0, atol=1e-10)
    assert NegativeBinomialNLLLoss(reduction="none")(mu, y).dtype == F64
    assert NegativeBinomialNLLLoss(learn_theta=True, reduction="none")(mu, y).dtype == F64


# ---------------------------------------------------------------------------
# A-LOSS-015: censored losses in the far tail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_A_LOSS_015_censored_tail_saturation(dtype: torch.dtype) -> None:
    m = torch.zeros(2, dtype=dtype, requires_grad=True)
    lv = torch.zeros(2, dtype=dtype)
    y = torch.tensor([6.0, 8.0], dtype=dtype)
    nll = CensoredGaussianNLLLoss(reduction="none")(
        (m, lv), y, censoring=torch.ones(2, dtype=torch.long)
    )
    ref = torch.tensor(-st.norm.logsf([6.0, 8.0]), dtype=dtype)
    torch.testing.assert_close(nll.detach(), ref, rtol=1e-4, atol=1e-4)
    nll.sum().backward()
    assert m.grad is not None and (m.grad < -1.0).all(), m.grad

    # Left-censored upper limit far below the prediction.
    m8 = torch.tensor([8.0], dtype=F64)
    left = CensoredGaussianNLLLoss(reduction="none")(
        (m8, torch.zeros(1, dtype=F64)),
        torch.zeros(1, dtype=F64),
        censoring=-torch.ones(1, dtype=torch.long),
    )
    assert abs(left.item() + st.norm.logcdf(-8.0)) < 1e-6

    # AFT right-censored far tail.
    aft = AFTLoss(reduction="none")(
        (torch.zeros(1, dtype=F64), torch.zeros(1, dtype=F64)),
        torch.tensor([math.exp(7.0)], dtype=F64),
        censoring=torch.ones(1, dtype=torch.long),
    )
    assert abs(aft.item() + st.norm.logsf(7.0)) < 1e-6


def test_A_LOSS_015_interval_censoring_log_difference() -> None:
    m = torch.zeros(3, dtype=F64, requires_grad=True)
    lo = torch.tensor([-1.0, 7.0, -12.0], dtype=F64)
    up = torch.tensor([1.0, 8.0, -11.0], dtype=F64)
    nll = CensoredGaussianNLLLoss(reduction="none")(
        (m, torch.zeros(3, dtype=F64)), torch.zeros(3, dtype=F64), lower_bound=lo, upper_bound=up
    )
    ref = -np.log(st.norm.cdf(up.numpy()) - st.norm.cdf(lo.numpy()))
    ref[1] = -np.log(st.norm.sf(7.0) - st.norm.sf(8.0))
    np.testing.assert_allclose(nll.detach().numpy(), ref, rtol=1e-8)
    nll.sum().backward()
    assert m.grad is not None and torch.isfinite(m.grad).all()

    aft = AFTLoss(reduction="none")(
        (torch.zeros(1, dtype=F64), torch.zeros(1, dtype=F64)),
        torch.ones(1, dtype=F64),
        lower_bound=torch.tensor([math.exp(7.0)], dtype=F64),
        upper_bound=torch.tensor([math.exp(8.0)], dtype=F64),
    )
    assert abs(aft.item() - ref[1]) < 1e-8


# ---------------------------------------------------------------------------
# A-LOSS-016: evidential NIG NLL (Amini et al. 2020, Eq. 8)
# ---------------------------------------------------------------------------


def test_A_LOSS_016_evidential_omega() -> None:
    g = torch.tensor([[0.5], [0.0], [1.0]], dtype=F64)
    nu = torch.tensor([[2.0], [0.5], [10.0]], dtype=F64)
    a = torch.tensor([[3.0], [1.5], [5.0]], dtype=F64)
    b = torch.tensor([[1.5], [0.7], [2.0]], dtype=F64)
    y = torch.tensor([[1.0], [-1.0], [3.0]], dtype=F64)
    loss = EvidentialRegressionLoss(coeff_nig=0.0, reduction="none", unconstrained_inputs=False)
    got = loss((g, nu, a, b), y)
    scale = np.sqrt((b * (1 + nu) / (nu * a)).numpy())
    ref = -st.t.logpdf(y.numpy(), df=(2 * a).numpy(), loc=g.numpy(), scale=scale)
    np.testing.assert_allclose(got.detach().numpy(), ref, rtol=1e-8)

    # predict_interval's 95% interval carries 95% of the trained density.
    lo, hi = loss.predict_interval(torch.cat([g, nu, a, b], dim=-1), confidence=0.95)
    for i in range(3):
        df, sc = 2 * a[i, 0].item(), float(scale[i, 0])
        mass = st.t.cdf(hi[i, 0].item(), df, g[i, 0].item(), sc) - st.t.cdf(
            lo[i, 0].item(), df, g[i, 0].item(), sc
        )
        assert abs(mass - 0.95) < 1e-6, (i, mass)


# ---------------------------------------------------------------------------
# A-LOSS-017: matrix square root gradient at repeated eigenvalues
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["covariance", "cholesky"])
def test_A_LOSS_017_wasserstein_eigh_nan_grad(mode: str) -> None:
    pc = torch.eye(3, dtype=F64).repeat(2, 1, 1).requires_grad_(True)
    tc = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=F64)).to(dtype=F64).repeat(2, 1, 1)
    m = torch.zeros(2, 3, dtype=F64)
    GaussianWassersteinBoundLoss(covariance_parameterization=mode)(m, m, pc, tc).backward()
    assert pc.grad is not None and torch.isfinite(pc.grad).all(), pc.grad


def test_A_LOSS_017_matrix_sqrt_grad_values() -> None:
    a = torch.eye(2, dtype=F64).requires_grad_(True)
    (g,) = torch.autograd.grad(symmetric_spd_matrix_sqrt(a).diagonal().sum(), a)
    torch.testing.assert_close(g, 0.5 * torch.eye(2, dtype=F64), rtol=1e-5, atol=1e-6)

    # Distinct eigenvalues: matches finite differences (gradcheck).
    torch.manual_seed(0)
    x = torch.randn(2, 3, 3, dtype=F64)
    spd = (x @ x.transpose(-1, -2) + 0.5 * torch.eye(3, dtype=F64)).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda s: symmetric_spd_matrix_sqrt(0.5 * (s + s.transpose(-1, -2))), (spd,)
    )
    # Repeated eigenvalue in a rotated basis: still finite and correct
    # (divided difference 1 / (s_i + s_j)).
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=F64))
    deg = (
        q @ torch.diag(torch.tensor([4.0, 4.0, 1.0], dtype=F64)).to(dtype=F64) @ q.T
    ).requires_grad_(True)
    (g2,) = torch.autograd.grad(symmetric_spd_matrix_sqrt(deg).diagonal().sum(), deg)
    expected = q @ torch.diag(torch.tensor([0.25, 0.25, 0.5], dtype=F64)).to(dtype=F64) @ q.T
    torch.testing.assert_close(g2, expected, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# A-LOSS-018: __all__ exports
# ---------------------------------------------------------------------------


def test_A_LOSS_018_all_exports() -> None:
    for name in ("StudentTLoss", "ExpectileCrossover", "QuantileCrossover"):
        assert name in L.__all__, name
        assert hasattr(L, name), name
    assert len(L.__all__) == len(set(L.__all__))


# ---------------------------------------------------------------------------
# A-LOSS-020: FaithfulGaussianLoss mean-term flag follows the buffer
# ---------------------------------------------------------------------------


def test_A_LOSS_020_faithful_stale_mean_flag() -> None:
    torch.manual_seed(0)
    m = torch.randn(4, 2, requires_grad=True)
    lv, y = torch.randn(4, 2), torch.randn(4, 2)

    ref = FaithfulGaussianLoss(mean_weight=1.0)
    fn = FaithfulGaussianLoss(mean_weight=0.0)
    fn.load_state_dict(ref.state_dict())
    torch.testing.assert_close(fn((m, lv), y), ref((m, lv), y))

    ramp = FaithfulGaussianLoss(mean_weight=0.0)
    lv_req = lv.clone().requires_grad_(True)
    (g0,) = torch.autograd.grad(ramp((m, lv_req), y), m, allow_unused=True)
    assert g0 is None or g0.abs().sum() == 0
    ramp.mean_weight.fill_(1.0)  # warm-up finished
    (g1,) = torch.autograd.grad(ramp((m, lv), y), m)
    assert g1.abs().sum() > 0
