"""Closed-form identity tests for flexible-shape distributional losses (F5)."""

from __future__ import annotations

import math

import pytest
import torch

from torchregress.losses.families import (
    AsymmetricLaplaceNLLLoss,
    BetaRegressionNLLLoss,
    GEVNLLLoss,
    JohnsonSUNLLLoss,
    SinhArcsinhNLLLoss,
    SkewNormalNLLLoss,
    SkewTLoss,
    SQRLoss,
    asymmetric_laplace_nll,
    beta_regression_nll,
    gev_nll,
    johnson_su_nll,
    sinh_arcsinh_nll,
    skew_normal_nll,
    skew_t_nll,
    sqr_loss,
)
from torchregress.losses.loss_registry import list_regression_losses

F64 = torch.float64


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return 0.5 * math.log(2 * math.pi) + torch.log(sigma) + (y - mu) ** 2 / (2 * sigma**2)


def _raw_for_softplus(target: float) -> float:
    return math.log(math.expm1(target))


def _ald_pred(mu: float, sigma: float, kappa: float) -> torch.Tensor:
    return torch.tensor([[mu, _raw_for_softplus(sigma), _raw_for_softplus(kappa)]], dtype=F64)


# ---------------------------------------------------------------------------
# skew-normal
# ---------------------------------------------------------------------------


def test_skew_normal_alpha_zero_reduces_to_gaussian():
    y = torch.tensor([-1.2, 0.0, 0.7, 2.3], dtype=F64)
    mu, omega_raw = 0.3, 0.9
    y_pred = torch.tensor([[mu, omega_raw, 0.0]], dtype=F64).expand(4, 3)
    got = skew_normal_nll(y_pred, y, eps=0.0, reduction="none")
    omega = torch.nn.functional.softplus(torch.tensor(omega_raw, dtype=F64))
    expected = gaussian_nll(y, torch.tensor(mu, dtype=F64), omega)
    torch.testing.assert_close(got, expected)


def test_skew_normal_matches_hand_computed_value():
    # Azzalini density f(y) = 2/omega * phi(z) * Phi(alpha*z); NLL = -log f.
    y = 0.7
    xi, omega, alpha = 0.1, 1.2, 1.5
    z = (y - xi) / omega
    phi_val = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    Phi = 0.5 * math.erfc(-alpha * z / math.sqrt(2))
    expected = -math.log(2.0 / omega * phi_val * Phi)
    y_pred = torch.tensor([[xi, _raw_for_softplus(omega), alpha]], dtype=F64)
    got = skew_normal_nll(y_pred, torch.tensor([y], dtype=F64), eps=0.0)
    assert abs(got.item() - expected) < 1e-10


def test_skew_normal_class_functional_and_registry():
    torch.manual_seed(0)
    y_pred, target = torch.randn(6, 3, dtype=F64), torch.randn(6, dtype=F64)
    cls_val = SkewNormalNLLLoss(reduction="sum")(y_pred, target)
    fn_val = skew_normal_nll(y_pred, target, reduction="sum")
    torch.testing.assert_close(cls_val, fn_val)
    assert "skew_normal_nll" in list_regression_losses()


# ---------------------------------------------------------------------------
# skew-t
# ---------------------------------------------------------------------------


def test_skew_t_alpha_zero_reduces_to_student_t():
    y = torch.tensor([-2.0, 0.4, 1.7], dtype=F64)
    xi, omega, nu = -0.2, 1.1, 5.0
    y_pred = torch.tensor(
        [[xi, _raw_for_softplus(omega), 0.0, _raw_for_softplus(nu)]], dtype=F64
    ).expand(3, 4)
    got = skew_t_nll(y_pred, y, eps=0.0, reduction="none")
    z = (y - xi) / omega
    log_pdf = (
        math.lgamma((nu + 1) / 2)
        - math.lgamma(nu / 2)
        - 0.5 * math.log(nu * math.pi)
        - math.log(omega)
        - ((nu + 1) / 2) * torch.log1p(z * z / nu)
    )
    torch.testing.assert_close(got, -log_pdf)


def test_skew_t_density_integrates_to_one():
    grid = torch.linspace(-60.0, 60.0, 200001, dtype=F64)
    y_pred = torch.tensor([[0.2, _raw_for_softplus(1.3), 2.0, _raw_for_softplus(5.0)]], dtype=F64)
    nll = skew_t_nll(y_pred, grid, eps=0.0, reduction="none")
    integral = torch.trapz(torch.exp(-nll), grid)
    assert abs(integral.item() - 1.0) < 1e-3


def test_skew_t_registered():
    assert "skew_t_nll" in list_regression_losses()
    assert isinstance(SkewTLoss(), torch.nn.Module)


# ---------------------------------------------------------------------------
# beta regression
# ---------------------------------------------------------------------------


def test_beta_regression_closed_form_value():
    # mu=0.4 (sigmoid of logit), phi=6 -> alpha=2.4, beta=3.6; evaluated at y=mu.
    mu, phi, y = 0.4, 6.0, 0.4
    alpha, beta = mu * phi, (1 - mu) * phi
    logit = math.log(mu / (1 - mu))
    expected = -(
        math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + (alpha - 1) * math.log(y)
        + (beta - 1) * math.log1p(-y)
    )
    y_pred = torch.tensor([[logit, _raw_for_softplus(phi)]], dtype=F64)
    got = beta_regression_nll(y_pred, torch.tensor([y], dtype=F64), eps=0.0)
    assert abs(got.item() - expected) < 1e-12


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_beta_regression_rejects_targets_outside_open_unit_interval(bad):
    y_pred = torch.randn(2, 2)
    target = torch.tensor([bad, 0.5])
    with pytest.raises(ValueError, match="strictly inside"):
        beta_regression_nll(y_pred, target)


def test_beta_regression_registered():
    assert "beta_regression_nll" in list_regression_losses()
    BetaRegressionNLLLoss()(torch.randn(2, 2), torch.tensor([0.5, 0.5]))


# ---------------------------------------------------------------------------
# Johnson SU
# ---------------------------------------------------------------------------


def test_johnson_su_hand_computed_value():
    y = 0.7
    xi, lam, gamma, delta = 0.0, 1.0, 0.5, 1.3
    u = (y - xi) / lam
    z = gamma + delta * math.asinh(u)
    expected = (
        0.5 * math.log(2 * math.pi)
        - math.log(delta)
        + math.log(lam)
        + 0.5 * math.log1p(u * u)
        + 0.5 * z * z
    )
    y_pred = torch.tensor(
        [[xi, _raw_for_softplus(lam), gamma, _raw_for_softplus(delta)]], dtype=F64
    )
    got = johnson_su_nll(y_pred, torch.tensor([y], dtype=F64), eps=0.0)
    assert abs(got.item() - expected) < 1e-12


def test_johnson_su_density_integrates_to_one():
    grid = torch.linspace(-40.0, 40.0, 200001, dtype=F64)
    y_pred = torch.tensor([[0.1, _raw_for_softplus(0.8), -0.7, _raw_for_softplus(1.1)]], dtype=F64)
    nll = johnson_su_nll(y_pred, grid, eps=0.0, reduction="none")
    integral = torch.trapz(torch.exp(-nll), grid)
    assert abs(integral.item() - 1.0) < 1e-3


def test_johnson_su_registered():
    assert "johnson_su_nll" in list_regression_losses()


# ---------------------------------------------------------------------------
# sinh-arcsinh
# ---------------------------------------------------------------------------


def test_sinh_arcsinh_epsilon_zero_delta_one_is_gaussian():
    y = torch.tensor([-1.5, 0.2, 2.2], dtype=F64)
    mu, sigma = 0.3, 0.9
    y_pred = torch.tensor(
        [[mu, _raw_for_softplus(sigma), 0.0, _raw_for_softplus(1.0)]], dtype=F64
    ).expand(3, 4)
    got = sinh_arcsinh_nll(y_pred, y, eps=0.0, reduction="none")
    expected = gaussian_nll(
        y,
        torch.tensor(mu, dtype=F64),
        torch.nn.functional.softplus(torch.tensor(_raw_for_softplus(sigma), dtype=F64)),
    )
    torch.testing.assert_close(got, expected)


def test_sinh_arcsinh_density_integrates_to_one():
    grid = torch.linspace(-40.0, 40.0, 200001, dtype=F64)
    y_pred = torch.tensor([[0.0, _raw_for_softplus(1.0), 1.5, _raw_for_softplus(0.8)]], dtype=F64)
    nll = sinh_arcsinh_nll(y_pred, grid, eps=0.0, reduction="none")
    integral = torch.trapz(torch.exp(-nll), grid)
    assert abs(integral.item() - 1.0) < 1e-3


def test_sinh_arcsinh_registered():
    assert "sinh_arcsinh_nll" in list_regression_losses()
    assert isinstance(SinhArcsinhNLLLoss(), torch.nn.Module)


# ---------------------------------------------------------------------------
# GEV
# ---------------------------------------------------------------------------


def test_gev_gumbel_limit_matches_general_formula_at_small_xi():
    # xi=1e-8 triggers the analytic Gumbel branch inside the threshold; the
    # general Coles formula evaluated in float64 must agree closely.
    import numpy as np

    xi, mu, sigma = 1e-8, 0.3, 1.7
    ys = np.array([-1.4, 0.2, 1.9])
    z = (ys - mu) / sigma
    t = 1.0 + xi * z
    expected = np.log(sigma) + (1.0 + 1.0 / xi) * np.log(t) + t ** (-1.0 / xi)
    y_pred = torch.tensor([[mu, _raw_for_softplus(sigma), xi]], dtype=F64)
    got = gev_nll(y_pred, torch.from_numpy(ys).to(F64), eps=0.0, reduction="none")
    torch.testing.assert_close(got, torch.from_numpy(expected).to(F64), rtol=1e-6, atol=1e-6)


def test_gev_shape_xi_hand_computed_value():
    xi, mu, sigma, y = 0.5, 0.0, 1.0, 0.6
    z = (y - mu) / sigma
    t = 1 + xi * z
    expected = math.log(sigma) + (1 + 1 / xi) * math.log(t) + t ** (-1 / xi)
    y_pred = torch.tensor([[mu, _raw_for_softplus(sigma), xi]], dtype=F64)
    got = gev_nll(y_pred, torch.tensor([y], dtype=F64), eps=0.0, reduction="none").item()
    assert abs(got - expected) < 1e-12


def test_gev_out_of_support_is_infinite():
    y_pred = torch.tensor([[0.0, _raw_for_softplus(1.0), 0.5]], dtype=F64)
    # support requires 1 + xi*z > 0, i.e. y > mu - sigma/xi = -2 here.
    out = gev_nll(y_pred, torch.tensor([-3.0], dtype=F64), reduction="none")
    inside = gev_nll(y_pred, torch.tensor([-1.0], dtype=F64), reduction="none")
    assert torch.isinf(out).all() and torch.isfinite(inside).all()


def test_gev_registered():
    assert "gev_nll" in list_regression_losses()
    assert isinstance(GEVNLLLoss(), torch.nn.Module)


# ---------------------------------------------------------------------------
# asymmetric Laplace <-> pinball correspondence (F5 acceptance)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.5, 1.0, 2.3])
def test_asymmetric_laplace_reduces_to_scaled_pinball_at_tau(kappa):
    """ALD NLL equals its normalization plus the scaled pinball (check) loss at
    tau = 1/(1+kappa^2): the kappa-tau correspondence."""
    mu, sigma = 0.2, 1.1
    tau = 1.0 / (1.0 + kappa**2)
    scale = math.sqrt(2.0) * (1.0 + kappa**2) / (sigma * kappa)
    log_norm = math.log(math.sqrt(2.0) * kappa / (sigma * (1.0 + kappa**2)))

    us = torch.linspace(-2.0, 2.0, 41, dtype=F64)
    pred = _ald_pred(mu, sigma, kappa)
    got = asymmetric_laplace_nll(pred.expand(us.numel(), 3), mu + us, eps=0.0, reduction="none")

    def pinball(u: torch.Tensor, level: float) -> torch.Tensor:
        return torch.maximum(level * u, (level - 1.0) * u)

    expected = -log_norm + scale * pinball(us, tau)
    torch.testing.assert_close(got, expected)

    # Numerical slope check: one-sided dNLL/d|u| above/below the mode has
    # ratio (1 - tau)/tau = 1/kappa^2 — the same correspondence verified
    # purely numerically.
    h = 1e-6
    mode = asymmetric_laplace_nll(pred, torch.tensor([mu], dtype=F64), eps=0.0)
    slope_up = asymmetric_laplace_nll(pred, torch.tensor([mu + h], dtype=F64), eps=0.0) - mode
    slope_dn = asymmetric_laplace_nll(pred, torch.tensor([mu - h], dtype=F64), eps=0.0) - mode
    ratio = (slope_up / slope_dn).item()
    assert abs(ratio - 1 / kappa**2) < 1e-3


def test_asymmetric_laplace_kappa_one_is_symmetric_laplace():
    mu, sigma, y = 0.3, 1.4, -0.8
    got = asymmetric_laplace_nll(_ald_pred(mu, sigma, 1.0), torch.tensor([y], dtype=F64), eps=0.0)
    expected = math.log(math.sqrt(2) * sigma) + math.sqrt(2) * abs(y - mu) / sigma
    assert abs(got.item() - expected) < 1e-12


def test_asymmetric_laplace_registered():
    assert "asymmetric_laplace_nll" in list_regression_losses()
    assert isinstance(AsymmetricLaplaceNLLLoss(), torch.nn.Module)


# ---------------------------------------------------------------------------
# SQR loss
# ---------------------------------------------------------------------------


def test_sqr_constant_predictions_equal_mean_pinball():
    levels = 32
    mu, y = 0.7, -1.3
    taus = torch.arange(1, levels + 1, dtype=F64) / (levels + 1.0)
    u = y - mu
    expected = torch.maximum(taus * u, (taus - 1.0) * u).mean().item()
    y_pred = torch.full((1, levels), mu, dtype=F64)
    got = sqr_loss(y_pred, torch.tensor([y], dtype=F64)).item()
    assert abs(got - expected) < 1e-12


def test_sqr_cummax_enforces_sorted_levels():
    levels = 8
    ascending = torch.linspace(-2, 2, levels, dtype=F64).unsqueeze(0)
    descending = ascending.flip(dims=[1])
    y = torch.tensor([0.5], dtype=F64)
    # Running max keeps an ascending profile unchanged: the loss equals the
    # raw pinball on the unsorted input.
    taus = torch.arange(1, levels + 1, dtype=F64) / (levels + 1.0)
    u = y - ascending
    expected_asc = torch.maximum(taus * u, (taus - 1.0) * u).mean().item()
    assert abs(sqr_loss(ascending, y, n_levels=levels).item() - expected_asc) < 1e-12
    # ...and maps a descending profile onto its running maximum.
    taus = torch.arange(1, levels + 1, dtype=F64) / (levels + 1.0)
    u = y - torch.cummax(descending, dim=1).values
    expected = torch.maximum(taus * u, (taus - 1.0) * u).mean().item()
    assert abs(sqr_loss(descending, y, n_levels=levels).item() - expected) < 1e-12


def test_sqr_mask_excludes_entries_from_mean():
    y_pred = torch.randn(4, 32, dtype=torch.float32)
    target = torch.randn(4, dtype=torch.float32)
    mask = torch.tensor([True, False, True, True])
    full = sqr_loss(y_pred[[0, 2, 3]], target[[0, 2, 3]])
    masked = sqr_loss(y_pred, target, mask=mask)
    torch.testing.assert_close(masked, full)


def test_sqr_level_count_validated_and_registered():
    with pytest.raises(ValueError, match="levels"):
        sqr_loss(torch.randn(2, 8), torch.randn(2), n_levels=32)
    assert "sqr_loss" in list_regression_losses()
    SQRLoss(n_levels=16)(torch.randn(2, 16), torch.randn(2))


@pytest.mark.parametrize(
    "fn,k",
    [
        (skew_normal_nll, 3),
        (skew_t_nll, 4),
        (beta_regression_nll, 2),
        (johnson_su_nll, 4),
        (sinh_arcsinh_nll, 4),
        (asymmetric_laplace_nll, 3),
        (gev_nll, 3),
    ],
)
def test_none_reduction_preserves_shape(fn, k):
    torch.manual_seed(1)
    target = torch.rand(5) if fn is beta_regression_nll else torch.randn(5)
    out = fn(torch.randn(5, k), target, reduction="none")
    assert out.shape == (5,)


# ---------------------------------------------------------------------------
# unconstrained_inputs flag — avoids double softplus (NEW-MED-01)
# ---------------------------------------------------------------------------


def test_skew_normal_unconstrained_inputs_flag():
    y_raw = torch.tensor([[0.3, _raw_for_softplus(1.2), 0.5]], dtype=F64)
    y_pos = torch.tensor([[0.3, 1.2, 0.5]], dtype=F64)
    target = torch.tensor([0.7], dtype=F64)
    loss_true = SkewNormalNLLLoss(unconstrained_inputs=True)(y_raw, target)
    loss_pos = SkewNormalNLLLoss(unconstrained_inputs=False)(y_pos, target)
    torch.testing.assert_close(loss_pos, loss_true, atol=1e-6, rtol=1e-6)
    assert SkewNormalNLLLoss(unconstrained_inputs=False).unconstrained_inputs is False


def test_skew_t_unconstrained_inputs_flag():
    y_raw = torch.tensor([[0.1, _raw_for_softplus(0.9), 0.4, _raw_for_softplus(4.0)]], dtype=F64)
    y_pos = torch.tensor([[0.1, 0.9, 0.4, 4.0]], dtype=F64)
    target = torch.tensor([-0.5], dtype=F64)
    loss_true = SkewTLoss(unconstrained_inputs=True)(y_raw, target)
    loss_pos = SkewTLoss(unconstrained_inputs=False)(y_pos, target)
    torch.testing.assert_close(loss_pos, loss_true, atol=1e-6, rtol=1e-6)
    assert SkewTLoss(unconstrained_inputs=False).unconstrained_inputs is False


def test_beta_unconstrained_inputs_flag():
    y_raw = torch.tensor([[0.2, _raw_for_softplus(2.0)]], dtype=F64)
    y_pos = torch.tensor([[0.2, 2.0]], dtype=F64)
    target = torch.tensor([0.4], dtype=F64)
    loss_true = BetaRegressionNLLLoss(unconstrained_inputs=True)(y_raw, target)
    loss_pos = BetaRegressionNLLLoss(unconstrained_inputs=False)(y_pos, target)
    torch.testing.assert_close(loss_pos, loss_true, atol=1e-6, rtol=1e-6)
    assert BetaRegressionNLLLoss(unconstrained_inputs=False).unconstrained_inputs is False


def test_johnson_su_unconstrained_inputs_flag():
    y_raw = torch.tensor([[0.1, _raw_for_softplus(0.8), -0.3, _raw_for_softplus(1.5)]], dtype=F64)
    y_pos = torch.tensor([[0.1, 0.8, -0.3, 1.5]], dtype=F64)
    target = torch.tensor([0.2], dtype=F64)
    loss_true = JohnsonSUNLLLoss(unconstrained_inputs=True)(y_raw, target)
    loss_pos = JohnsonSUNLLLoss(unconstrained_inputs=False)(y_pos, target)
    torch.testing.assert_close(loss_pos, loss_true, atol=1e-6, rtol=1e-6)
    assert JohnsonSUNLLLoss(unconstrained_inputs=False).unconstrained_inputs is False


def test_sinh_arcsinh_unconstrained_inputs_flag():
    y_raw = torch.tensor([[0.0, _raw_for_softplus(1.1), 0.2, _raw_for_softplus(0.9)]], dtype=F64)
    y_pos = torch.tensor([[0.0, 1.1, 0.2, 0.9]], dtype=F64)
    target = torch.tensor([0.5], dtype=F64)
    loss_true = SinhArcsinhNLLLoss(unconstrained_inputs=True)(y_raw, target)
    loss_pos = SinhArcsinhNLLLoss(unconstrained_inputs=False)(y_pos, target)
    torch.testing.assert_close(loss_pos, loss_true, atol=1e-6, rtol=1e-6)
    assert SinhArcsinhNLLLoss(unconstrained_inputs=False).unconstrained_inputs is False


def test_gev_unconstrained_inputs_flag():
    y_raw = torch.tensor([[0.0, _raw_for_softplus(1.0), 0.1]], dtype=F64)
    y_pos = torch.tensor([[0.0, 1.0, 0.1]], dtype=F64)
    target = torch.tensor([0.3], dtype=F64)
    loss_true = GEVNLLLoss(unconstrained_inputs=True)(y_raw, target)
    loss_pos = GEVNLLLoss(unconstrained_inputs=False)(y_pos, target)
    torch.testing.assert_close(loss_pos, loss_true, atol=1e-6, rtol=1e-6)
    assert GEVNLLLoss(unconstrained_inputs=False).unconstrained_inputs is False


def test_asymmetric_laplace_unconstrained_inputs_flag():
    y_raw = torch.tensor([[0.2, _raw_for_softplus(1.3), _raw_for_softplus(0.7)]], dtype=F64)
    y_pos = torch.tensor([[0.2, 1.3, 0.7]], dtype=F64)
    target = torch.tensor([0.0], dtype=F64)
    loss_true = AsymmetricLaplaceNLLLoss(unconstrained_inputs=True)(y_raw, target)
    loss_pos = AsymmetricLaplaceNLLLoss(unconstrained_inputs=False)(y_pos, target)
    torch.testing.assert_close(loss_pos, loss_true, atol=1e-6, rtol=1e-6)
    assert AsymmetricLaplaceNLLLoss(unconstrained_inputs=False).unconstrained_inputs is False
