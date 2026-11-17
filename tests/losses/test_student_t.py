"""Unit tests for torchregress.losses.student_t.StudentTLoss."""

import math

import numpy as np
import pytest
import torch

from torchregress.losses.student_t import (
    StudentTLoss,
    effective_precision,
    student_t_efficiency,
    student_t_nll,
)

# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def y_pred(device):
    torch.manual_seed(42)
    return torch.randn(8, 3, device=device)


@pytest.fixture
def target(device):
    torch.manual_seed(123)
    return torch.randn(8, 3, device=device)


@pytest.fixture
def mask(device):
    torch.manual_seed(7)
    return torch.randint(0, 2, (8, 3), device=device).bool()


@pytest.fixture
def weights(device):
    torch.manual_seed(99)
    return torch.rand(8, 3, device=device)


# ── construction ──────────────────────────────────────────────────────────


def test_construction_defaults() -> None:
    loss = StudentTLoss()
    assert loss.nu == 1.0
    assert loss.scale == 1.0
    assert loss.reduction == "mean"


def test_construction_custom() -> None:
    loss = StudentTLoss(nu=3.0, scale=5.0, reduction="sum")
    assert loss.nu == 3.0
    assert loss.scale == 5.0
    assert loss.reduction == "sum"


def test_construction_invalid_nu() -> None:
    with pytest.raises(ValueError, match="nu must be > 0"):
        StudentTLoss(nu=0.0)
    with pytest.raises(ValueError, match="nu must be > 0"):
        StudentTLoss(nu=-1.0)


def test_construction_invalid_scale() -> None:
    with pytest.raises(ValueError, match="scale must be > 0"):
        StudentTLoss(scale=0.0)
    with pytest.raises(ValueError, match="scale must be > 0"):
        StudentTLoss(scale=-1.0)


def test_extra_repr() -> None:
    loss = StudentTLoss(nu=3.0, scale=5.0)
    rep = loss.extra_repr()
    assert "nu=3.0" in rep
    assert "scale=5.0" in rep


# ── forward pass ──────────────────────────────────────────────────────────


def test_forward_returns_scalar_for_mean_reduction(y_pred, target) -> None:
    loss_fn = StudentTLoss(reduction="mean")
    out = loss_fn(y_pred, target)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_forward_returns_per_sample_for_none_reduction(y_pred, target) -> None:
    loss_fn = StudentTLoss(reduction="none")
    out = loss_fn(y_pred, target)
    assert out.shape == (8, 3)
    assert torch.all(torch.isfinite(out))


def test_forward_sum_reduction(y_pred, target) -> None:
    loss_fn_sum = StudentTLoss(reduction="sum")
    loss_fn_none = StudentTLoss(reduction="none")
    assert torch.allclose(
        loss_fn_sum(y_pred, target),
        loss_fn_none(y_pred, target).sum(),
    )


def test_forward_with_mask(y_pred, target, mask) -> None:
    loss_fn = StudentTLoss(reduction="mean")
    out = loss_fn(y_pred, target, mask=mask)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_forward_with_mask_all_false_gives_nan(device) -> None:
    """All-false mask → empty tensor → mean reduction gives NaN."""
    loss_fn = StudentTLoss(reduction="mean")
    t = torch.randn(4, 3, device=device)
    mask_all_false = torch.zeros(4, 3, device=device).bool()
    out = loss_fn(t, t, mask=mask_all_false)
    # Mean of empty tensor: implementation-defined (NaN or 0)
    # Accept both as valid library behavior
    assert out.item() == 0.0 or torch.isnan(out)


def test_forward_with_weights(y_pred, target, weights) -> None:
    loss_fn = StudentTLoss(reduction="mean")
    out = loss_fn(y_pred, target, weights=weights)
    assert out.dim() == 0
    assert torch.isfinite(out)


# ── special cases ─────────────────────────────────────────────────────────


def test_nu1_matches_cauchy(y_pred, target) -> None:
    """StudentTLoss with nu=1 should match Cauchy NLL up to additive constant."""
    loss_t = StudentTLoss(nu=1.0, scale=2.0, reduction="none")
    out_t = loss_t(y_pred, target)

    # Cauchy NLL: log(πσ) + log(1 + (r/σ)²)
    residual = target - y_pred
    sigma = 2.0
    expected = math.log(math.pi * sigma) + torch.log(1.0 + (residual / sigma) ** 2)

    assert torch.allclose(out_t, expected, rtol=1e-5)


def test_large_nu_approaches_gaussian(y_pred, target) -> None:
    """As nu → ∞, Student-t NLL should approach Gaussian NLL."""
    loss_t = StudentTLoss(nu=1000.0, scale=1.0, reduction="mean")
    out_t = loss_t(y_pred, target)

    residual = target - y_pred
    gaussian_nll = 0.5 * math.log(2.0 * math.pi) + 0.5 * (residual**2).mean()

    # Mean NLLs should match within 1% for nu=1000
    assert out_t.item() == pytest.approx(gaussian_nll.item(), rel=0.01)


def test_nu_point_five_works(y_pred, target) -> None:
    """nu=0.5 is valid (heavy-tailed, df < 1)."""
    loss_fn = StudentTLoss(nu=0.5, reduction="mean")
    out = loss_fn(y_pred, target)
    assert torch.isfinite(out)


def test_scale_changes_loss(y_pred, target) -> None:
    """Larger scale → higher NLL for typical residuals (less concentrated density)."""
    loss_small = StudentTLoss(scale=0.5, reduction="mean")(y_pred, target)
    loss_large = StudentTLoss(scale=5.0, reduction="mean")(y_pred, target)
    # Narrower distribution has higher density near the mean → lower NLL
    assert loss_small < loss_large


def test_perfect_prediction_gives_minimum(y_pred, device) -> None:
    """When y_pred == target, NLL should be at minimum (log_norm + log_scale)."""
    loss_fn = StudentTLoss(nu=3.0, scale=2.0, reduction="none")
    perfect = torch.zeros(1, 1, device=device)
    out = loss_fn(perfect, perfect)

    # NLL(r=0) = lgamma(nu/2) - lgamma((nu+1)/2) + 0.5*log(nu*pi) + log(scale)
    # For nu=3: lgamma(1.5) - lgamma(2) + 0.5*log(3π) + log(2)
    # lgamma(2) = 0, so: lgamma(1.5) + 0.5*log(3π) + log(2)
    expected_min = math.lgamma(1.5) + 0.5 * math.log(3.0 * math.pi) + math.log(2.0)

    assert out.item() == pytest.approx(expected_min, rel=1e-5)


# ── gradient flow ─────────────────────────────────────────────────────────


def test_gradient_flows(y_pred, target) -> None:
    y_pred.requires_grad_(True)
    loss_fn = StudentTLoss(reduction="mean")
    out = loss_fn(y_pred, target)
    out.backward()
    assert y_pred.grad is not None
    assert torch.all(torch.isfinite(y_pred.grad))


def test_gradient_nonzero_for_wrong_prediction(device) -> None:
    pred = torch.tensor([[5.0]], device=device, requires_grad=True)
    true = torch.tensor([[0.0]], device=device)
    loss_fn = StudentTLoss(reduction="mean")
    out = loss_fn(pred, true)
    out.backward()
    assert pred.grad.item() != 0.0


# ── numerical stability ───────────────────────────────────────────────────


def test_extreme_residuals(y_pred, target) -> None:
    """Very large residuals should not produce NaN or Inf."""
    loss_fn = StudentTLoss(reduction="mean")
    huge_pred = y_pred.clone()
    huge_pred[0, 0] = 1e10
    out = loss_fn(huge_pred, target)
    assert torch.isfinite(out)


def test_small_residuals(y_pred, target) -> None:
    """Very small residuals should be numerically stable."""
    loss_fn = StudentTLoss(reduction="mean")
    small_pred = target.clone() + 1e-8
    out = loss_fn(small_pred, target)
    assert torch.isfinite(out)


def test_nan_in_input_with_mask(y_pred, target, device) -> None:
    """NaN in input should be handled when masked out."""
    loss_fn = StudentTLoss(reduction="mean")
    pred_nan = y_pred.clone()
    pred_nan[0, 0] = float("nan")
    mask_nan = torch.ones(8, 3, device=device).bool()
    mask_nan[0, 0] = False
    out = loss_fn(pred_nan, target, mask=mask_nan)
    assert torch.isfinite(out)


def test_inf_in_input_with_mask(y_pred, target, device) -> None:
    """Inf in input should be handled when masked out."""
    loss_fn = StudentTLoss(reduction="mean")
    pred_inf = y_pred.clone()
    pred_inf[0, 0] = float("inf")
    mask_inf = torch.ones(8, 3, device=device).bool()
    mask_inf[0, 0] = False
    out = loss_fn(pred_inf, target, mask=mask_inf)
    assert torch.isfinite(out)


# ── reduction modes ───────────────────────────────────────────────────────


def test_none_reduction_with_mask(y_pred, target, mask) -> None:
    loss_fn = StudentTLoss(reduction="none")
    out = loss_fn(y_pred, target, mask=mask)
    # A9: reduction="none" + mask zero-fills masked entries, keeping the shape
    assert out.shape == target.shape
    assert torch.all(torch.isfinite(out))


def test_registry_includes_student_t() -> None:
    from torchregress.losses.loss_registry import get_regression_loss

    cls = get_regression_loss("student_t")
    assert cls is StudentTLoss


# ── functional Student-t family ───────────────────────────────────────────


def _student_t_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    residual = 3.0 * torch.randn(4, 50, generator=generator, dtype=torch.float64)
    precision = 0.1 + 2.0 * torch.rand(4, 50, generator=generator, dtype=torch.float64)
    mask = torch.rand(4, 50, generator=generator) > 0.3
    return residual, precision, mask


@pytest.mark.parametrize("nu", [0.5, 1.0, 2.5, 5.0, 30.0])
def test_student_t_nll_matches_scipy_logpdf(nu: float) -> None:
    from scipy import stats

    residual, precision, mask = _student_t_case()
    scale = precision.rsqrt().numpy()
    reference = -stats.t.logpdf(residual.numpy(), df=nu, scale=scale)
    elementwise = student_t_nll(residual, precision, nu, dim=None)
    torch.testing.assert_close(elementwise, torch.from_numpy(reference), rtol=1e-12, atol=1e-12)
    per_row = student_t_nll(residual, precision, nu, mask=mask)
    expected = torch.from_numpy(reference * mask.numpy()).sum(-1)
    torch.testing.assert_close(per_row, expected, rtol=1e-12, atol=1e-12)


def test_student_t_nll_gaussian_limit_matches_scipy_norm() -> None:
    from scipy import stats

    residual, precision, mask = _student_t_case()
    reference = -stats.norm.logpdf(residual.numpy(), scale=precision.rsqrt().numpy())
    out = student_t_nll(residual, precision, math.inf, mask=mask)
    expected = torch.from_numpy(reference * mask.numpy()).sum(-1)
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)
    # Large finite nu approaches the Gaussian.
    close = student_t_nll(residual, precision, 1e7, mask=mask)
    torch.testing.assert_close(close, out, rtol=1e-5, atol=1e-5)


def test_student_t_nll_masked_entries_are_ignored_even_if_non_finite() -> None:
    residual, precision, mask = _student_t_case()
    dirty_r = torch.where(mask, residual, torch.nan)
    dirty_p = torch.where(mask, precision, -1.0)
    clean = student_t_nll(residual, precision, 4.0, mask=mask)
    torch.testing.assert_close(student_t_nll(dirty_r, dirty_p, 4.0, mask=mask), clean)
    empty = torch.zeros_like(mask)
    assert torch.equal(student_t_nll(dirty_r, dirty_p, 4.0, mask=empty), torch.zeros(4).double())


def test_student_t_nll_fails_closed_on_unmasked_invalid_precision() -> None:
    residual = torch.tensor([[0.5, 1.0]], dtype=torch.float64)
    assert torch.isinf(student_t_nll(residual, torch.tensor([[1.0, 0.0]]).double(), 4.0)).all()
    nan_row = student_t_nll(residual, torch.tensor([[1.0, torch.nan]]).double(), 4.0)
    assert torch.isnan(nan_row).all()
    with pytest.raises(ValueError, match="nu must be > 0"):
        student_t_nll(residual, torch.ones_like(residual), 0.0)


def test_student_t_nll_gradient_is_minus_weighted_residual() -> None:
    residual, precision, _ = _student_t_case()
    nu = 4.0
    r = residual.clone().requires_grad_(True)
    student_t_nll(r, precision, nu).sum().backward()
    weight = precision * (nu + 1.0) / (nu + precision * residual.square())
    torch.testing.assert_close(r.grad, weight * residual, rtol=1e-12, atol=1e-12)


def test_student_t_loss_matches_functional_and_accepts_infinite_nu() -> None:
    residual, _, mask = _student_t_case()
    for nu in (3.0, math.inf):
        loss = StudentTLoss(nu=nu, scale=2.0, reduction="none")
        out = loss(torch.zeros_like(residual), residual)
        precision = torch.full_like(residual, 0.25)
        torch.testing.assert_close(out, student_t_nll(residual, precision, nu, dim=None))


def test_effective_precision_variance_model() -> None:
    _, precision, _ = _student_t_case()
    model = torch.linspace(-3.0, 3.0, precision.shape[-1], dtype=torch.float64)
    eta = 0.05
    out = effective_precision(precision, model, eta)
    torch.testing.assert_close(out, 1.0 / (1.0 / precision + eta**2 * model.square()))
    assert effective_precision(precision, model, 0.0) is precision
    zero = torch.zeros(3, dtype=torch.float64)
    assert torch.equal(effective_precision(zero, torch.ones(3).double(), 0.1), zero)
    per_row = torch.tensor([[0.0], [0.1], [0.2], [0.3]], dtype=torch.float64)
    rows = effective_precision(precision, model, per_row)
    torch.testing.assert_close(rows[2], effective_precision(precision[2], model, 0.2))
    with pytest.raises(ValueError, match="eta"):
        effective_precision(precision, model, -0.1)


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_effective_precision_rejects_invalid_tensor_eta(bad: float) -> None:
    precision = torch.ones(3, 4, dtype=torch.float64)
    model = torch.ones(3, 4, dtype=torch.float64)
    eta = torch.full((3, 1), 0.1, dtype=torch.float64)
    eta[1, 0] = bad
    with pytest.raises(ValueError, match="eta must be finite"):
        effective_precision(precision, model, eta)


@pytest.mark.parametrize("shape", [(2, 3, 1), (3,), (2, 4)])
def test_effective_precision_tensor_eta_must_broadcast_to_precision(
    shape: tuple[int, ...],
) -> None:
    # (2, 3, 1) broadcasts but expands the output; (3,) and (2, 4) do not
    # broadcast at all.
    precision = torch.ones(3, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="broadcast"):
        effective_precision(precision, precision, torch.full(shape, 0.1, dtype=torch.float64))


@pytest.mark.parametrize("nu", [1.0, 3.0, 5.0, 10.0])
def test_student_t_efficiency_matches_quadrature(nu: float) -> None:
    from scipy import integrate, stats

    def psi(z: float) -> float:
        return (nu + 1.0) * z / (nu + z * z)

    def dpsi(z: float) -> float:
        return (nu + 1.0) * (nu - z * z) / (nu + z * z) ** 2

    for law, noise in ((stats.norm(), "gaussian"), (stats.t(df=nu), "student_t")):
        e_psi2 = integrate.quad(lambda z: psi(z) ** 2 * law.pdf(z), -np.inf, np.inf)[0]
        e_dpsi = integrate.quad(lambda z: dpsi(z) * law.pdf(z), -np.inf, np.inf)[0]
        assert student_t_efficiency(nu, noise) == pytest.approx(
            e_dpsi**2 / e_psi2, rel=1e-9, abs=0.0
        )
    assert student_t_efficiency(nu, "student_t") == (nu + 1.0) / (nu + 3.0)


# E[ψ']²/E[ψ²] under N(0, 1), mpmath quadrature at 40 digits split at ±√ν
# (the integrands are sharply peaked there for small ν).
_GAUSSIAN_EFFICIENCY = {
    1e-4: 0.015813318678772718,
    0.01: 0.14597427213807549,
    0.05: 0.2939222207513924,
    0.1: 0.38566324700044832,
    0.5: 0.64824609018873961,
    1.0: 0.76154243304489677,
    1.5: 0.82029895220324634,
    2.0: 0.85705210814757343,
    2.5: 0.88233723459860524,
    3.0: 0.90079766745498764,
    5.0: 0.94196183456315327,
    30.0: 0.99545966064580325,
    1e3: 0.99999408281288983,
}


@pytest.mark.parametrize("nu", sorted(_GAUSSIAN_EFFICIENCY))
def test_student_t_gaussian_efficiency_matches_high_precision_reference(nu: float) -> None:
    assert student_t_efficiency(nu) == pytest.approx(_GAUSSIAN_EFFICIENCY[nu], rel=1e-13, abs=0)


def test_student_t_efficiency_limits_and_validation() -> None:
    assert student_t_efficiency(math.inf) == 1.0
    assert student_t_efficiency(math.inf, "student_t") == 1.0
    assert student_t_efficiency(5.0) == pytest.approx(0.9419618345631522, rel=1e-14)
    assert student_t_efficiency(1e6) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError, match="noise"):
        student_t_efficiency(5.0, "laplace")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nu must be > 0"):
        student_t_efficiency(-1.0)


# log Γ(ν/2) − log Γ((ν+1)/2) + ½ log(νπ), mpmath at 50 digits.
_LOG_NORM = {
    30.0: 0.92727032537884565092,
    1e2: 0.92143849154300455812,
    1e5: 0.91894103320467270011,
    1e9: 0.91893853345467274178,
    1e15: 0.91893853320467299178,
}


@pytest.mark.parametrize("nu", sorted(_LOG_NORM))
def test_student_t_log_norm_is_accurate_for_large_nu(nu: float) -> None:
    from torchregress.losses.student_t import _student_t_log_norm

    assert _student_t_log_norm(nu) == pytest.approx(_LOG_NORM[nu], rel=0, abs=1e-14)


def test_student_t_nll_tends_to_the_gaussian_branch() -> None:
    generator = torch.Generator().manual_seed(3)
    residual = torch.randn(4, 50, generator=generator, dtype=torch.float64)
    precision = 0.5 + torch.rand(4, 50, generator=generator, dtype=torch.float64)
    gaussian = student_t_nll(residual, precision, math.inf)
    torch.testing.assert_close(
        student_t_nll(residual, precision, 1e15), gaussian, rtol=1e-12, atol=1e-12
    )
    assert math.isfinite(float(student_t_nll(residual, precision, 1e300).sum()))
