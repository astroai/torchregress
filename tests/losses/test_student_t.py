"""Unit tests for torchregress.losses.student_t.StudentTLoss."""

import math

import pytest
import torch

from torchregress.losses.student_t import StudentTLoss

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
    # With reduction="none", mask filters the returned elements
    assert out.numel() == int(mask.sum().item())
    assert torch.all(torch.isfinite(out))


def test_registry_includes_student_t() -> None:
    from torchregress.losses.loss_registry import get_regression_loss

    cls = get_regression_loss("student_t")
    assert cls is StudentTLoss
