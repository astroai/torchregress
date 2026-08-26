import math

import pytest
import torch

from torchregress.losses.beta_nll import BetaNLLLoss, beta_nll_loss
from torchregress.losses.gaussian import GaussianNLLLoss
from torchregress.losses.loss_registry import create_loss_from_config


def test_beta_zero_matches_gaussian_nll() -> None:
    torch.manual_seed(0)
    mean = torch.randn(5, 3)
    log_var = torch.randn(5, 3) * 0.5
    target = torch.randn(5, 3)
    kwargs = dict(min_variance=1e-6, eps=1e-8, reduction="mean")
    b = BetaNLLLoss(beta=0.0, **kwargs)
    y_pred = (mean, log_var)
    # ponytail: since TR-COR-07 GaussianNLL reduces element-wise (B·D mean);
    # BetaNLL still sums over D pre-reduce, so with beta=0 (identity
    # weighting) it equals the per-element NLL summed over features.
    g_none = GaussianNLLLoss(reduction="none")
    per_row = g_none(y_pred, target).sum(dim=-1)
    torch.testing.assert_close(b(y_pred, target), per_row.mean())


def test_create_loss_from_config_beta_nll() -> None:
    loss = create_loss_from_config({"type": "beta_nll", "beta": 0.25})
    assert isinstance(loss, BetaNLLLoss)
    assert loss.beta == 0.25


def test_beta_nll_loss_functional_matches_module() -> None:
    mean = torch.tensor([[0.0], [1.0]])
    log_var = torch.tensor([[0.0], [0.0]])
    target = torch.tensor([[0.5], [2.0]])
    beta = 0.3
    m = BetaNLLLoss(beta=beta, reduction="mean")
    f = beta_nll_loss((mean, log_var), target, beta, reduction="mean")
    torch.testing.assert_close(f, m((mean, log_var), target))


def test_finiteness_and_no_nan_extreme_log_var() -> None:
    mean = torch.zeros(4, 2)
    log_var = torch.tensor([[10.0, 10.0], [-30.0, 0.0], [0.0, 0.0], [5.0, -5.0]])
    target = torch.randn(4, 2)
    loss_fn = BetaNLLLoss(beta=0.5, min_variance=1e-6, eps=1e-8)
    out = loss_fn((mean, log_var), target)
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


def test_negative_beta_raises() -> None:
    with pytest.raises(ValueError, match="beta must be non-negative"):
        BetaNLLLoss(beta=-0.1)


def test_mask_mean_reduction() -> None:
    mean = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    log_var = torch.zeros_like(mean)
    target = torch.tensor([[0.0, 10.0], [2.0, 3.0]])
    mask = torch.tensor([[True, True], [False, False]])
    loss_fn = BetaNLLLoss(beta=0.5, reduction="mean")
    full = loss_fn((mean, log_var), target)
    masked = loss_fn((mean, log_var), target, mask=mask)
    assert masked.shape == full.shape == torch.Size([])
    assert masked != full
    # ponytail: mask is per-sample (collapsed via .any(dim=-1) when loss is [B])


def test_reduction_none_shape() -> None:
    mean = torch.randn(3, 4)
    log_var = torch.randn(3, 4)
    target = torch.randn(3, 4)
    out = BetaNLLLoss(beta=0.4, reduction="none")((mean, log_var), target)
    # ponytail: loss is per-sample [B], not per-element [B, D]; consistent with MultivariateGaussianLoss
    assert out.shape == mean.shape[:1]


def test_weights_broadcast() -> None:
    mean = torch.zeros(2, 3)
    log_var = torch.zeros(2, 3)
    target = torch.ones(2, 3)
    w = torch.tensor([2.0, 0.5])
    loss_fn = BetaNLLLoss(beta=0.5, reduction="mean")
    out = loss_fn((mean, log_var), target, weights=w)
    assert torch.isfinite(out)


def test_gradient_finite_mean_and_log_var() -> None:
    mean = torch.zeros(2, 1, requires_grad=True)
    log_var = torch.zeros(2, 1, requires_grad=True)
    target = torch.ones(2, 1)
    loss = BetaNLLLoss(beta=0.5)((mean, log_var), target)
    loss.backward()
    assert mean.grad is not None
    assert log_var.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(log_var.grad).all()


def test_grads_match_manual_detached_prefactor() -> None:
    """Gradients match explicit NLL times ``(var+eps).detach().pow(-beta)``."""
    log_var = torch.tensor([[0.0]], requires_grad=True)
    mean = torch.tensor([[0.0]], requires_grad=True)
    target = torch.tensor([[0.0]])
    eps = 1e-8
    min_v = 1e-6
    beta = 0.5
    var = torch.exp(log_var).clamp(min=min_v)
    nll = 0.5 * (math.log(2 * math.pi) + torch.log(var + eps) + (target - mean) ** 2 / (var + eps))
    coef = (var + eps).detach().pow(-beta)
    loss_m = (coef * nll).mean()
    g_mean_m, g_lv_m = torch.autograd.grad(loss_m, (mean, log_var))

    mean2 = torch.tensor([[0.0]], requires_grad=True)
    log_var2 = torch.tensor([[0.0]], requires_grad=True)
    loss_c = BetaNLLLoss(beta=beta, eps=eps, min_variance=min_v)((mean2, log_var2), target)
    g_mean_c, g_lv_c = torch.autograd.grad(loss_c, (mean2, log_var2))
    torch.testing.assert_close(g_mean_c, g_mean_m)
    torch.testing.assert_close(g_lv_c, g_lv_m)


def test_weighting_differs_from_nll_when_beta_positive_and_heteroscedastic() -> None:
    """With varying variance, beta-NLL mean differs from plain NLL mean for beta>0."""
    mean = torch.zeros(2, 1)
    log_var = torch.tensor([[0.0], [3.0]])  # different variances
    target = torch.ones(2, 1)
    nll = GaussianNLLLoss(reduction="mean")((mean, log_var), target)
    bnll = BetaNLLLoss(beta=0.5, reduction="mean")((mean, log_var), target)
    assert not torch.allclose(nll, bnll)


def test_concatenated_input_matches_tuple() -> None:
    mean = torch.randn(3, 2)
    log_var = torch.randn(3, 2)
    target = torch.randn(3, 2)
    cat = torch.cat([mean, log_var], dim=-1)
    fn = BetaNLLLoss(beta=0.25, reduction="mean")
    torch.testing.assert_close(fn((mean, log_var), target), fn(cat, target))
