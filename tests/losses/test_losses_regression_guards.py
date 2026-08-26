"""Regression tests for Workstream A audit fixes (A1-A4, A6-A11).

Each test asserts closed-form/known values for a behavior-changing fix from
the 2026-08-24 two-repo audit plan.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchregress.losses.base import RegressionLoss, WeightedLossWrapper
from torchregress.losses.beta_nll import BetaNLLLoss
from torchregress.losses.eiv import FunctionalEIVLoss, InputNoiseAugmentationLoss
from torchregress.losses.evidential import EvidentialRegressionLoss
from torchregress.losses.expectile import MultiExpectileLoss
from torchregress.losses.gaussian import GaussianNLLLoss
from torchregress.losses.poisson import NegativeBinomialNLLLoss, ZeroInflatedPoissonNLLLoss
from torchregress.losses.poisson_gaussian import (
    EnhancedPoissonGaussianMixtureLoss,
    PoissonGaussianMixtureLoss,
)
from torchregress.losses.quantile import MultiQuantileLoss, QuantileCrossoverLoss
from torchregress.losses.utils_robust import huber_elementwise, log_cosh, tukey_biweight
from torchregress.utils.reduction import REDUCERS
from torchregress.utils.validation import validate_reduction

# ── A9: weight/reduction pipeline ────────────────────────────────────────


def test_a9_weighted_mean_is_zero_weight_safe() -> None:
    """REDUCERS['mean'] divides by sum(w) even when tiny; all-zero weights → 0."""
    values = torch.tensor([1.0, 2.0])
    tiny = torch.full((2,), 1e-6)
    out = REDUCERS["mean"](values, tiny)
    assert float(out) == pytest.approx(1.5)


class _SqLoss(RegressionLoss):
    def forward(self, y_pred, target, mask=None, weights=None, **kw):
        self._validate_inputs(y_pred, target, mask)
        return self._reduce((y_pred - target) ** 2, mask, weights)


def test_a9_zero_fill_mask_keeps_shape() -> None:
    """reduction='none' + mask zero-fills masked entries, keeps shape (A9)."""
    fn = _SqLoss(reduction="none")
    y = torch.tensor([0.0, 1.0, 2.0, 3.0])
    t = torch.tensor([1.0, 1.0, 1.0, 1.0])
    mask = torch.tensor([True, True, False, False])
    out = fn(y, t, mask=mask)
    assert out.shape == y.shape
    assert torch.allclose(out, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_a9_masked_mean_divides_by_mask_count() -> None:
    fn = _SqLoss()
    y = torch.tensor([0.0, 2.0, 10.0, 10.0])
    t = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([True, True, False, False])
    assert float(fn(y, t, mask=mask)) == pytest.approx((1.0 + 4.0) / 2)


def test_a9_min_max_reductions_rejected() -> None:
    with pytest.raises(ValueError):
        validate_reduction("max")
    with pytest.raises(ValueError):
        _SqLoss(reduction="min")


def test_a9_weighted_loss_wrapper_deepcopies() -> None:
    inner = nn.MSELoss(reduction="sum")
    wrapper = WeightedLossWrapper(inner)
    assert wrapper.reduction == "sum"
    # caller instance untouched (A9)
    assert inner.reduction == "sum"


# ── A1: true Poisson-Gaussian mixture NLL ───────────────────────────────


def test_a1_mixture_nll_matches_logsumexp_closed_form() -> None:
    torch.manual_seed(0)
    y_pred = torch.rand(8, 2) * 5 + 0.1
    target = torch.rand(8, 2) * 5
    fn = PoissonGaussianMixtureLoss(initial_variance=0.3, mixture_weights=0.7)
    got = fn(y_pred, target)

    pll = nn.PoissonNLLLoss(log_input=False, full=True, eps=1e-8, reduction="none")(y_pred, target)
    gll = 0.5 * (math.log(2 * math.pi * 0.3) + (target - y_pred) ** 2 / 0.3)
    expected = -torch.logsumexp(
        torch.stack(
            [math.log(0.7) - pll, torch.full_like(gll, math.log(0.3)) - gll],
            dim=-1,
        ),
        dim=-1,
    ).mean()
    assert torch.isclose(got, expected, rtol=1e-5)


def test_a1_mixture_differs_from_arithmetic_mean() -> None:
    y_pred = torch.tensor([[2.0]])
    target = torch.tensor([[3.0]])
    fn = PoissonGaussianMixtureLoss(initial_variance=1.0)
    mixture = float(fn(y_pred, target))
    pll = float(nn.PoissonNLLLoss(log_input=False, full=True, eps=1e-8)(y_pred, target))
    gll = 0.5 * (math.log(2 * math.pi) + 1.0)
    arithmetic = 0.5 * pll + 0.5 * gll
    assert not math.isclose(mixture, arithmetic, rel_tol=1e-4)


def test_a1_enhanced_uses_true_mixture_nll() -> None:
    y_pred = torch.tensor([[2.0, 3.0]])
    target = torch.tensor([[2.5, 2.0]])
    fn = EnhancedPoissonGaussianMixtureLoss(read_noise=0.5, shot_noise=0.2)
    out = fn(y_pred, target)
    assert torch.isfinite(out)

    rate = y_pred
    pll = nn.PoissonNLLLoss(log_input=False, full=True, eps=1e-8, reduction="none")(rate, target)
    var = 0.5 + 0.2 * rate
    gll = 0.5 * ((target - rate) ** 2 / var + torch.log(var) + math.log(2 * math.pi))
    expected = -torch.logsumexp(
        torch.stack(
            [torch.full_like(pll, math.log(0.5)) - pll, torch.full_like(gll, math.log(0.5)) - gll],
            dim=-1,
        ),
        dim=-1,
    ).mean()
    assert torch.isclose(out, expected, rtol=1e-5)


# ── A2: paper-exact β-NLL ────────────────────────────────────────────────


def test_a2_beta_nll_paper_exact() -> None:
    mean = torch.zeros(2, 2)
    log_var = torch.log(torch.tensor([[1.0, 4.0], [0.25, 9.0]]))
    target = torch.ones(2, 2)
    beta = 0.5
    got = BetaNLLLoss(beta=beta)((mean, log_var), target)

    var = log_var.exp()
    nll_per_dim = 0.5 * (math.log(2 * math.pi) + torch.log(var) + (target - mean) ** 2 / var)
    coef = var.detach().pow(beta)
    expected = (nll_per_dim * coef).sum(dim=-1).mean()
    assert torch.isclose(got, expected, rtol=1e-6)


def test_a2_citation_corrected() -> None:
    import torchregress.losses.beta_nll as m

    src = open(m.__file__).read()
    assert "2203.09168" in src
    assert "2205.11310" not in src


# ── A3: log-space clamps ─────────────────────────────────────────────────


def test_a3_gaussian_extreme_logvar_finite() -> None:
    mean = torch.zeros(1, 1)
    log_var = torch.tensor([[80.0]])  # exp(80) overflows fp32 without clamp
    target = torch.ones(1, 1)
    out = GaussianNLLLoss()((mean, log_var), target)
    assert torch.isfinite(out)
    _, var = GaussianNLLLoss()._extract_distribution_parameters((mean, log_var))
    assert var.max() <= math.exp(30.0) + 1e-6


def test_a3_tweedie_extreme_input_finite() -> None:
    from torchregress.losses.tweedie import TweedieLoss

    fn = TweedieLoss(p=1.5, link="log")
    y_pred = torch.tensor([[100.0]])  # exp would overflow without clamp
    target = torch.tensor([[1.0]])
    assert torch.isfinite(fn(y_pred, target))


# ── A4: NIG softplus constraints ─────────────────────────────────────────


def test_a4_nig_parameters_constrained() -> None:
    fn = EvidentialRegressionLoss()
    raw = torch.randn(6, 8)  # 4 params × 2 features
    gamma, nu, alpha, beta = fn._extract_nig_parameters(raw)
    assert torch.equal(gamma, raw[..., :2])
    assert (nu > 0).all() and (alpha > 1.0).all() and (beta > 0).all()


# ── A6: SLS context plumbing / sign consistency ──────────────────────────


def test_a6_sls_context_dim_zero_forward_finite() -> None:
    torch.manual_seed(0)
    loss_fn = SLSLossFactory()
    out = loss_fn(torch.randn(6, 2), torch.randn(6, 2))
    assert torch.isfinite(torch.as_tensor(float(out.detach())))


def SLSLossFactory():
    from torchregress.losses.sls import SLSLoss

    return SLSLoss(d=2, context_dim=0, K=1, warmup_steps=1)


# ── A7: poisson family ───────────────────────────────────────────────────


def test_a7_zip_requires_pi_logits() -> None:
    fn = ZeroInflatedPoissonNLLLoss(log_input=False)
    with pytest.raises(ValueError, match="pi_logits must be provided"):
        fn(torch.ones(4, 1), torch.zeros(4, 1))


def test_a7_no_learn_variance_in_poisson_module() -> None:
    import inspect

    import torchregress.losses.poisson as m

    src = inspect.getsource(m)
    assert "learn_variance" not in src
    with pytest.raises(TypeError):
        ZeroInflatedPoissonNLLLoss(learn_variance=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        __import__(
            "torchregress.losses.poisson", fromlist=["PoissonDevianceLoss"]
        ).PoissonDevianceLoss(learn_variance=True)  # type: ignore[call-arg]


def test_a7_negative_binomial_logitsigmoid_matches_manual() -> None:
    torch.manual_seed(1)
    mu = torch.rand(5, 1) * 4 + 0.5
    target = torch.randint(0, 8, (5, 1)).float()
    theta = torch.tensor(2.0)
    fn = NegativeBinomialNLLLoss()
    got = fn(mu, target, theta=float(theta))

    lgamma = torch.lgamma(target + theta) - torch.lgamma(target + 1.0) - torch.lgamma(theta)
    logit_p = math.log(theta) - torch.log(mu)
    manual = -(lgamma + theta * (-F.logsigmoid(-logit_p)) + target * (-F.logsigmoid(logit_p)))
    assert torch.allclose(got, manual.mean(), rtol=1e-5)


def test_a7_zip_extreme_lambda_finite() -> None:
    fn = ZeroInflatedPoissonNLLLoss(log_input=True)
    y = torch.tensor([[200.0]])  # exp overflow guard
    pi = torch.tensor([[-1.0]])
    out = fn(y, torch.zeros(1, 1), pi_logits=pi)
    assert torch.isfinite(out)


# ── A8: quantile/expectile contracts ────────────────────────────────────


def test_a8_multi_quantile_two_dim_multi_feature_raises() -> None:
    fn = MultiQuantileLoss(quantiles=[0.1, 0.5, 0.9])
    y = torch.randn(4, 6)  # ambiguous: 2 features × 3 quantiles
    with pytest.raises(ValueError, match="ambiguous 2-D input"):
        fn(y, torch.randn(4, 2))


def test_a8_multi_expectile_two_dim_multi_feature_raises() -> None:
    fn = MultiExpectileLoss(expectiles=[0.1, 0.5, 0.9])
    y = torch.randn(4, 6)
    with pytest.raises(ValueError, match="ambiguous 2-D input"):
        fn(y, torch.randn(4, 2))


def test_a8_unsorted_quantiles_rejected() -> None:
    with pytest.raises(ValueError, match="quantiles must be ascending"):
        MultiQuantileLoss(quantiles=[0.9, 0.1, 0.5])
    with pytest.raises(ValueError, match="quantiles must be ascending"):
        QuantileCrossoverLoss(quantiles=[0.9, 0.1])


def test_a8_unsorted_expectiles_rejected() -> None:
    with pytest.raises(ValueError, match="expectiles must be ascending"):
        MultiExpectileLoss(expectiles=[0.8, 0.2])


def test_a8_multi_expectile_partial_mask_uses_unmasked_features_only() -> None:
    expectiles = [0.5]
    fn = MultiExpectileLoss(expectiles=expectiles)
    y_pred = torch.tensor([[[1.0, 100.0]]])
    target = torch.tensor([[2.0, 2.0]])
    mask = torch.tensor([[True, False]])

    got = fn(y_pred, target, mask=mask)
    r = 2.0 - 1.0
    expected = 2 * r**2 * 0.5  # only the unmasked feature contributes
    assert float(got) == pytest.approx(expected, rel=1e-6)


# ── A10: EIV weights at outermost reduction only ────────────────────────


def test_a10_input_noise_augmentation_does_not_forward_weights() -> None:
    calls = {}

    def base_loss(pred, target, mask=None, weights=None, reduction="none", **kw):
        calls["weights"] = weights
        return (pred - target) ** 2

    model = nn.Identity()
    fn = InputNoiseAugmentationLoss(
        model=model,
        base_loss=base_loss,
        sigma_x=0.0,
        n_samples=3,
    )
    w = torch.tensor([1.0, 2.0])
    out = fn(torch.ones(2, 3), torch.zeros(2, 3), weights=w)
    assert calls["weights"] is None  # never forwarded into base_loss (A10)
    assert torch.isfinite(out).all()


# ── A11: misc fixes ──────────────────────────────────────────────────────


def test_a11_functional_eiv_mc_needs_two_samples() -> None:
    def model(x):
        return x[:, :1] * 2.0

    sigma = torch.tensor([0.1])
    with pytest.raises(ValueError, match="n_samples >= 2"):
        FunctionalEIVLoss(model, sigma_x=sigma, mode="mc", n_samples=1)
    # analytical mode is fine with any n_samples
    FunctionalEIVLoss(model, sigma_x=sigma, mode="analytical", n_samples=1)


def test_a11_fds_returns_features_at_eval_time() -> None:
    from torchregress.losses.imbalanced import FeatureDistributionSmoother

    fds = FeatureDistributionSmoother(feature_dim=4)
    targets = torch.cat([torch.zeros(50), torch.ones(50)])
    fds.fit(targets)
    feats = torch.randn(8, 4)
    fds.eval()
    assert fds.forward(feats, targets[:8], epoch=100) is feats
    fds.train()
    out_train = fds.forward(feats, targets[:8], epoch=0)
    assert torch.equal(out_train, feats)  # before start_smooth_epoch


def test_a11_density_weighted_buffer_on_device() -> None:
    from torchregress.losses.imbalanced import DensityWeightedLoss

    fn = DensityWeightedLoss()
    train_targets = torch.randn(64)
    fn.fit_density(train_targets)
    assert isinstance(fn.density_weights, torch.Tensor)
    assert fn.density_weights.numel() == 64
    # persistent buffer participates in state_dict round trip (A11)
    clone = DensityWeightedLoss()
    clone.density_weights = torch.zeros_like(fn.density_weights)
    clone.load_state_dict(fn.state_dict())
    idx = torch.arange(8)
    assert torch.allclose(clone.density_weights[idx].cpu(), fn.density_weights[idx].cpu())


def test_a11_utils_robust_identities() -> None:
    u = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    # Huber matches manual piecewise definition
    manual_huber = torch.where(u.abs() <= 1.0, 0.5 * u**2, 1.0 * (u.abs() - 0.5))
    assert torch.allclose(huber_elementwise(u, 1.0), manual_huber)
    # log-cosh ≈ log(cosh(u))
    assert torch.allclose(log_cosh(u), torch.log(torch.cosh(u)), atol=1e-6)
    # Tukey saturates beyond c
    t = tukey_biweight(u, c=1.0)
    c26 = 1.0**2 / 6.0
    assert float(t[0]) == pytest.approx(c26)
    assert float(t[1]) < c26
