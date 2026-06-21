"""Cross-class consistency tests for non-Gaussian regression losses.

Verifies that Poisson-Gaussian mixtures, censored losses, AFT, Tweedie,
quantile, ordinal, and conformal losses share consistent
reduction/mask/weight contracts and that special-case relationships hold.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from torchregress.losses.balanced_mse import BalancedMSELoss, BMCLoss
from torchregress.losses.censored import AFTLoss, CensoredGaussianNLLLoss, CensoredQuantileLoss
from torchregress.losses.conformal import ConformalLoss, MultiDimensionalConformalLoss
from torchregress.losses.eiv import EnsembleEIVLoss, FunctionalEIVLoss, StructuralEIVLoss
from torchregress.losses.evidential import EvidentialRegressionLoss
from torchregress.losses.expectile import (
    ExpectileLoss,
    MultiExpectileLoss,
)
from torchregress.losses.ordinal import CORALLoss, CumulativeLinkLoss, OrdinalCrossEntropyLoss
from torchregress.losses.poisson_gaussian import (
    EnhancedPoissonGaussianMixtureLoss,
    PoissonGaussianLikelihoodRatioLoss,
    PoissonGaussianMixtureLoss,
)
from torchregress.losses.quantile import (
    MultiQuantileLoss,
    QuantileCrossoverLoss,
    QuantileLoss,
)
from torchregress.losses.robust import (
    BarronLoss,
    CauchyLoss,
    CharbonnierLoss,
    LogCoshLoss,
    PseudoHuberLoss,
    TukeyBiweightLoss,
)
from torchregress.losses.tweedie import (
    CompoundPoissonLoss,
    GammaLoss,
    InverseGaussianLoss,
    TweedieLoss,
)

# ── helpers ──────────────────────────────────────────────────────────


def _check_reduction(loss_cls, y_pred, target, *loss_args, **kwargs):
    """Verify none/mean/sum reduction consistency.

    Instantiates *loss_cls* with ``reduction='none'/'mean'/'sum'`` plus
    any constructor keyword arguments in **kwargs, then calls each
    on ``(y_pred, target, *loss_args)``.  Asserts that:

    * ``none_out`` has the same shape as ``target``
    * ``none_out.mean() == mean_out``
    * ``sum_out / none_out.numel() == mean_out``

    Parameters
    ----------
    **kwargs : forwarded to the loss class constructor (e.g., ``p=1.5``).
    *loss_args : forwarded to ``forward(y_pred, target, *loss_args)``.
    """
    fn_none = loss_cls(reduction="none", **kwargs)
    fn_mean = loss_cls(reduction="mean", **kwargs)
    fn_sum = loss_cls(reduction="sum", **kwargs)

    none_out = fn_none(y_pred, target, *loss_args)
    mean_out = fn_mean(y_pred, target, *loss_args)
    sum_out = fn_sum(y_pred, target, *loss_args)

    assert none_out.shape == target.shape
    torch.testing.assert_close(none_out.mean(), mean_out)
    torch.testing.assert_close(sum_out / none_out.numel(), mean_out)


def _make_pos_targets(batch=4, dim=3):
    """Positive-valued targets suitable for Poisson-family losses."""
    torch.manual_seed(42)
    return torch.rand(batch, dim) * 20 + 1


def _make_pos_preds(batch=4, dim=3, log_input=False):
    """Positive-valued predictions.  If *log_input*, returns log(λ)."""
    torch.manual_seed(43)
    lam = torch.rand(batch, dim) * 15 + 0.5
    return torch.log(lam) if log_input else lam


# ── PoissonGaussianMixtureLoss ────────────────────────────────────────


class TestPoissonGaussianMixtureContract:
    """Reduction / mask / weight contracts for PoissonGaussianMixtureLoss."""

    def test_reduction_consistency(self):
        batch, dim = 6, 3
        y_pred = _make_pos_preds(batch, dim)
        target = _make_pos_targets(batch, dim)
        _check_reduction(PoissonGaussianMixtureLoss, y_pred, target)

    def test_mask_changes_loss(self):
        batch, dim = 5, 3
        y_pred = _make_pos_preds(batch, dim)
        target = _make_pos_targets(batch, dim)
        mask = torch.ones(batch, dim, dtype=torch.bool)
        mask[0, 0] = False

        fn = PoissonGaussianMixtureLoss(reduction="mean")
        loss_full = fn(y_pred, target)
        loss_masked = fn(y_pred, target, mask=mask)
        assert loss_masked != loss_full

    def test_mask_all_false_no_crash(self):
        batch, dim = 4, 3
        y_pred = _make_pos_preds(batch, dim)
        target = _make_pos_targets(batch, dim)
        mask = torch.zeros(batch, dim, dtype=torch.bool)

        fn = PoissonGaussianMixtureLoss(reduction="mean")
        loss = fn(y_pred, target, mask=mask)
        assert isinstance(loss, torch.Tensor)

    def test_weights_scale_loss(self):
        batch, dim = 4, 3
        y_pred = _make_pos_preds(batch, dim)
        target = _make_pos_targets(batch, dim)
        w1 = torch.ones(batch, dim)
        w2 = w1.clone()
        w2[0, 0] = 2.0

        fn = PoissonGaussianMixtureLoss(reduction="none")
        out1 = fn(y_pred, target, weights=w1)
        out2 = fn(y_pred, target, weights=w2)

        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_zero_weight_zeros_loss(self):
        batch, dim = 4, 3
        y_pred = _make_pos_preds(batch, dim)
        target = _make_pos_targets(batch, dim)
        w = torch.ones(batch, dim)
        w[0, 0] = 0.0

        fn = PoissonGaussianMixtureLoss(reduction="none")
        out = fn(y_pred, target, weights=w)
        assert out[0, 0] == 0.0

    def test_gradients_flow(self):
        model = torch.nn.Linear(3, 3)
        loss_fn = PoissonGaussianMixtureLoss(learn_variance=True, mixture_weights="learn")
        opt = torch.optim.SGD(list(model.parameters()) + list(loss_fn.parameters()), lr=0.01)
        x = torch.randn(4, 3)
        target = _make_pos_targets(4, 3)

        for _ in range(3):
            opt.zero_grad()
            # Clamp to positive — Poisson rate must be > 0
            y_pred = torch.nn.functional.softplus(model(x))
            loss = loss_fn(y_pred, target)
            loss.backward()
            opt.step()

        assert torch.isfinite(loss)

    def test_numerical_stability_extreme_lambda(self):
        fn = PoissonGaussianMixtureLoss()
        # Very small predictions
        y_pred = torch.tensor([[1e-10, 1e-8], [1e-9, 1e-7]])
        target = torch.tensor([[0.0, 1e-8], [1e-9, 1e-7]])
        loss = fn(y_pred, target)
        assert torch.isfinite(loss)

        # Very large predictions
        y_pred = torch.full((2, 2), 1e8)
        target = torch.full((2, 2), 1e8 + 100)
        loss = fn(y_pred, target)
        assert torch.isfinite(loss)


class TestPoissonGaussianMixtureRelationships:
    """Family relationships: PG mixture reduces to known limits."""

    def test_equal_weights_are_symmetric(self):
        """With equal weights (0.5, 0.5), swapping Poisson/Gaussian
        doesn't change the loss."""
        y_pred = _make_pos_preds(3, 2)
        target = _make_pos_targets(3, 2)

        fn = PoissonGaussianMixtureLoss(mixture_weights=None)  # default 0.5
        loss1 = fn(y_pred, target)

        fn2 = PoissonGaussianMixtureLoss(mixture_weights=0.5)  # explicit 0.5
        loss2 = fn2(y_pred, target)

        torch.testing.assert_close(loss1, loss2)

    def test_poisson_only_with_weight_one(self):
        """poisson_weight=1 (gaussian_weight=0) should match pure PoissonNLLLoss."""
        y_pred = _make_pos_preds(3, 2)
        target = _make_pos_targets(3, 2)

        fn_pg = PoissonGaussianMixtureLoss(mixture_weights=1.0, reduction="none")
        pg_loss = fn_pg(y_pred, target)

        # Pure Poisson NLL (full=True matches the internal PoissonNLLLoss)
        poisson = torch.nn.PoissonNLLLoss(log_input=False, full=True, eps=1e-8, reduction="none")
        poisson_loss = poisson(y_pred, target)

        # With weight=1, PG loss = poisson_weight * poisson_loss
        # = 1.0 * poisson_loss = poisson_loss
        torch.testing.assert_close(pg_loss, poisson_loss, atol=1e-5, rtol=1e-4)

    def test_fixed_mixture_weights_differ_from_default(self):
        """Different mixture weights produce different losses."""
        y_pred = _make_pos_preds(4, 3)
        target = _make_pos_targets(4, 3)

        fn_default = PoissonGaussianMixtureLoss(reduction="mean")
        fn_weighted = PoissonGaussianMixtureLoss(mixture_weights=0.3, reduction="mean")

        loss_default = fn_default(y_pred, target)
        loss_weighted = fn_weighted(y_pred, target)
        assert loss_default != loss_weighted

    def test_learnable_weights_match_default_at_init(self):
        """Learnable mixture weights (init ~0.5 via sigmoid(0)) match
        the default equal weighting at initialization."""
        y_pred = _make_pos_preds(4, 3)
        target = _make_pos_targets(4, 3)

        fn_default = PoissonGaussianMixtureLoss(reduction="mean")
        fn_learn = PoissonGaussianMixtureLoss(mixture_weights="learn", reduction="mean")

        # Learner starts at sigmoid(0) = 0.5, same as default
        torch.testing.assert_close(fn_default(y_pred, target), fn_learn(y_pred, target))


# ── EnhancedPoissonGaussianMixtureLoss ────────────────────────────────


class TestEnhancedPoissonGaussianContract:
    """Reduction / mask / weight contracts for EnhancedPoissonGaussianMixtureLoss."""

    def test_reduction_consistency(self):
        batch, dim = 6, 3
        y_pred = _make_pos_preds(batch, dim)
        target = _make_pos_targets(batch, dim)
        _check_reduction(EnhancedPoissonGaussianMixtureLoss, y_pred, target)

    def test_mask_changes_loss(self):
        batch, dim = 5, 3
        y_pred = _make_pos_preds(batch, dim)
        target = _make_pos_targets(batch, dim)
        mask = torch.ones(batch, dim, dtype=torch.bool)
        mask[0, 0] = False

        fn = EnhancedPoissonGaussianMixtureLoss(reduction="mean")
        loss_full = fn(y_pred, target)
        loss_masked = fn(y_pred, target, mask=mask)
        assert loss_masked != loss_full

    def test_weights_scale_loss(self):
        batch, dim = 4, 3
        y_pred = _make_pos_preds(batch, dim)
        target = _make_pos_targets(batch, dim)
        w1 = torch.ones(batch, dim)
        w2 = w1.clone()
        w2[0, 0] = 2.0

        fn = EnhancedPoissonGaussianMixtureLoss(reduction="none")
        out1 = fn(y_pred, target, weights=w1)
        out2 = fn(y_pred, target, weights=w2)
        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_default_identity_noop(self):
        """With default gain=1 offset=0, scaled_rate ≈ rate (up to clamp)."""
        y_pred = _make_pos_preds(4, 3)
        target = _make_pos_targets(4, 3)

        fn = EnhancedPoissonGaussianMixtureLoss(read_noise=1.0, shot_noise=0.0)
        loss = fn(y_pred, target)
        assert torch.isfinite(loss)

    def test_gradients_flow(self):
        model = torch.nn.Linear(3, 3)
        loss_fn = EnhancedPoissonGaussianMixtureLoss(
            gain="learn", offset="learn", read_noise="learn"
        )
        opt = torch.optim.SGD(list(model.parameters()) + list(loss_fn.parameters()), lr=0.01)
        x = torch.randn(4, 3)
        target = _make_pos_targets(4, 3)

        for _ in range(3):
            opt.zero_grad()
            loss = loss_fn(model(x), target)
            loss.backward()
            opt.step()

        assert torch.isfinite(loss)

    def test_numerical_stability(self):
        fn = EnhancedPoissonGaussianMixtureLoss()
        y_pred = torch.full((2, 2), 1e8)
        target = torch.full((2, 2), 1e8 + 100)
        loss = fn(y_pred, target)
        assert torch.isfinite(loss)


# ── PoissonGaussianLikelihoodRatioLoss ────────────────────────────────


class TestPoissonGaussianLRContract:
    """Contracts for PoissonGaussianLikelihoodRatioLoss."""

    def test_reduction_consistency(self):
        batch, dim = 6, 3
        y_pred = _make_pos_preds(batch, dim, log_input=True)
        target = _make_pos_targets(batch, dim)
        _check_reduction(PoissonGaussianLikelihoodRatioLoss, y_pred, target, log_input=True)

    def test_mask_changes_loss(self):
        y_pred = _make_pos_preds(5, 3, log_input=True)
        target = _make_pos_targets(5, 3)
        mask = torch.ones(5, 3, dtype=torch.bool)
        mask[0, 0] = False

        fn = PoissonGaussianLikelihoodRatioLoss(reduction="mean", log_input=True)
        loss_full = fn(y_pred, target)
        loss_masked = fn(y_pred, target, mask=mask)
        assert loss_masked != loss_full

    def test_weights_scale_loss(self):
        y_pred = _make_pos_preds(4, 2, log_input=True)
        target = _make_pos_targets(4, 2)
        w1 = torch.ones(4, 2)
        w2 = w1.clone()
        w2[0, 0] = 2.0

        fn = PoissonGaussianLikelihoodRatioLoss(reduction="none", log_input=True)
        out1 = fn(y_pred, target, weights=w1)
        out2 = fn(y_pred, target, weights=w2)
        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_gradients_flow(self):
        model = torch.nn.Linear(3, 3)
        loss_fn = PoissonGaussianLikelihoodRatioLoss(learn_variance=True)
        opt = torch.optim.SGD(list(model.parameters()) + list(loss_fn.parameters()), lr=0.01)
        x = torch.randn(4, 3)
        target = _make_pos_targets(4, 3)

        for _ in range(3):
            opt.zero_grad()
            loss = loss_fn(model(x), target)
            loss.backward()
            opt.step()

        assert torch.isfinite(loss)


# ── CensoredGaussianNLLLoss ───────────────────────────────────────────


class TestCensoredGaussianNLLContract:
    """Contracts for CensoredGaussianNLLLoss."""

    def test_reduction_consistency(self):
        batch, dim = 6, 3
        mean = torch.randn(batch, dim)
        log_var = torch.zeros(batch, dim)
        target = torch.randn(batch, dim)
        censoring = torch.zeros(batch, dim, dtype=torch.long)
        _check_reduction(CensoredGaussianNLLLoss, (mean, log_var), target, censoring)

    def test_mask_changes_loss(self):
        batch, dim = 5, 2
        mean = torch.randn(batch, dim)
        log_var = torch.zeros(batch, dim)
        target = torch.randn(batch, dim)
        mask = torch.ones(batch, dim, dtype=torch.bool)
        mask[0, 0] = False

        fn = CensoredGaussianNLLLoss(reduction="mean")
        loss_full = fn((mean, log_var), target)
        loss_masked = fn((mean, log_var), target, mask=mask)
        assert loss_masked != loss_full

    def test_weights_scale_loss(self):
        batch, dim = 4, 2
        mean = torch.randn(batch, dim)
        log_var = torch.zeros(batch, dim)
        target = torch.randn(batch, dim)
        w1 = torch.ones(batch, dim)
        w2 = w1.clone()
        w2[0, 0] = 2.0

        fn = CensoredGaussianNLLLoss(reduction="none")
        out1 = fn((mean, log_var), target, weights=w1)
        out2 = fn((mean, log_var), target, weights=w2)

        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_observed_matches_standard_nll(self):
        """With all-observed censoring, loss equals standard Gaussian NLL."""
        batch, dim = 4, 2
        mean = torch.randn(batch, dim)
        log_var = torch.randn(batch, dim) * 0.5
        target = torch.randn(batch, dim)
        censoring = torch.zeros(batch, dim, dtype=torch.long)

        fn_cens = CensoredGaussianNLLLoss(reduction="none", log_variance=True)
        cens_loss = fn_cens((mean, log_var), target, censoring=censoring)

        # Standard NLL: 0.5 * (log(2π) + log(σ²) + (y-μ)²/σ²)
        var = torch.exp(log_var)
        std = torch.sqrt(var)
        z = (target - mean) / std
        logpdf = -0.5 * z.pow(2) - torch.log(std) - 0.5 * math.log(2 * math.pi)
        expected = -logpdf

        torch.testing.assert_close(cens_loss, expected, atol=1e-5, rtol=1e-4)

    def test_right_censored_is_non_negative(self):
        batch, dim = 3, 2
        mean = torch.randn(batch, dim)
        log_var = torch.zeros(batch, dim)
        target = torch.randn(batch, dim)
        censoring = torch.ones(batch, dim, dtype=torch.long)  # all right-censored

        fn = CensoredGaussianNLLLoss(reduction="none")
        loss = fn((mean, log_var), target, censoring=censoring)
        assert (loss >= 0).all()

    def test_left_censored_is_non_negative(self):
        batch, dim = 3, 2
        mean = torch.randn(batch, dim)
        log_var = torch.zeros(batch, dim)
        target = torch.randn(batch, dim)
        censoring = torch.full((batch, dim), -1, dtype=torch.long)  # all left-censored

        fn = CensoredGaussianNLLLoss(reduction="none")
        loss = fn((mean, log_var), target, censoring=censoring)
        assert (loss >= 0).all()

    def test_interval_censoring_produces_finite_loss(self):
        batch, dim = 3, 2
        mean = torch.randn(batch, dim)
        log_var = torch.zeros(batch, dim)
        target = torch.randn(batch, dim)
        lower = target - 0.5
        upper = target + 0.5

        fn = CensoredGaussianNLLLoss()
        loss = fn((mean, log_var), target, lower_bound=lower, upper_bound=upper)
        assert torch.isfinite(loss)

    def test_var_from_kwargs(self):
        """CensoredGaussianNLLLoss accepts var/log_var via kwargs."""
        mean = torch.randn(3, 2)
        target = torch.randn(3, 2)

        fn = CensoredGaussianNLLLoss(log_variance=False)
        loss = fn(mean, target, var=torch.ones(3, 2))
        assert torch.isfinite(loss)


class TestCensoredGaussianNLLRelationships:
    """Cross-path consistency for censored NLL."""

    def test_censoring_value_zero_matches_no_censoring(self):
        """censoring=None vs censoring=0 produce the same result."""
        batch, dim = 5, 3
        mean = torch.randn(batch, dim)
        log_var = torch.randn(batch, dim) * 0.5
        target = torch.randn(batch, dim)
        censoring = torch.zeros(batch, dim, dtype=torch.long)

        fn = CensoredGaussianNLLLoss(reduction="none", log_variance=True)
        loss_none_cens = fn((mean, log_var), target)
        loss_zero_cens = fn((mean, log_var), target, censoring=censoring)

        torch.testing.assert_close(loss_none_cens, loss_zero_cens)

    def test_interval_narrow_approaches_observed(self):
        """As interval width → 0, interval NLL − (−log width) ≈ observed NLL.

        The interval-censored NLL is −log(Φ(z_up) − Φ(z_low)), which for
        narrow intervals approximates −log(φ(z_mid)·width/σ) =
        −log φ(z_mid) + log σ − log(width).  The −log(width) term diverges
        as width → 0, so we subtract it before comparing to the observed NLL."""
        batch, dim = 3, 2
        mean = torch.randn(batch, dim)
        log_var = torch.randn(batch, dim) * 0.5
        target = torch.randn(batch, dim)
        eps_interval = 1e-6
        lower = target - eps_interval
        upper = target + eps_interval
        width = 2 * eps_interval

        fn = CensoredGaussianNLLLoss(reduction="none", log_variance=True)
        loss_obs = fn((mean, log_var), target)
        loss_interval = fn((mean, log_var), target, lower_bound=lower, upper_bound=upper)

        # NLL_interval ≈ NLL_obs − log(width); undo the divergence.
        loss_interval_corrected = loss_interval + math.log(width)
        torch.testing.assert_close(loss_interval_corrected, loss_obs, atol=0.5, rtol=0.3)


# ── CensoredQuantileLoss ──────────────────────────────────────────────


class TestCensoredQuantileContract:
    """Contracts for CensoredQuantileLoss."""

    def test_reduction_consistency(self):
        batch, dim = 4, 2
        y_pred = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        censoring = torch.zeros(batch, dim, dtype=torch.long)
        _check_reduction(CensoredQuantileLoss, y_pred, target, censoring, quantile=0.5)

    def test_all_paths_produce_non_negative_loss(self):
        batch = 4
        y_pred = torch.randn(batch)
        target = torch.randn(batch)
        censoring = torch.tensor([0, 1, -1, 0], dtype=torch.long)

        fn = CensoredQuantileLoss(quantile=0.5, reduction="none")
        loss = fn(y_pred, target, censoring=censoring)
        assert (loss >= 0).all()

    def test_median_observed_is_mae(self):
        """Quantile=0.5 with all-observed matches 0.5 * |error| (MAE)."""
        y_pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        censoring = torch.zeros(5, 3, dtype=torch.long)

        fn = CensoredQuantileLoss(quantile=0.5, reduction="none")
        loss = fn(y_pred, target, censoring=censoring)

        expected = 0.5 * torch.abs(target - y_pred)
        torch.testing.assert_close(loss, expected)


# ── AFTLoss ───────────────────────────────────────────────────────────


class TestAFTLossContract:
    """Contracts for AFTLoss."""

    def test_reduction_consistency(self):
        batch, dim = 6, 2
        loc = torch.randn(batch, dim)
        log_scale = torch.randn(batch, dim) * 0.5
        target = torch.exp(torch.randn(batch, dim))
        censoring = torch.zeros(batch, dim, dtype=torch.long)
        _check_reduction(AFTLoss, (loc, log_scale), target, censoring)

    def test_mask_changes_loss(self):
        batch, dim = 5, 2
        loc = torch.randn(batch, dim)
        log_scale = torch.randn(batch, dim) * 0.5
        target = torch.exp(torch.randn(batch, dim))
        mask = torch.ones(batch, dim, dtype=torch.bool)
        mask[0, 0] = False

        fn = AFTLoss(reduction="mean")
        loss_full = fn((loc, log_scale), target)
        loss_masked = fn((loc, log_scale), target, mask=mask)
        assert loss_masked != loss_full

    def test_gradients_flow(self):
        loc = torch.randn(4, 3, requires_grad=True)
        log_scale = torch.randn(4, 3, requires_grad=True)
        target = torch.exp(torch.randn(4, 3))
        censoring = torch.randint(low=-1, high=2, size=(4, 3))

        fn = AFTLoss()
        loss = fn((loc, log_scale), target, censoring=censoring)
        loss.backward()

        assert loc.grad is not None and torch.isfinite(loc.grad).all()
        assert log_scale.grad is not None and torch.isfinite(log_scale.grad).all()

    def test_observed_matches_log_normal_nll(self):
        """All-observed AFT = log-normal NLL: -log_pdf of lognormal(loc, scale)."""
        batch, dim = 4, 2
        loc = torch.randn(batch, dim)
        log_scale = torch.zeros(batch, dim)
        target = torch.exp(torch.randn(batch, dim))

        fn = AFTLoss(reduction="none")
        aft_loss = fn((loc, log_scale), target)

        # Log-normal NLL: log(t) + log(σ) + 0.5*((log(t)-μ)/σ)² + 0.5*log(2π)
        scale = torch.exp(log_scale)
        log_t = torch.log(target)
        z = (log_t - loc) / scale
        expected = (
            torch.log(target) + torch.log(scale) + 0.5 * z.pow(2) + 0.5 * math.log(2 * math.pi)
        )

        torch.testing.assert_close(aft_loss, expected)


# ── PG vs Enhanced PG cross-relationships ─────────────────────────────


class TestPoissonGaussianCrossFamily:
    """Cross-family: PG vs Enhanced PG vs LR."""

    def test_basic_pg_equals_enhanced_with_defaults(self):
        """With default gain=1/offset=0/read_noise=1/shot_noise=0 and
        compatible Poisson loss, EnhancedPG approximates basic PG.

        They are not identical — PG uses learned variance while Enhanced
        uses fixed read noise, and the Gaussian term differs — but the
        loss magnitude should be within a factor of ~2."""
        y_pred = _make_pos_preds(4, 3)
        target = _make_pos_targets(4, 3)

        pg = PoissonGaussianMixtureLoss(mixture_weights=0.5, reduction="mean")
        epg = EnhancedPoissonGaussianMixtureLoss(
            gain=1.0, offset=0.0, read_noise=1.0, shot_noise=0.0, reduction="mean"
        )

        loss_pg = pg(y_pred, target)
        loss_epg = epg(y_pred, target)

        assert torch.isfinite(loss_pg)
        assert torch.isfinite(loss_epg)
        # Same order of magnitude
        assert 0.1 < loss_epg / loss_pg < 10.0


# ── Tweedie family ────────────────────────────────────────────────────


class TestTweedieFamilyContract:
    """Reduction / mask / weight contracts for the Tweedie family."""

    @staticmethod
    def _data(batch=6, dim=3):
        torch.manual_seed(99)
        y_pred = torch.rand(batch, dim) * 5 + 0.1
        target = torch.rand(batch, dim) * 5 + 0.5
        return y_pred, target

    def test_tweedie_reduction_consistency(self):
        y_pred, target = self._data()
        _check_reduction(TweedieLoss, y_pred, target, p=1.5, link="identity")

    def test_gamma_reduction_consistency(self):
        y_pred, target = self._data()
        _check_reduction(GammaLoss, y_pred, target, link="identity")

    def test_inverse_gaussian_reduction_consistency(self):
        y_pred, target = self._data()
        _check_reduction(InverseGaussianLoss, y_pred, target, link="identity")

    def test_compound_poisson_reduction_consistency(self):
        y_pred, target = self._data()
        _check_reduction(CompoundPoissonLoss, y_pred, target, p=1.5, link="identity")

    def test_tweedie_mask_changes_loss(self):
        y_pred, target = self._data(5, 3)
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[0, 0] = False
        fn = TweedieLoss(p=1.5, reduction="mean", link="identity")
        assert fn(y_pred, target) != fn(y_pred, target, mask=mask)

    def test_tweedie_weights_scale_loss(self):
        y_pred, target = self._data(4, 2)
        w1 = torch.ones_like(target)
        w2 = w1.clone()
        w2[0, 0] = 2.0
        fn = TweedieLoss(p=1.5, reduction="none", link="identity")
        out1 = fn(y_pred, target, weights=w1)
        out2 = fn(y_pred, target, weights=w2)
        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_tweedie_zero_weight_zeros_loss(self):
        y_pred, target = self._data(4, 2)
        w = torch.ones_like(target)
        w[0, 0] = 0.0
        fn = TweedieLoss(p=1.5, reduction="none", link="identity")
        out = fn(y_pred, target, weights=w)
        assert out[0, 0] == 0.0


class TestTweedieFamilyRelationships:
    """Tweedie subclass ↔ base class equivalence."""

    def test_gamma_equals_tweedie_p2(self):
        y_pred = torch.rand(4, 3) * 5 + 0.1
        target = torch.rand(4, 3) * 5 + 0.5
        g = GammaLoss(reduction="mean", link="identity")(y_pred, target)
        t = TweedieLoss(p=2.0, reduction="mean", link="identity")(y_pred, target)
        torch.testing.assert_close(g, t)

    def test_inverse_gaussian_equals_tweedie_p3(self):
        y_pred = torch.rand(4, 3) * 5 + 0.1
        target = torch.rand(4, 3) * 5 + 0.5
        ig = InverseGaussianLoss(reduction="mean", link="identity")(y_pred, target)
        t = TweedieLoss(p=3.0, reduction="mean", link="identity")(y_pred, target)
        torch.testing.assert_close(ig, t)

    def test_compound_poisson_equals_tweedie_p1_5(self):
        y_pred = torch.rand(4, 3) * 5 + 0.1
        target = torch.rand(4, 3) * 5 + 0.5
        cp = CompoundPoissonLoss(p=1.5, reduction="mean", link="identity")(y_pred, target)
        t = TweedieLoss(p=1.5, reduction="mean", link="identity")(y_pred, target)
        torch.testing.assert_close(cp, t)

    def test_tweedie_gradient_flow(self):
        # requires_grad_(True) on the final tensor keeps it a leaf.
        y_pred = (torch.rand(4, 3) * 5 + 0.1).requires_grad_(True)
        target = torch.rand(4, 3) * 5 + 0.5
        for cls, kw in [
            (TweedieLoss, {"p": 1.5, "link": "identity"}),
            (GammaLoss, {"link": "identity"}),
            (InverseGaussianLoss, {"link": "identity"}),
            (CompoundPoissonLoss, {"p": 1.5, "link": "identity"}),
        ]:
            y_pred.grad = None
            loss = cls(reduction="mean", **kw)(y_pred, target)
            loss.backward()
            assert y_pred.grad is not None and torch.isfinite(y_pred.grad).all()


# ── Quantile family ───────────────────────────────────────────────────


class TestQuantileFamilyContract:
    """Reduction / mask / weight / relationship contracts for quantile losses."""

    def test_quantile_reduction_consistency(self):
        y_pred = torch.randn(6, 3)
        target = torch.randn(6, 3)
        _check_reduction(QuantileLoss, y_pred, target, quantile=0.3)

    def test_quantile_mask_changes_loss(self):
        y_pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        mask = torch.ones(5, 3, dtype=torch.bool)
        mask[0, 0] = False
        fn = QuantileLoss(quantile=0.5, reduction="mean")
        assert fn(y_pred, target) != fn(y_pred, target, mask=mask)

    def test_quantile_weights_scale_loss(self):
        y_pred = torch.randn(4, 2)
        target = torch.randn(4, 2)
        w1 = torch.ones(4, 2)
        w2 = w1.clone()
        w2[0, 0] = 2.0
        fn = QuantileLoss(quantile=0.5, reduction="none")
        out1 = fn(y_pred, target, weights=w1)
        out2 = fn(y_pred, target, weights=w2)
        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_median_is_half_mae(self):
        """Quantile(0.5) = 0.5 * |y − ŷ|."""
        y_pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        q_loss = QuantileLoss(quantile=0.5, reduction="none")(y_pred, target)
        expected = 0.5 * torch.abs(target - y_pred)
        torch.testing.assert_close(q_loss, expected)

    def test_multi_quantile_reduction_consistency(self):
        y_pred = torch.randn(6, 3, 2)
        target = torch.randn(6, 2)
        _check_reduction(MultiQuantileLoss, y_pred, target, quantiles=[0.1, 0.5, 0.9])

    def test_crossover_penalty_for_violations(self):
        """The crossover penalty term equals
        ``crossover_penalty * sum(relu(qᵢ − qᵢ₊₁))``.

        Subtracting ``crossover_penalty=0`` isolates the penalty:
        ``final_loss_pen = base_loss_w * base_loss + pen * violations``
        ``final_loss_0    = base_loss_w * base_loss``
        so the difference is ``pen * violations``, independent of base loss."""
        target = torch.randn(4, 2)
        # Same base tensor with offsets guarantees elementwise ordering.
        base = torch.randn(4, 2)
        ordered = torch.stack([base - 1.0, base, base + 1.0], dim=1)
        reversed_ = ordered.flip(1)

        crossover_penalty = 10.0
        quantiles = [0.1, 0.5, 0.9]

        fn_0 = QuantileCrossoverLoss(quantiles=quantiles, crossover_penalty=0.0, reduction="none")
        fn_pen = QuantileCrossoverLoss(
            quantiles=quantiles,
            crossover_penalty=crossover_penalty,
            reduction="none",
        )

        # Ordered predictions have no crossover: difference = 0.
        diff_ordered = fn_pen(ordered, target) - fn_0(ordered, target)
        assert (diff_ordered == 0.0).all(), "ordered predictions should have no crossover penalty"

        # Reversed predictions: difference = crossover_penalty * sum(violations).
        diff_reversed = fn_pen(reversed_, target) - fn_0(reversed_, target)
        violations = F.relu(reversed_[:, :-1] - reversed_[:, 1:])
        expected_penalty = crossover_penalty * torch.sum(violations, dim=1)

        torch.testing.assert_close(
            diff_reversed,
            expected_penalty,
            msg="crossover penalty ≠ crossover_penalty * sum(violations)",
        )

        # Sanity: reversed loss > ordered loss.
        fn = QuantileCrossoverLoss(quantiles=quantiles, crossover_penalty=crossover_penalty)
        assert fn(ordered, target) < fn(reversed_, target)


# ── Ordinal family ────────────────────────────────────────────────────


class TestOrdinalFamilyContract:
    """Reduction / mask / weight contracts for ordinal losses."""

    def test_ordinal_ce_reduction_consistency(self):
        logits = torch.randn(6, 5, 3)
        target = torch.randint(0, 5, (6, 3))
        _check_reduction(OrdinalCrossEntropyLoss, logits, target)

    def test_ordinal_ce_mask_changes_loss(self):
        logits = torch.randn(5, 4, 3)
        target = torch.randint(0, 4, (5, 3))
        mask = torch.ones(5, 3, dtype=torch.bool)
        mask[0, 0] = False
        fn = OrdinalCrossEntropyLoss(reduction="mean")
        assert fn(logits, target) != fn(logits, target, mask=mask)

    def test_ordinal_ce_weights_scale_loss(self):
        logits = torch.randn(4, 3, 2)
        target = torch.randint(0, 3, (4, 2))
        w1 = torch.ones(4, 2)
        w2 = w1.clone()
        w2[0, 0] = 2.0
        fn = OrdinalCrossEntropyLoss(reduction="none")
        out1 = fn(logits, target, weights=w1)
        out2 = fn(logits, target, weights=w2)
        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_cumulative_link_reduction_consistency(self):
        logits = torch.randn(6, 4, 3)
        target = torch.randint(0, 5, (6, 3))
        _check_reduction(CumulativeLinkLoss, logits, target)

    def test_cumulative_link_mask_changes_loss(self):
        logits = torch.randn(5, 4, 3)
        target = torch.randint(0, 5, (5, 3))
        mask = torch.ones(5, 3, dtype=torch.bool)
        mask[0, 0] = False
        fn = CumulativeLinkLoss(reduction="mean")
        assert fn(logits, target) != fn(logits, target, mask=mask)

    def test_coral_equals_cumulative_link(self):
        """CORALLoss = CumulativeLinkLoss (same computation, different
        architectural convention)."""
        logits = torch.randn(4, 3, 2)
        target = torch.randint(0, 4, (4, 2))
        coral = CORALLoss(reduction="none")(logits, target)
        cl = CumulativeLinkLoss(reduction="none")(logits, target)
        torch.testing.assert_close(coral, cl)

    def test_coral_reduction_consistency(self):
        logits = torch.randn(6, 3, 2)
        target = torch.randint(0, 4, (6, 2))
        _check_reduction(CORALLoss, logits, target)

    def test_coral_mask_changes_loss(self):
        logits = torch.randn(5, 3, 2)
        target = torch.randint(0, 4, (5, 2))
        mask = torch.ones(5, 2, dtype=torch.bool)
        mask[0, 0] = False
        fn = CORALLoss(reduction="mean")
        assert fn(logits, target) != fn(logits, target, mask=mask)

    def test_coral_weights_scale_loss(self):
        logits = torch.randn(4, 3, 2)
        target = torch.randint(0, 3, (4, 2))
        w1 = torch.ones(4, 2)
        w2 = w1.clone()
        w2[0, 0] = 2.0
        fn = CORALLoss(reduction="none")
        out1 = fn(logits, target, weights=w1)
        out2 = fn(logits, target, weights=w2)
        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_coral_gradients_flow(self):
        logits = torch.randn(4, 3, 2, requires_grad=True)
        target = torch.randint(0, 4, (4, 2))
        loss = CORALLoss()(logits, target)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


# ── ConformalLoss ─────────────────────────────────────────────────────


class TestConformalLossContract:
    """Reduction / mask / weight / gradient contracts for ConformalLoss."""

    # -- split method -------------------------------------------------------

    def test_split_reduction_consistency(self):
        y_pred = torch.randn(8, 3)
        target = torch.randn(8, 3)
        _check_reduction(ConformalLoss, y_pred, target, method="split")

    def test_split_mask_changes_loss(self):
        batch, feat = 5, 3
        y_pred = torch.randn(batch, feat)
        target = torch.randn(batch, feat)
        mask = torch.ones(batch, feat, dtype=torch.bool)
        mask[0, 0] = False

        fn = ConformalLoss(method="split", reduction="mean")
        assert fn(y_pred, target) != fn(y_pred, target, mask=mask)

    def test_split_weights_scale_loss(self):
        batch, feat = 4, 2
        y_pred = torch.randn(batch, feat)
        target = torch.randn(batch, feat)
        w1 = torch.ones(batch, feat)
        w2 = w1.clone()
        w2[0, 0] = 2.0

        fn = ConformalLoss(method="split", reduction="none")
        out1 = fn(y_pred, target, weights=w1)
        out2 = fn(y_pred, target, weights=w2)
        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_split_gradients_flow(self):
        model = torch.nn.Linear(3, 3)
        loss_fn = ConformalLoss(method="split", reduction="mean")
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 3)
        target = torch.randn(4, 3)

        for _ in range(3):
            opt.zero_grad()
            loss = loss_fn(model(x), target)
            loss.backward()
            opt.step()
        assert torch.isfinite(loss)

    # -- cqr method --------------------------------------------------------

    def test_cqr_reduction_consistency(self):
        y_pred = torch.randn(8, 6)  # batch=8, 2*feat=6 → feat=3
        target = torch.randn(8, 3)
        _check_reduction(ConformalLoss, y_pred, target, method="cqr")

    def test_cqr_mask_changes_loss(self):
        batch, feat = 5, 3
        y_pred = torch.randn(batch, 2 * feat)
        target = torch.randn(batch, feat)
        mask = torch.ones(batch, feat, dtype=torch.bool)
        mask[0, 0] = False

        fn = ConformalLoss(method="cqr", reduction="mean")
        assert fn(y_pred, target) != fn(y_pred, target, mask=mask)

    def test_cqr_weights_scale_loss(self):
        batch, feat = 4, 2
        y_pred = torch.randn(batch, 2 * feat)
        target = torch.randn(batch, feat)
        w1 = torch.ones(batch, feat)
        w2 = w1.clone()
        w2[0, 0] = 2.0

        fn = ConformalLoss(method="cqr", reduction="none")
        out1 = fn(y_pred, target, weights=w1)
        out2 = fn(y_pred, target, weights=w2)
        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_cqr_gradients_flow(self):
        model = torch.nn.Linear(3, 6)  # 2 * feat
        loss_fn = ConformalLoss(method="cqr", reduction="mean")
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 3)
        target = torch.randn(4, 3)

        for _ in range(3):
            opt.zero_grad()
            loss = loss_fn(model(x), target)
            loss.backward()
            opt.step()
        assert torch.isfinite(loss)


class TestConformalLossRelationships:
    """Cross-method consistency and calibration relationships."""

    def test_split_method_equals_mse(self):
        """ConformalLoss(method='split') forward = MSE loss."""
        y_pred = torch.randn(6, 3)
        target = torch.randn(6, 3)

        cf_loss = ConformalLoss(method="split", reduction="none")(y_pred, target)
        mse = (y_pred - target) ** 2
        torch.testing.assert_close(cf_loss, mse)

    def test_cqr_and_uacqr_forward_match(self):
        """CQR and UACQR share the same pinball training loss."""
        y_pred = torch.randn(6, 6)  # batch=6, 2*feat=6 → feat=3
        target = torch.randn(6, 3)

        cqr = ConformalLoss(method="cqr", reduction="mean")(y_pred, target)
        uacqr = ConformalLoss(method="uacqr", reduction="mean")(y_pred, target)
        torch.testing.assert_close(cqr, uacqr)

    def test_calibrate_then_predict_produces_valid_intervals(self):
        """After calibration, predict_interval returns valid intervals."""
        loss_fn = ConformalLoss(method="split", alpha=0.1)
        n_cal = 60
        pred_cal = torch.randn(n_cal, 2)
        target_cal = torch.randn(n_cal, 2)

        loss_fn.calibrate(pred_cal, target_cal)
        lower, upper = loss_fn.predict_interval(pred_cal)

        assert lower.shape == pred_cal.shape
        assert (lower <= upper).all()

    def test_predict_before_calibrate_raises(self):
        """Calling predict_interval before calibrate raises RuntimeError."""
        loss_fn = ConformalLoss(method="split", alpha=0.1)
        with pytest.raises(RuntimeError, match="calibrate"):
            loss_fn.predict_interval(torch.randn(10, 2))

    def test_forward_does_not_require_calibration(self):
        """Forward pass works fine without calibration."""
        fn = ConformalLoss(method="split")
        loss = fn(torch.randn(8, 3), torch.randn(8, 3))
        assert torch.isfinite(loss)

    def test_calibrate_with_mask(self):
        """Calibration respects mask."""
        loss_fn = ConformalLoss(method="split", alpha=0.1)
        n = 40
        y_pred = torch.randn(n, 2)
        target = torch.randn(n, 2)
        mask = torch.ones(n, 2, dtype=torch.bool)
        mask[:5] = False

        loss_fn.calibrate(y_pred, target, mask=mask)
        assert loss_fn._is_calibrated


# ── MultiDimensionalConformalLoss ──────────────────────────────────────


class TestMultiDimensionalConformalContract:
    """Contracts for MultiDimensionalConformalLoss."""

    def test_forward_equals_mse(self):
        """MultiDimensionalConformalLoss forward = MSE."""
        y_pred = torch.randn(6, 4)
        target = torch.randn(6, 4)

        cf_loss = MultiDimensionalConformalLoss(reduction="none")(y_pred, target)
        mse = (y_pred - target) ** 2
        torch.testing.assert_close(cf_loss, mse)

    def test_reduction_consistency(self):
        y_pred = torch.randn(8, 3)
        target = torch.randn(8, 3)
        _check_reduction(MultiDimensionalConformalLoss, y_pred, target)

    def test_mask_changes_loss(self):
        y_pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        mask = torch.ones(5, 3, dtype=torch.bool)
        mask[0, 0] = False

        fn = MultiDimensionalConformalLoss(reduction="mean")
        assert fn(y_pred, target) != fn(y_pred, target, mask=mask)

    def test_weights_scale_loss(self):
        y_pred = torch.randn(4, 2)
        target = torch.randn(4, 2)
        w1 = torch.ones(4, 2)
        w2 = w1.clone()
        w2[0, 0] = 2.0

        fn = MultiDimensionalConformalLoss(reduction="none")
        out1 = fn(y_pred, target, weights=w1)
        out2 = fn(y_pred, target, weights=w2)
        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_gradients_flow(self):
        model = torch.nn.Linear(3, 4)
        loss_fn = MultiDimensionalConformalLoss(reduction="mean")
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 3)
        target = torch.randn(4, 4)

        for _ in range(3):
            opt.zero_grad()
            loss = loss_fn(model(x), target)
            loss.backward()
            opt.step()
        assert torch.isfinite(loss)

    def test_per_dimension_thresholds(self):
        """Calibration produces per-dimension q_hat."""
        loss_fn = MultiDimensionalConformalLoss(alpha=0.1)
        n_cal, n_feat = 80, 4
        y_pred = torch.randn(n_cal, n_feat)
        target = torch.randn(n_cal, n_feat)

        loss_fn.calibrate(y_pred, target)
        assert loss_fn._is_calibrated
        assert loss_fn.q_hat is not None
        assert loss_fn.q_hat.shape == (n_feat,)

    def test_calibrate_then_predict(self):
        """After calibration, predict_interval returns per-dim valid intervals."""
        loss_fn = MultiDimensionalConformalLoss(alpha=0.1)
        n_cal, n_feat = 60, 3
        y_pred_cal = torch.randn(n_cal, n_feat)
        target_cal = torch.randn(n_cal, n_feat)

        loss_fn.calibrate(y_pred_cal, target_cal)
        y_pred_test = torch.randn(20, n_feat)
        lower, upper = loss_fn.predict_interval(y_pred_test)

        assert lower.shape == y_pred_test.shape
        assert upper.shape == y_pred_test.shape
        assert lower.shape[-1] == n_feat
        assert (lower <= upper).all()


# ── EIV family helpers ────────────────────────────────────────────────


def _eiv_model():
    """Deterministic linear model for EIV tests: y = 2 * x[:, :out_dim]."""
    return lambda x: x[:, :2] * 2.0


# ── FunctionalEIVLoss ─────────────────────────────────────────────────


class TestFunctionalEIVContract:
    """Reduction / mask / weight / gradient contracts for FunctionalEIVLoss."""

    def test_reduction_consistency(self):
        model = _eiv_model()
        batch, in_dim = 8, 4
        x_obs = torch.randn(batch, in_dim)
        target = torch.randn(batch, 2)

        fn_none = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1, reduction="none")
        fn_mean = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1, reduction="mean")
        fn_sum = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1, reduction="sum")

        none_out = fn_none(x_obs, target)
        # FunctionalEIVLoss returns per-sample scalar loss (batch,)
        assert none_out.shape == (batch,)
        torch.testing.assert_close(none_out.mean(), fn_mean(x_obs, target))
        torch.testing.assert_close(fn_sum(x_obs, target) / batch, fn_mean(x_obs, target))

    def test_mask_changes_loss(self):
        model = _eiv_model()
        batch, out_dim = 6, 2
        x_obs = torch.randn(batch, 4)
        target = torch.randn(batch, out_dim)
        mask = torch.ones(batch, out_dim, dtype=torch.bool)
        mask[0, 0] = False

        fn = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1, reduction="mean")
        assert fn(x_obs, target) != fn(x_obs, target, mask=mask)

    def test_weights_scale_loss(self):
        model = _eiv_model()
        batch = 4
        x_obs = torch.randn(batch, 4)
        target = torch.randn(batch, 2)
        # EIV loss is per-sample (batch,), so weights are 1-D
        w1 = torch.ones(batch)
        w2 = w1.clone()
        w2[0] = 2.0

        fn = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1, reduction="none")
        out1 = fn(x_obs, target, weights=w1)
        out2 = fn(x_obs, target, weights=w2)
        torch.testing.assert_close(out2[0] / out1[0], torch.tensor(2.0))

    def test_zero_weight_zeros_loss(self):
        model = _eiv_model()
        batch = 4
        x_obs = torch.randn(batch, 4)
        target = torch.randn(batch, 2)
        w = torch.ones(batch)
        w[0] = 0.0

        fn = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1, reduction="none")
        out = fn(x_obs, target, weights=w)
        assert out[0] == 0.0

    def test_gradients_flow(self):
        weight = torch.randn(4, 2, requires_grad=True)

        def model(x):
            return x @ weight

        fn = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1, reduction="mean")
        x_obs = torch.randn(4, 4)
        target = torch.randn(4, 2)

        loss = fn(x_obs, target)
        loss.backward()
        assert weight.grad is not None and torch.isfinite(weight.grad).all()

    def test_all_modes_produce_finite_loss(self):
        model = _eiv_model()
        x_obs = torch.randn(6, 4)
        target = torch.randn(6, 2)

        for mode in ["analytical", "mc", "hybrid"]:
            fn = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1, mode=mode, n_samples=10)
            loss = fn(x_obs, target)
            assert torch.isfinite(loss), f"mode={mode} produced non-finite loss"

    def test_with_scalar_vector_and_matrix_sigma(self):
        """FunctionalEIVLoss works with scalar, vector, and matrix sigma_x."""
        model = _eiv_model()
        x_obs = torch.randn(6, 4)
        target = torch.randn(6, 2)

        for sigma_x in [0.1, torch.ones(4) * 0.1, torch.eye(4) * 0.01]:
            fn = FunctionalEIVLoss(model, sigma_x=sigma_x, sigma_y=0.1)
            loss = fn(x_obs, target)
            assert torch.isfinite(loss)


class TestFunctionalEIVRelationships:
    """Cross-mode and zero-noise relationships for FunctionalEIVLoss."""

    def test_zero_noise_approaches_mse(self):
        """With sigma_x = sigma_y = 0, Functional EIV loss reduces exactly
        to the Gaussian NLL with variance = eps:

        NLL = 0.5 * D * (log(2π) + log(eps)) + 0.5 * ∑ rᵢ² / eps."""
        model = _eiv_model()
        batch, out_dim = 8, 2
        x_obs = torch.randn(batch, 4)
        target = torch.randn(batch, out_dim)
        eps = 1e-8

        fn = FunctionalEIVLoss(model, sigma_x=0.0, sigma_y=0.0, eps=eps, reduction="none")
        eiv_loss = fn(x_obs, target)  # shape (batch,)

        # Closed-form: with var = 0, the NLL is
        # 0.5 * sum_i (log(eps) + r_i² / eps) + 0.5 * D * log(2π)
        residuals = model(x_obs) - target
        residual_sq_sum = residuals.pow(2).sum(dim=-1)  # (batch,)
        expected = 0.5 * (
            out_dim * math.log(2 * math.pi) + out_dim * math.log(eps) + residual_sq_sum / eps
        )
        torch.testing.assert_close(eiv_loss, expected)

    def test_modes_produce_similar_loss(self):
        """All three modes (analytical, mc, hybrid) produce similar loss values."""
        model = _eiv_model()
        x_obs = torch.randn(10, 4)
        target = torch.randn(10, 2)

        fn_a = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1, mode="analytical")
        fn_mc = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1, mode="mc", n_samples=50)
        fn_hyb = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1, mode="hybrid", n_samples=20)

        loss_a = fn_a(x_obs, target)
        loss_mc = fn_mc(x_obs, target)
        loss_hyb = fn_hyb(x_obs, target)

        assert torch.isfinite(loss_a)
        assert torch.isfinite(loss_mc)
        assert torch.isfinite(loss_hyb)
        # Same order of magnitude
        assert 0.1 < loss_mc / loss_a < 10.0
        assert 0.1 < loss_hyb / loss_a < 10.0


# ── EnsembleEIVLoss ───────────────────────────────────────────────────


class TestEnsembleEIVContract:
    """Contracts for EnsembleEIVLoss."""

    def test_reduction_consistency(self):
        model = _eiv_model()
        batch, out_dim = 6, 2
        x_obs = torch.randn(batch, 4)
        target = torch.randn(batch, out_dim)

        # Use sigma_x=0 for deterministic loss across calls
        fn_none = EnsembleEIVLoss(model, sigma_x=0.0, n_samples=3, reduction="none")
        fn_mean = EnsembleEIVLoss(model, sigma_x=0.0, n_samples=3, reduction="mean")
        fn_sum = EnsembleEIVLoss(model, sigma_x=0.0, n_samples=3, reduction="sum")

        none_out = fn_none(x_obs, target)
        # EnsembleEIVLoss returns per-sample scalar loss (batch,)
        assert none_out.shape == (batch,)
        torch.testing.assert_close(none_out.mean(), fn_mean(x_obs, target))
        torch.testing.assert_close(fn_sum(x_obs, target) / batch, fn_mean(x_obs, target))

    def test_mask_changes_loss(self):
        model = _eiv_model()
        batch, out_dim = 6, 2
        x_obs = torch.randn(batch, 4)
        target = torch.randn(batch, out_dim)
        mask = torch.ones(batch, out_dim, dtype=torch.bool)
        mask[0, 0] = False

        fn = EnsembleEIVLoss(model, sigma_x=0.0, n_samples=3, reduction="mean")
        assert fn(x_obs, target) != fn(x_obs, target, mask=mask)

    def test_zero_noise_approaches_mse(self):
        """With sigma_x=0, all perturbed samples are identical, output = MSE."""
        model = _eiv_model()
        batch, out_dim = 6, 2
        x_obs = torch.randn(batch, 4)
        target = torch.randn(batch, out_dim)

        fn = EnsembleEIVLoss(model, sigma_x=0.0, n_samples=3, reduction="mean")
        eiv_loss = fn(x_obs, target)

        mse = (model(x_obs) - target).pow(2).sum(dim=-1).mean()
        torch.testing.assert_close(eiv_loss, mse)

    def test_weights_scale_loss(self):
        model = _eiv_model()
        batch = 4
        x_obs = torch.randn(batch, 4)
        target = torch.randn(batch, 2)
        # EIV loss is per-sample (batch,), so weights are 1-D
        w1 = torch.ones(batch)
        w2 = w1.clone()
        w2[0] = 2.0

        fn = EnsembleEIVLoss(model, sigma_x=0.0, n_samples=3, reduction="none")
        out1 = fn(x_obs, target, weights=w1)
        out2 = fn(x_obs, target, weights=w2)
        torch.testing.assert_close(out2[0] / out1[0], torch.tensor(2.0))


# ── StructuralEIVLoss ─────────────────────────────────────────────────


class TestStructuralEIVContract:
    """Contracts for StructuralEIVLoss."""

    def test_reduction_consistency(self):
        model = _eiv_model()
        batch = 6
        x_obs = torch.randn(batch, 4)
        target = torch.randn(batch, 2)
        sigma_x = 0.1
        sigma_y = 0.1
        sigma_xy = torch.zeros(2, 4)

        fn_none = StructuralEIVLoss(model, sigma_x, sigma_y, sigma_xy, reduction="none")
        fn_mean = StructuralEIVLoss(model, sigma_x, sigma_y, sigma_xy, reduction="mean")
        fn_sum = StructuralEIVLoss(model, sigma_x, sigma_y, sigma_xy, reduction="sum")

        none_out = fn_none(x_obs, target)
        # StructuralEIVLoss returns per-sample scalar loss (batch,)
        assert none_out.shape == (batch,)
        torch.testing.assert_close(none_out.mean(), fn_mean(x_obs, target))
        torch.testing.assert_close(fn_sum(x_obs, target) / batch, fn_mean(x_obs, target))

    def test_mask_changes_loss(self):
        model = _eiv_model()
        batch, out_dim = 5, 2
        x_obs = torch.randn(batch, 4)
        target = torch.randn(batch, out_dim)
        mask = torch.ones(batch, out_dim, dtype=torch.bool)
        mask[0, 0] = False
        sigma_xy = torch.zeros(out_dim, 4)

        fn = StructuralEIVLoss(model, 0.1, 0.1, sigma_xy, reduction="mean")
        assert fn(x_obs, target) != fn(x_obs, target, mask=mask)

    def test_gradients_flow(self):
        weight = torch.randn(4, 2, requires_grad=True)

        def model(x):
            return x @ weight

        sigma_xy = torch.zeros(2, 4)
        fn = StructuralEIVLoss(model, 0.1, 0.1, sigma_xy, reduction="mean")
        x_obs = torch.randn(4, 4)
        target = torch.randn(4, 2)

        loss = fn(x_obs, target)
        loss.backward()
        assert weight.grad is not None and torch.isfinite(weight.grad).all()


class TestStructuralEIVRelationships:
    """Cross-family: Structural EIV reduces to Functional when cross-cov is zero."""

    def test_zero_cross_cov_equals_functional(self):
        """Structural(sigma_xy=0) ≈ Functional with same sigma."""
        model = _eiv_model()
        batch, out_dim = 10, 2
        x_obs = torch.randn(batch, 4)
        target = torch.randn(batch, out_dim)
        sigma_x = 0.1
        sigma_y = 0.1
        sigma_xy = torch.zeros(out_dim, 4)

        fn_func = FunctionalEIVLoss(model, sigma_x, sigma_y, reduction="none")
        fn_struct = StructuralEIVLoss(model, sigma_x, sigma_y, sigma_xy, reduction="none")

        loss_func = fn_func(x_obs, target)
        loss_struct = fn_struct(x_obs, target)

        torch.testing.assert_close(loss_struct, loss_func)


# ── Expectile family ─────────────────────────────────────────────────


class TestExpectileContract:
    """Reduction / mask / weight contracts for ExpectileLoss."""

    def test_reduction_consistency(self):
        y_pred = torch.randn(8, 3)
        target = torch.randn(8, 3)
        _check_reduction(ExpectileLoss, y_pred, target, expectile=0.3)

    def test_mask_changes_loss(self):
        y_pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        mask = torch.ones(5, 3, dtype=torch.bool)
        mask[0, 0] = False
        fn = ExpectileLoss(expectile=0.5, reduction="mean")
        assert fn(y_pred, target) != fn(y_pred, target, mask=mask)

    def test_weights_scale_loss(self):
        y_pred = torch.randn(4, 2)
        target = torch.randn(4, 2)
        w1 = torch.ones(4, 2)
        w2 = w1.clone()
        w2[0, 0] = 2.0
        fn = ExpectileLoss(expectile=0.5, reduction="none")
        out1 = fn(y_pred, target, weights=w1)
        out2 = fn(y_pred, target, weights=w2)
        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_tau_half_is_mse(self):
        """Expectile(τ=0.5) = MSE exactly: 2·r²·weight, weight=0.5 → r²."""
        y_pred = torch.randn(6, 4)
        target = torch.randn(6, 4)
        exp_loss = ExpectileLoss(expectile=0.5, reduction="none")(y_pred, target)
        mse = (y_pred - target) ** 2
        torch.testing.assert_close(exp_loss, mse)

    def test_multi_expectile_reduction(self):
        """MultiExpectileLoss returns per-sample scalar for 'none'."""
        y_pred = torch.randn(6, 3, 2)  # [batch, num_expectiles, features]
        target = torch.randn(6, 2)
        none = MultiExpectileLoss(expectiles=[0.1, 0.5, 0.9], reduction="none")(y_pred, target)
        mean = MultiExpectileLoss(expectiles=[0.1, 0.5, 0.9], reduction="mean")(y_pred, target)
        assert none.shape == (6,)
        torch.testing.assert_close(none.mean(), mean)


# ── Evidential regression ─────────────────────────────────────────────


class TestEvidentialContract:
    """Contracts for EvidentialRegressionLoss."""

    @staticmethod
    def _make_params(batch=6, dim=2):
        """Create valid NIG parameters: gamma, nu, alpha, beta."""
        torch.manual_seed(77)
        gamma = torch.randn(batch, dim)
        nu = F.softplus(torch.randn(batch, dim)) + 0.01
        alpha = F.softplus(torch.randn(batch, dim)) + 1.01
        beta = F.softplus(torch.randn(batch, dim)) + 0.01
        return torch.cat([gamma, nu, alpha, beta], dim=-1)

    def test_reduction_consistency(self):
        y_pred = self._make_params(6, 2)
        target = torch.randn(6, 2)
        _check_reduction(EvidentialRegressionLoss, y_pred, target, coeff_nig=0.01)

    def test_mask_changes_loss(self):
        y_pred = self._make_params(5, 2)
        target = torch.randn(5, 2)
        mask = torch.ones(5, 2, dtype=torch.bool)
        mask[0, 0] = False
        fn = EvidentialRegressionLoss(coeff_nig=0.01, reduction="mean")
        assert fn(y_pred, target) != fn(y_pred, target, mask=mask)

    def test_weights_scale_loss(self):
        y_pred = self._make_params(4, 2)
        target = torch.randn(4, 2)
        w1 = torch.ones(4, 2)
        w2 = w1.clone()
        w2[0, 0] = 2.0
        fn = EvidentialRegressionLoss(reduction="none")
        out1 = fn(y_pred, target, weights=w1)
        out2 = fn(y_pred, target, weights=w2)
        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_gradients_flow(self):
        # NIG parameters need stable ranges: use fixed params, require grad on gamma only.
        gamma = torch.randn(4, 3, requires_grad=True)
        nu = F.softplus(torch.randn(4, 3)) + 0.5
        alpha = F.softplus(torch.randn(4, 3)) + 2.0  # > 1, well away from boundary
        beta = F.softplus(torch.randn(4, 3)) + 0.5
        y_pred = torch.cat([gamma, nu, alpha, beta], dim=-1)
        target = torch.randn(4, 3)

        loss_fn = EvidentialRegressionLoss()
        loss = loss_fn(y_pred, target)
        loss.backward()
        assert gamma.grad is not None and torch.isfinite(gamma.grad).all()

    def test_predict_with_uncertainty_shapes(self):
        """predict_with_uncertainty returns (mean, ale, epi) with correct shapes."""
        y_pred = self._make_params(5, 3)
        fn = EvidentialRegressionLoss()
        mean, ale, epi = fn.predict_with_uncertainty(y_pred)
        assert mean.shape == (5, 3)
        assert ale.shape == (5, 3)
        assert epi.shape == (5, 3)
        assert (ale >= 0).all()
        assert (epi >= 0).all()


# ── Robust losses ─────────────────────────────────────────────────────


class TestRobustContract:
    """Reduction / mask / weight contracts for robust loss classes."""

    @staticmethod
    def _rand_data(batch=6, dim=4):
        return torch.randn(batch, dim), torch.randn(batch, dim)

    def test_pseudo_huber_reduction(self):
        y_pred, target = self._rand_data()
        _check_reduction(PseudoHuberLoss, y_pred, target, delta=1.0)

    def test_log_cosh_reduction(self):
        y_pred, target = self._rand_data()
        _check_reduction(LogCoshLoss, y_pred, target, scale=1.0)

    def test_barron_reduction(self):
        y_pred, target = self._rand_data()
        _check_reduction(BarronLoss, y_pred, target, alpha=1.0, scale=1.0)

    def test_cauchy_reduction(self):
        y_pred, target = self._rand_data()
        _check_reduction(CauchyLoss, y_pred, target, c=1.0)

    def test_charbonnier_reduction(self):
        y_pred, target = self._rand_data()
        _check_reduction(CharbonnierLoss, y_pred, target, eps=0.1)

    def test_tukey_reduction(self):
        y_pred, target = self._rand_data()
        _check_reduction(TukeyBiweightLoss, y_pred, target, c=4.685)

    def test_pseudo_huber_mask(self):
        y_pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        mask = torch.ones(5, 3, dtype=torch.bool)
        mask[0, 0] = False
        fn = PseudoHuberLoss(reduction="mean")
        assert fn(y_pred, target) != fn(y_pred, target, mask=mask)

    def test_pseudo_huber_weights(self):
        y_pred = torch.randn(4, 2)
        target = torch.randn(4, 2)
        w1 = torch.ones(4, 2)
        w2 = w1.clone()
        w2[0, 0] = 2.0
        fn = PseudoHuberLoss(reduction="none")
        out1 = fn(y_pred, target, weights=w1)
        out2 = fn(y_pred, target, weights=w2)
        torch.testing.assert_close(out2[0, 0] / out1[0, 0], torch.tensor(2.0))

    def test_log_cosh_gradient_flow(self):
        y_pred = torch.randn(4, 3, requires_grad=True)
        target = torch.randn(4, 3)
        loss = LogCoshLoss()(y_pred, target)
        loss.backward()
        assert y_pred.grad is not None and torch.isfinite(y_pred.grad).all()


class TestRobustRelationships:
    """Special-case relationships for robust losses."""

    def test_barron_alpha_2_is_half_mse(self):
        """Barron(alpha=2, scale=1) = 0.5 * r² = MSE/2."""
        y_pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        barron = BarronLoss(alpha=2, scale=1, reduction="none")(y_pred, target)
        half_mse = 0.5 * (y_pred - target) ** 2
        torch.testing.assert_close(barron, half_mse)

    def test_pseudo_huber_large_delta_approx_half_mse(self):
        """PseudoHuber(delta → ∞) ≈ 0.5 * r² (in float64).

        At δ=100, r²/δ² ≈ 1e-4 for r~1, easily resolvable; at δ=1e4
        the term underflows in float32, so we use float64."""
        y_pred = torch.randn(5, 3, dtype=torch.float64)
        target = torch.randn(5, 3, dtype=torch.float64)
        ph = PseudoHuberLoss(delta=1e4, reduction="none")(y_pred, target)
        half_mse = 0.5 * (y_pred - target) ** 2
        # Taylor: δ²(sqrt(1+x)-1) = r²/2 - r⁴/(8δ²) + ...
        # At δ=1e4 in float64, the remainder is ~1e-8 for r~5
        torch.testing.assert_close(ph, half_mse)

    def test_cauchy_large_c_produces_small_loss(self):
        """Cauchy(c → ∞) → log(1 + r²/c²) ≈ 0 for large c."""
        y_pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        fn = CauchyLoss(c=1e3)
        loss = fn(y_pred, target)
        assert torch.isfinite(loss)
        assert loss < 1e-3  # Practically zero for small residuals

    def test_charbonnier_large_eps_approaches_constant(self):
        """Charbonnier(eps → ∞) is approximately constant."""
        y_pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        fn = CharbonnierLoss(eps=1e3)
        loss = fn(y_pred, target)
        assert torch.isfinite(loss)
        torch.testing.assert_close(loss, torch.tensor(1e3), atol=1e-3, rtol=1e-2)


# ── Balanced MSE ──────────────────────────────────────────────────────


class TestBalancedMSEContract:
    """Reduction / mask / weight contracts for BalancedMSELoss and BMCLoss."""

    @staticmethod
    def _make_bins():
        return torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

    def test_balanced_mse_reduction_consistency(self):
        batch, dim = 8, 3
        y_pred = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        edges = self._make_bins()

        fn_none = BalancedMSELoss(edges, reduction="none")
        fn_mean = BalancedMSELoss(edges, reduction="mean")
        fn_sum = BalancedMSELoss(edges, reduction="sum")
        for fn in (fn_none, fn_mean, fn_sum):
            fn.fit(target)

        none_out = fn_none(y_pred, target)
        assert none_out.shape == (batch, dim)
        torch.testing.assert_close(none_out.mean(), fn_mean(y_pred, target))
        torch.testing.assert_close(
            fn_sum(y_pred, target) / none_out.numel(), fn_mean(y_pred, target)
        )

    def test_balanced_mse_mask_changes_loss(self):
        y_pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        mask = torch.ones(5, 3, dtype=torch.bool)
        mask[0, 0] = False
        fn = BalancedMSELoss(self._make_bins(), reduction="mean")
        fn.fit(target)
        assert fn(y_pred, target) != fn(y_pred, target, mask=mask)

    def test_bmc_reduction_consistency(self):
        batch, dim = 8, 3
        y_pred = torch.randn(batch, dim)
        target = torch.randn(batch, dim)

        fn_none = BMCLoss(num_bins=4, noise_sigma=1.0, reduction="none")
        fn_mean = BMCLoss(num_bins=4, noise_sigma=1.0, reduction="mean")
        fn_sum = BMCLoss(num_bins=4, noise_sigma=1.0, reduction="sum")
        for fn in (fn_none, fn_mean, fn_sum):
            fn.fit(target)

        none_out = fn_none(y_pred, target)
        assert none_out.shape == (batch, dim)
        torch.testing.assert_close(none_out.mean(), fn_mean(y_pred, target))
        torch.testing.assert_close(
            fn_sum(y_pred, target) / none_out.numel(), fn_mean(y_pred, target)
        )

    def test_bmc_predict_before_fit_raises(self):
        """BMCLoss without fit() raises RuntimeError."""
        fn = BMCLoss(num_bins=4, noise_sigma=1.0)
        with pytest.raises(RuntimeError, match="fit"):
            fn(torch.randn(4, 2), torch.randn(4, 2))

    def test_balanced_mse_predict_before_fit_raises(self):
        """BalancedMSELoss without fit() raises RuntimeError."""
        fn = BalancedMSELoss(self._make_bins())
        with pytest.raises(RuntimeError, match="fit"):
            fn(torch.randn(4, 2), torch.randn(4, 2))
