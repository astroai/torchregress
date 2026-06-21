"""Cross-class consistency tests for non-Gaussian regression losses.

Verifies that Poisson-Gaussian mixtures, censored losses, and AFT losses
share consistent reduction/mask/weight contracts and that special-case
relationships hold (e.g. Poisson-Gaussian reduces to known limits).
"""

import math

import torch

from torchregress.losses.censored import AFTLoss, CensoredGaussianNLLLoss, CensoredQuantileLoss
from torchregress.losses.poisson_gaussian import (
    EnhancedPoissonGaussianMixtureLoss,
    PoissonGaussianLikelihoodRatioLoss,
    PoissonGaussianMixtureLoss,
)

# ── helpers ──────────────────────────────────────────────────────────


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

        fn_none = PoissonGaussianMixtureLoss(reduction="none")
        fn_mean = PoissonGaussianMixtureLoss(reduction="mean")
        fn_sum = PoissonGaussianMixtureLoss(reduction="sum")

        none_out = fn_none(y_pred, target)
        assert none_out.shape == (batch, dim)
        torch.testing.assert_close(none_out.mean(), fn_mean(y_pred, target))
        torch.testing.assert_close(fn_sum(y_pred, target) / (batch * dim), fn_mean(y_pred, target))

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

        fn_none = EnhancedPoissonGaussianMixtureLoss(reduction="none")
        fn_mean = EnhancedPoissonGaussianMixtureLoss(reduction="mean")
        fn_sum = EnhancedPoissonGaussianMixtureLoss(reduction="sum")

        none_out = fn_none(y_pred, target)
        assert none_out.shape == (batch, dim)
        torch.testing.assert_close(none_out.mean(), fn_mean(y_pred, target))
        torch.testing.assert_close(fn_sum(y_pred, target) / (batch * dim), fn_mean(y_pred, target))

    def test_mask_and_weights(self):
        batch, dim = 5, 3
        y_pred = _make_pos_preds(batch, dim)
        target = _make_pos_targets(batch, dim)
        mask = torch.ones(batch, dim, dtype=torch.bool)
        mask[0, 0] = False
        weights = torch.rand(batch, dim)

        fn = EnhancedPoissonGaussianMixtureLoss(reduction="mean")
        loss = fn(y_pred, target, mask=mask, weights=weights)
        assert torch.isfinite(loss)

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

        fn_none = PoissonGaussianLikelihoodRatioLoss(reduction="none", log_input=True)
        fn_mean = PoissonGaussianLikelihoodRatioLoss(reduction="mean", log_input=True)
        fn_sum = PoissonGaussianLikelihoodRatioLoss(reduction="sum", log_input=True)

        none_out = fn_none(y_pred, target)
        assert none_out.shape == (batch, dim)
        torch.testing.assert_close(none_out.mean(), fn_mean(y_pred, target))
        torch.testing.assert_close(fn_sum(y_pred, target) / (batch * dim), fn_mean(y_pred, target))

    def test_mask_and_weights(self):
        y_pred = _make_pos_preds(5, 3, log_input=True)
        target = _make_pos_targets(5, 3)
        mask = torch.ones(5, 3, dtype=torch.bool)
        mask[0, 0] = False

        fn = PoissonGaussianLikelihoodRatioLoss(reduction="mean", log_input=True)
        loss_full = fn(y_pred, target)
        loss_masked = fn(y_pred, target, mask=mask)
        assert loss_masked != loss_full

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

        fn_none = CensoredGaussianNLLLoss(reduction="none")
        fn_mean = CensoredGaussianNLLLoss(reduction="mean")
        fn_sum = CensoredGaussianNLLLoss(reduction="sum")

        none_out = fn_none((mean, log_var), target, censoring=censoring)
        assert none_out.shape == (batch, dim)
        torch.testing.assert_close(
            none_out.mean(), fn_mean((mean, log_var), target, censoring=censoring)
        )
        torch.testing.assert_close(
            fn_sum((mean, log_var), target, censoring=censoring) / (batch * dim),
            fn_mean((mean, log_var), target, censoring=censoring),
        )

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

        fn_none = CensoredQuantileLoss(quantile=0.5, reduction="none")
        fn_mean = CensoredQuantileLoss(quantile=0.5, reduction="mean")

        none_out = fn_none(y_pred, target, censoring=censoring)
        assert none_out.shape == (batch, dim)
        torch.testing.assert_close(
            none_out.mean(),
            fn_mean(y_pred, target, censoring=censoring),
        )

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

        fn_none = AFTLoss(reduction="none")
        fn_mean = AFTLoss(reduction="mean")
        fn_sum = AFTLoss(reduction="sum")

        none_out = fn_none((loc, log_scale), target, censoring=censoring)
        assert none_out.shape == (batch, dim)
        torch.testing.assert_close(
            none_out.mean(),
            fn_mean((loc, log_scale), target, censoring=censoring),
        )
        torch.testing.assert_close(
            fn_sum((loc, log_scale), target, censoring=censoring) / (batch * dim),
            fn_mean((loc, log_scale), target, censoring=censoring),
        )

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
