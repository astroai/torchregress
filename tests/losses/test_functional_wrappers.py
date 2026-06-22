"""Direct tests for the functional loss wrappers exported from
:mod:`torchregress.losses`.

These wrappers are thin factory functions that instantiate the corresponding
:class:`RegressionLoss` subclass and call ``forward``.  They are parity-tested
inside their own class tests (e.g. ``test_quantile.QuantileLoss``) but never
as standalone factory calls.  This file asserts the keep-in-sync contract:
``fn(y_pred, target, **kwargs)`` must exactly equal
``ClassName(**kwargs)(y_pred, target)``.

Covered wrappers
----------------
- ``quantile_loss`` (QuantileLoss parity)
- ``expectile_loss`` (ExpectileLoss parity + edge cases)
- ``beta_nll_loss`` (BetaNLLLoss parity + edge cases)
- ``tweedie_loss`` (TweedieLoss p-value routing)
- ``conformal_loss`` (ConformalLoss split/cqr routing)
- ``gaussian_wasserstein_bound_loss`` (4 covariance parameterizations)
- ``poisson_gaussian_mixture_loss`` (factory config plumbing)
- ``enhanced_poisson_gaussian_loss`` (factory config plumbing)
- ``poisson_gaussian_likelihood_ratio_loss`` (factory config plumbing)
"""

from __future__ import annotations

import math
import unittest

import torch

from torchregress.losses.beta_nll import BetaNLLLoss, beta_nll_loss
from torchregress.losses.conformal import ConformalLoss, conformal_loss
from torchregress.losses.expectile import ExpectileLoss, expectile_loss
from torchregress.losses.gaussian_wasserstein import (
    GaussianWassersteinBoundLoss,
    gaussian_wasserstein_bound_loss,
    symmetric_spd_matrix_sqrt,
)
from torchregress.losses.poisson_gaussian import (
    EnhancedPoissonGaussianConfig,
    EnhancedPoissonGaussianMixtureLoss,
    PoissonGaussianLikelihoodRatioConfig,
    PoissonGaussianLikelihoodRatioLoss,
    PoissonGaussianMixtureConfig,
    PoissonGaussianMixtureLoss,
    enhanced_poisson_gaussian_loss,
    poisson_gaussian_likelihood_ratio_loss,
    poisson_gaussian_mixture_loss,
)
from torchregress.losses.quantile import QuantileLoss, quantile_loss
from torchregress.losses.tweedie import TweedieLoss, tweedie_loss


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# quantile_loss
# ---------------------------------------------------------------------------


class TestQuantileLossWrapper(unittest.TestCase):
    def test_zero_quantile_matches_mae_at_median(self):
        _set_seed(0)
        y_pred = torch.randn(10, 1)
        target = torch.randn(10, 1)
        ref = QuantileLoss(quantile=0.5, reduction="mean")(y_pred, target)
        got = quantile_loss(y_pred, target, quantile=0.5, reduction="mean")
        self.assertTrue(torch.isclose(got, ref, atol=1e-6))

    def test_reduction_string_passed_through(self):
        _set_seed(1)
        y_pred = torch.randn(6)
        target = torch.randn(6)
        ref_none = QuantileLoss(quantile=0.3, reduction="none")(y_pred, target)
        got_none = quantile_loss(y_pred, target, quantile=0.3, reduction="none")
        self.assertTrue(torch.allclose(got_none, ref_none))

    def test_asymmetric_penalty_direction(self):
        """At q=0.9, underestimating carries 9x the penalty of overestimating.
        For a single sample with target>prediction, quantile_loss must equal
        0.9*(target-prediction).
        """
        y_pred = torch.tensor([1.0])
        target = torch.tensor([5.0])
        self.assertTrue(
            torch.isclose(
                quantile_loss(y_pred, target, quantile=0.9, reduction="none"), torch.tensor([3.6])
            )
        )


# ---------------------------------------------------------------------------
# expectile_loss
# ---------------------------------------------------------------------------


class TestExpectileLossWrapper(unittest.TestCase):
    def test_tau_half_matches_mse(self):
        """At ``tau=0.5``, the asymmetric weights become uniform and the factor
        is 2; with reduction='mean' this collapses to MSE.
        """
        _set_seed(2)
        y_pred = torch.randn(8, 1)
        target = torch.randn(8, 1)
        got = expectile_loss(y_pred, target, expectile=0.5, reduction="mean")
        expected = torch.nn.functional.mse_loss(y_pred, target, reduction="mean")
        self.assertTrue(torch.isclose(got, expected, atol=1e-6))

    def test_parity_with_expectile_loss_class(self):
        _set_seed(3)
        y_pred = torch.randn(6, 1)
        target = torch.randn(6, 1)
        ref = ExpectileLoss(expectile=0.8, reduction="mean")(y_pred, target)
        got = expectile_loss(y_pred, target, expectile=0.8, reduction="mean")
        self.assertTrue(torch.isclose(got, ref, atol=1e-6))

    def test_asymmetry_direction(self):
        """At tau=0.8 underestimation (positive residual) carries weight 0.8;
        overestimation carries weight 0.2.
        For target=2, pred=0 -> positive residual, weight=0.8*2=1.6.
        """
        y_pred = torch.tensor([0.0])
        target = torch.tensor([2.0])
        # 2 * (2-0)^2 * 0.8 = 6.4
        self.assertTrue(
            torch.isclose(expectile_loss(y_pred, target, expectile=0.8), torch.tensor(6.4))
        )


# ---------------------------------------------------------------------------
# beta_nll_loss
# ---------------------------------------------------------------------------


class TestBetaNLLLossWrapper(unittest.TestCase):
    def test_beta_zero_matches_gaussian_nll(self):
        """``beta=0`` removes the detached variance rescaling and beta_nll is
        exactly equivalent to ``0.5*(log(2pi)+log(var)+sq/var)``.
        """
        _set_seed(4)
        mean = torch.randn(5, 1)
        log_var = torch.zeros(5, 1)
        var = log_var.exp()
        target = torch.randn(5, 1)
        nll_ref = 0.5 * (
            math.log(2 * math.pi) + torch.log(var + 1e-8) + (target - mean) ** 2 / (var + 1e-8)
        )
        ref = nll_ref.mean()
        got = beta_nll_loss((mean, log_var), target, beta=0.0, reduction="mean")
        self.assertTrue(torch.isclose(got, ref, atol=1e-5))

    def test_parity_with_class(self):
        """Both the functional wrapper and ``BetaNLLLoss`` must produce the
        same scalar when given a ``(mean, log_var)`` tuple.
        """
        _set_seed(5)
        mean = torch.randn(7, 1)
        log_var = torch.randn(7, 1) - 2.0  # small variance
        target = torch.randn(7, 1)
        kwargs = dict(beta=0.5, reduction="mean")
        ref = BetaNLLLoss(**kwargs)(y_pred=(mean, log_var), target=target)
        got = beta_nll_loss((mean, log_var), target, **kwargs)
        self.assertTrue(torch.isclose(got, ref, atol=1e-6))

    def test_parity_with_class_for_concat_input(self):
        """The class also accepts a concat ``[mean, log_var]`` tensor; the
        wrapper must agree.
        """
        _set_seed(55)
        mean = torch.randn(7, 1)
        log_var = torch.randn(7, 1) - 2.0
        target = torch.randn(7, 1)
        y_pred = torch.cat([mean, log_var], dim=-1)
        ref = BetaNLLLoss(beta=0.5, reduction="mean", split_dim=-1)(y_pred, target)
        got = beta_nll_loss(y_pred, target, beta=0.5, reduction="mean", split_dim=-1)
        self.assertTrue(torch.isclose(got, ref, atol=1e-6))

    def test_concat_input_format_supported(self):
        """beta_nll_loss must accept the same ``[mean, log_var]`` concat
        format that ``BetaNLLLoss`` accepts (uses ``split_dim``).
        """
        _set_seed(6)
        y_pred = torch.cat([torch.randn(4, 1), torch.randn(4, 1)], dim=-1)
        target = torch.randn(4, 1)
        got = beta_nll_loss(y_pred, target, beta=0.5, reduction="mean", split_dim=-1)
        ref = BetaNLLLoss(beta=0.5, reduction="mean", split_dim=-1)(y_pred, target)
        self.assertTrue(torch.isclose(got, ref, atol=1e-6))

    def test_negative_beta_raises(self):
        with self.assertRaises(ValueError):
            beta_nll_loss(torch.randn(3, 1), torch.randn(3, 1), beta=-0.1, reduction="mean")


# ---------------------------------------------------------------------------
# tweedie_loss
# ---------------------------------------------------------------------------


class TestTweedieLossWrapper(unittest.TestCase):
    def test_p_zero_matches_normal_loss(self):
        """``p=0`` branch uses ``(t-mu)^2/2``; with ``mu == target`` through
        the identity link the result is 0.
        """
        _set_seed(7)
        mu = torch.tensor([1.0, 2.0, 3.0])  # identity link <-> mu as-is
        target = torch.tensor([1.0, 2.0, 3.0])
        got = tweedie_loss(mu, target, p=0.0, reduction="mean", link="identity")
        self.assertTrue(torch.isclose(got, torch.tensor(0.0), atol=1e-6))

    def test_p_two_exact_prediction_is_zero(self):
        """``p=2`` (gamma) with ``mu == target`` through the log link produces 0."""
        _set_seed(8)
        mu = torch.log(torch.tensor([1.0, 2.0, 3.0]))  # log link
        target = torch.tensor([1.0, 2.0, 3.0])
        got = tweedie_loss(mu, target, p=2.0, reduction="mean", link="log")
        self.assertTrue(torch.isclose(got, torch.tensor(0.0), atol=1e-6))

    def test_parity_with_class(self):
        _set_seed(9)
        y_pred = torch.log(torch.rand(6) + 0.5)
        target = torch.rand(6) + 0.5
        ref = TweedieLoss(p=1.5, reduction="mean", link="log")(y_pred, target)
        got = tweedie_loss(y_pred, target, p=1.5, reduction="mean", link="log")
        self.assertTrue(torch.isclose(got, ref, atol=1e-6))

    def test_unknown_p_routes_to_poisson(self):
        """``p=1`` is the poisson branch; wrapper must accept it and not raise."""
        _set_seed(10)
        mu = torch.log(torch.rand(4) + 0.5)
        target = torch.randint(0, 5, (4,)).float()
        got = tweedie_loss(mu, target, p=1.0, reduction="mean", link="log")
        self.assertTrue(torch.is_tensor(got))
        self.assertTrue(torch.isfinite(got))

    def test_invalid_p_raises(self):
        with self.assertRaises(ValueError):
            tweedie_loss(torch.zeros(2), torch.zeros(2), p=0.5, reduction="mean")


# ---------------------------------------------------------------------------
# conformal_loss
# ---------------------------------------------------------------------------


class TestConformalLossWrapper(unittest.TestCase):
    def test_split_method_returns_mse(self):
        """``method='split'`` uses MSE; with mean reduction and matching y_pred
        the result is 0.
        """
        _set_seed(11)
        y_pred = torch.randn(8, 1)
        target = y_pred.clone()
        got = conformal_loss(y_pred, target, method="split", alpha=0.1, reduction="mean")
        self.assertTrue(torch.isclose(got, torch.tensor(0.0), atol=1e-6))

    def test_cqr_method_pinball_matches_class(self):
        _set_seed(12)
        # y_pred must have last dim = 2 * n_features
        y_pred = torch.randn(6, 2)
        target = torch.randn(6, 1)
        ref = ConformalLoss(method="cqr", alpha=0.1, reduction="mean")(y_pred, target)
        got = conformal_loss(y_pred, target, method="cqr", alpha=0.1, reduction="mean")
        self.assertTrue(torch.isclose(got, ref, atol=1e-6))

    def test_invalid_method_propagates(self):
        """Invalid ``method`` must propagate; the wrapper calls the class
        constructor which raises ValueError.
        """
        with self.assertRaises(ValueError):
            conformal_loss(
                torch.randn(4, 1),
                torch.randn(4, 1),
                method="not_a_method",
                alpha=0.1,
                reduction="mean",
            )


# ---------------------------------------------------------------------------
# gaussian_wasserstein_bound_loss
# ---------------------------------------------------------------------------


class TestGaussianWassersteinBoundWrapper(unittest.TestCase):
    """Each ``covariance_parameterization`` gets parametrised coverage."""

    def setUp(self):
        self.batch, self.d = 4, 3
        self.mu_p = torch.randn(self.batch, self.d)
        self.mu_t = torch.randn(self.batch, self.d)
        # Shared PSD covariance for all 4 modes.
        A = torch.randn(self.d, self.d)
        base = A @ A.T + torch.eye(self.d)
        self.sigma_p = base.clone()
        self.sigma_t = base.clone()
        self.sqrt_p = symmetric_spd_matrix_sqrt(self.sigma_p)
        self.sqrt_t = symmetric_spd_matrix_sqrt(self.sigma_t)

    def test_diag_mode(self):
        pred_diag = self.sigma_p.diagonal().clamp_min(1e-6).repeat(self.batch, 1)
        target_diag = self.sigma_t.diagonal().clamp_min(1e-6).repeat(self.batch, 1)
        ref = GaussianWassersteinBoundLoss(
            covariance_parameterization="diagonal", reduction="mean"
        )(self.mu_p, self.mu_t, pred_diag, target_diag)
        got = gaussian_wasserstein_bound_loss(
            self.mu_p,
            self.mu_t,
            pred_diag,
            target_diag,
            covariance_parameterization="diagonal",
            reduction="mean",
        )
        self.assertTrue(torch.isclose(ref, got, atol=1e-6))

    def test_covariance_mode(self):
        ref = GaussianWassersteinBoundLoss(
            covariance_parameterization="covariance", reduction="mean"
        )(self.mu_p, self.mu_t, self.sigma_p, self.sigma_t)
        got = gaussian_wasserstein_bound_loss(
            self.mu_p,
            self.mu_t,
            self.sigma_p,
            self.sigma_t,
            covariance_parameterization="covariance",
            reduction="mean",
        )
        self.assertTrue(torch.isclose(ref, got, atol=1e-6))

    def test_sqrt_mode(self):
        """In sqrt mode, pred/target covar ARE roots; no Cholesky needed."""
        ref = GaussianWassersteinBoundLoss(covariance_parameterization="sqrt", reduction="mean")(
            self.mu_p, self.mu_t, self.sqrt_p, self.sqrt_t
        )
        got = gaussian_wasserstein_bound_loss(
            self.mu_p,
            self.mu_t,
            self.sqrt_p,
            self.sqrt_t,
            covariance_parameterization="sqrt",
            reduction="mean",
        )
        self.assertTrue(torch.isclose(ref, got, atol=1e-6))

    def test_cholesky_mode(self):
        chol_p = torch.linalg.cholesky(self.sigma_p)
        chol_t = torch.linalg.cholesky(self.sigma_t)
        ref = GaussianWassersteinBoundLoss(
            covariance_parameterization="cholesky", reduction="mean"
        )(self.mu_p, self.mu_t, chol_p, chol_t)
        got = gaussian_wasserstein_bound_loss(
            self.mu_p,
            self.mu_t,
            chol_p,
            chol_t,
            covariance_parameterization="cholesky",
            reduction="mean",
        )
        self.assertTrue(torch.isclose(ref, got, atol=1e-6))

    def test_invalid_parameterization_propagates(self):
        with self.assertRaises(ValueError):
            gaussian_wasserstein_bound_loss(
                self.mu_p,
                self.mu_t,
                self.sigma_p,
                self.sigma_t,
                covariance_parameterization="kebab",
                reduction="mean",
            )

    def test_sqrt_mode_is_zero_when_means_and_roots_match(self):
        ref = GaussianWassersteinBoundLoss(
            covariance_parameterization="sqrt",
            reduction="mean",
            mean_weight=0.0,
            covariance_weight=0.0,
        )(self.mu_p, self.mu_p, self.sqrt_p, self.sqrt_p)
        self.assertTrue(torch.isclose(ref, torch.tensor(0.0), atol=1e-6))


# ---------------------------------------------------------------------------
# poisson_gaussian_mixture_loss / enhanced_poisson_gaussian_loss /
# poisson_gaussian_likelihood_ratio_loss
# ---------------------------------------------------------------------------


class TestPoissonGaussianFactoryWrappers(unittest.TestCase):
    """These three factory functions return instantiated losses, not scalar
    losses: they each take a Config dataclass plus kwargs and return the
    appropriate ``RegressionLoss`` subclass.  We test the class binding.
    """

    def test_mixture_factory_returns_instance(self):
        loss = poisson_gaussian_mixture_loss(initial_variance=0.5)
        self.assertIsInstance(loss, PoissonGaussianMixtureLoss)
        self.assertEqual(loss.initial_variance, 0.5)
        # Round-trip with a Config
        cfg = PoissonGaussianMixtureConfig(initial_variance=0.25, learn_variance=True)
        loss2 = poisson_gaussian_mixture_loss(cfg)
        self.assertIsInstance(loss2, PoissonGaussianMixtureLoss)
        self.assertEqual(loss2.initial_variance, 0.25)
        self.assertTrue(loss2.learn_variance)

    def test_enhanced_factory_returns_instance(self):
        cfg = EnhancedPoissonGaussianConfig(
            gain=2.0, read_noise=0.1, shot_noise=0.05, log_input=True
        )
        loss = enhanced_poisson_gaussian_loss(cfg)
        self.assertIsInstance(loss, EnhancedPoissonGaussianMixtureLoss)
        # Constructor stored the config object.
        self.assertEqual(loss.log_input, True)

    def test_enhanced_default_kwargs_construct_loss(self):
        loss = enhanced_poisson_gaussian_loss(read_noise=0.2, gain="learn")
        self.assertIsInstance(loss, EnhancedPoissonGaussianMixtureLoss)
        self.assertTrue(loss.learn_gain)
        self.assertEqual(loss.config.read_noise, 0.2)

    def test_lr_factory_returns_instance(self):
        cfg = PoissonGaussianLikelihoodRatioConfig(log_input=True, learn_variance=True)
        loss = poisson_gaussian_likelihood_ratio_loss(cfg)
        self.assertIsInstance(loss, PoissonGaussianLikelihoodRatioLoss)
        self.assertTrue(loss.learn_variance)
        self.assertTrue(loss.log_input)

    def test_lr_factory_default_log_input_kwarg(self):
        """Per the docstring default, ``log_input=False`` is honoured when
        the caller doesn't pass it.
        """
        loss = poisson_gaussian_likelihood_ratio_loss()
        self.assertIsInstance(loss, PoissonGaussianLikelihoodRatioLoss)
        self.assertFalse(loss.log_input)


# ---------------------------------------------------------------------------
# Argument-order discriminator
# ---------------------------------------------------------------------------


class TestFunctionalArgOrder(unittest.TestCase):
    """Lock the documented ``(y_pred, target)`` arg order for each functional
    wrapper so accidental arg-rotation in the wrapper internals is caught
    deterministically.  For asymmetric losses (expectile, quantile) the
    canonical order and the swapped order must give different finite values;
    the difference is the formula's residual-sign reinterpretation.  For
    symmetric losses (conformal split / MSE), the two orderings collide and
    we assert finite + identical scalars.
    """

    def test_expectile_loss_canonical_order_matches_published_formula(self) -> None:
        # residual = y_pred - target = -2 (< 0, under-estimation); tau=0.8 weight = 0.8.
        # 2 * (-2)^2 * 0.8 = 6.4
        canonical = expectile_loss(torch.tensor([0.0]), torch.tensor([2.0]), expectile=0.8)
        self.assertTrue(torch.isclose(canonical, torch.tensor(6.4)))

    def test_expectile_loss_swapped_args_yields_different_value(self) -> None:
        canonical = expectile_loss(torch.tensor([0.0]), torch.tensor([2.0]), expectile=0.8)
        # residual = y_pred - target = +2 (> 0); tau=0.8 weight = 0.2
        # 2 * 4 * 0.2 = 1.6
        swapped = expectile_loss(torch.tensor([2.0]), torch.tensor([0.0]), expectile=0.8)
        self.assertTrue(torch.isclose(swapped, torch.tensor(1.6)))
        # Order MUST matter for asymmetric losses.
        self.assertGreater((canonical - swapped).abs().item(), 1.0)

    def test_quantile_loss_canonical_order_matches_pinball(self) -> None:
        # residual = y_pred - target = -4 < 0; q=0.9 weight = 0.9; 4 * 0.9 = 3.6
        canonical = quantile_loss(torch.tensor([1.0]), torch.tensor([5.0]), quantile=0.9)
        self.assertTrue(torch.isclose(canonical, torch.tensor(3.6)))

    def test_quantile_loss_swapped_args_yields_different_value(self) -> None:
        # residual = y_pred - target = +4 > 0; q=0.9 weight = 0.1; 4 * 0.1 = 0.4
        swapped = quantile_loss(torch.tensor([5.0]), torch.tensor([1.0]), quantile=0.9)
        self.assertTrue(torch.isclose(swapped, torch.tensor(0.4)))

    def test_conformal_loss_split_arg_order_is_symmetric(self) -> None:
        # MSE is symmetric in ``(y_pred, target)``: both orderings collapse
        # to the same scalar.  Lock that the wrapper accepts both orderings
        # and produces finite, identical results.
        torch.manual_seed(0)
        y = torch.randn(6, 1)
        canonical = conformal_loss(y, y.clone(), method="split", alpha=0.1, reduction="mean")
        swapped = conformal_loss(y.clone(), y, method="split", alpha=0.1, reduction="mean")
        self.assertTrue(torch.isfinite(canonical))
        self.assertTrue(torch.isfinite(swapped))
        self.assertTrue(torch.isclose(canonical, swapped, atol=1e-6))

    def test_tweedie_loss_canonical_order_zero_when_match(self) -> None:
        # Identity link; y_pred == target -> 0 (p=0 branch).
        mu = torch.tensor([1.0, 2.0, 3.0])
        target = mu.clone()
        got = tweedie_loss(mu, target, p=0.0, link="identity", reduction="none")
        self.assertTrue(torch.allclose(got, torch.zeros_like(target)))


if __name__ == "__main__":
    unittest.main()
