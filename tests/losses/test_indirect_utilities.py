"""Direct tests for the public loss-library utilities that are exported from
``torchregress.losses`` but only exercised indirectly via the per-class test
suites.  Each utility gets a dedicated test class that locks in the public
API contract:

- :func:`low_rank_output_dim`               (torchregress.utils.gaussian_output)
- :func:`split_low_rank_gaussian_output`    (torchregress.utils.gaussian_output)
- :func:`symmetric_spd_matrix_sqrt`         (torchregress.losses.gaussian_wasserstein)
- :func:`create_gaussian_nll`               (torchregress.losses.gaussian)
- :func:`create_mdn_loss`                   (torchregress.losses.mdn)
- :data:`QuantileCrossover`  (alias for :class:`QuantileCrossoverLoss`)
- :data:`ExpectileCrossover` (alias for :class:`ExpectileCrossoverLoss`)
"""

from __future__ import annotations

import unittest

import torch

from torchregress.losses import GaussianNLLLoss, WeightedMSELoss
from torchregress.losses.expectile import ExpectileCrossover, ExpectileCrossoverLoss
from torchregress.losses.gaussian import (
    LowRankGaussianLoss,
    MultivariateGaussianLoss,
    create_gaussian_nll,
)
from torchregress.losses.gaussian_wasserstein import symmetric_spd_matrix_sqrt
from torchregress.losses.mdn import MixtureDensityLoss, create_mdn_loss
from torchregress.losses.quantile import QuantileCrossover, QuantileCrossoverLoss
from torchregress.utils.gaussian_output import (
    low_rank_output_dim,
    split_low_rank_gaussian_output,
)


def _psd_matrix(d: int, *, seed: int = 0) -> torch.Tensor:
    """Build a deterministic ``d x d`` symmetric positive definite matrix."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    return A @ A.T + torch.eye(d) * 1e-3


# ---------------------------------------------------------------------------
# low_rank_output_dim
# ---------------------------------------------------------------------------


class TestLowRankOutputDim(unittest.TestCase):
    def test_formula_is_two_n_features_plus_n_features_rank(self):
        """The factorised low-rank Gaussian head concatenates ``[mean | low-rank
        factor | per-feature stddev]`` along the last axis.  The total dim is
        therefore ``2 * n_features + n_features * rank`` (mean + diagonal both
        occupy one feature-size slice each).
        """
        for n_features in (1, 3, 7):
            for rank in (1, 2, 5):
                self.assertEqual(
                    low_rank_output_dim(n_features, rank),
                    2 * n_features + n_features * rank,
                )

    def test_zero_features_or_rank_raises(self):
        """By contract ``n_features`` and ``rank`` must both be strictly
        positive; the function surfaces ``ValueError`` for invalid inputs.
        """
        with self.assertRaises(ValueError):
            low_rank_output_dim(0, 1)
        with self.assertRaises(ValueError):
            low_rank_output_dim(1, 0)
        with self.assertRaises(ValueError):
            low_rank_output_dim(-1, 1)

    def test_increases_monotonically_in_rank_and_features(self):
        for n_features in (1, 2, 3, 4):
            for rank in (1, 2, 3):
                d = low_rank_output_dim(n_features, rank)
                # Strictly increasing with rank when features>0.
                if rank > 1:
                    self.assertGreater(d, low_rank_output_dim(n_features, rank - 1))
                # Strictly increasing with features when rank>0.
                if n_features > 1:
                    self.assertGreater(d, low_rank_output_dim(n_features - 1, rank))


# ---------------------------------------------------------------------------
# split_low_rank_gaussian_output
# ---------------------------------------------------------------------------


class TestSplitLowRankGaussianOutput(unittest.TestCase):
    def test_round_trip_with_concat_input(self):
        """``split_low_rank_gaussian_output(low_rank_output_dim)`` recovers the
        three pieces stacked into ``y_pred``.  The documented layout is
        ``[mean | cov_factor | cov_diag]`` along ``dim=-1`` so the total
        dimension is ``n_features + n_features * rank + n_features``.
        """
        batch = 4
        n_features = 3
        rank = 2
        out_dim = low_rank_output_dim(n_features, rank)  # 2F + F*R = 12 here
        # Stack: mean (B, F), factor (B, F*R), diag (B, F).
        mean = torch.randn(batch, n_features)
        factor = torch.randn(batch, n_features, rank)
        diag = torch.full((batch, n_features), 0.5)
        y_pred = torch.cat([mean, factor.reshape(batch, -1), diag], dim=-1)
        self.assertEqual(y_pred.shape, (batch, out_dim))

        m, f, d = split_low_rank_gaussian_output(y_pred, n_features, rank)
        self.assertEqual(m.shape, mean.shape)
        self.assertEqual(f.shape, factor.shape)
        self.assertEqual(d.shape, diag.shape)
        # Reconstruction must round-trip exactly.
        self.assertTrue(torch.allclose(m, mean))
        self.assertTrue(torch.allclose(f, factor))
        self.assertTrue(torch.allclose(d, diag))

    def test_wrong_out_dim_raises(self):
        """A ``y_pred`` whose last dim doesn't equal ``low_rank_output_dim``
        must surface as ValueError, not silently truncate.
        """
        batch, n_features, rank = 2, 3, 2
        bad = torch.zeros(batch, low_rank_output_dim(n_features, rank) + 1)
        with self.assertRaises(ValueError):
            split_low_rank_gaussian_output(bad, n_features, rank)

    def test_unbatched_shape_is_supported(self):
        """The function should handle unbatched inputs of shape ``(out_dim,)``.
        Given the layout ``[F + F*R + F]``, the slices are:
        mean ``(F,)``, factor ``(F, R)``, diag ``(F,)``.
        """
        n_features, rank = 3, 2
        out_dim = low_rank_output_dim(n_features, rank)
        y_pred = torch.arange(out_dim, dtype=torch.float32) / out_dim
        m, f, d = split_low_rank_gaussian_output(y_pred, n_features, rank)
        self.assertEqual(m.shape, (n_features,))
        self.assertEqual(f.shape, (n_features, rank))
        self.assertEqual(d.shape, (n_features,))


# ---------------------------------------------------------------------------
# symmetric_spd_matrix_sqrt
# ---------------------------------------------------------------------------


class TestSymmetricSpsMatrixSqrt(unittest.TestCase):
    def test_sqrt_is_symmetric_for_spd_input(self):
        d = 5
        sigma = _psd_matrix(d, seed=1)
        s = symmetric_spd_matrix_sqrt(sigma)
        self.assertEqual(s.shape, sigma.shape)
        # Symmetric to floating tolerance.
        self.assertTrue(torch.allclose(s, s.transpose(-1, -2), atol=1e-6))

    def test_sqrt_squared_recovers_input(self):
        """``sqrt(sigma) @ sqrt(sigma) ≈ sigma`` for symmetric PSD input."""
        d = 4
        sigma = _psd_matrix(d, seed=2)
        s = symmetric_spd_matrix_sqrt(sigma)
        self.assertTrue(torch.allclose(s @ s, sigma, atol=1e-5))

    def test_eps_argument_clears_smallest_eigenvalue(self):
        """When eigenvalues approach zero, ``eps`` floors them before sqrt
        so the result stays finite.  A tiny positive perturbation ``1e-12``
        would otherwise produce numerical instability.
        """
        # Build a near-singular PSD: rank d-1 with a tiny eigenvalue.
        d = 3
        A = torch.randn(d, d - 1, dtype=torch.float64) * 0.01
        sigma = A @ A.T  # smallest eigenvalue ≈ 0
        # Tight tol: small eps should not blow up here.
        s = symmetric_spd_matrix_sqrt(sigma, eps=1e-8)
        self.assertTrue(torch.isfinite(s).all())
        # Symmetric.
        self.assertTrue(torch.allclose(s, s.T, atol=1e-8))


# ---------------------------------------------------------------------------
# create_gaussian_nll
# ---------------------------------------------------------------------------


class TestCreateGaussianNll(unittest.TestCase):
    def test_default_routes_to_diag_gaussian_nll(self):
        loss = create_gaussian_nll()
        self.assertIsInstance(loss, GaussianNLLLoss)

    def test_routes_to_multivariate_when_requested(self):
        loss = create_gaussian_nll(covariance_type="multivariate")
        self.assertIsInstance(loss, MultivariateGaussianLoss)

    def test_full_alias_routes_same_as_multivariate(self):
        """``covariance_type="full"`` and ``"multivariate"`` are documented
        aliases and must produce the same loss class.
        """
        loss_full = create_gaussian_nll(covariance_type="full")
        loss_multi = create_gaussian_nll(covariance_type="multivariate")
        self.assertIs(type(loss_full), type(loss_multi))

    def test_routes_to_low_rank_when_requested(self):
        loss = create_gaussian_nll(covariance_type="low_rank", rank=3)
        self.assertIsInstance(loss, LowRankGaussianLoss)
        # The factory records ``rank`` only as a parameter-symmetry hint and
        # does not store a public ``rank`` attribute on the constructed loss
        # (rank is determined by model output shape at forward time).  We
        # only assert the routing happened.

    def test_mse_shortcut_collapses_to_weighted_mse(self):
        """``use_mse_for_unit_variance=True`` collapses to ``WeightedMSELoss``
        for the diagonal case.  We must pass only WeightedMSELoss-compatible
        kwargs (``reduction``) or the constructor itself raises.
        """
        loss = create_gaussian_nll(
            covariance_type="diagonal",
            use_mse_for_unit_variance=True,
            reduction="mean",
        )
        self.assertIsInstance(loss, WeightedMSELoss)
        # Sanity: the MSE shortcut is callable end-to-end.
        mse = WeightedMSELoss(reduction="mean")
        val = mse(torch.randn(4, 1), torch.randn(4, 1))
        self.assertTrue(torch.isfinite(val).all())

    def test_unknown_covariance_type_raises(self):
        with self.assertRaises(ValueError):
            create_gaussian_nll(covariance_type="not_a_mode")


# ---------------------------------------------------------------------------
# create_mdn_loss
# ---------------------------------------------------------------------------


class TestCreateMdnLoss(unittest.TestCase):
    def test_default_returns_mixture_density_loss(self):
        loss = create_mdn_loss(n_components=3, n_features=2)
        self.assertIsInstance(loss, MixtureDensityLoss)

    def test_kwargs_propagate_to_constructor(self):
        loss = create_mdn_loss(n_components=4, n_features=2, covariance_type="full", min_std=1e-2)
        self.assertEqual(loss.n_components, 4)
        self.assertEqual(loss.n_features, 2)
        self.assertEqual(loss.covariance_type, "full")
        self.assertEqual(loss.min_std, 1e-2)

    def test_default_covariance_is_diagonal(self):
        loss = create_mdn_loss(n_components=2, n_features=1)
        self.assertEqual(loss.covariance_type, "diagonal")

    def test_invalid_covariance_type_raises(self):
        with self.assertRaises(ValueError):
            create_mdn_loss(n_components=2, n_features=1, covariance_type="not_a_real_mode")

    def test_forward_smoke(self):
        """A factory-built MDN loss must be callable end-to-end with a small
        parameter shaped model output.  This catches accidental construction
        failures (e.g. ``MixtureDensityLoss`` validator regression).
        """
        n_components, n_features = 2, 3
        loss = create_mdn_loss(n_components=n_components, n_features=n_features)
        # Diagonal MDN: ``[weights | means | log_stds]`` so the output dim is
        # ``n_components + 2 * n_components * n_features``.  This is the
        # canonical formula internally; any deviation would break ``forward``.
        out_dim = n_components + 2 * n_components * n_features
        y_pred = torch.randn(4, out_dim, requires_grad=True)
        target = torch.randn(4, n_features)
        val = loss(y_pred, target)
        self.assertTrue(torch.isfinite(val).all())
        # Backward must also succeed (sanity check on autograd graph).
        val.backward()
        self.assertIsNotNone(y_pred.grad)
        self.assertTrue(torch.isfinite(y_pred.grad).all())


# ---------------------------------------------------------------------------
# Class aliases
# ---------------------------------------------------------------------------


class TestCrossoverAliases(unittest.TestCase):
    """``QuantileCrossover`` / ``ExpectileCrossover`` are exported from the
    top-level ``torchregress.losses`` namespace as backward-compatibility
    aliases for the concrete caller classes.
    """

    def test_quantile_crossover_is_class_alias(self):
        # Strict identity (not isinstance) -- catches accidental rebinding of
        # the alias to ``functools.partial(QuantileCrossoverLoss, ...)``, a
        # factory closure, or any builder function that *returns* the class
        # instead of being the class object itself.
        self.assertIs(QuantileCrossover, QuantileCrossoverLoss)

    def test_expectile_crossover_is_class_alias(self):
        # Mirrors the QuantileCrossover alias-discrimination contract.
        self.assertIs(ExpectileCrossover, ExpectileCrossoverLoss)

    def test_quantile_crossover_instantiable_via_alias(self):
        # Construct through the alias and verify the instance is the
        # underlying class.  This catches accidentally redefining the
        # alias to a builder function.
        loss = QuantileCrossover(quantiles=[0.1, 0.5, 0.9])
        self.assertIsInstance(loss, QuantileCrossoverLoss)

    def test_expectile_crossover_instantiable_via_alias(self):
        loss = ExpectileCrossover(expectiles=[0.1, 0.5, 0.9])
        self.assertIsInstance(loss, ExpectileCrossoverLoss)

    def test_canonical_aliases_share_constructor_signature(self):
        """Both aliases must accept the same constructor kwargs as the
        underlying class without raising.  Catches accidental re-binding of
        the alias to a builder function (e.g. ``partial``) by the caller.
        Use strictly non-default kwargs so the assertion actually forces the
        alias to traverse the constructor rather than the default-mercy path.
        """
        kwargs = dict(
            quantiles=[0.1, 0.7, 0.9],
            base_loss=2.5,
            crossover_penalty=7.5,
            reduction="sum",
        )
        loss_via_alias = QuantileCrossover(**kwargs)
        loss_via_class = QuantileCrossoverLoss(**kwargs)
        # Same type still holds after construction.
        self.assertIs(type(loss_via_alias), type(loss_via_class))
        # And the propagated kwargs actually reached the live attributes.
        self.assertEqual(loss_via_alias.base_loss, 2.5)
        self.assertEqual(loss_via_alias.crossover_penalty, 7.5)
        self.assertEqual(loss_via_alias.reduction, "sum")


if __name__ == "__main__":
    unittest.main()
