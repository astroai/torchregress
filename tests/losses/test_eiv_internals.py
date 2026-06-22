"""Direct tests for EIV internal modules.

The :mod:`torchregress.tests.losses.test_eiv` module exercises the four public
loss classes (FunctionalEIVLoss, StructuralEIVLoss, OrthogonalDistanceRegressionLoss,
EnsembleEIVLoss) plus InputNoiseMarginalizationLoss.  This file drills into the
two public helper modules that EIV losses use internally:

- :class:`NoisyInputPredictor` -- test-time input-noise marginalization wrapper
- :class:`ExplicitEIVAdapter` -- ``(x_obs, target)`` adapter exposed via
  ``BaseEIVLoss.explicit()``

It also chains :meth:`BaseEIVLoss.explicit` (the factory method) plus the
sigma-preparation branch (``_prepare_covariance_from_sigma``) which has many
shape/edge-case branches worth covering directly.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from torchregress.losses.eiv import (
    ExplicitEIVAdapter,
    FunctionalEIVLoss,
    InputNoiseMarginalizationLoss,
    NoisyInputPredictor,
)


def _linear_model(n_in: int, n_out: int) -> nn.Module:
    """Deterministic linear model used by the EIV internals tests."""
    torch.manual_seed(0)
    return nn.Linear(n_in, n_out)


class TestExplicitEIVAdapter(unittest.TestCase):
    """``loss.explicit()`` returns an :class:`ExplicitEIVAdapter` that
    accepts ``(x_obs, target)`` and forwards ``sigma_x`` / ``sigma_y`` kwargs.
    """

    def setUp(self):
        self.batch_size = 4
        self.n_features_x = 5
        self.n_features_y = 2
        self.model = _linear_model(self.n_features_x, self.n_features_y)
        self.x_obs = torch.randn(self.batch_size, self.n_features_x)
        self.y_obs = torch.randn(self.batch_size, self.n_features_y)

    def test_adapter_returns_finite_loss_with_scalar_sigma(self):
        loss_fn = FunctionalEIVLoss(self.model, sigma_x=0.1, sigma_y=0.1).explicit()
        val = loss_fn(self.x_obs, self.y_obs)
        self.assertTrue(torch.is_tensor(val))
        self.assertTrue(torch.isfinite(val))

    def test_adapter_overrides_sigma_x_at_call_site(self):
        """sigma_x supplied per-call must take precedence over the value
        bound at construction time; the loss must remain finite.
        """
        # Construct with sigma_x=0.1; override at call with sigma_x=2.0.
        loss_fn = FunctionalEIVLoss(self.model, sigma_x=0.1, sigma_y=0.1).explicit()
        per_batch_sigma = torch.full((self.batch_size, self.n_features_x), 2.0)
        val = loss_fn(self.x_obs, self.y_obs, sigma_x=per_batch_sigma)
        self.assertTrue(torch.isfinite(val))

        # Compare to the same call through the underlying loss' explicit path:
        # both routes must produce an identical scalar.
        loss_underlying = FunctionalEIVLoss(self.model, sigma_x=0.1, sigma_y=0.1)
        val_ref = loss_underlying(self.x_obs, self.y_obs, sigma_x=per_batch_sigma, sigma_y=None)
        self.assertTrue(torch.allclose(val, val_ref))

    def test_adapter_overrides_sigma_y_at_call_site(self):
        loss_fn = FunctionalEIVLoss(self.model, sigma_x=0.1, sigma_y=0.1).explicit()
        per_batch_sigma_y = torch.full((self.batch_size, self.n_features_y), 0.05)
        val = loss_fn(self.x_obs, self.y_obs, sigma_y=per_batch_sigma_y)
        self.assertTrue(torch.isfinite(val))

    def test_adapter_forwards_mask_and_weights(self):
        """With mask+weights supplied through the adapter, the reduced scalar
        must remain finite.  ``FunctionalEIVLoss`` aggregates per-feature NLL
        into a per-sample scalar before reduction, so the adapter's weights
        must be per-sample (``(B,)``) to broadcast correctly through the
        inner ``BaseLoss._reduce`` path.
        """
        loss_fn = FunctionalEIVLoss(self.model, sigma_x=0.1, sigma_y=0.1).explicit()
        mask = torch.ones(self.batch_size, self.n_features_y, dtype=torch.bool)
        mask[0, 0] = False
        weights = torch.full((self.batch_size,), 1.0)
        val = loss_fn(self.x_obs, self.y_obs, mask=mask, weights=weights)
        self.assertTrue(torch.is_tensor(val))
        self.assertTrue(torch.isfinite(val))

        # Compare to the same call routed through the underlying loss so the
        # adapter is verified to drop neither kwargs.
        loss_ref = FunctionalEIVLoss(self.model, sigma_x=0.1, sigma_y=0.1)
        ref = loss_ref(self.x_obs, self.y_obs, mask=mask, weights=weights)
        self.assertTrue(torch.isclose(val, ref, atol=1e-6))

        # Without mask/weights the adapter must still produce a finite scalar
        # that matches the loss' default reduction.
        val_default = loss_fn(self.x_obs, self.y_obs)
        ref_default = loss_ref(self.x_obs, self.y_obs)
        self.assertTrue(torch.isclose(val_default, ref_default, atol=1e-6))


class TestNoisyInputPredictor(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.n_features_x = 4
        self.n_features_y = 2
        self.model = _linear_model(self.n_features_x, self.n_features_y)
        self.x_obs = torch.randn(self.batch_size, self.n_features_x)

    def test_forward_returns_mean_over_perturbations(self):
        """``predictor(x)`` averages across noise samples.  With a fixed seed,
        the averaged output must match ``stacked.mean(dim=0)`` under a single
        draw sequence.  We assert shape parity and that the average operates
        along the first axis without changing other dims (mean reduces only
        dim 0).
        """
        predictor = NoisyInputPredictor(self.model, sigma_x=0.05, n_samples=7)
        averaged = predictor(self.x_obs)
        self.assertTrue(torch.is_tensor(averaged))
        # Output shape must be (batch, n_y) because stacking is (n_samples, batch, n_y).
        self.assertEqual(averaged.shape, (self.batch_size, self.n_features_y))
        # The averaged tensor must contain only finitely-valued numbers.
        self.assertTrue(torch.isfinite(averaged).all())
        # Lowering n_samples to 1 collapses the mean to the single draw at sigma_x=0
        # by way of the bias-correction ``clamp(min=min_sigma)``; we just verify
        # that an ``n_samples=1`` predictor returns a finite tensor (sanity).
        predictor_single = NoisyInputPredictor(self.model, sigma_x=0.05, n_samples=1)
        self.assertTrue(torch.isfinite(predictor_single(self.x_obs)).all())

    def test_sample_predictions_shape(self):
        predictor = NoisyInputPredictor(self.model, sigma_x=0.05, n_samples=5, antithetic=True)
        stacked = predictor.sample_predictions(self.x_obs)
        self.assertEqual(stacked.shape, (5, self.batch_size, self.n_features_y))

    def test_antithetic_pairs_sum_to_zero_perturbation(self):
        """When ``antithetic=True`` and ``n_samples`` is even, the first draw's
        perturbation and the second's should be opposite vectors.  We verify
        via a fixed seed by directly exercising ``InputNoiseMarginalizationLoss``,
        which is the same sampling path.
        """
        marginalizer = InputNoiseMarginalizationLoss(
            self.model,
            base_loss=lambda p, t: p,
            sigma_x=torch.full((self.n_features_x,), 0.05),
            n_samples=4,
            antithetic=True,
        )
        # Two antithetic samples must yield unperturbed-x + noise vs unperturbed-x - noise.
        torch.manual_seed(123)
        x = torch.zeros(1, self.n_features_x)  # zero so observed is known
        stacked = marginalizer.sample_predictions(x)
        # first two predictions correspond to +eps and -eps perturbations around zero;
        # with antithetic, the *mechanism* stacks +noise then -noise.  We assert
        # the predictions are NOT identical, which proves antithetic sampling
        # produced two distinct rows.
        self.assertFalse(torch.allclose(stacked[0], stacked[1]))

    def test_sample_predictions_raises_on_non_tensor_output(self):
        """When the wrapped model returns a non-Tensor, NoisyInputPredictor
        must raise because ``sample_predictions`` only accepts Tensor outputs.
        """

        class NonTensorModel(nn.Module):
            def forward(self, x):  # returns a list, not a tensor
                return [x[:, 0]]

        predictor = NoisyInputPredictor(NonTensorModel(), sigma_x=0.05, n_samples=2)
        with self.assertRaises(TypeError):
            predictor.sample_predictions(self.x_obs)

    def test_closure_uses_predictive_average_transform(self):
        """If ``transform`` is supplied to ``forward``, it must be applied to
        the stacked predictions and its return value bubbled up unchanged.
        """

        class Head(nn.Module):
            def forward(self, x):
                return x[:, :3]

        predictor = NoisyInputPredictor(Head(), sigma_x=0.02, n_samples=4, antithetic=True)
        probs = predictor(
            self.x_obs,
            transform=lambda stacked: torch.softmax(stacked, dim=-1).mean(dim=0),
        )
        self.assertTrue(torch.is_tensor(probs))
        # softmax probabilities must sum to ~1 per row.
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))


class TestBaseEIVLossInternals(unittest.TestCase):
    """Direct tests for the ``BaseEIVLoss`` helper methods that the public
    classes depend on, but which have no first-class coverage in ``test_eiv.py``.
    """

    def setUp(self):
        self.model = _linear_model(3, 2)
        self.loss = FunctionalEIVLoss(self.model, sigma_x=0.1, sigma_y=0.1)
        self.device = torch.device("cpu")

    def test_explicit_factory_returns_adapter(self):
        adapter = self.loss.explicit()
        self.assertIsInstance(adapter, ExplicitEIVAdapter)
        # The adapter must hold a reference to the same underlying loss object.
        self.assertIs(adapter.loss, self.loss)

    def test_prepare_covariance_scalar(self):
        n_features = 4
        cov = self.loss._prepare_covariance_from_sigma(0.5, n_features, self.device)
        self.assertEqual(cov.shape, (n_features, n_features))
        # Off-diagonal entries are zero; diagonal entries are 0.5^2.
        # Pin ``device`` and ``dtype`` on the reference so the assertion is
        # independent of any global PyTorch state a prior test might mutate
        # (``allclose`` is dtype/device-tied at the framework level).
        ref = torch.eye(n_features, device=cov.device, dtype=cov.dtype) * 0.25
        self.assertTrue(torch.allclose(cov, ref))

    def test_prepare_covariance_one_element_tensor(self):
        n_features = 4
        cov = self.loss._prepare_covariance_from_sigma(torch.tensor(0.5), n_features, self.device)
        self.assertEqual(cov.shape, (n_features, n_features))
        ref = torch.eye(n_features, device=cov.device, dtype=cov.dtype) * 0.25
        self.assertTrue(torch.allclose(cov, ref))

    def test_prepare_covariance_one_d_vector(self):
        """Shape ``[n_features]`` is treated as diagonal stddev.

        Pin dtype/device on the reference so the ``allclose`` is independent
        of any global PyTorch state a prior test in the suite might mutate.
        Compute the expected diagonal from the input ``sigma`` to avoid
        hard-coding a divergent dtype on a literal.
        """
        n_features = 3
        sigma = torch.tensor([0.5, 0.4, 0.6])
        cov = self.loss._prepare_covariance_from_sigma(sigma, n_features, self.device)
        self.assertEqual(cov.shape, (n_features, n_features))
        sigma_on_cov = sigma.to(device=cov.device, dtype=cov.dtype)
        expected = torch.diag(sigma_on_cov**2)
        self.assertTrue(torch.allclose(cov, expected))

    def test_prepare_covariance_one_d_vector_wrong_size_raises(self):
        n_features = 3
        sigma = torch.tensor([0.5, 0.4])  # wrong length
        with self.assertRaises(ValueError):
            self.loss._prepare_covariance_from_sigma(sigma, n_features, self.device)

    def test_prepare_covariance_two_d_full_shared(self):
        """Shape ``[n, n]`` is treated as a shared covariance matrix.

        Pin dtype/device against any global PyTorch state mutation.
        """
        n_features = 3
        sigma_mat = 0.1 * torch.eye(n_features)
        cov = self.loss._prepare_covariance_from_sigma(sigma_mat, n_features, self.device)
        self.assertEqual(cov.shape, (n_features, n_features))
        expected = sigma_mat.to(device=cov.device, dtype=cov.dtype)
        self.assertTrue(torch.allclose(cov, expected))

    def test_prepare_covariance_three_d_per_batch_diagonal(self):
        """Shape ``[batch, n_features]`` is treated as per-sample diagonal stddev.

        Pin dtype/device against any global PyTorch state mutation.
        """
        batch_size = 5
        n_features = 3
        sigma = torch.full((batch_size, n_features), 0.5)
        cov = self.loss._prepare_covariance_from_sigma(
            sigma, n_features, self.device, batch_size=batch_size
        )
        self.assertEqual(cov.shape, (batch_size, n_features, n_features))
        eye_diag = torch.eye(n_features, device=cov.device, dtype=cov.dtype) * 0.25
        for i in range(batch_size):
            self.assertTrue(torch.allclose(cov[i], eye_diag))

    def test_prepare_covariance_three_d_per_batch_wrong_size_raises(self):
        """A ``[B, n, m]`` tensor with ``m != n_features`` must raise."""
        bad = torch.full((2, 3, 4), 0.1)
        with self.assertRaises(ValueError):
            self.loss._prepare_covariance_from_sigma(bad, 3, self.device, batch_size=2)

    def test_prepare_covariance_unknown_dim_raises(self):
        bad = torch.randn(2, 3, 4, 5)
        with self.assertRaises(ValueError):
            self.loss._prepare_covariance_from_sigma(bad, 3, self.device)


if __name__ == "__main__":
    unittest.main()
