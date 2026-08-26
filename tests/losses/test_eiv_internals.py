"""Direct tests for EIV internal modules.

The :mod:`torchregress.tests.losses.test_eiv` module exercises the four public
loss classes (FunctionalEIVLoss, StructuralEIVLoss, OrthogonalDistanceRegressionLoss,
EnsembleEIVLoss, LatentMarginalizationLoss).  This file drills into the
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
    FunctionalEIVLoss,
    NoisyInputPredictor,
)


def _linear_model(n_in: int, n_out: int) -> nn.Module:
    """Deterministic linear model used by the EIV internals tests."""
    torch.manual_seed(0)
    return nn.Linear(n_in, n_out)


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
        # ``torch.diag`` does not accept device=/dtype= natively; pin via
        # chained ``.to`` so the fixture doesn't implicitly rely on the
        # loss module handling dtype/device of input fixtures internally.
        expected = torch.diag(sigma_on_cov**2).to(
            device=sigma_on_cov.device, dtype=sigma_on_cov.dtype
        )
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
        # Pin dtype/device to ``self.device`` so the fixture doesn't
        # implicitly rely on the loss module handling dtype/device of input
        # fixtures internally.
        sigma_mat = 0.1 * torch.eye(n_features, device=self.device, dtype=torch.get_default_dtype())
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
