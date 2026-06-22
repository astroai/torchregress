import unittest

import torch
import torch.nn as nn

from torchregress.algorithms.rc import RegressionCalibration
from torchregress.algorithms.simex import PredictionSIMEX
from torchregress.losses import WeightedMSELoss
from torchregress.losses.eiv import (
    InputNoiseAugmentationLoss,
    InputNoiseMarginalizationLoss,
    LatentMarginalizationLoss,
    OrthogonalDistanceRegressionLoss,
)
from torchregress.losses.gaussian import GaussianNLLLoss


class TestEIVCorrectness(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 8
        self.n_features_x = 4
        self.n_features_y = 1

        # Simple linear model
        self.model = nn.Linear(self.n_features_x, self.n_features_y).to(self.device)
        # Fix weights for deterministic behavior
        with torch.no_grad():
            self.model.weight.fill_(1.0)
            self.model.bias.fill_(0.0)

        self.x_obs = torch.randn(self.batch_size, self.n_features_x, device=self.device)
        self.y_obs = torch.randn(self.batch_size, self.n_features_y, device=self.device)
        self.sigma_x = torch.full((self.n_features_x,), 0.1, device=self.device)

    def test_input_noise_augmentation_vectorization(self):
        # Base loss is MSE (used as dummy regression loss)
        base_loss = WeightedMSELoss(reduction="none")

        # Set up loss
        loss_fn = InputNoiseAugmentationLoss(
            model=self.model,
            base_loss=base_loss,
            sigma_x=self.sigma_x,
            n_samples=10,
        ).to(self.device)

        # Verify it returns a finite tensor
        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.isfinite(loss))

        # Test deprecated alias behaves exactly the same
        with self.assertWarns(DeprecationWarning):
            legacy_loss_fn = InputNoiseMarginalizationLoss(
                model=self.model,
                base_loss=base_loss,
                sigma_x=self.sigma_x,
                n_samples=10,
            ).to(self.device)
            legacy_loss = legacy_loss_fn(self.x_obs, self.y_obs)
            self.assertTrue(torch.is_tensor(legacy_loss))

    def test_latent_marginalization_gaussian_prior(self):
        # Base loss is Gaussian NLL loss (expects mean and log_var)
        # Let's use a dummy model that outputs mean and log_var
        class DummyGaussianModel(nn.Module):
            def __init__(self, in_features):
                super().__init__()
                self.linear_mean = nn.Linear(in_features, 1)
                self.linear_logvar = nn.Linear(in_features, 1)

            def forward(self, x):
                return torch.cat([self.linear_mean(x), self.linear_logvar(x)], dim=-1)

        model = DummyGaussianModel(self.n_features_x).to(self.device)
        base_loss = GaussianNLLLoss()

        # Pre-fit RegressionCalibration prior
        rc = RegressionCalibration(sigma_u=self.sigma_x)
        rc.fit(self.x_obs)

        # Use RC as posterior sampler
        loss_fn = LatentMarginalizationLoss(
            model=model,
            base_loss=base_loss,
            posterior_sampler=rc,
            n_samples=8,
        ).to(self.device)

        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.isfinite(loss))

        # Use analytical prior parameters directly (should also work)
        prior_mean = self.x_obs.mean(dim=0)
        prior_cov = torch.eye(self.n_features_x, device=self.device) * 0.5

        loss_fn_analytic = LatentMarginalizationLoss(
            model=model,
            base_loss=base_loss,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            sigma_u=self.sigma_x,
            n_samples=8,
        ).to(self.device)

        loss_analytic = loss_fn_analytic(self.x_obs, self.y_obs)
        self.assertTrue(torch.isfinite(loss_analytic))

    def test_latent_marginalization_custom_sampler(self):
        # A custom posterior sampler that just perturbs observed inputs with a different scale
        def custom_sampler(x_obs, n_samples):
            # Returns shape [n_samples, batch_size, n_features]
            noise = (
                torch.randn(n_samples, x_obs.shape[0], x_obs.shape[1], device=x_obs.device) * 0.05
            )
            return x_obs.unsqueeze(0) + noise

        base_loss = WeightedMSELoss(reduction="none")
        loss_fn = LatentMarginalizationLoss(
            model=self.model,
            base_loss=base_loss,
            posterior_sampler=custom_sampler,
            n_samples=5,
        ).to(self.device)

        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertTrue(torch.isfinite(loss))

    def test_odr_no_gradient_pollution(self):
        # Make a copy of the model parameters to check they are not polluted during forward pass
        original_params = [p.clone().detach() for p in self.model.parameters()]

        # Zero any existing gradients
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        sigma_x = torch.ones(self.n_features_x, device=self.device) * 0.1
        sigma_y = torch.tensor([0.1], device=self.device)

        # Instantiating ODR with envelope mode (default)
        loss_fn = OrthogonalDistanceRegressionLoss(
            model=self.model,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            learning_rate=0.1,
            max_iterations=5,
            gradient_mode="envelope",
        ).to(self.device)

        # Compute forward pass
        loss = loss_fn(self.x_obs, self.y_obs)

        # Check that model parameters didn't change
        for p, p_orig in zip(self.model.parameters(), original_params):
            self.assertTrue(torch.allclose(p, p_orig))

        # Check that none of the model parameters have gradients
        for p in self.model.parameters():
            if p.grad is not None:
                self.assertTrue(
                    torch.all(p.grad == 0.0),
                    "Gradients accumulated on model parameters during forward pass!",
                )

        # Now do the backward pass
        loss.backward()

        # Check that model parameters now DO have gradients (since we backpropagated)
        has_grad = False
        for p in self.model.parameters():
            if p.grad is not None and torch.sum(torch.abs(p.grad)) > 0.0:
                has_grad = True
        self.assertTrue(
            has_grad, "No gradients propagated to model parameters after backward pass!"
        )

        # Check that optimality residuals are stored
        self.assertIsNotNone(loss_fn.last_optimality_residual_)

    def test_odr_unrolled_gradients(self):
        # ODR in unrolled mode allows gradients to flow back through the optimization trajectory.
        sigma_x = torch.ones(self.n_features_x, device=self.device) * 0.1
        sigma_y = torch.tensor([0.1], device=self.device)

        loss_fn = OrthogonalDistanceRegressionLoss(
            model=self.model,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            learning_rate=0.1,
            max_iterations=3,
            gradient_mode="unrolled",
        ).to(self.device)

        # Reset model gradients
        self.model.zero_grad()

        # Compute loss
        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertTrue(torch.isfinite(loss))

        # Check model parameters don't have gradients yet (forward pass doesn't pollute in unrolled either)
        for p in self.model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.all(p.grad == 0.0))

        # Backward pass
        loss.backward()

        # Verify gradients are propagated
        has_grad = False
        for p in self.model.parameters():
            if p.grad is not None and torch.sum(torch.abs(p.grad)) > 0.0:
                has_grad = True
        self.assertTrue(has_grad)

    def test_prediction_simex_predict_pipeline(self):
        # Create a simple toy problem where we know the true slope is 3.0
        torch.manual_seed(42)
        n_samples = 100

        X_true = torch.randn(n_samples, 1, device=self.device)
        Y = 3.0 * X_true + torch.randn(n_samples, 1, device=self.device) * 0.05

        # Add noise to input
        noise_std = 0.5
        W_obs = X_true + torch.randn(n_samples, 1, device=self.device) * noise_std

        def train_func(model, X, y):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            loss_fn = nn.MSELoss()
            for _ in range(50):
                optimizer.zero_grad()
                loss = loss_fn(model(X), y)
                loss.backward()
                optimizer.step()
            return model

        def model_factory():
            m = nn.Linear(1, 1)
            with torch.no_grad():
                m.weight.fill_(1.0)
                m.bias.fill_(0.0)
            return m

        # Verify PredictionSIMEX works (the only public entry left after the
        # extrapolate_estimand yagni cut; classical parameter SIMEX is now
        # reachable only via the predict() extrapolated output).
        pred_simex = PredictionSIMEX(
            model_factory=model_factory,
            train_func=train_func,
            sigma_u=noise_std,
            lambdas=[0.5, 1.0, 1.5, 2.0],
            n_simulations=3,
            extrapolation_order=2,
        )
        pred_simex.fit(W_obs, Y)
        preds = pred_simex.predict(W_obs[:5])
        self.assertEqual(preds.shape[0], 5)

    def test_simex_extrapolate_estimand_removed(self):
        """Audit-trail stub: the classical parameter-level SIMEX
        (`extrapolate_estimand`) was removed as part of the SIMEX yagni cut
        (commit "complete over-engineering trims: SIMEX yagni + archive
        cleanup + CI Codecov drop"). The original test asserted numerical
        correctness -- that the corrected slope lied in (2.0, 4.0) when the
        true slope was 3.0 -- on the path through `extrapolate_estimand`,
        which was the SIMEX method's only caller in the repo. To
        reintroduce classical SIMEX: restore the ``extrapolate_estimand``
        method on ``SIMEX`` (the original method body remains in
        git history at the pre-trim commit) and re-apply this stub as a
        live assertion. See ``tests/losses/test_eiv_correctness.py`` git
        log for the prior version of this test method.
        """
        self.skipTest(
            "extrapolate_estimand removed in SIMEX yagni cut; "
            "see docstring for reintroducing instructions."
        )
