import unittest

import torch
from torch.autograd import gradcheck

from torchregress.losses.eiv import (
    EnsembleEIVLoss,
    FunctionalEIVLoss,
    OrthogonalDistanceRegressionLoss,
    StructuralEIVLoss,
)


class TestEIVLoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_features_x = 5
        self.n_features_y = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Simple linear model for testing
        self.model = lambda x: x[:, : self.n_features_y] * 2.0

        # observed features (x_obs)
        self.x_obs = torch.randn(self.batch_size, self.n_features_x, device=self.device)
        # noisy targets (y_obs)
        self.y_obs = torch.randn(self.batch_size, self.n_features_y, device=self.device)

        self.mask = torch.ones(self.batch_size, self.n_features_y, device=self.device).bool()
        self.mask[0, 0] = False

    def test_functional_eiv_loss(self):
        # Test with scalar sigma
        loss_fn = FunctionalEIVLoss(self.model, sigma_x=0.1, sigma_y=0.1).to(self.device)
        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))

        # Test with vector sigma (diagonal)
        sigma_x = torch.ones(self.n_features_x, device=self.device) * 0.1
        loss_fn = FunctionalEIVLoss(self.model, sigma_x=sigma_x, sigma_y=0.1).to(self.device)
        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertFalse(torch.isnan(loss))

        # Test with matrix sigma (covariance)
        sigma_x_mat = torch.eye(self.n_features_x, device=self.device) * 0.01
        loss_fn = FunctionalEIVLoss(self.model, sigma_x=sigma_x_mat, sigma_y=0.1).to(self.device)
        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertFalse(torch.isnan(loss))

        # Test with per-sample diagonal sigma
        sigma_x_sample = torch.ones(self.batch_size, self.n_features_x, device=self.device) * 0.1
        loss_fn = FunctionalEIVLoss(self.model, sigma_x=sigma_x_sample, sigma_y=0.1).to(self.device)
        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertFalse(torch.isnan(loss))

        # Test with per-sample full covariance sigma
        sigma_x_sample_mat = (
            torch.eye(self.n_features_x, device=self.device)
            .unsqueeze(0)
            .expand(self.batch_size, -1, -1)
            * 0.01
        )
        loss_fn = FunctionalEIVLoss(self.model, sigma_x=sigma_x_sample_mat, sigma_y=0.1).to(
            self.device
        )
        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertFalse(torch.isnan(loss))

        # Test with Monte Carlo mode
        loss_fn_mc = FunctionalEIVLoss(self.model, sigma_x=0.1, sigma_y=0.1, monte_carlo=True).to(
            self.device
        )
        loss_mc = loss_fn_mc(self.x_obs, self.y_obs)
        self.assertFalse(torch.isnan(loss_mc))

    def test_structural_eiv_loss(self):
        sigma_x = 0.1
        sigma_y = 0.1
        sigma_xy = torch.zeros(self.n_features_y, self.n_features_x, device=self.device)

        loss_fn = StructuralEIVLoss(self.model, sigma_x, sigma_y, sigma_xy).to(self.device)
        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))

        # Test with matrix sigma
        sigma_x_mat = torch.eye(self.n_features_x, device=self.device) * 0.01
        sigma_y_mat = torch.eye(self.n_features_y, device=self.device) * 0.01
        loss_fn = StructuralEIVLoss(self.model, sigma_x_mat, sigma_y_mat, sigma_xy).to(self.device)
        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertFalse(torch.isnan(loss))

    def test_odr_loss(self):
        loss_fn = OrthogonalDistanceRegressionLoss(self.model, sigma_x=0.1, sigma_y=0.1).to(
            self.device
        )
        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))

    def test_ensemble_eiv_loss(self):
        loss_fn = EnsembleEIVLoss(self.model, sigma_x=0.1).to(self.device)
        loss = loss_fn(self.x_obs, self.y_obs)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))

    def test_eiv_edge_cases(self):
        """Test EIVLoss with edge cases: zeros, empty tensors, extreme values, NaN/Inf."""
        loss_fn = FunctionalEIVLoss(self.model, sigma_x=0.1, sigma_y=0.1).to(self.device)

        # Test with zeros
        x_zeros = torch.zeros(10, self.n_features_x, device=self.device)
        y_zeros = torch.zeros(10, self.n_features_y, device=self.device)
        self.assertTrue(torch.isfinite(loss_fn(x_zeros, y_zeros)))

        # Test with empty tensors
        x_empty = torch.tensor([]).reshape(0, self.n_features_x).to(self.device)
        y_empty = torch.tensor([]).reshape(0, self.n_features_y).to(self.device)

        loss = loss_fn(x_empty, y_empty)
        # Mean reduction on empty returns NaN in PyTorch
        self.assertTrue(torch.isnan(loss) or loss.numel() == 0)

        # Test with extreme values
        x_large = torch.ones(1, self.n_features_x, device=self.device) * 1e5
        y_large = torch.ones(1, self.n_features_y, device=self.device) * 1e5
        assert torch.isfinite(loss_fn(x_large, y_large))

        # Test with NaN/Inf and masks
        x_with_nan = torch.randn(3, self.n_features_x, device=self.device)
        x_with_nan[1, 0] = float("nan")
        y_obs = torch.randn(3, self.n_features_y, device=self.device)
        mask = torch.tensor([[True, True], [False, False], [True, True]], device=self.device).bool()

        # We need to make sure FunctionalEIVLoss handles mask correctly during gradient computation
        # In current implementation, apply_mask is used on target and model_output.
        loss = loss_fn(x_with_nan, y_obs, mask=mask)
        self.assertTrue(torch.isfinite(loss))


class TestEIVLossNumericalStability(unittest.TestCase):
    def test_functional_eiv_gradient_flow(self):
        """Test that gradients flow through FunctionalEIVLoss properly."""
        batch_size = 4
        n_features_x = 3
        n_features_y = 2

        # Simple linear model with parameters that require gradients
        weight = torch.randn(n_features_x, n_features_y, requires_grad=True, dtype=torch.double)
        bias = torch.randn(n_features_y, requires_grad=True, dtype=torch.double)

        def model(x):
            return x @ weight + bias

        x_obs = torch.randn(batch_size, n_features_x, requires_grad=True, dtype=torch.double)
        y_obs = torch.randn(batch_size, n_features_y, requires_grad=True, dtype=torch.double)

        # Analytical mode
        loss_fn = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1, reduction="mean")

        # Ensure we use double for gradcheck
        x_obs.data = x_obs.data.to(torch.double)
        y_obs.data = y_obs.data.to(torch.double)
        weight.data = weight.data.to(torch.double)
        bias.data = bias.data.to(torch.double)

        # gradcheck for x_obs and y_obs
        assert gradcheck(lambda x, y: loss_fn(x, y), (x_obs, y_obs), eps=1e-6, atol=1e-4)

        # Monte Carlo mode (using smaller eps and more samples for stability in gradcheck if needed)
        # But MC gradients might be noisy, gradcheck might be tough.
        # Let's at least check they are finite.
        loss_fn_mc = FunctionalEIVLoss(
            model, sigma_x=0.1, sigma_y=0.1, monte_carlo=True, n_samples=100
        )
        # Note: MC gradients are stochastic, don't use gradcheck.
        # Ensure we check finite gradients on leaf.
        x_obs = torch.randn(batch_size, n_features_x, requires_grad=True, dtype=torch.double)
        loss_mc = loss_fn_mc(x_obs, y_obs)
        loss_mc.backward()
        for param in (x_obs, y_obs, weight, bias):
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    def test_structural_eiv_gradient_flow(self):
        """Test that gradients flow through StructuralEIVLoss properly."""
        batch_size = 4
        n_features_x = 3
        n_features_y = 2

        weight = torch.randn(n_features_x, n_features_y, requires_grad=True, dtype=torch.double)

        def model(x):
            return x @ weight

        x_obs = torch.randn(batch_size, n_features_x, requires_grad=True, dtype=torch.double)
        y_obs = torch.randn(batch_size, n_features_y, requires_grad=True, dtype=torch.double)
        sigma_xy = torch.zeros(n_features_y, n_features_x, dtype=torch.double)

        loss_fn = StructuralEIVLoss(model, sigma_x=0.1, sigma_y=0.1, sigma_xy=sigma_xy)

        assert gradcheck(lambda x, y: loss_fn(x, y), (x_obs, y_obs), eps=1e-6, atol=1e-4)

    def test_odr_gradient_flow(self):
        """Test that gradients flow through OrthogonalDistanceRegressionLoss properly."""
        batch_size = 4
        n_features_x = 3
        n_features_y = 2

        weight = torch.randn(n_features_x, n_features_y, requires_grad=True, dtype=torch.double)

        def model(x):
            return x @ weight

        x_obs = torch.randn(batch_size, n_features_x, requires_grad=True, dtype=torch.double)
        y_obs = torch.randn(batch_size, n_features_y, requires_grad=True, dtype=torch.double)

        loss_fn = OrthogonalDistanceRegressionLoss(model, sigma_x=0.1, sigma_y=0.1)

        # ODR has an inner loop and detaches the final latent x.
        # It's not fully differentiable w.r.t. inputs for gradcheck.
        # Let's at least check gradients are finite for model parameters if any,
        # or for the input itself.
        loss = loss_fn(x_obs, y_obs)
        loss.backward()
        for param in (x_obs, y_obs, weight):
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    def test_ensemble_eiv_gradient_flow(self):
        """Test that gradients flow through EnsembleEIVLoss properly."""
        batch_size = 4
        n_features_x = 3
        n_features_y = 2

        weight = torch.randn(n_features_x, n_features_y, requires_grad=True, dtype=torch.double)

        def model(x):
            return x @ weight

        x_obs = torch.randn(batch_size, n_features_x, requires_grad=True, dtype=torch.double)
        y_obs = torch.randn(batch_size, n_features_y, requires_grad=True, dtype=torch.double)

        loss_fn = EnsembleEIVLoss(model, sigma_x=0.1, n_samples=10)

        # Ensemble is just a sum of forward passes.
        # It's stochastic if it uses random perturbations, so gradcheck might fail
        # unless we fix the seed. Let's at least check gradients are finite.
        loss = loss_fn(x_obs, y_obs)
        loss.backward()
        for param in (x_obs, y_obs, weight):
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    def test_extreme_values(self):
        """Test stability with extreme values."""
        n_features_x = 3
        n_features_y = 2

        def model(x):
            return x[:, :n_features_y] * 2.0

        # Large values
        x_obs_large = (torch.ones(2, n_features_x) * 1e4).requires_grad_(True)
        y_obs_large = (torch.ones(2, n_features_y) * 1e4).requires_grad_(True)

        loss_fn = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1)
        loss = loss_fn(x_obs_large, y_obs_large)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(x_obs_large.grad).all()

    def test_nan_inf_handling(self):
        """Test how EIV loss handles NaN and Inf values with masks."""
        n_features_y = 1

        def model(x):
            return x[:, :n_features_y] * 2.0

        x_obs = torch.tensor(
            [[1.0, 2.0, 3.0], [float("nan"), 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
        )
        y_obs = torch.tensor([[1.0], [3.0], [float("inf")]], requires_grad=True)
        mask = torch.tensor([[True], [False], [False]])

        loss_fn = FunctionalEIVLoss(model, sigma_x=0.1, sigma_y=0.1)
        loss = loss_fn(x_obs, y_obs, mask=mask)
        assert torch.isfinite(loss)
        loss.backward()

        # Only row 0 should have gradients
        assert torch.isfinite(x_obs.grad[0]).all()
        assert (x_obs.grad[1:] == 0).all()

    def test_zero_variance_handling(self):
        """Test handling of zero or very small variance values."""
        n_features_x = 3
        n_features_y = 2

        def model(x):
            return x[:, :n_features_y] * 2.0

        x_obs = torch.randn(5, n_features_x, requires_grad=True)
        y_obs = torch.randn(5, n_features_y, requires_grad=True)

        # Very small variance values
        loss_fn = FunctionalEIVLoss(model, sigma_x=1e-10, sigma_y=1e-10)

        loss = loss_fn(x_obs, y_obs)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(x_obs.grad).all()


if __name__ == "__main__":
    unittest.main()
