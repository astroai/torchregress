import torch
import unittest
from torchregression.losses.eiv import EIVLoss
from torch.autograd import gradcheck


class TestEIVLoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_features = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.y_true = torch.randn(self.batch_size, device=self.device)
        self.y_pred = torch.randn(self.batch_size, device=self.device)
        self.x = torch.randn(self.batch_size, self.n_features, device=self.device)
        self.mask = torch.randint(0, 2, (self.batch_size,), device=self.device).bool()

    def test_eiv_loss(self):
        loss_fn = EIVLoss(error_ratio=1.0).to(self.device)

        # Test with mask
        loss = loss_fn(self.y_pred, self.y_true, x=self.x, mask=self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test without mask
        loss_no_mask = loss_fn(self.y_pred, self.y_true, x=self.x)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())

        # Test with different error_ratio values
        loss_fn_small_ratio = EIVLoss(error_ratio=0.1).to(self.device)
        loss_fn_large_ratio = EIVLoss(error_ratio=10.0).to(self.device)

        loss_small_ratio = loss_fn_small_ratio(self.y_pred, self.y_true, x=self.x)
        loss_large_ratio = loss_fn_large_ratio(self.y_pred, self.y_true, x=self.x)

        self.assertTrue(torch.is_tensor(loss_small_ratio))
        self.assertFalse(torch.isnan(loss_small_ratio).any())

        self.assertTrue(torch.is_tensor(loss_large_ratio))
        self.assertFalse(torch.isnan(loss_large_ratio).any())

    def test_eiv_edge_cases(self):
        """Test EIVLoss with edge cases: zeros, empty tensors, extreme values, NaN/Inf."""
        loss_fn = EIVLoss(error_ratio=1.0)

        # Test with zeros
        y_pred_zeros = torch.zeros(10)
        y_true_zeros = torch.zeros(10)
        x_zeros = torch.zeros(10, 5)  # Features
        assert torch.isfinite(loss_fn(y_pred_zeros, y_true_zeros, x=x_zeros))

        # Test with empty tensors
        y_pred_empty = torch.tensor([])
        y_true_empty = torch.tensor([])
        x_empty = torch.tensor([]).reshape(0, 5)  # Empty features
        assert loss_fn(y_pred_empty, y_true_empty, x=x_empty).numel() == 0

        # Test with extreme values
        y_pred_large = torch.tensor([1e10])
        y_true_small = torch.tensor([0.0])
        x_large = torch.ones(1, 5) * 1e10
        assert torch.isfinite(loss_fn(y_pred_large, y_true_small, x=x_large))

        # Test with very small values
        y_pred_small = torch.tensor([1e-10])
        y_true_small = torch.tensor([1e-10])
        x_small = torch.ones(1, 5) * 1e-10
        assert torch.isfinite(loss_fn(y_pred_small, y_true_small, x=x_small))

        # Test with NaN/Inf and masks
        y_pred = torch.tensor([1.0, float("nan"), 3.0])
        y_true = torch.tensor([1.5, 2.5, float("inf")])
        x_with_nan = torch.ones(3, 5)
        x_with_nan[1, 0] = float("nan")
        mask = torch.tensor([True, False, False])
        assert torch.isfinite(loss_fn(y_pred, y_true, x=x_with_nan, mask=mask))

        # Test with different error_ratio values
        loss_fn_small_ratio = EIVLoss(error_ratio=0.1)
        loss_fn_large_ratio = EIVLoss(error_ratio=10.0)

        assert torch.isfinite(loss_fn_small_ratio(y_pred_zeros, y_true_zeros, x=x_zeros))
        assert torch.isfinite(loss_fn_large_ratio(y_pred_zeros, y_true_zeros, x=x_zeros))


class TestEIVLossNumericalStability:
    def test_eiv_gradient_flow(self):
        """Test that gradients flow through EIVLoss properly."""
        from torchregression.losses.eiv import EIVLoss

        # Create inputs that require gradients
        batch_size = 8
        x_obs = torch.randn(batch_size, 3, requires_grad=True, dtype=torch.double)
        y_obs = torch.randn(batch_size, 2, requires_grad=True, dtype=torch.double)
        x_var = torch.exp(torch.randn(batch_size, 3, requires_grad=True, dtype=torch.double))
        y_var = torch.exp(torch.randn(batch_size, 2, requires_grad=True, dtype=torch.double))

        # Define a function for gradcheck
        def loss_fn(x_o, y_o, x_v, y_v):
            return EIVLoss(reduction="mean")(x_o, y_o, x_v, y_v)

        # Test with gradcheck
        assert gradcheck(loss_fn, (x_obs, y_obs, x_var, y_var), eps=1e-6, atol=1e-4)

    def test_multi_eiv_gradient_flow(self):
        """Test that gradients flow through MultiOutputEIVLoss properly."""
        from torchregression.losses.eiv import MultiOutputEIVLoss

        # Create inputs that require gradients
        batch_size = 8
        x_dims = 3
        y_dims = 2

        x_obs = torch.randn(batch_size, x_dims, requires_grad=True, dtype=torch.double)
        y_obs = torch.randn(batch_size, y_dims, requires_grad=True, dtype=torch.double)
        coefficients = torch.randn(x_dims, y_dims, requires_grad=True, dtype=torch.double)
        bias = torch.randn(y_dims, requires_grad=True, dtype=torch.double)
        x_var = torch.exp(torch.randn(batch_size, x_dims, requires_grad=True, dtype=torch.double))
        y_var = torch.exp(torch.randn(batch_size, y_dims, requires_grad=True, dtype=torch.double))

        # Define a function for gradcheck
        def loss_fn(x_o, y_o, coef, b, x_v, y_v):
            return MultiOutputEIVLoss(reduction="mean")(x_o, y_o, coef, b, x_v, y_v)

        # Test with gradcheck - use smaller inputs if needed for computational feasibility
        assert gradcheck(
            loss_fn, (x_obs, y_obs, coefficients, bias, x_var, y_var), eps=1e-6, atol=1e-4
        )

    def test_extreme_values(self):
        """Test stability with extreme values."""
        from torchregression.losses.eiv import EIVLoss

        # Large values
        x_obs_large = torch.tensor([[1e3, 1e3, 1e3], [1e5, 1e5, 1e5]], requires_grad=True)
        y_obs_large = torch.tensor([[1e3, 1e3], [1e5, 1e5]], requires_grad=True)
        x_var_large = torch.tensor([[1e2, 1e2, 1e2], [1e4, 1e4, 1e4]], requires_grad=True)
        y_var_large = torch.tensor([[1e2, 1e2], [1e4, 1e4]], requires_grad=True)

        # Small values
        x_obs_small = torch.tensor([[1e-3, 1e-3, 1e-3], [1e-5, 1e-5, 1e-5]], requires_grad=True)
        y_obs_small = torch.tensor([[1e-3, 1e-3], [1e-5, 1e-5]], requires_grad=True)
        x_var_small = torch.tensor([[1e-2, 1e-2, 1e-2], [1e-4, 1e-4, 1e-4]], requires_grad=True)
        y_var_small = torch.tensor([[1e-2, 1e-2], [1e-4, 1e-4]], requires_grad=True)

        eiv_loss = EIVLoss(reduction="mean")

        # Test with large values
        loss = eiv_loss(x_obs_large, y_obs_large, x_var_large, y_var_large)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(x_obs_large.grad))
        assert torch.all(torch.isfinite(y_obs_large.grad))
        assert torch.all(torch.isfinite(x_var_large.grad))
        assert torch.all(torch.isfinite(y_var_large.grad))

        # Reset gradients
        x_obs_small.grad = None
        y_obs_small.grad = None
        x_var_small.grad = None
        y_var_small.grad = None

        # Test with small values
        loss = eiv_loss(x_obs_small, y_obs_small, x_var_small, y_var_small)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(x_obs_small.grad))
        assert torch.all(torch.isfinite(y_obs_small.grad))
        assert torch.all(torch.isfinite(x_var_small.grad))
        assert torch.all(torch.isfinite(y_var_small.grad))

    def test_nan_inf_handling(self):
        """Test how EIV loss handles NaN and Inf values with masks."""
        from torchregression.losses.eiv import EIVLoss

        # Create data with some NaNs and Infs
        x_obs = torch.tensor(
            [[1.0, 2.0, 3.0], [float("nan"), 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
        )
        y_obs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [float("inf"), 6.0]], requires_grad=True)
        x_var = torch.tensor(
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]], requires_grad=True
        )
        y_var = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], requires_grad=True)

        # Create mask that excludes rows with NaNs/Infs
        mask = torch.tensor([True, False, False])

        eiv_loss = EIVLoss(reduction="mean")

        # This should only use the valid elements
        loss = eiv_loss(x_obs, y_obs, x_var, y_var, mask=mask)
        assert torch.isfinite(loss)
        loss.backward()

        # Only the unmasked elements should have gradients
        assert torch.all(torch.isfinite(x_obs.grad[0]))
        assert torch.all(torch.isfinite(y_obs.grad[0]))
        assert torch.all(torch.isfinite(x_var.grad[0]))
        assert torch.all(torch.isfinite(y_var.grad[0]))

        # Masked elements should have zero gradient
        assert torch.all(x_obs.grad[1] == 0.0)
        assert torch.all(y_obs.grad[1] == 0.0)
        assert torch.all(x_var.grad[1] == 0.0)
        assert torch.all(y_var.grad[1] == 0.0)

        assert torch.all(x_obs.grad[2] == 0.0)
        assert torch.all(y_obs.grad[2] == 0.0)
        assert torch.all(x_var.grad[2] == 0.0)
        assert torch.all(y_var.grad[2] == 0.0)

    def test_reduction_modes(self):
        """Test different reduction modes for backward pass."""
        from torchregression.losses.eiv import EIVLoss

        x_obs = torch.randn(10, 3, requires_grad=True)
        y_obs = torch.randn(10, 2, requires_grad=True)
        x_var = torch.exp(torch.randn(10, 3, requires_grad=True))
        y_var = torch.exp(torch.randn(10, 2, requires_grad=True))

        # Test mean reduction
        eiv_mean = EIVLoss(reduction="mean")
        loss = eiv_mean(x_obs, y_obs, x_var, y_var)
        loss.backward()

        x_mean_grad = x_obs.grad.clone()
        y_mean_grad = y_obs.grad.clone()
        x_var_mean_grad = x_var.grad.clone()
        y_var_mean_grad = y_var.grad.clone()

        # Reset gradients
        x_obs.grad = None
        y_obs.grad = None
        x_var.grad = None
        y_var.grad = None

        # Test sum reduction
        eiv_sum = EIVLoss(reduction="sum")
        loss = eiv_sum(x_obs, y_obs, x_var, y_var)
        loss.backward()

        x_sum_grad = x_obs.grad.clone()
        y_sum_grad = y_obs.grad.clone()
        x_var_sum_grad = x_var.grad.clone()
        y_var_sum_grad = y_var.grad.clone()

        # Reset gradients
        x_obs.grad = None
        y_obs.grad = None
        x_var.grad = None
        y_var.grad = None

        # Test none reduction
        eiv_none = EIVLoss(reduction="none")
        loss = eiv_none(x_obs, y_obs, x_var, y_var)
        loss.mean().backward()

        x_none_grad = x_obs.grad.clone()
        y_none_grad = y_obs.grad.clone()
        x_var_none_grad = x_var.grad.clone()
        y_var_none_grad = y_var.grad.clone()

        # Mean and sum should give different gradients
        assert not torch.allclose(x_mean_grad, x_sum_grad)
        assert not torch.allclose(y_mean_grad, y_sum_grad)

        # Mean and manual mean over none should be similar
        assert torch.allclose(x_mean_grad, x_none_grad, rtol=1e-5, atol=1e-5)
        assert torch.allclose(y_mean_grad, y_none_grad, rtol=1e-5, atol=1e-5)

    def test_zero_variance_handling(self):
        """Test handling of zero or very small variance values."""
        from torchregression.losses.eiv import EIVLoss

        x_obs = torch.randn(5, 3, requires_grad=True)
        y_obs = torch.randn(5, 2, requires_grad=True)

        # Very small variance values
        x_var_small = torch.ones_like(x_obs) * 1e-10
        y_var_small = torch.ones_like(y_obs) * 1e-10
        x_var_small.requires_grad_(True)
        y_var_small.requires_grad_(True)

        eiv_loss = EIVLoss(reduction="mean")

        # Should handle very small variance values without numerical issues
        loss = eiv_loss(x_obs, y_obs, x_var_small, y_var_small)
        assert torch.isfinite(loss)
        loss.backward()

        assert torch.all(torch.isfinite(x_obs.grad))
        assert torch.all(torch.isfinite(y_obs.grad))
        assert torch.all(torch.isfinite(x_var_small.grad))
        assert torch.all(torch.isfinite(y_var_small.grad))


if __name__ == "__main__":
    unittest.main()
