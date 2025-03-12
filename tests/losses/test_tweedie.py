import torch
import unittest
from torch.autograd import gradcheck
from torchregression.losses.tweedie import (
    TweedieLoss,
    GammaLoss,
    InverseGaussianLoss,
    CompoundPoissonLoss,
)


class TestTweedieLoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_features = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.y_true = torch.randn(self.batch_size, self.n_features, device=self.device).float()
        # For Gamma and compound tests, we need positive y_true values
        self.y_true_pos = (
            torch.rand(self.batch_size, self.n_features, device=self.device) * 5 + 0.1
        )  # Positive true values
        self.y_pred = (
            torch.rand(self.batch_size, self.n_features, device=self.device) * 5 + 0.1
        )  # Positive predictions
        self.mask = torch.randint(
            0, 2, (self.batch_size, self.n_features), device=self.device
        ).bool()
        self.weights = torch.rand(
            self.batch_size, self.n_features, device=self.device
        )  # test weights

    def test_tweedie_loss_gaussian(self):
        """Test Tweedie loss with p=0 (Gaussian distribution)"""
        loss_fn = TweedieLoss(p=0.0).to(self.device)  # Gaussian
        loss = loss_fn(self.y_true, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

    def test_tweedie_loss_poisson(self):
        """Test Tweedie loss with p=1 (Poisson distribution)"""
        loss_fn = TweedieLoss(p=1.0).to(self.device)  # Poisson
        loss = loss_fn(self.y_true_pos, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

    def test_tweedie_loss_gamma(self):
        """Test Tweedie loss with p=2 (Gamma distribution)"""
        loss_fn = TweedieLoss(p=2.0).to(self.device)  # Gamma
        loss = loss_fn(self.y_true_pos, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

    def test_tweedie_loss_compound(self):
        """Test Tweedie loss with p=1.5 (Compound Poisson-Gamma distribution)"""
        loss_fn = TweedieLoss(p=1.5).to(self.device)  # Compound Poisson-Gamma
        loss = loss_fn(self.y_true_pos, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

    def test_tweedie_loss_weighted(self):
        """Test Tweedie loss with sample weights"""
        loss_fn = TweedieLoss(p=1.5).to(self.device)
        loss_unweighted = loss_fn(self.y_true_pos, self.y_pred, self.mask)
        loss_weighted = loss_fn(self.y_true_pos, self.y_pred, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss_weighted))
        self.assertFalse(torch.isnan(loss_weighted).any())
        # Weighted and unweighted should be different
        self.assertNotEqual(loss_unweighted.item(), loss_weighted.item())

    def test_tweedie_loss_reduction(self):
        """Test different reduction methods"""
        # Test mean reduction (default)
        loss_fn_mean = TweedieLoss(p=1.0, reduction="mean").to(self.device)
        loss_mean = loss_fn_mean(self.y_true_pos, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss_mean))
        self.assertEqual(loss_mean.dim(), 0)  # Scalar output

        # Test sum reduction
        loss_fn_sum = TweedieLoss(p=1.0, reduction="sum").to(self.device)
        loss_sum = loss_fn_sum(self.y_true_pos, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss_sum))
        self.assertEqual(loss_sum.dim(), 0)  # Scalar output

        # Test none reduction
        loss_fn_none = TweedieLoss(p=1.0, reduction="none").to(self.device)
        loss_none = loss_fn_none(self.y_true_pos, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss_none))
        self.assertEqual(loss_none.shape, self.y_true_pos.shape)  # Per-element loss

    def test_tweedie_loss_nans(self):
        """Test that NaNs are handled correctly in inputs"""
        loss_fn = TweedieLoss(p=1.5).to(self.device)

        # Test NaN in true values
        y_true_nan = self.y_true_pos.clone()
        y_true_nan[0, 0] = float("nan")
        loss = loss_fn(y_true_nan, self.y_pred, self.mask)
        self.assertFalse(torch.isnan(loss).any())

        # Test NaN in predictions
        y_pred_nan = self.y_pred.clone()
        y_pred_nan[0, 0] = float("nan")
        loss = loss_fn(self.y_true_pos, y_pred_nan, self.mask)
        self.assertFalse(torch.isnan(loss).any())

        # Test NaN in mask - using boolean mask
        mask_nan = self.mask.clone()
        # Set an entry to False that would have a NaN in the calculation
        y_true_nan[0, 1] = float("nan")
        mask_nan[0, 1] = False  # Mask out the NaN value
        loss = loss_fn(y_true_nan, self.y_pred, mask_nan)
        self.assertFalse(torch.isnan(loss).any())

    def test_tweedie_loss_edge_cases(self):
        """Test edge cases like zero predictions and extreme values"""
        loss_fn = TweedieLoss(p=1.5).to(self.device)

        # Test with zero predictions (should handle numerical stability)
        y_pred_zeros = torch.zeros_like(self.y_pred)
        # Make a mask that avoids mathematical impossibilities
        safe_mask = self.mask.clone()
        safe_mask[self.y_true_pos == 0] = False
        loss = loss_fn(self.y_true_pos, y_pred_zeros, safe_mask)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())

        # Test with very small predictions
        y_pred_small = torch.ones_like(self.y_pred) * 1e-10
        loss = loss_fn(self.y_true_pos, y_pred_small, safe_mask)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())

        # Test with very large predictions
        y_pred_large = torch.ones_like(self.y_pred) * 1e10
        loss = loss_fn(self.y_true_pos, y_pred_large, safe_mask)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())

    def test_tweedie_loss_invalid_p(self):
        """Test that invalid p values raise appropriate errors"""
        with self.assertRaises(ValueError):
            TweedieLoss(p=0.5)  # Invalid p
        with self.assertRaises(ValueError):
            TweedieLoss(p=-1)  # Invalid p

    # Add tests for specialized loss classes
    def test_gamma_loss(self):
        """Test the GammaLoss class"""
        # Test initialization
        loss_fn = GammaLoss().to(self.device)
        self.assertEqual(loss_fn.p, 2.0)  # Gamma uses p=2

        # Test calculation
        loss = loss_fn(self.y_pred, self.y_true_pos, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())

        # Compare with base TweedieLoss with p=2
        base_loss_fn = TweedieLoss(p=2.0).to(self.device)
        base_loss = base_loss_fn(self.y_pred, self.y_true_pos, self.mask)
        self.assertAlmostEqual(loss.item(), base_loss.item(), places=5)

    def test_inverse_gaussian_loss(self):
        """Test the InverseGaussianLoss class"""
        # Test initialization
        loss_fn = InverseGaussianLoss().to(self.device)
        self.assertEqual(loss_fn.p, 3.0)  # Inverse Gaussian uses p=3

        # Test calculation
        loss = loss_fn(self.y_pred, self.y_true_pos, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())

        # Compare with base TweedieLoss with p=3
        base_loss_fn = TweedieLoss(p=3.0).to(self.device)
        base_loss = base_loss_fn(self.y_pred, self.y_true_pos, self.mask)
        self.assertAlmostEqual(loss.item(), base_loss.item(), places=5)

    def test_compound_poisson_loss(self):
        """Test the CompoundPoissonLoss class"""
        # Test initialization
        loss_fn = CompoundPoissonLoss(p=1.5).to(self.device)
        self.assertEqual(loss_fn.p, 1.5)  # Using p=1.5

        # Test with invalid p values
        with self.assertRaises(ValueError):
            CompoundPoissonLoss(p=0.9)  # p < 1
        with self.assertRaises(ValueError):
            CompoundPoissonLoss(p=2.1)  # p > 2

        # Test calculation
        loss = loss_fn(self.y_pred, self.y_true_pos, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())

        # Compare with base TweedieLoss with p=1.5
        base_loss_fn = TweedieLoss(p=1.5).to(self.device)
        base_loss = base_loss_fn(self.y_pred, self.y_true_pos, self.mask)
        self.assertAlmostEqual(loss.item(), base_loss.item(), places=5)

        # Test with zero values
        y_mixed = self.y_true_pos.clone()
        y_mixed[0, :] = 0.0  # First batch element has zeros
        loss = loss_fn(self.y_pred, y_mixed, self.mask)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())

    def test_link_functions(self):
        """Test different link functions"""
        # Prepare data with positive predictions for both link functions
        y_pred_pos = torch.rand(self.batch_size, self.n_features, device=self.device) * 5 + 0.1

        # For identity link, y_pred is directly the mean
        loss_fn_identity = TweedieLoss(p=1.5, link="identity").to(self.device)
        loss_identity = loss_fn_identity(y_pred_pos, self.y_true_pos, self.mask)

        # For log link, y_pred is log(mean)
        y_pred_log = torch.log(y_pred_pos)
        loss_fn_log = TweedieLoss(p=1.5, link="log").to(self.device)
        loss_log = loss_fn_log(y_pred_log, self.y_true_pos, self.mask)

        # Both should produce valid losses
        self.assertFalse(torch.isnan(loss_identity).any())
        self.assertFalse(torch.isnan(loss_log).any())

        # Invalid link function
        with self.assertRaises(ValueError):
            TweedieLoss(p=1.5, link="invalid")

    def test_gradient_flow(self):
        """Test that gradients flow properly through the loss"""
        # Create a tensor that requires gradients
        y_pred = torch.rand(
            self.batch_size, self.n_features, device=self.device, requires_grad=True
        )

        # Compute loss for different p values
        for p in [0.0, 1.0, 1.5, 2.0, 3.0]:
            loss_fn = TweedieLoss(p=p).to(self.device)
            loss = loss_fn(y_pred, self.y_true_pos, self.mask)

            # Check gradient flow
            loss.backward(retain_graph=True)
            self.assertIsNotNone(y_pred.grad)
            self.assertFalse(torch.isnan(y_pred.grad).any())
            self.assertFalse(torch.isinf(y_pred.grad).any())

            # Reset gradients for next iteration
            y_pred.grad.zero_()

    def test_extreme_inputs(self):
        """Test with extreme input values"""
        loss_fn = TweedieLoss(p=1.5).to(self.device)

        # Very large target values
        y_true_large = torch.ones_like(self.y_true_pos) * 1e6
        loss = loss_fn(self.y_pred, y_true_large, self.mask)
        self.assertFalse(torch.isnan(loss).any())

        # Mix of zero and positive values in targets
        y_mixed = self.y_true_pos.clone()
        y_mixed[0, :] = 0.0
        loss = loss_fn(self.y_pred, y_mixed, self.mask)
        self.assertFalse(torch.isnan(loss).any())

        # Test with sparse masks (few True values)
        sparse_mask = torch.zeros_like(self.mask).bool()
        sparse_mask[0, 0] = True  # Only one True value
        loss = loss_fn(self.y_pred, self.y_true_pos, sparse_mask)
        self.assertFalse(torch.isnan(loss).any())

    def test_tweedie_edge_cases(self):
        """Test TweedieLoss with edge cases: zeros, empty tensors, extreme values, NaN/Inf."""
        loss_fn = TweedieLoss(power=1.5)  # Power between 1-2 for compound Poisson-gamma

        # Test with zeros - for some powers, this requires special handling
        y_pred_zeros = torch.ones(10) * 1e-6  # Small positive values
        y_true_zeros = torch.zeros(10)
        assert torch.isfinite(loss_fn(y_pred_zeros, y_true_zeros))

        # Test with empty tensors
        y_pred_empty = torch.tensor([])
        y_true_empty = torch.tensor([])
        assert loss_fn(y_pred_empty, y_true_empty).numel() == 0

        # Test with extreme values
        y_pred_large = torch.tensor([1e10])
        y_true_large = torch.tensor([1e5])
        assert torch.isfinite(loss_fn(y_pred_large, y_true_large))

        # Test with small positive values
        y_pred_small = torch.tensor([1e-5])
        y_true_small = torch.tensor([1e-5])
        assert torch.isfinite(loss_fn(y_pred_small, y_true_small))

        # Test with NaN/Inf and masks
        y_pred = torch.tensor([1.0, float("nan"), 3.0])
        y_true = torch.tensor([1.5, 2.5, float("inf")])
        mask = torch.tensor([True, False, False])
        assert torch.isfinite(loss_fn(y_pred, y_true, mask))


class TestTweedieLossNumericalStability:
    def test_tweedie_gradient_flow(self):
        """Test that gradients flow through TweedieLoss properly."""
        from torchregression.losses.tweedie import TweedieLoss

        # Create inputs that require gradients
        y_pred = torch.exp(torch.randn(10, 1, requires_grad=True, dtype=torch.double))
        y_true = torch.abs(torch.randn(10, 1, dtype=torch.double)) + 0.5  # positive values

        # Test with gradcheck for different p values
        for p in [1.1, 1.5, 1.9]:
            loss_fn = TweedieLoss(reduction="mean", p=p, log_input=False)
            assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-4)

        # Test with log input
        y_pred_log = torch.log(y_pred.detach()).requires_grad_(True)
        loss_fn_log = TweedieLoss(reduction="mean", p=1.5, log_input=True)
        assert gradcheck(loss_fn_log, (y_pred_log, y_true), eps=1e-6, atol=1e-4)

    def test_extreme_values(self):
        """Test stability with extreme values."""
        from torchregression.losses.tweedie import TweedieLoss

        # Small values (close to zero)
        y_pred_small = torch.tensor([1e-3, 1e-4, 1e-5], requires_grad=True)
        # Large values
        y_pred_large = torch.tensor([1e3, 1e4, 1e5], requires_grad=True)

        y_true = torch.tensor([1.0, 10.0, 100.0])

        # Test with non-log input
        tweedie_loss = TweedieLoss(reduction="mean", p=1.5, log_input=False)

        # Small values test
        loss = tweedie_loss(y_pred_small, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_small.grad))

        # Large values test
        y_pred_large.grad = None
        loss = tweedie_loss(y_pred_large, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_large.grad))

        # Test with log input
        log_tweedie_loss = TweedieLoss(reduction="mean", p=1.5, log_input=True)

        # Log of small/large values
        y_pred_log_small = torch.log(y_pred_small.detach()).requires_grad_(True)
        y_pred_log_large = torch.log(y_pred_large.detach()).requires_grad_(True)

        # Small values test (log input)
        loss = log_tweedie_loss(y_pred_log_small, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_log_small.grad))

        # Large values test (log input)
        y_pred_log_large.grad = None
        loss = log_tweedie_loss(y_pred_log_large, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_log_large.grad))

    def test_zero_targets(self):
        """Test handling of zero values in targets."""
        from torchregression.losses.tweedie import TweedieLoss

        # Predictions with gradient
        y_pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        # Targets with zeros
        y_true = torch.tensor([0.0, 1.0, 0.0])

        # Non-log input
        tweedie_loss = TweedieLoss(reduction="mean", p=1.5, log_input=False)
        loss = tweedie_loss(y_pred, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred.grad))

        # Log input
        y_pred.grad = None
        log_pred = torch.log(y_pred).requires_grad_(True)
        log_tweedie_loss = TweedieLoss(reduction="mean", p=1.5, log_input=True)
        loss = log_tweedie_loss(log_pred, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(log_pred.grad))

    def test_nan_inf_handling(self):
        """Test how tweedie loss handles NaN and Inf values with masks."""
        from torchregression.losses.tweedie import TweedieLoss

        # Create data with some NaNs and Infs
        y_pred = torch.tensor([1.0, float("nan"), 3.0, float("inf")], requires_grad=True)
        y_true = torch.tensor([1.1, 2.0, float("nan"), 4.0])
        mask = torch.tensor([True, False, False, True])  # Mask out NaNs and Infs

        tweedie_loss = TweedieLoss(reduction="mean", p=1.5)

        # This should only use the valid elements
        loss = tweedie_loss(y_pred, y_true, mask=mask)
        assert torch.isfinite(loss)
        loss.backward()
        # Only the unmasked elements should have gradients
        assert torch.isfinite(y_pred.grad[0])
        assert torch.isfinite(y_pred.grad[3])
        # Masked elements should have zero gradient
        assert y_pred.grad[1] == 0.0
        assert y_pred.grad[2] == 0.0

    def test_reduction_modes(self):
        """Test different reduction modes for backward pass."""
        from torchregression.losses.tweedie import TweedieLoss

        y_pred = torch.exp(torch.randn(10, 1, requires_grad=True))
        y_true = torch.abs(torch.randn(10, 1)) + 0.1  # positive values

        # Test mean reduction
        tweedie_mean = TweedieLoss(reduction="mean", p=1.5, log_input=False)
        loss = tweedie_mean(y_pred, y_true)
        loss.backward()
        mean_grad = y_pred.grad.clone()

        # Test sum reduction
        y_pred.grad = None
        tweedie_sum = TweedieLoss(reduction="sum", p=1.5, log_input=False)
        loss = tweedie_sum(y_pred, y_true)
        loss.backward()
        sum_grad = y_pred.grad.clone()

        # Test none reduction
        y_pred.grad = None
        tweedie_none = TweedieLoss(reduction="none", p=1.5, log_input=False)
        loss = tweedie_none(y_pred, y_true)
        loss.mean().backward()
        none_grad = y_pred.grad.clone()

        # Mean and sum should give different gradients
        assert not torch.allclose(mean_grad, sum_grad)
        # Mean and manual mean over none should be similar
        assert torch.allclose(mean_grad, none_grad, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
