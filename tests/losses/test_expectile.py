import torch
import unittest
from torch.autograd import gradcheck
from torchregression.losses.expectile import (
    ExpectileLoss,
    MultiExpectileLoss,
    ExpectileCrossoverLoss,
    AsymmetricLeastSquaresLoss,
)


class TestExpectileLoss(unittest.TestCase):
    """Test suite for Expectile Loss function."""

    def setUp(self):
        """Initialize common test data."""
        self.batch_size = 4
        self.n_features = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use consistent random seed for reproducibility
        torch.manual_seed(42)
        self.y_true = torch.randn(self.batch_size, self.n_features, device=self.device)
        self.y_pred = torch.randn(self.batch_size, self.n_features, device=self.device)
        self.mask = torch.randint(
            0, 2, (self.batch_size, self.n_features), device=self.device
        ).bool()

    def test_expectile_loss(self):
        """Test basic functionality of ExpectileLoss."""
        # Test with non-median tau
        loss_fn = ExpectileLoss(tau=0.7).to(self.device)
        loss = loss_fn(self.y_true, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test without mask
        loss_no_mask = loss_fn(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())

        # Verify consistent dimensions
        self.assertEqual(loss.dim(), 0, "Loss should be a scalar")

    def test_expectile_asymmetry(self):
        """Test asymmetric behavior of expectile loss with different tau values."""
        # Create data where predictions are all above true values
        y_true_below = torch.zeros(self.batch_size, self.n_features, device=self.device)
        y_pred_above = torch.ones(self.batch_size, self.n_features, device=self.device)

        # Create data where predictions are all below true values
        y_true_above = torch.ones(self.batch_size, self.n_features, device=self.device)
        y_pred_below = torch.zeros(self.batch_size, self.n_features, device=self.device)

        # Test with low tau (e.g., 0.2)
        tau_low = 0.2
        loss_fn_low = ExpectileLoss(tau=tau_low).to(self.device)

        # Test with high tau (e.g., 0.8)
        tau_high = 0.8
        loss_fn_high = ExpectileLoss(tau=tau_high).to(self.device)

        # For tau=0.2, overestimation should be penalized less
        loss_low_over = loss_fn_low(y_pred_above, y_true_below)
        loss_high_over = loss_fn_high(y_pred_above, y_true_below)
        self.assertLess(loss_low_over.item(), loss_high_over.item())

        # For tau=0.8, underestimation should be penalized less
        loss_low_under = loss_fn_low(y_pred_below, y_true_above)
        loss_high_under = loss_fn_high(y_pred_below, y_true_above)
        self.assertGreater(loss_low_under.item(), loss_high_under.item())

        # Check mathematical property: expectile weighting for overestimation
        diff = 1.0  # Difference between y_pred and y_true
        manual_low_over = tau_low * diff**2  # For overestimation with tau_low
        manual_high_over = tau_high * diff**2  # For overestimation with tau_high
        ratio = manual_low_over / manual_high_over
        self.assertAlmostEqual(ratio, tau_low / tau_high, places=5)

    def test_expectile_with_tau_0_5(self):
        """Test that expectile with tau=0.5 is equivalent to MSE."""
        loss_fn = ExpectileLoss(tau=0.5).to(self.device)
        expectile_loss = loss_fn(self.y_true, self.y_pred)

        # MSE loss
        mse_loss = torch.nn.MSELoss()(self.y_true, self.y_pred)

        # With tau=0.5, expectile loss should be identical to MSE
        self.assertAlmostEqual(expectile_loss.item(), mse_loss.item(), places=5)

        # Test with various scales of data to ensure consistency
        for scale in [0.01, 1.0, 100.0]:
            scaled_y_true = self.y_true * scale
            scaled_y_pred = self.y_pred * scale
            expectile_scaled = loss_fn(scaled_y_true, scaled_y_pred).item()
            mse_scaled = torch.nn.MSELoss()(scaled_y_true, scaled_y_pred).item()
            self.assertAlmostEqual(expectile_scaled, mse_scaled, places=5)

    def test_tau_validation(self):
        """Test validation of tau parameter."""
        # tau must be between 0 and 1
        with self.assertRaises(ValueError):
            ExpectileLoss(tau=-0.1)

        with self.assertRaises(ValueError):
            ExpectileLoss(tau=1.1)

        # Edge cases should work
        try:
            ExpectileLoss(tau=0.0)
            ExpectileLoss(tau=1.0)
        except ValueError:
            self.fail("ExpectileLoss raised ValueError unexpectedly for tau=0.0 or tau=1.0")

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        loss_fn = ExpectileLoss(tau=0.7).to(self.device)

        # Test with large differences
        y_true_extreme = torch.ones(self.batch_size, self.n_features, device=self.device) * 1e5
        y_pred_extreme = torch.zeros(self.batch_size, self.n_features, device=self.device)

        loss_extreme = loss_fn(y_true_extreme, y_pred_extreme)
        self.assertFalse(torch.isnan(loss_extreme).any())
        self.assertFalse(torch.isinf(loss_extreme).any())

        # Test with small differences
        y_true_small = torch.ones(self.batch_size, self.n_features, device=self.device) * 1e-5
        y_pred_small = torch.zeros(self.batch_size, self.n_features, device=self.device)

        loss_small = loss_fn(y_true_small, y_pred_small)
        self.assertFalse(torch.isnan(loss_small).any())
        self.assertFalse(torch.isinf(loss_small).any())

        # Verify scale-appropriate results
        # For small values, loss should be proportional to square of values
        ratio = (loss_extreme / loss_small).item()
        expected_ratio = (1e5 / 1e-5) ** 2  # Ratio should be (1e5/1e-5)^2 = 1e20
        # Allow significant tolerance due to floating point precision
        self.assertTrue(abs(ratio / expected_ratio - 1.0) < 0.01)

    def test_gradient(self):
        """Test gradient computation and correctness."""
        tau = 0.7
        loss_fn = ExpectileLoss(tau=tau).to(self.device)

        # Create inputs that require gradients
        y_true = torch.tensor([[1.0, 2.0]], device=self.device)
        y_pred = torch.tensor([[1.5, 1.5]], device=self.device, requires_grad=True)

        # Compute loss and gradients
        loss = loss_fn(y_true, y_pred)
        loss.backward()

        # Verify gradients exist
        self.assertIsNotNone(y_pred.grad)

        # For overestimation (y_pred[0,0] > y_true[0,0]), gradient should be 2*tau*(y_pred-y_true)
        # For underestimation (y_pred[0,1] < y_true[0,1]), gradient should be 2*(1-tau)*(y_pred-y_true)
        expected_grad_0 = 2 * tau * (y_pred[0, 0] - y_true[0, 0])  # overestimation
        expected_grad_1 = 2 * (1 - tau) * (y_pred[0, 1] - y_true[0, 1])  # underestimation

        self.assertAlmostEqual(y_pred.grad[0, 0].item(), expected_grad_0.item(), places=5)
        self.assertAlmostEqual(y_pred.grad[0, 1].item(), expected_grad_1.item(), places=5)

    def test_nans(self):
        """Test handling of NaN values."""
        loss_fn = ExpectileLoss().to(self.device)

        # Set up mask to ignore NaN positions
        mask = self.mask.clone()
        mask[0, 0] = False  # Ensure this position is masked out

        y_true_nan = self.y_true.clone()
        y_true_nan[0, 0] = float("nan")
        loss = loss_fn(y_true_nan, self.y_pred, mask)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle masked NaNs")

        y_pred_nan = self.y_pred.clone()
        y_pred_nan[0, 0] = float("nan")
        loss = loss_fn(self.y_true, y_pred_nan, mask)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle masked NaNs")

        # Test NaN in mask (should be treated as False)
        mask_nan = mask.clone().float()
        mask_nan[0, 0] = float("nan")
        loss = loss_fn(self.y_true, self.y_pred, mask_nan.bool())  # Explicit bool cast
        self.assertFalse(torch.isnan(loss).any(), "NaN in mask should be treated as False")

    def test_inf(self):
        """Test handling of infinite values."""
        loss_fn = ExpectileLoss().to(self.device)

        # Proper masking for infinite values
        mask = self.mask.clone()
        mask[0, 0] = False  # Ensure this position is masked out

        y_true_inf = self.y_true.clone()
        y_true_inf[0, 0] = float("inf")
        loss = loss_fn(y_true_inf, self.y_pred, mask)
        # With proper masking, we should get finite loss
        self.assertFalse(torch.isinf(loss).any(), "Masked infinity should give finite loss")

        y_pred_inf = self.y_pred.clone()
        y_pred_inf[0, 0] = float("inf")
        loss = loss_fn(self.y_true, y_pred_inf, mask)
        # With proper masking, we should get finite loss
        self.assertFalse(torch.isinf(loss).any(), "Masked infinity should give finite loss")

        # Test with default complete mask (should handle infinite values gracefully)
        mask_inf = torch.ones_like(self.mask).bool()
        mask_inf[0, 0] = False  # Exclude the infinite value
        loss = loss_fn(y_true_inf, self.y_pred, mask_inf)
        self.assertFalse(
            torch.isinf(loss).any(), "Properly masked infinities should not produce infinite loss"
        )

    def test_batch_consistency(self):
        """Test that loss calculations are consistent across batch dimensions."""
        # Create a batch with identical data
        batch_size = 10
        y_true_batch = torch.ones(batch_size, 1, device=self.device)
        y_pred_batch = torch.zeros(batch_size, 1, device=self.device)

        # Test each tau value
        for tau in [0.2, 0.5, 0.8]:
            loss_fn = ExpectileLoss(tau=tau).to(self.device)
            # Single sample loss
            single_loss = loss_fn(y_true_batch[0:1], y_pred_batch[0:1])
            # Batch loss
            batch_loss = loss_fn(y_true_batch, y_pred_batch)

            # The average loss should be the same regardless of batch size
            self.assertAlmostEqual(
                single_loss.item(),
                batch_loss.item(),
                places=5,
                msg=f"Batch consistency failed for tau={tau}",
            )

    def test_reduction_modes(self):
        """Test different reduction modes."""
        # Test with reduction='none'
        loss_fn_none = ExpectileLoss(tau=0.7, reduction="none").to(self.device)
        loss_none = loss_fn_none(self.y_true, self.y_pred)
        self.assertEqual(
            loss_none.shape,
            self.y_true.shape,
            "With reduction='none', output should match input shape",
        )

        # Test with reduction='sum'
        loss_fn_sum = ExpectileLoss(tau=0.7, reduction="sum").to(self.device)
        loss_sum = loss_fn_sum(self.y_true, self.y_pred)
        self.assertEqual(loss_sum.dim(), 0, "With reduction='sum', output should be scalar")

        # Test with reduction='mean' (default)
        loss_fn_mean = ExpectileLoss(tau=0.7, reduction="mean").to(self.device)
        loss_mean = loss_fn_mean(self.y_true, self.y_pred)
        self.assertEqual(loss_mean.dim(), 0, "With reduction='mean', output should be scalar")

        # Verify that sum = mean * n_elements for unmasked data
        n_elements = self.y_true.numel()
        self.assertAlmostEqual(loss_sum.item(), loss_mean.item() * n_elements, places=5)

    def test_weights(self):
        """Test that weights are properly applied to the loss."""
        loss_fn = ExpectileLoss(tau=0.7).to(self.device)

        # Create uniform prediction/target data
        y_true = torch.ones(self.batch_size, self.n_features, device=self.device)
        y_pred = torch.zeros(self.batch_size, self.n_features, device=self.device)

        # Test with uniform weights (should be same as unweighted)
        uniform_weights = torch.ones(self.batch_size, self.n_features, device=self.device)
        loss_weighted_uniform = loss_fn(y_true, y_pred, weights=uniform_weights)
        loss_unweighted = loss_fn(y_true, y_pred)
        self.assertAlmostEqual(loss_weighted_uniform.item(), loss_unweighted.item(), places=5)

        # Test with non-uniform weights
        non_uniform_weights = torch.ones(self.batch_size, self.n_features, device=self.device)
        non_uniform_weights[:, 0] = 2.0  # Double weight for first feature
        loss_weighted_nonuniform = loss_fn(y_true, y_pred, weights=non_uniform_weights)

        # Calculate expected weighted loss manually for verification
        diff = (y_true - y_pred) ** 2  # Squared differences
        alpha = torch.ones_like(diff) * 0.7  # tau=0.7, and all diffs are positive (1-0=1)
        expected_weighted_loss = (alpha * diff * non_uniform_weights).mean()

        self.assertAlmostEqual(
            loss_weighted_nonuniform.item(), expected_weighted_loss.item(), places=5
        )

        # Test with per-sample weights (batch dimension only)
        per_sample_weights = torch.ones(self.batch_size, device=self.device)
        per_sample_weights[0] = 2.0  # Double weight for first sample
        loss_weighted_per_sample = loss_fn(y_true, y_pred, weights=per_sample_weights)

        # This should weight the first sample's loss twice as much
        self.assertNotEqual(loss_weighted_per_sample.item(), loss_unweighted.item())

        # Test with mask and weights
        mask = torch.ones(self.batch_size, self.n_features, device=self.device).bool()
        mask[0, 0] = False  # Mask out one element
        loss_masked_weighted = loss_fn(y_true, y_pred, mask=mask, weights=non_uniform_weights)
        self.assertFalse(torch.isnan(loss_masked_weighted).any())

    def test_asymmetric_least_squares_alias(self):
        """Test AsymmetricLeastSquaresLoss as alias for ExpectileLoss."""
        # Create same expectile value with different class names
        tau = 0.7
        expectile_loss = ExpectileLoss(expectile=tau).to(self.device)
        als_loss = AsymmetricLeastSquaresLoss(tau=tau).to(self.device)

        # Create test data
        y_true = torch.randn(self.batch_size, self.n_features, device=self.device)
        y_pred = torch.randn(self.batch_size, self.n_features, device=self.device)

        # Both should give identical results
        loss1 = expectile_loss(y_pred, y_true)
        loss2 = als_loss(y_pred, y_true)

        self.assertAlmostEqual(loss1.item(), loss2.item(), places=6)

        # Test with mask
        mask = torch.randint(0, 2, (self.batch_size, self.n_features), device=self.device).bool()
        loss1_masked = expectile_loss(y_pred, y_true, mask)
        loss2_masked = als_loss(y_pred, y_true, mask)

        self.assertAlmostEqual(loss1_masked.item(), loss2_masked.item(), places=6)

        # Test with weights
        weights = torch.rand(self.batch_size, self.n_features, device=self.device)
        loss1_weighted = expectile_loss(y_pred, y_true, weights=weights)
        loss2_weighted = als_loss(y_pred, y_true, weights=weights)

        self.assertAlmostEqual(loss1_weighted.item(), loss2_weighted.item(), places=6)

    def test_multi_expectile_loss_basics(self):
        """Test basic functionality of MultiExpectileLoss."""
        expectiles = [0.1, 0.5, 0.9]
        loss_fn = MultiExpectileLoss(expectiles=expectiles).to(self.device)

        # Test with joint prediction format [batch_size, num_expectiles, n_features]
        batch_size = 5
        n_features = 3
        num_expectiles = len(expectiles)

        y_true = torch.randn(batch_size, n_features, device=self.device)
        y_pred = torch.randn(batch_size, num_expectiles, n_features, device=self.device)

        loss = loss_fn(y_pred, y_true)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0, "Loss should be a scalar")
        self.assertFalse(torch.isnan(loss).any())

        # Test with flat prediction format [batch_size, n_features * num_expectiles]
        y_pred_flat = y_pred.reshape(batch_size, num_expectiles * n_features)
        loss_flat = loss_fn(y_pred_flat, y_true)

        # Both formats should produce the same loss
        self.assertAlmostEqual(loss.item(), loss_flat.item(), places=5)

    def test_multi_expectile_separate_predictions(self):
        """Test MultiExpectileLoss with separate predictions."""
        expectiles = [0.1, 0.5, 0.9]
        loss_fn = MultiExpectileLoss(expectiles=expectiles, joint_prediction=False).to(self.device)

        batch_size = 5
        n_features = 3

        y_true = torch.randn(batch_size, n_features, device=self.device)

        # Create list of predictions, one per expectile level
        y_pred_list = [
            torch.randn(batch_size, n_features, device=self.device) for _ in range(len(expectiles))
        ]

        loss = loss_fn(y_pred_list, y_true)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0, "Loss should be a scalar")
        self.assertFalse(torch.isnan(loss).any())

        # Test with incorrect list length
        wrong_pred_list = y_pred_list[:-1]  # One fewer than expected
        with self.assertRaises(TypeError):
            loss_fn(wrong_pred_list, y_true)

    def test_multi_expectile_mask_and_weights(self):
        """Test MultiExpectileLoss with mask and weights."""
        expectiles = [0.2, 0.5, 0.8]
        loss_fn = MultiExpectileLoss(expectiles=expectiles).to(self.device)

        batch_size = 4
        n_features = 3
        num_expectiles = len(expectiles)

        y_true = torch.randn(batch_size, n_features, device=self.device)
        y_pred = torch.randn(batch_size, num_expectiles, n_features, device=self.device)

        # Create mask and weights
        mask = torch.randint(0, 2, (batch_size, n_features), device=self.device).bool()
        weights = torch.rand(batch_size, n_features, device=self.device)

        # Test with mask
        loss_masked = loss_fn(y_pred, y_true, mask=mask)
        self.assertFalse(torch.isnan(loss_masked).any())

        # Test with weights
        loss_weighted = loss_fn(y_pred, y_true, weights=weights)
        self.assertFalse(torch.isnan(loss_weighted).any())

        # Test with both mask and weights
        loss_both = loss_fn(y_pred, y_true, mask=mask, weights=weights)
        self.assertFalse(torch.isnan(loss_both).any())

        # Test with per-sample weights [batch_size]
        per_sample_weights = torch.rand(batch_size, device=self.device)
        loss_per_sample = loss_fn(y_pred, y_true, weights=per_sample_weights)
        self.assertFalse(torch.isnan(loss_per_sample).any())

    def test_multi_expectile_reduction(self):
        """Test different reduction modes for MultiExpectileLoss."""
        expectiles = [0.2, 0.5, 0.8]

        batch_size = 4
        n_features = 3
        num_expectiles = len(expectiles)

        y_true = torch.randn(batch_size, n_features, device=self.device)
        y_pred = torch.randn(batch_size, num_expectiles, n_features, device=self.device)

        # Test with reduction='none'
        loss_fn_none = MultiExpectileLoss(expectiles=expectiles, reduction="none").to(self.device)
        loss_none = loss_fn_none(y_pred, y_true)
        self.assertEqual(loss_none.shape, (batch_size,))

        # Test with reduction='sum'
        loss_fn_sum = MultiExpectileLoss(expectiles=expectiles, reduction="sum").to(self.device)
        loss_sum = loss_fn_sum(y_pred, y_true)
        self.assertEqual(loss_sum.dim(), 0)

        # Test with reduction='mean' (default)
        loss_fn_mean = MultiExpectileLoss(expectiles=expectiles, reduction="mean").to(self.device)
        loss_mean = loss_fn_mean(y_pred, y_true)
        self.assertEqual(loss_mean.dim(), 0)

        # Verify sum = mean * batch_size
        self.assertAlmostEqual(loss_sum.item(), loss_mean.item() * batch_size, places=5)

    def test_expectile_crossover_basics(self):
        """Test basic functionality of ExpectileCrossoverLoss loss."""
        expectiles = [0.1, 0.5, 0.9]
        loss_fn = ExpectileCrossoverLoss(expectiles=expectiles).to(self.device)

        batch_size = 4
        n_features = 3
        num_expectiles = len(expectiles)

        y_true = torch.randn(batch_size, n_features, device=self.device)

        # Create well-ordered predictions (no crossover)
        # For each sample, expectiles are strictly increasing
        y_pred_good = torch.zeros(batch_size, num_expectiles, n_features, device=self.device)
        for i in range(num_expectiles):
            y_pred_good[:, i, :] = i

        # Create predictions with crossovers
        y_pred_bad = torch.zeros(batch_size, num_expectiles, n_features, device=self.device)
        # Reverse order to create crossovers
        for i in range(num_expectiles):
            y_pred_bad[:, i, :] = num_expectiles - i - 1

        # Loss should be higher for predictions with crossovers
        loss_good = loss_fn(y_pred_good, y_true)
        loss_bad = loss_fn(y_pred_bad, y_true)

        self.assertLess(
            loss_good.item(), loss_bad.item(), "Loss with crossover violations should be higher"
        )

    def test_expectile_crossover_penalty(self):
        """Test that crossover penalty works correctly."""
        expectiles = [0.1, 0.5, 0.9]
        base_loss_weight = 1.0
        crossover_penalty = 10.0

        loss_fn = ExpectileCrossoverLoss(
            expectiles=expectiles, base_loss=base_loss_weight, crossover_penalty=crossover_penalty
        ).to(self.device)

        batch_size = 1  # Simplified test with single sample
        n_features = 1  # And single feature
        num_expectiles = len(expectiles)

        # Create target value in the middle
        y_true = torch.zeros(batch_size, n_features, device=self.device)

        # Case 1: Perfect ordering, no crossover
        y_pred_good = torch.tensor([[[-1.0], [0.0], [1.0]]], device=self.device)
        loss_good = loss_fn(y_pred_good, y_true)

        # Case 2: Complete reversal, maximum crossover
        y_pred_bad = torch.tensor([[[1.0], [0.0], [-1.0]]], device=self.device)
        loss_bad = loss_fn(y_pred_bad, y_true)

        # Calculate expected penalties
        # No violation in good case
        expected_good_penalty = 0.0
        # In bad case: first crossover = 1.0-0.0=1.0, second crossover = 0.0-(-1.0)=1.0, total=2.0
        expected_bad_penalty = 2.0 * crossover_penalty

        # Base losses should be similar, but overall loss should differ by penalty amount
        expected_diff = expected_bad_penalty - expected_good_penalty
        actual_diff = loss_bad.item() - loss_good.item()

        # Allow for some calculation differences, but should be close to the expected difference
        self.assertAlmostEqual(actual_diff, expected_diff, delta=0.5)

    def test_expectile_crossover_mask_and_weights(self):
        """Test ExpectileCrossoverLoss with mask and weights."""
        expectiles = [0.1, 0.5, 0.9]
        loss_fn = ExpectileCrossoverLoss(expectiles=expectiles).to(self.device)

        batch_size = 4
        n_features = 3
        num_expectiles = len(expectiles)

        y_true = torch.randn(batch_size, n_features, device=self.device)
        y_pred = torch.randn(batch_size, num_expectiles, n_features, device=self.device)

        # Create mask and weights
        mask = torch.randint(0, 2, (batch_size, n_features), device=self.device).bool()
        weights = torch.rand(batch_size, n_features, device=self.device)

        # Test with mask
        loss_masked = loss_fn(y_pred, y_true, mask=mask)
        self.assertFalse(torch.isnan(loss_masked).any())

        # Test with weights
        loss_weighted = loss_fn(y_pred, y_true, weights=weights)
        self.assertFalse(torch.isnan(loss_weighted).any())

        # Test with both mask and weights
        loss_both = loss_fn(y_pred, y_true, mask=mask, weights=weights)
        self.assertFalse(torch.isnan(loss_both).any())

    def test_expectile_crossover_reduction(self):
        """Test different reduction modes for ExpectileCrossoverLoss."""
        expectiles = [0.1, 0.5, 0.9]

        batch_size = 4
        n_features = 3
        num_expectiles = len(expectiles)

        y_true = torch.randn(batch_size, n_features, device=self.device)
        y_pred = torch.randn(batch_size, num_expectiles, n_features, device=self.device)

        # Test with reduction='none'
        loss_fn_none = ExpectileCrossoverLoss(expectiles=expectiles, reduction="none").to(self.device)
        loss_none = loss_fn_none(y_pred, y_true)
        self.assertEqual(loss_none.shape, (batch_size,))

        # Test with reduction='sum'
        loss_fn_sum = ExpectileCrossoverLoss(expectiles=expectiles, reduction="sum").to(self.device)
        loss_sum = loss_fn_sum(y_pred, y_true)
        self.assertEqual(loss_sum.dim(), 0)

        # Test with reduction='mean' (default)
        loss_fn_mean = ExpectileCrossoverLoss(expectiles=expectiles, reduction="mean").to(self.device)
        loss_mean = loss_fn_mean(y_pred, y_true)
        self.assertEqual(loss_mean.dim(), 0)

        # Verify sum = mean * batch_size
        self.assertAlmostEqual(loss_sum.item(), loss_mean.item() * batch_size, places=5)

    def test_multi_expectile_nan_inf(self):
        """Test MultiExpectileLoss with NaN and Inf values."""
        expectiles = [0.2, 0.5, 0.8]
        loss_fn = MultiExpectileLoss(expectiles=expectiles).to(self.device)

        batch_size = 4
        n_features = 3
        num_expectiles = len(expectiles)

        y_true = torch.randn(batch_size, n_features, device=self.device)
        y_pred = torch.randn(batch_size, num_expectiles, n_features, device=self.device)

        # Create mask that excludes problematic values
        mask = torch.ones(batch_size, n_features, dtype=torch.bool, device=self.device)
        mask[0, 0] = False  # Mask first element

        # Insert NaN into predictions
        y_pred_nan = y_pred.clone()
        y_pred_nan[0, 0, 0] = float("nan")
        loss_nan = loss_fn(y_pred_nan, y_true, mask=mask)
        self.assertFalse(torch.isnan(loss_nan).any(), "Loss should handle masked NaNs")

        # Insert Inf into predictions
        y_pred_inf = y_pred.clone()
        y_pred_inf[0, 0, 0] = float("inf")
        loss_inf = loss_fn(y_pred_inf, y_true, mask=mask)
        self.assertFalse(torch.isinf(loss_inf).any(), "Loss should handle masked Infs")

        # Insert NaN into targets
        y_true_nan = y_true.clone()
        y_true_nan[0, 0] = float("nan")
        loss_target_nan = loss_fn(y_pred, y_true_nan, mask=mask)
        self.assertFalse(
            torch.isnan(loss_target_nan).any(), "Loss should handle masked NaNs in targets"
        )

    def test_expectile_crossover_nan_inf(self):
        """Test ExpectileCrossoverLoss with NaN and Inf values."""
        expectiles = [0.2, 0.5, 0.8]
        loss_fn = ExpectileCrossoverLoss(expectiles=expectiles).to(self.device)

        batch_size = 4
        n_features = 3
        num_expectiles = len(expectiles)

        y_true = torch.randn(batch_size, n_features, device=self.device)
        y_pred = torch.randn(batch_size, num_expectiles, n_features, device=self.device)

        # Create mask that excludes problematic values
        mask = torch.ones(batch_size, n_features, dtype=torch.bool, device=self.device)
        mask[0, 0] = False  # Mask first element

        # Insert NaN into predictions
        y_pred_nan = y_pred.clone()
        y_pred_nan[0, 0, 0] = float("nan")
        loss_nan = loss_fn(y_pred_nan, y_true, mask=mask)
        self.assertFalse(torch.isnan(loss_nan).any(), "Loss should handle masked NaNs")

        # Insert Inf into predictions
        y_pred_inf = y_pred.clone()
        y_pred_inf[0, 0, 0] = float("inf")
        loss_inf = loss_fn(y_pred_inf, y_true, mask=mask)
        self.assertFalse(torch.isinf(loss_inf).any(), "Loss should handle masked Infs")

        # Insert NaN into targets
        y_true_nan = y_true.clone()
        y_true_nan[0, 0] = float("nan")
        loss_target_nan = loss_fn(y_pred, y_true_nan, mask=mask)
        self.assertFalse(
            torch.isnan(loss_target_nan).any(), "Loss should handle masked NaNs in targets"
        )

    def test_multi_expectile_gradient(self):
        """Test gradient computation for MultiExpectileLoss."""
        expectiles = [0.2, 0.5, 0.8]
        loss_fn = MultiExpectileLoss(expectiles=expectiles).to(self.device)

        batch_size = 2
        n_features = 2
        num_expectiles = len(expectiles)

        # Create simple test data
        y_true = torch.ones(batch_size, n_features, device=self.device)
        y_pred = torch.zeros(
            batch_size, num_expectiles, n_features, device=self.device, requires_grad=True
        )

        # Forward pass
        loss = loss_fn(y_pred, y_true)
        loss.backward()

        # Check gradients exist and have expected shape
        self.assertIsNotNone(y_pred.grad)
        self.assertEqual(y_pred.grad.shape, y_pred.shape)

        # For y_pred < y_true, gradients should be negative for all expectiles
        # (pushing predictions up toward target)
        self.assertTrue((y_pred.grad <= 0).all())

        # Higher expectile levels should have larger absolute gradients
        avg_grad_low = y_pred.grad[:, 0, :].abs().mean().item()
        avg_grad_mid = y_pred.grad[:, 1, :].abs().mean().item()
        avg_grad_high = y_pred.grad[:, 2, :].abs().mean().item()

        # For underestimation, gradients should be more negative as tau increases
        self.assertLess(avg_grad_low, avg_grad_high)

    def test_expectile_crossover_gradient(self):
        """Test gradient computation for ExpectileCrossoverLoss."""
        expectiles = [0.2, 0.5, 0.8]
        base_loss = 1.0
        crossover_penalty = 10.0

        loss_fn = ExpectileCrossoverLoss(
            expectiles=expectiles, base_loss=base_loss, crossover_penalty=crossover_penalty
        ).to(self.device)

        batch_size = 2
        n_features = 2
        num_expectiles = len(expectiles)

        # Create targets
        y_true = torch.zeros(batch_size, n_features, device=self.device)

        # Test with crossover violation (requires_grad=True)
        # Higher expectile prediction below lower expectile prediction
        y_pred = torch.zeros(
            batch_size, num_expectiles, n_features, device=self.device, requires_grad=True
        )
        # Create crossover: 0.2 expectile > 0.5 expectile > 0.8 expectile
        y_pred[:, 0, :] = 1.0  # 0.2 expectile
        y_pred[:, 1, :] = 0.0  # 0.5 expectile
        y_pred[:, 2, :] = -1.0  # 0.8 expectile

        # Forward and backward pass
        loss = loss_fn(y_pred, y_true)
        loss.backward()

        # Check gradients exist and have expected shape
        self.assertIsNotNone(y_pred.grad)
        self.assertEqual(y_pred.grad.shape, y_pred.shape)

        # Gradient for lower expectiles (that are too high) should be positive
        # (penalty pushing them down)
        self.assertTrue((y_pred.grad[:, 0, :] > 0).all())

        # Gradient for higher expectiles (that are too low) should be negative
        # (penalty pushing them up)
        self.assertTrue((y_pred.grad[:, 2, :] < 0).all())

    def test_expectile_edge_cases(self):
        """Test ExpectileLoss with edge cases: zeros, empty tensors, extreme values, NaN/Inf."""
        loss_fn = ExpectileLoss(expectile=0.5)

        # Test with zeros
        y_pred_zeros = torch.zeros(10)
        y_true_zeros = torch.zeros(10)
        assert torch.isclose(loss_fn(y_pred_zeros, y_true_zeros), torch.tensor(0.0))

        # Test with empty tensors
        y_pred_empty = torch.tensor([])
        y_true_empty = torch.tensor([])
        assert loss_fn(y_pred_empty, y_true_empty).numel() == 0

        # Test with extreme values
        y_pred_large = torch.tensor([1e10])
        y_true_small = torch.tensor([0.0])
        assert torch.isfinite(loss_fn(y_pred_large, y_true_small))

        # Test with NaN/Inf and masks
        y_pred = torch.tensor([1.0, float("nan"), 3.0])
        y_true = torch.tensor([1.5, 2.5, float("inf")])
        mask = torch.tensor([True, False, False])
        assert torch.isfinite(loss_fn(y_pred, y_true, mask))


class TestExpectileLossNumericalStability:
    def test_expectile_gradient_flow(self):
        """Test that gradients flow through ExpectileLoss properly."""
        from torchregression.losses.expectile import ExpectileLoss

        # Create inputs that require gradients
        y_pred = torch.randn(10, 1, requires_grad=True, dtype=torch.double)
        y_true = torch.randn(10, 1, dtype=torch.double)

        # Test with gradcheck for default expectile (tau=0.5)
        loss_fn = ExpectileLoss(reduction="mean", expectile=0.5)
        assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-4)

        # Test with gradcheck for lower expectile (tau=0.1)
        loss_fn = ExpectileLoss(reduction="mean", expectile=0.1)
        assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-4)

        # Test with gradcheck for upper expectile (tau=0.9)
        loss_fn = ExpectileLoss(reduction="mean", expectile=0.9)
        assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-4)

    def test_extreme_values(self):
        """Test stability with extreme values."""
        from torchregression.losses.expectile import ExpectileLoss

        # Large values
        y_pred_large = torch.tensor([1e5, 1e7, 1e9], requires_grad=True)
        y_true_large = torch.tensor([1e5 + 1, 1e7 + 10, 1e9 + 100])

        # Small values
        y_pred_small = torch.tensor([1e-5, 1e-7, 1e-9], requires_grad=True)
        y_true_small = torch.tensor([1e-5 + 1e-6, 1e-7 + 1e-8, 1e-9 + 1e-10])

        # Test with extreme values for symmetric expectile (tau=0.5)
        expectile_loss = ExpectileLoss(reduction="mean", expectile=0.5)

        # Large values test
        loss = expectile_loss(y_pred_large, y_true_large)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_large.grad))

        # Small values test
        y_pred_small.grad = None
        loss = expectile_loss(y_pred_small, y_true_small)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_small.grad))

    def test_nan_inf_handling(self):
        """Test how expectile loss handles NaN and Inf values with masks."""
        from torchregression.losses.expectile import ExpectileLoss

        # Create data with some NaNs and Infs
        y_pred = torch.tensor([1.0, float("nan"), 3.0, float("inf")], requires_grad=True)
        y_true = torch.tensor([1.1, 2.0, float("nan"), 4.0])
        mask = torch.tensor([True, False, False, True])  # Mask out NaNs and Infs

        expectile_loss = ExpectileLoss(reduction="mean", expectile=0.5)

        # This should only use the valid elements
        loss = expectile_loss(y_pred, y_true, mask=mask)
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
        from torchregression.losses.expectile import ExpectileLoss

        y_pred = torch.randn(10, 1, requires_grad=True)
        y_true = torch.randn(10, 1)

        # Test mean reduction
        expectile_mean = ExpectileLoss(reduction="mean", expectile=0.5)
        loss = expectile_mean(y_pred, y_true)
        loss.backward()
        mean_grad = y_pred.grad.clone()

        # Test sum reduction
        y_pred.grad = None
        expectile_sum = ExpectileLoss(reduction="sum", expectile=0.5)
        loss = expectile_sum(y_pred, y_true)
        loss.backward()
        sum_grad = y_pred.grad.clone()

        # Test none reduction
        y_pred.grad = None
        expectile_none = ExpectileLoss(reduction="none", expectile=0.5)
        loss = expectile_none(y_pred, y_true)
        loss.mean().backward()
        none_grad = y_pred.grad.clone()

        # Mean and sum should give different gradients
        assert not torch.allclose(mean_grad, sum_grad)
        # Mean and manual mean over none should be similar
        assert torch.allclose(mean_grad, none_grad)


if __name__ == "__main__":
    unittest.main()
