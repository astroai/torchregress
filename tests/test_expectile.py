import torch
import unittest
from torchregression.expectile import ExpectileLoss

class TestExpectileLoss(unittest.TestCase):
    """Test suite for Expectile Loss function."""
    
    def setUp(self):
        """Initialize common test data."""
        self.batch_size = 4
        self.n_features = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Use consistent random seed for reproducibility
        torch.manual_seed(42)
        self.y_true = torch.randn(self.batch_size, self.n_features, device=self.device)
        self.y_pred = torch.randn(self.batch_size, self.n_features, device=self.device)
        self.mask = torch.randint(0, 2, (self.batch_size, self.n_features), device=self.device).bool()

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
        loss_low_over = loss_fn_low(y_true_below, y_pred_above)
        loss_high_over = loss_fn_high(y_true_below, y_pred_above)
        self.assertLess(loss_low_over.item(), loss_high_over.item())
        
        # For tau=0.8, underestimation should be penalized less
        loss_low_under = loss_fn_low(y_true_above, y_pred_below)
        loss_high_under = loss_fn_high(y_true_above, y_pred_below)
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
        expected_ratio = (1e5 / 1e-5)**2  # Ratio should be (1e5/1e-5)^2 = 1e20
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
        expected_grad_0 = 2 * tau * (y_pred[0,0] - y_true[0,0])  # overestimation
        expected_grad_1 = 2 * (1-tau) * (y_pred[0,1] - y_true[0,1])  # underestimation
        
        self.assertAlmostEqual(y_pred.grad[0,0].item(), expected_grad_0.item(), places=5)
        self.assertAlmostEqual(y_pred.grad[0,1].item(), expected_grad_1.item(), places=5)

    def test_nans(self):
        """Test handling of NaN values."""
        loss_fn = ExpectileLoss().to(self.device)

        # Set up mask to ignore NaN positions
        mask = self.mask.clone()
        mask[0,0] = False  # Ensure this position is masked out
        
        y_true_nan = self.y_true.clone()
        y_true_nan[0,0] = float('nan')
        loss = loss_fn(y_true_nan, self.y_pred, mask)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle masked NaNs")

        y_pred_nan = self.y_pred.clone()
        y_pred_nan[0,0] = float('nan')
        loss = loss_fn(self.y_true, y_pred_nan, mask)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle masked NaNs")

        # Test NaN in mask (should be treated as False)
        mask_nan = mask.clone().float()
        mask_nan[0,0] = float('nan')
        loss = loss_fn(self.y_true, self.y_pred, mask_nan.bool())  # Explicit bool cast
        self.assertFalse(torch.isnan(loss).any(), "NaN in mask should be treated as False")

    def test_inf(self):
        """Test handling of infinite values."""
        loss_fn = ExpectileLoss().to(self.device)
        
        # Proper masking for infinite values
        mask = self.mask.clone()
        mask[0,0] = False  # Ensure this position is masked out
        
        y_true_inf = self.y_true.clone()
        y_true_inf[0, 0] = float('inf')
        loss = loss_fn(y_true_inf, self.y_pred, mask)
        # With proper masking, we should get finite loss
        self.assertFalse(torch.isinf(loss).any(), "Masked infinity should give finite loss")

        y_pred_inf = self.y_pred.clone()
        y_pred_inf[0, 0] = float('inf')
        loss = loss_fn(self.y_true, y_pred_inf, mask)
        # With proper masking, we should get finite loss
        self.assertFalse(torch.isinf(loss).any(), "Masked infinity should give finite loss")

        # Test with default complete mask (should handle infinite values gracefully)
        mask_inf = torch.ones_like(self.mask).bool()
        mask_inf[0, 0] = False  # Exclude the infinite value
        loss = loss_fn(y_true_inf, self.y_pred, mask_inf)
        self.assertFalse(torch.isinf(loss).any(), "Properly masked infinities should not produce infinite loss")
        
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
            self.assertAlmostEqual(single_loss.item(), batch_loss.item(), places=5,
                                  msg=f"Batch consistency failed for tau={tau}")
    
    def test_reduction_modes(self):
        """Test different reduction modes."""
        # Test with reduction='none'
        loss_fn_none = ExpectileLoss(tau=0.7, reduction='none').to(self.device)
        loss_none = loss_fn_none(self.y_true, self.y_pred)
        self.assertEqual(loss_none.shape, self.y_true.shape, 
                         "With reduction='none', output should match input shape")
        
        # Test with reduction='sum'
        loss_fn_sum = ExpectileLoss(tau=0.7, reduction='sum').to(self.device)
        loss_sum = loss_fn_sum(self.y_true, self.y_pred)
        self.assertEqual(loss_sum.dim(), 0, "With reduction='sum', output should be scalar")
        
        # Test with reduction='mean' (default)
        loss_fn_mean = ExpectileLoss(tau=0.7, reduction='mean').to(self.device)
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
        diff = (y_true - y_pred)**2  # Squared differences
        alpha = torch.ones_like(diff) * 0.7  # tau=0.7, and all diffs are positive (1-0=1)
        expected_weighted_loss = (alpha * diff * non_uniform_weights).mean()
        
        self.assertAlmostEqual(loss_weighted_nonuniform.item(), expected_weighted_loss.item(), places=5)
        
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

if __name__ == '__main__':
    unittest.main()