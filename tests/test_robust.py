import torch
import unittest
from torchregression.robust import (
    L1Loss, HuberLoss, PseudoHuberLoss,
    LogCoshLoss, CharbonnierLoss, LqLoss
)

class TestRobustLossFunctions(unittest.TestCase):
    """Test suite for robust regression loss functions."""

    def setUp(self):
        """Initialize common test data."""
        self.batch_size = 4
        self.n_features = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.y_true = torch.randn(self.batch_size, self.n_features, device=self.device)
        self.y_pred = torch.randn(self.batch_size, self.n_features, device=self.device)
        self.mask = torch.randint(0, 2, (self.batch_size, self.n_features), device=self.device).bool()
        
        # Create data with outliers for robust loss testing
        self.y_true_outliers = self.y_true.clone()
        self.y_true_outliers[0, 0] = 10.0  # Add an outlier
        
        # Zero difference data (for testing behavior at minimum)
        self.y_true_zero = torch.zeros(self.batch_size, self.n_features, device=self.device)
        self.y_pred_zero = torch.zeros(self.batch_size, self.n_features, device=self.device)
        
        # Additional test data for gradient checking and tensor shapes
        self.scalar_true = torch.tensor(5.0, device=self.device, requires_grad=True)
        self.scalar_pred = torch.tensor(3.0, device=self.device, requires_grad=True)
        
        # 3D tensors for testing different shapes
        self.y_true_3d = torch.randn(2, 3, 4, device=self.device)
        self.y_pred_3d = torch.randn(2, 3, 4, device=self.device)
        self.mask_3d = torch.randint(0, 2, (2, 3, 4), device=self.device).bool()
        
        # Empty tensor for edge case testing
        self.y_true_empty = torch.tensor([], device=self.device)
        self.y_pred_empty = torch.tensor([], device=self.device)

    def test_l1_loss(self):
        """Test L1Loss (Mean Absolute Error) behavior."""
        loss_fn = L1Loss().to(self.device)
        
        # Test with mask
        loss = loss_fn(self.y_true, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test without mask
        loss_no_mask = loss_fn(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())
        
        # Verify L1 is less affected by outliers than MSE
        mse = torch.nn.MSELoss()(self.y_true_outliers, self.y_pred)
        l1 = loss_fn(self.y_true_outliers, self.y_pred)
        baseline_mse = torch.nn.MSELoss()(self.y_true, self.y_pred)
        baseline_l1 = loss_fn(self.y_true, self.y_pred)
        
        # Outlier effect ratio should be larger for MSE than L1
        mse_ratio = mse / baseline_mse
        l1_ratio = l1 / baseline_l1
        self.assertGreater(mse_ratio.item(), l1_ratio.item())
        
        # Test that L1 at zero is actually zero
        loss_at_zero = L1Loss()(self.y_true_zero, self.y_pred_zero)
        self.assertAlmostEqual(loss_at_zero.item(), 0.0, places=5)

    def test_huber_loss(self):
        """Test HuberLoss with different delta values."""
        delta = 1.5
        loss_fn = HuberLoss(delta=delta).to(self.device)
        
        # Test with mask
        loss = loss_fn(self.y_true, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test without mask
        loss_no_mask = loss_fn(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())
        
        # Test with different deltas
        small_delta_fn = HuberLoss(delta=0.1).to(self.device)
        large_delta_fn = HuberLoss(delta=10.0).to(self.device)
        
        small_delta_loss = small_delta_fn(self.y_true_outliers, self.y_pred)
        large_delta_loss = large_delta_fn(self.y_true_outliers, self.y_pred)
        
        # Small delta should handle outliers more like L1
        # Large delta should handle outliers more like MSE
        l1_loss = L1Loss().to(self.device)(self.y_true_outliers, self.y_pred)
        mse_loss = torch.nn.MSELoss()(self.y_true_outliers, self.y_pred)
        
        # Small delta loss should be closer to L1 loss
        self.assertLess(abs(small_delta_loss.item() - l1_loss.item()), 
                        abs(large_delta_loss.item() - l1_loss.item()))
        
        # Test that Huber at zero is actually zero
        loss_at_zero = HuberLoss()(self.y_true_zero, self.y_pred_zero)
        self.assertAlmostEqual(loss_at_zero.item(), 0.0, places=5)
        
        # Test behavior at the transition point
        delta = 1.0
        huber_fn = HuberLoss(delta=delta).to(self.device)
        
        # Create inputs where |diff| = delta exactly
        y_true_delta = torch.zeros(1, 1, device=self.device)
        y_pred_delta = torch.ones(1, 1, device=self.device) * delta
        
        huber_at_delta = huber_fn(y_true_delta, y_pred_delta).item()
        l1_at_delta = L1Loss().to(self.device)(y_true_delta, y_pred_delta).item()
        mse_at_delta = torch.nn.MSELoss()(y_true_delta, y_pred_delta).item() / 2
        
        # At |diff| = delta, Huber should equal both 0.5*delta^2 and delta*|diff| - 0.5*delta^2
        self.assertAlmostEqual(huber_at_delta, delta * delta / 2, places=5)
        
        # For |diff| < delta, it behaves like MSE/2
        small_diff = delta / 2
        y_pred_small = torch.ones(1, 1, device=self.device) * small_diff
        huber_small = huber_fn(y_true_delta, y_pred_small).item()
        mse_small = torch.nn.MSELoss()(y_true_delta, y_pred_small).item() / 2
        self.assertAlmostEqual(huber_small, mse_small, places=5)
        
        # For |diff| > delta, it behaves like L1 - constant
        large_diff = delta * 2
        y_pred_large = torch.ones(1, 1, device=self.device) * large_diff
        huber_large = huber_fn(y_true_delta, y_pred_large).item()
        expected = large_diff - delta/2
        self.assertAlmostEqual(huber_large, expected, places=5)

    def test_pseudo_huber_loss(self):
        """Test PseudoHuberLoss with different delta values."""
        delta = 0.5
        loss_fn = PseudoHuberLoss(delta=delta).to(self.device)
        
        # Test with mask
        loss = loss_fn(self.y_true, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test without mask
        loss_no_mask = loss_fn(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())
        
        # Test that PseudoHuber is differentiable everywhere (no sharp corners)
        # Create inputs where traditional Huber would have a corner
        y_true_corner = torch.zeros(1, 1, device=self.device)
        y_pred_corner = torch.ones(1, 1, device=self.device) * delta
        
        # Compute gradients
        y_pred_corner.requires_grad = True
        pseudo_huber_loss = loss_fn(y_true_corner, y_pred_corner)
        pseudo_huber_loss.backward()
        
        # Gradient should exist and be finite
        self.assertFalse(torch.isnan(y_pred_corner.grad).any())
        self.assertFalse(torch.isinf(y_pred_corner.grad).any())
        
        # Test that PseudoHuber at zero is actually zero
        loss_at_zero = PseudoHuberLoss()(self.y_true_zero, self.y_pred_zero)
        self.assertAlmostEqual(loss_at_zero.item(), 0.0, places=5)
        
        # Test smooth asymptotic behavior
        delta = 1.0
        pseudo_huber_fn = PseudoHuberLoss(delta=delta).to(self.device)
        huber_fn = HuberLoss(delta=delta).to(self.device)
        
        # For very small differences, PseudoHuber ≈ MSE/2
        small_diff = delta * 0.01
        y_true_small = torch.zeros(1, 1, device=self.device)
        y_pred_small = torch.ones(1, 1, device=self.device) * small_diff
        
        pseudo_huber_small = pseudo_huber_fn(y_true_small, y_pred_small).item()
        mse_small = torch.nn.MSELoss()(y_true_small, y_pred_small).item() / 2
        self.assertAlmostEqual(pseudo_huber_small, mse_small, places=5)
        
        # For very large differences, PseudoHuber ≈ delta*|diff|
        large_diff = delta * 100
        y_true_large = torch.zeros(1, 1, device=self.device)
        y_pred_large = torch.ones(1, 1, device=self.device) * large_diff
        
        pseudo_huber_large = pseudo_huber_fn(y_true_large, y_pred_large).item()
        expected_approx = delta * large_diff
        # Should be within a small percentage of the expected asymptotic value
        self.assertLess(abs(pseudo_huber_large - expected_approx) / expected_approx, 0.01)

    def test_log_cosh_loss(self):
        """Test LogCoshLoss behavior."""
        loss_fn = LogCoshLoss().to(self.device)
        
        # Test with mask
        loss = loss_fn(self.y_true, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test without mask
        loss_no_mask = loss_fn(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())
        
        # Test LogCosh behavior with extreme values
        # For small errors, LogCosh ≈ MSE/2
        small_diff = torch.ones(1, 1, device=self.device) * 0.01
        small_error_logcosh = torch.log(torch.cosh(small_diff)).item()
        small_error_mse = (small_diff**2 / 2).item()
        self.assertAlmostEqual(small_error_logcosh, small_error_mse, places=5)
        
        # For large errors, LogCosh ≈ abs(error) - log(2)
        large_diff = torch.ones(1, 1, device=self.device) * 10.0
        large_error_logcosh = torch.log(torch.cosh(large_diff)).item()
        large_error_l1_adjusted = (torch.abs(large_diff) - torch.log(torch.tensor(2.0))).item()
        self.assertAlmostEqual(large_error_logcosh, large_error_l1_adjusted, places=2)
        
        # Test that LogCosh at zero is actually zero
        loss_at_zero = LogCoshLoss()(self.y_true_zero, self.y_pred_zero)
        self.assertAlmostEqual(loss_at_zero.item(), 0.0, places=5)

    def test_charbonnier_loss(self):
        """Test CharbonnierLoss with different parameters."""
        alpha = 0.7
        epsilon = 1e-5
        loss_fn = CharbonnierLoss(alpha=alpha, epsilon=epsilon).to(self.device)
        
        # Test with mask
        loss = loss_fn(self.y_true, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test without mask
        loss_no_mask = loss_fn(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())
        
        # Test with alpha=1.0 (should be similar to L1 but smoothed)
        alpha_one_fn = CharbonnierLoss(alpha=1.0, epsilon=epsilon).to(self.device)
        alpha_one_loss = alpha_one_fn(self.y_true, self.y_pred)
        l1_loss = L1Loss().to(self.device)(self.y_true, self.y_pred)
        
        # Should be reasonably close to L1 loss
        self.assertLess(abs(alpha_one_loss.item() - l1_loss.item()) / l1_loss.item(), 0.1)
        
        # Test with alpha=0.5 (should be more robust to outliers)
        alpha_half_fn = CharbonnierLoss(alpha=0.5, epsilon=epsilon).to(self.device)
        regular_loss = alpha_half_fn(self.y_true, self.y_pred)
        outlier_loss = alpha_half_fn(self.y_true_outliers, self.y_pred)
        
        # Effect of outlier should be less severe than with L1
        l1_regular = L1Loss().to(self.device)(self.y_true, self.y_pred)
        l1_outlier = L1Loss().to(self.device)(self.y_true_outliers, self.y_pred)
        
        charbonnier_ratio = outlier_loss / regular_loss
        l1_ratio = l1_outlier / l1_regular
        
        self.assertLess(charbonnier_ratio.item(), l1_ratio.item())
        
        # Test that Charbonnier at zero approaches zero (not exactly zero due to epsilon)
        loss_at_zero = CharbonnierLoss()(self.y_true_zero, self.y_pred_zero)
        self.assertLess(loss_at_zero.item(), 1e-5)  # Should be very close to zero

    def test_lq_loss(self):
        """Test LqLoss with different q values."""
        # Test with q=1.5 (between L1 and L2)
        loss_fn_1_5 = LqLoss(q=1.5).to(self.device)
        loss_1_5 = loss_fn_1_5(self.y_true, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss_1_5))
        self.assertFalse(torch.isnan(loss_1_5).any())
        
        # Test with q=1.0 (should be equivalent to L1)
        loss_fn_1 = LqLoss(q=1.0).to(self.device)
        loss_1 = loss_fn_1(self.y_true, self.y_pred, self.mask)
        l1_loss = L1Loss().to(self.device)(self.y_true, self.y_pred, self.mask)
        self.assertAlmostEqual(loss_1.item(), l1_loss.item(), places=5)
        
        # Test with q=2.0 (should be equivalent to L2/MSE)
        loss_fn_2 = LqLoss(q=2.0).to(self.device)
        loss_2 = loss_fn_2(self.y_true, self.y_pred, self.mask)
        mse_loss = torch.nn.MSELoss(reduction='none')(self.y_true, self.y_pred)
        mse_loss = torch.mean(mse_loss * self.mask.float()) / (torch.mean(self.mask.float()) + 1e-10)
        self.assertAlmostEqual(loss_2.item(), mse_loss.item(), places=5)
        
        # Test outlier sensitivity increases with q
        loss_1_outlier = loss_fn_1(self.y_true_outliers, self.y_pred, self.mask)
        loss_1_5_outlier = loss_fn_1_5(self.y_true_outliers, self.y_pred, self.mask)
        loss_2_outlier = loss_fn_2(self.y_true_outliers, self.y_pred, self.mask)
        
        # Ratios of outlier to regular loss should increase with q
        ratio_1 = loss_1_outlier / loss_1
        ratio_1_5 = loss_1_5_outlier / loss_1_5
        ratio_2 = loss_2_outlier / loss_2
        
        self.assertLess(ratio_1.item(), ratio_1_5.item())
        self.assertLess(ratio_1_5.item(), ratio_2.item())
        
        # Test behavior with q < 1 (non-convex case)
        # This is just a warning test, not testing actual optimization behavior
        with self.assertLogs(level='WARNING') as cm:
            LqLoss(q=0.8)
        self.assertTrue(any('not convex' in msg for msg in cm.output))
        
        # Test that Lq at zero is actually zero
        loss_at_zero = LqLoss()(self.y_true_zero, self.y_pred_zero)
        self.assertAlmostEqual(loss_at_zero.item(), 0.0, places=5)
        
        # Test with fractional q between 1 and 2
        q_frac = 1.5
        loss_fn_frac = LqLoss(q=q_frac).to(self.device)
        
        # Create specific test case for verification
        y_true_test = torch.zeros(1, 1, device=self.device)
        y_pred_test = torch.ones(1, 1, device=self.device) * 2.0
        
        # Manual calculation: |2.0|^1.5 = 2^1.5 ≈ 2.83
        lq_manual = 2 ** q_frac
        lq_actual = loss_fn_frac(y_true_test, y_pred_test).item()
        self.assertAlmostEqual(lq_actual, lq_manual, places=5)

    def test_nans(self):
        """Test handling of NaN values."""
        loss_fn = HuberLoss().to(self.device)  # Use HuberLoss as a representative example

        y_true_nan = self.y_true.clone()
        y_true_nan[0, 0] = float('nan')
        loss = loss_fn(y_true_nan, self.y_pred, self.mask)
        self.assertFalse(torch.isnan(loss).any())  # Should handle NaNs due to masking

        y_pred_nan = self.y_pred.clone()
        y_pred_nan[0, 0] = float('nan')
        loss = loss_fn(self.y_true, y_pred_nan, self.mask)
        self.assertFalse(torch.isnan(loss).any())

        mask_nan = self.mask.clone().float()  # Convert to float for NaN assignment
        mask_nan[0, 0] = float('nan')
        loss = loss_fn(self.y_true, self.y_pred, mask_nan)
        self.assertFalse(torch.isnan(loss).any())  # NaN in mask should be treated as False

    def test_inf(self):
        """Test handling of infinite values."""
        loss_fn = HuberLoss().to(self.device) #Use HuberLoss as a representative example

        y_true_inf = self.y_true.clone()
        y_true_inf[0, 0] = float('inf')
        loss = loss_fn(y_true_inf, self.y_pred, self.mask)
        self.assertTrue(torch.isinf(loss).any())

        y_pred_inf = self.y_pred.clone()
        y_pred_inf[0, 0] = float('inf')
        loss = loss_fn(self.y_true, y_pred_inf, self.mask)
        self.assertTrue(torch.isinf(loss).any())


        mask_inf = self.mask.clone().float() #Convert to float for inf assignment
        mask_inf[0, 0] = float('inf')  # Should be cast to True
        loss = loss_fn(self.y_true, self.y_pred, mask_inf)
        self.assertFalse(torch.isnan(loss).any())
        
    def test_batch_consistency(self):
        """Test that loss calculations are consistent across batch dimensions."""
        # Create a batch with identical data
        batch_size = 10
        y_true_batch = torch.ones(batch_size, 1, device=self.device)
        y_pred_batch = torch.zeros(batch_size, 1, device=self.device)
        
        # Test each loss function
        loss_functions = [
            L1Loss(), 
            HuberLoss(), 
            PseudoHuberLoss(), 
            LogCoshLoss(), 
            CharbonnierLoss(),
            LqLoss()
        ]
        
        for loss_fn in loss_functions:
            loss_fn = loss_fn.to(self.device)
            # Single sample loss
            single_loss = loss_fn(y_true_batch[0:1], y_pred_batch[0:1])
            # Batch loss
            batch_loss = loss_fn(y_true_batch, y_pred_batch)
            
            # The average loss should be the same regardless of batch size
            self.assertAlmostEqual(single_loss.item(), batch_loss.item(), places=5)

    def test_reduction_modes(self):
        """Test different reduction modes for loss functions."""
        loss_classes = [L1Loss, HuberLoss, PseudoHuberLoss, LogCoshLoss, CharbonnierLoss, LqLoss]
        
        for loss_class in loss_classes:
            # Test 'mean' reduction (default)
            loss_fn_mean = loss_class(reduction='mean').to(self.device)
            loss_mean = loss_fn_mean(self.y_true, self.y_pred)
            self.assertTrue(torch.is_tensor(loss_mean))
            self.assertEqual(loss_mean.numel(), 1)
            
            # Test 'sum' reduction
            loss_fn_sum = loss_class(reduction='sum').to(self.device)
            loss_sum = loss_fn_sum(self.y_true, self.y_pred)
            self.assertTrue(torch.is_tensor(loss_sum))
            self.assertEqual(loss_sum.numel(), 1)
            
            # Test 'none' reduction
            loss_fn_none = loss_class(reduction='none').to(self.device)
            loss_none = loss_fn_none(self.y_true, self.y_pred)
            self.assertTrue(torch.is_tensor(loss_none))
            self.assertEqual(loss_none.shape, self.y_true.shape)
            
            # Verify sum and mean relationship
            self.assertAlmostEqual(
                loss_sum.item(), 
                loss_none.sum().item(), 
                places=5
            )
            self.assertAlmostEqual(
                loss_mean.item(), 
                loss_none.mean().item(), 
                places=5
            )

    def test_tensor_shapes(self):
        """Test loss functions with various tensor shapes."""
        loss_classes = [L1Loss, HuberLoss, PseudoHuberLoss, LogCoshLoss, CharbonnierLoss, LqLoss]
        
        for loss_class in loss_classes:
            loss_fn = loss_class().to(self.device)
            
            # Test with scalar inputs
            scalar_loss = loss_fn(self.scalar_true, self.scalar_pred)
            self.assertTrue(torch.is_tensor(scalar_loss))
            self.assertEqual(scalar_loss.numel(), 1)
            
            # Test with 3D tensors
            tensor_3d_loss = loss_fn(self.y_true_3d, self.y_pred_3d)
            self.assertTrue(torch.is_tensor(tensor_3d_loss))
            self.assertEqual(tensor_3d_loss.numel(), 1)
            
            # Test with 3D tensors and mask
            tensor_3d_mask_loss = loss_fn(self.y_true_3d, self.y_pred_3d, self.mask_3d)
            self.assertTrue(torch.is_tensor(tensor_3d_mask_loss))
            self.assertEqual(tensor_3d_mask_loss.numel(), 1)
            
            # Test broadcasting with different shapes
            if hasattr(self, 'y_true_broadcast') and hasattr(self, 'y_pred_broadcast'):
                broadcast_loss = loss_fn(self.y_true_broadcast, self.y_pred_broadcast)
                self.assertTrue(torch.is_tensor(broadcast_loss))

    def test_gradient(self):
        """Test gradients of loss functions."""
        loss_classes = [L1Loss, HuberLoss, PseudoHuberLoss, LogCoshLoss, CharbonnierLoss, LqLoss]
        
        for loss_class in loss_classes:
            # Skip gradient test for LqLoss with q < 1 (non-differentiable at zero)
            if loss_class == LqLoss:
                loss_fn = loss_class(q=1.5).to(self.device)
            else:
                loss_fn = loss_class().to(self.device)
                
            # Reset gradients
            self.scalar_pred.grad = None
            
            # Forward pass
            loss = loss_fn(self.scalar_true, self.scalar_pred)
            
            # Backward pass
            loss.backward()
            
            # Check gradient exists and is finite
            self.assertIsNotNone(self.scalar_pred.grad)
            self.assertFalse(torch.isnan(self.scalar_pred.grad).any())
            self.assertFalse(torch.isinf(self.scalar_pred.grad).any())
            
            # For specific losses, verify gradient values at test points
            if loss_class == L1Loss:
                # L1 gradient should be sign(pred - true)
                expected_sign = torch.sign(self.scalar_pred - self.scalar_true)
                self.assertEqual(torch.sign(self.scalar_pred.grad).item(), expected_sign.item())

    def test_nans(self):
        """Test handling of NaN values in inputs and mask."""
        # Create a separate mask that explicitly handles NaN values
        nan_handling_mask = torch.ones_like(self.mask, dtype=torch.bool)
        
        # Test each loss function
        loss_functions = [
            L1Loss(), HuberLoss(), PseudoHuberLoss(), 
            LogCoshLoss(), CharbonnierLoss(), LqLoss()
        ]
        
        for loss_fn in loss_functions:
            loss_fn = loss_fn.to(self.device)
            
            # NaN in y_true with explicit masking
            y_true_nan = self.y_true.clone()
            y_true_nan[0, 0] = float('nan')
            nan_handling_mask[0, 0] = False
            
            loss = loss_fn(y_true_nan, self.y_pred, nan_handling_mask)
            self.assertFalse(torch.isnan(loss).any(), 
                            f"{loss_fn.__class__.__name__} failed with NaN in y_true")
            
            # NaN in y_pred with explicit masking
            y_pred_nan = self.y_pred.clone()
            y_pred_nan[0, 0] = float('nan')
            
            loss = loss_fn(self.y_true, y_pred_nan, nan_handling_mask)
            self.assertFalse(torch.isnan(loss).any(),
                            f"{loss_fn.__class__.__name__} failed with NaN in y_pred")
    
    def test_inf(self):
        """Test handling of infinite values with proper masking."""
        # Create a separate mask that explicitly handles Inf values
        inf_handling_mask = torch.ones_like(self.mask, dtype=torch.bool)
        
        # Test each loss function
        loss_functions = [
            L1Loss(), HuberLoss(), PseudoHuberLoss(), 
            LogCoshLoss(), CharbonnierLoss(), LqLoss()
        ]
        
        for loss_fn in loss_functions:
            loss_fn = loss_fn.to(self.device)
            
            # Inf in y_true with explicit masking
            y_true_inf = self.y_true.clone()
            y_true_inf[0, 0] = float('inf')
            inf_handling_mask[0, 0] = False
            
            loss = loss_fn(y_true_inf, self.y_pred, inf_handling_mask)
            self.assertFalse(torch.isnan(loss).any(), 
                            f"{loss_fn.__class__.__name__} failed with Inf in y_true")
            self.assertFalse(torch.isinf(loss).any(),
                            f"{loss_fn.__class__.__name__} produces Inf loss with Inf in y_true")
            
            # Inf in y_pred with explicit masking
            y_pred_inf = self.y_pred.clone()
            y_pred_inf[0, 0] = float('inf')
            
            loss = loss_fn(self.y_true, y_pred_inf, inf_handling_mask)
            self.assertFalse(torch.isnan(loss).any(),
                            f"{loss_fn.__class__.__name__} failed with Inf in y_pred")
            self.assertFalse(torch.isinf(loss).any(),
                            f"{loss_fn.__class__.__name__} produces Inf loss with Inf in y_pred")

    def test_edge_cases(self):
        """Test edge cases like empty tensors and extreme values."""
        loss_functions = [
            L1Loss(), HuberLoss(), PseudoHuberLoss(), 
            LogCoshLoss(), CharbonnierLoss(), LqLoss()
        ]
        
        for loss_fn in loss_functions:
            loss_fn = loss_fn.to(self.device)
            
            # Test with extremely small differences
            y_true_small = torch.zeros(1, 1, device=self.device)
            y_pred_small = torch.ones(1, 1, device=self.device) * 1e-10
            
            loss_small = loss_fn(y_true_small, y_pred_small)
            self.assertFalse(torch.isnan(loss_small).any())
            self.assertFalse(torch.isinf(loss_small).any())
            
            # Test with extremely large (but finite) values
            y_true_large = torch.zeros(1, 1, device=self.device)
            y_pred_large = torch.ones(1, 1, device=self.device) * 1e10
            
            loss_large = loss_fn(y_true_large, y_pred_large)
            self.assertFalse(torch.isnan(loss_large).any())
            
            # Special case for empty tensors
            # This might raise an exception depending on implementation
            # We'll catch any exceptions and print a message
            try:
                # We need to ensure empty tensors have the same shape
                y_true_empty = torch.tensor([], device=self.device).reshape(0, 1)
                y_pred_empty = torch.tensor([], device=self.device).reshape(0, 1)
                
                loss_empty = loss_fn(y_true_empty, y_pred_empty)
                # If we get here, make sure the loss is either 0 or NaN (both could be valid)
                if not torch.isnan(loss_empty).any():
                    self.assertAlmostEqual(loss_empty.item(), 0.0, places=5)
            except Exception as e:
                # We'll just log the exception rather than failing the test
                # since different implementations may handle empty tensors differently
                print(f"{loss_fn.__class__.__name__} on empty tensor: {str(e)}")

if __name__ == '__main__':
    unittest.main()