import torch
import unittest
import pytest
import math
from torch.autograd import gradcheck
from torchregression.losses.gaussian import (
    WeightedMSELoss,
    DiagonalGaussianNLL,
    GaussianNLLWithCovariance,
    create_gaussian_nll,
    MSELoss,
    MAELoss,
    GaussianNLLLoss
)

class TestGaussianLosses(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            
        self.batch_size = 4
        self.n_features_cov = 5
        self.n_features_diag = 3
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Data for tests where covariance is NOT provided
        self.x = torch.randn(self.batch_size, self.n_features_diag, device=self.device)
        self.x_reconstructed = torch.randn(self.batch_size, self.n_features_diag, device=self.device)
        self.mask = torch.randint(0, 2, (self.batch_size, self.n_features_diag), device=self.device).bool()

        # Data for tests where covariance IS provided
        self.x_cov = torch.randn(self.batch_size, self.n_features_cov, device=self.device)
        self.x_cov_reconstructed = torch.randn(self.batch_size, self.n_features_cov, device=self.device)
        self.covariance_matrices = torch.randn(self.batch_size, self.n_features_cov, self.n_features_cov, device=self.device)
        # Ensure symmetric positive definiteness:
        self.covariance_matrices = self.covariance_matrices @ self.covariance_matrices.transpose(-1, -2)
        # Add small diagonal term to ensure positive definiteness
        self.covariance_matrices = self.covariance_matrices + torch.eye(self.n_features_cov, device=self.device) * 1e-3
        self.mask_cov = torch.randint(0, 2, (self.batch_size, self.n_features_cov), device=self.device).bool()

        # Simple case for verifying exact outputs
        self.simple_x = torch.ones(1, 1, device=self.device)
        self.simple_y = torch.ones(1, 1, device=self.device) * 2  # Prediction is 2, ground truth is 1
        self.simple_mask = torch.ones(1, 1, device=self.device).bool()
        
        # Edge cases
        self.zero_tensor = torch.zeros(self.batch_size, self.n_features_diag, device=self.device)
        self.extreme_values = torch.tensor([1e10, 1e-10, 0.0]).to(self.device)
        self.extreme_tensor = torch.cat([
            self.extreme_values.unsqueeze(0).expand(self.batch_size//3, self.n_features_diag),
            self.extreme_values.unsqueeze(0).expand(self.batch_size//3, self.n_features_diag),
            self.extreme_values.unsqueeze(0).expand(self.batch_size - 2*(self.batch_size//3), self.n_features_diag)
        ])

    def test_weighted_mse_loss(self):
        """Test WeightedMSELoss with and without mask, including exact value verification."""
        loss_fn = WeightedMSELoss().to(self.device)
        
        # Test with random data and mask
        weights = torch.rand(self.batch_size, self.n_features_diag, device=self.device)
        loss = loss_fn(self.x, self.x_reconstructed, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test with weights
        loss_weighted = loss_fn(self.x, self.x_reconstructed, weights=weights)
        self.assertTrue(torch.is_tensor(loss_weighted))
        self.assertFalse(torch.isnan(loss_weighted).any())
        
        # Test with weights and mask
        loss_weighted_masked = loss_fn(self.x, self.x_reconstructed, mask=self.mask, weights=weights)
        self.assertTrue(torch.is_tensor(loss_weighted_masked))
        self.assertFalse(torch.isnan(loss_weighted_masked).any())

        # Test without mask
        loss_no_mask = loss_fn(self.x, self.x_reconstructed)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())
        
        # Test with simple case for exact value verification
        simple_weights = torch.ones(1, 1, device=self.device)
        simple_loss = loss_fn(self.simple_x, self.simple_y, weights=simple_weights)
        self.assertAlmostEqual(simple_loss.item(), 1.0, places=5)  # MSE of (1-2)^2 = 1

        # Test with extreme values
        extreme_loss = loss_fn(self.extreme_tensor, self.extreme_tensor * 1.1)
        self.assertFalse(torch.isnan(extreme_loss).any())
        self.assertFalse(torch.isinf(extreme_loss).any())
        
        # Test with zero tensors 
        zero_loss = loss_fn(self.zero_tensor, self.zero_tensor)
        self.assertAlmostEqual(zero_loss.item(), 0.0, places=5)  # Should be exactly 0

    def test_diagonal_gaussian_nll(self):
        """Test DiagonalGaussianNLL with mask and gradient flow verification."""
        # Test with learned variance
        loss_fn = DiagonalGaussianNLL(self.n_features_diag).to(self.device)
        
        # Test with mask
        loss = loss_fn(self.x, self.x_reconstructed, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())
        
        # Test without mask
        loss_no_mask = loss_fn(self.x, self.x_reconstructed)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())
        
        # Check for gradient flow
        loss.backward()
        self.assertIsNotNone(loss_fn.log_variances.grad)
        
        # Test with fixed variance
        fixed_loss_fn = DiagonalGaussianNLL(learnable_variance=False, fixed_variance=0.5).to(self.device)
        fixed_loss = fixed_loss_fn(self.x, self.x_reconstructed)
        self.assertTrue(torch.is_tensor(fixed_loss))
        self.assertFalse(torch.isnan(fixed_loss).any())
        
        # Test with tuple input format (mean, logvar)
        log_var = torch.zeros_like(self.x)
        tuple_loss = fixed_loss_fn((self.x, log_var), self.x_reconstructed)
        self.assertTrue(torch.is_tensor(tuple_loss))
        self.assertFalse(torch.isnan(tuple_loss).any())
        
        # Test with concatenated input format [mean, logvar]
        concat_input = torch.cat([self.x, log_var], dim=-1)
        concat_loss = fixed_loss_fn(concat_input, self.x_reconstructed)
        self.assertTrue(torch.is_tensor(concat_loss))
        self.assertFalse(torch.isnan(concat_loss).any())
        
        # Test with extreme values to check numerical stability
        extreme_loss_fn = DiagonalGaussianNLL(learnable_variance=False).to(self.device)
        extreme_loss = extreme_loss_fn(self.extreme_tensor, self.extreme_tensor * 1.1)
        self.assertFalse(torch.isnan(extreme_loss).any(), "Loss should handle extreme values")
        self.assertFalse(torch.isinf(extreme_loss).any(), "Loss should not be infinite with extreme values")

    def test_gaussian_nll_with_covariance(self):
        """Test GaussianNLLWithCovariance with and without mask."""
        loss_fn = GaussianNLLWithCovariance().to(self.device)
        
        # Test with mask
        loss = loss_fn(self.x_cov, self.x_cov_reconstructed, self.covariance_matrices, self.mask_cov)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())
        
        # Test without mask
        loss_no_mask = loss_fn(self.x_cov, self.x_cov_reconstructed, self.covariance_matrices)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())
        
        # Test with single shared covariance matrix
        single_cov = self.covariance_matrices[0]
        loss_single_cov = loss_fn(self.x_cov, self.x_cov_reconstructed, single_cov)
        self.assertTrue(torch.is_tensor(loss_single_cov))
        self.assertFalse(torch.isnan(loss_single_cov).any())
        
        # Test with learnable adjustment
        adj_loss_fn = GaussianNLLWithCovariance(n_features=self.n_features_cov, learnable_adjustment=True).to(self.device)
        adj_loss = adj_loss_fn(self.x_cov, self.x_cov_reconstructed, self.covariance_matrices)
        self.assertTrue(torch.is_tensor(adj_loss))
        self.assertFalse(torch.isnan(adj_loss).any())
        
        # Check gradient flow for learnable adjustment
        adj_loss.backward()
        self.assertIsNotNone(adj_loss_fn.log_variance_adjustment.grad)
        
        # Test with different jitter values for numerical stability
        jitter_values = [1e-8, 1e-6, 1e-4, 1e-2]
        for jitter in jitter_values:
            with self.subTest(jitter=jitter):
                jitter_loss_fn = GaussianNLLWithCovariance(jitter=jitter).to(self.device)
                jitter_loss = jitter_loss_fn(self.x_cov, self.x_cov_reconstructed, self.covariance_matrices)
                self.assertTrue(torch.is_tensor(jitter_loss))
                self.assertFalse(torch.isnan(jitter_loss).any())

    def test_create_gaussian_nll_factory(self):
        """Test the factory function for creating Gaussian NLL losses."""
        # Test diagonal with fixed variance
        diag_fixed = create_gaussian_nll(
            n_features=self.n_features_diag, 
            covariance_type='diagonal',
            learnable_variance=False, 
            fixed_variance=0.5
        ).to(self.device)
        
        self.assertIsInstance(diag_fixed, DiagonalGaussianNLL)
        self.assertEqual(diag_fixed.fixed_variance.item(), 0.5)
        
        # Test diagonal with learned variance
        diag_learned = create_gaussian_nll(
            n_features=self.n_features_diag, 
            covariance_type='diagonal',
            learnable_variance=True
        ).to(self.device)
        
        self.assertIsInstance(diag_learned, DiagonalGaussianNLL)
        self.assertEqual(diag_learned.log_variances.shape[0], self.n_features_diag)
        
        # Test full covariance with adjustment
        full_cov = create_gaussian_nll(
            n_features=self.n_features_cov,
            covariance_type='full',
            learnable_variance=True,
            jitter=1e-5
        ).to(self.device)
        
        self.assertIsInstance(full_cov, GaussianNLLWithCovariance)
        self.assertEqual(full_cov.jitter, 1e-5)
        self.assertTrue(full_cov.learnable_adjustment)
        
        # Test invalid covariance type
        with self.assertRaises(ValueError):
            create_gaussian_nll(n_features=self.n_features_diag, covariance_type='invalid')

    def test_nans(self):
        """Test handling of NaN values in inputs."""
        loss_fn = DiagonalGaussianNLL(self.n_features_diag).to(self.device)

        # Test NaN in inputs
        x_nan = self.x.clone()
        x_nan[0,0] = float('nan')
        loss = loss_fn(x_nan, self.x_reconstructed, self.mask)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle NaN in inputs")

        # Test NaN in reconstructed values
        x_rec_nan = self.x_reconstructed.clone()
        x_rec_nan[0,0] = float('nan')
        loss = loss_fn(self.x, x_rec_nan, self.mask)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle NaN in reconstructed values")

        # Test NaN in mask
        mask_nan = self.mask.clone().float()
        mask_nan[0, 0] = float('nan')
        loss = loss_fn(self.x, self.x_reconstructed, mask_nan)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle NaN in mask")
        
        # Test NaN in variances for applicable losses
        variance_loss_fn = GaussianNLLWithCovariance().to(self.device)
        cov_nan = self.covariance_matrices.clone()
        cov_nan[0, 0, 0] = float('nan')
        with self.assertRaises(RuntimeError):
            # This should raise an error since NaN in covariance is not valid
            _ = variance_loss_fn(self.x_cov, self.x_cov_reconstructed, cov_nan, self.mask_cov)

    def test_inf(self):
        """Test handling of infinite values in inputs."""
        loss_fn = DiagonalGaussianNLL(self.n_features_diag).to(self.device)

        # Test infinity in inputs
        x_inf = self.x.clone()
        x_inf[0,0] = float('inf')
        with self.assertWarns(RuntimeWarning):
            loss = loss_fn(x_inf, self.x_reconstructed, self.mask)
            self.assertTrue(torch.isinf(loss).any(), "Loss should be inf with inf inputs")

        # Test infinity in reconstructed values where mask is True
        x_rec_inf = self.x_reconstructed.clone()
        x_rec_inf[0,0] = float('inf')
        mask_all_zero = torch.zeros_like(self.mask).bool()
        # With all-zero mask, inf should be masked out
        loss = loss_fn(self.x, x_rec_inf, mask_all_zero)
        self.assertFalse(torch.isinf(loss).any(), "Loss should not be inf when inf values are masked")
        
        # With regular mask, inf should affect the loss
        loss = loss_fn(self.x, x_rec_inf, self.mask)
        # Only check if mask[0,0] is True
        if self.mask[0,0]:
            self.assertTrue(torch.isinf(loss).any(), "Loss should be inf with inf reconstructed values")

        # Test infinity in mask - should be handled by casting to bool
        mask_inf = self.mask.clone().float()
        mask_inf[0,0] = float('inf')  # Will be cast to True
        loss = loss_fn(self.x, self.x_reconstructed, mask_inf)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle inf in mask")
        
    def test_numerical_stability(self):
        """Test the numerical stability of Gaussian losses."""
        # Test with very small/large variances
        variance_scales = [1e-8, 1e-4, 1.0, 1e4, 1e8]
        
        for scale in variance_scales:
            with self.subTest(variance_scale=scale):
                # Test fixed variance case
                fixed_var_loss = DiagonalGaussianNLL(
                    learnable_variance=False, 
                    fixed_variance=scale
                ).to(self.device)
                
                loss = fixed_var_loss(self.x, self.x_reconstructed)
                self.assertFalse(torch.isnan(loss).any(), f"Loss should be stable with variance scale {scale}")
                self.assertFalse(torch.isinf(loss).any(), f"Loss should not be infinite with variance scale {scale}")
                
                # Test log variance case
                log_var = torch.ones_like(self.x) * math.log(scale)
                loss = fixed_var_loss((self.x, log_var), self.x_reconstructed)
                self.assertFalse(torch.isnan(loss).any(), f"Loss should be stable with log variance scale {scale}")
                self.assertFalse(torch.isinf(loss).any(), f"Loss should not be infinite with log variance scale {scale}")
        
        # Test with very small differences between prediction and target
        small_diff = 1e-10
        close_recon = self.x + small_diff
        loss_fn = DiagonalGaussianNLL(self.n_features_diag).to(self.device)
        close_loss = loss_fn(self.x, close_recon)
        self.assertFalse(torch.isnan(close_loss).any(), "Loss should be stable with very small differences")
        self.assertFalse(torch.isinf(close_loss).any(), "Loss should not be infinite with very small differences")
        
        # Test with covariance matrices of different condition numbers
        for scale in [1e-2, 1.0, 1e2]:
            with self.subTest(condition_number_scale=scale):
                # Create covariance with controlled condition number
                U, _, V = torch.linalg.svd(self.covariance_matrices[0])
                S = torch.diag(torch.tensor([1.0, scale, scale**2, scale**3, scale**4], device=self.device))
                cov = U @ S @ V.T
                cov = cov @ cov.T  # Ensure symmetric positive definite
                
                cov_loss_fn = GaussianNLLWithCovariance().to(self.device)
                cond_loss = cov_loss_fn(self.x_cov[0:1], self.x_cov_reconstructed[0:1], cov)
                self.assertFalse(torch.isnan(cond_loss).any(), f"Loss should be stable with condition number scale {scale}")
                self.assertFalse(torch.isinf(cond_loss).any(), f"Loss should not be infinite with condition number scale {scale}")

    def test_api_consistency(self):
        """Test that all Gaussian losses have consistent APIs."""
        # All loss functions should accept y_pred, target, mask, and return a scalar loss
        loss_fns = [
            WeightedMSELoss().to(self.device),
            DiagonalGaussianNLL(self.n_features_diag).to(self.device),
            # GaussianNLLWithCovariance has a different signature, will test separately
        ]
        
        for loss_fn in loss_fns:
            with self.subTest(loss_fn=type(loss_fn).__name__):
                # Should accept target, predictions and mask
                loss = loss_fn(self.x, self.x_reconstructed, self.mask)
                self.assertTrue(torch.is_tensor(loss))
                self.assertTrue(loss.dim() == 0, f"Loss should be a scalar, got shape {loss.shape}")
                
                # Should work with various reduction modes
                for reduction in ['none', 'mean', 'sum']:
                    loss_fn.reduction = reduction
                    loss = loss_fn(self.x, self.x_reconstructed)
                    if reduction == 'none':
                        self.assertEqual(loss.dim(), 1, f"Reduction 'none' should return per-sample losses")
                        self.assertEqual(loss.shape[0], self.batch_size, f"Should have {self.batch_size} samples")
                    else:
                        self.assertEqual(loss.dim(), 0, f"Reduction '{reduction}' should return a scalar")
        
        # Test GaussianNLLWithCovariance separately due to different signature
        cov_loss_fn = GaussianNLLWithCovariance().to(self.device)
        loss = cov_loss_fn(self.x_cov, self.x_cov_reconstructed, self.covariance_matrices)
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(loss.dim() == 0, "Loss should be a scalar")
        
        # Test reduction modes
        for reduction in ['none', 'mean', 'sum']:
            cov_loss_fn.reduction = reduction
            loss = cov_loss_fn(self.x_cov, self.x_cov_reconstructed, self.covariance_matrices)
            if reduction == 'none':
                self.assertEqual(loss.dim(), 1, f"Reduction 'none' should return per-sample losses")
                self.assertEqual(loss.shape[0], self.batch_size, f"Should have {self.batch_size} samples")
            else:
                self.assertEqual(loss.dim(), 0, f"Reduction '{reduction}' should return a scalar")

    def test_mse_edge_cases(self):
        """Test MSELoss with edge cases: zeros, empty tensors, extreme values, NaN/Inf."""
        loss_fn = MSELoss()
        
        # Test with zeros
        y_pred_zeros = torch.zeros(10)
        y_true_zeros = torch.zeros(10)
        assert loss_fn(y_pred_zeros, y_true_zeros).item() == 0.0
        
        # Test with empty tensors
        y_pred_empty = torch.tensor([])
        y_true_empty = torch.tensor([])
        assert loss_fn(y_pred_empty, y_true_empty).numel() == 0
        
        # Test with extreme values
        y_pred_large = torch.tensor([1e10])
        y_true_small = torch.tensor([0.0])
        assert torch.isfinite(loss_fn(y_pred_large, y_true_small))
        
        # Test with NaN/Inf and masks
        y_pred = torch.tensor([1.0, float('nan'), 3.0])
        y_true = torch.tensor([1.5, 2.5, float('inf')])
        mask = torch.tensor([True, False, False])
        assert torch.isfinite(loss_fn(y_pred, y_true, mask))


class TestGaussianLossesNumericalStability:
    def test_mse_gradient_flow(self):
        """Test that gradients flow through MSELoss properly."""
        from torchregression.losses.gaussian import MSELoss
        
        # Create inputs that require gradients
        y_pred = torch.randn(10, 1, requires_grad=True, dtype=torch.double)
        y_true = torch.randn(10, 1, dtype=torch.double)
        
        # Test with gradcheck
        loss_fn = MSELoss(reduction='mean')
        assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-5)
    
    def test_mae_gradient_flow(self):
        """Test that gradients flow through MAELoss properly."""
        from torchregression.losses.gaussian import MAELoss
        
        # Create inputs that require gradients
        y_pred = torch.randn(10, 1, requires_grad=True, dtype=torch.double)
        y_true = torch.randn(10, 1, dtype=torch.double)
        
        # Test with gradcheck
        loss_fn = MAELoss(reduction='mean')
        assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-4)
    
    def test_gaussian_nll_gradient_flow(self):
        """Test that gradients flow through GaussianNLLLoss properly."""
        from torchregression.losses.gaussian import GaussianNLLLoss
        
        # Create inputs that require gradients
        y_pred = torch.randn(10, 1, requires_grad=True, dtype=torch.double)
        var = torch.exp(torch.randn(10, 1, requires_grad=True, dtype=torch.double))
        y_true = torch.randn(10, 1, dtype=torch.double)
        
        # Define a function for gradcheck that handles the multiple inputs
        def loss_fn(pred, variance, target):
            return GaussianNLLLoss(reduction='mean')(pred, target, variance)
        
        # Test with gradcheck
        assert gradcheck(loss_fn, (y_pred, var, y_true), eps=1e-6, atol=1e-4)
    
    def test_extreme_values(self):
        """Test stability with extreme values."""
        from torchregression.losses.gaussian import MSELoss, MAELoss, GaussianNLLLoss
        
        # Very large values
        y_pred_large = torch.tensor([1e10, 1e15, 1e20], dtype=torch.float, requires_grad=True)
        y_true_large = torch.tensor([1e10 + 1, 1e15 + 10, 1e20 + 100], dtype=torch.float)
        
        # Very small values
        y_pred_small = torch.tensor([1e-10, 1e-15, 1e-20], dtype=torch.float, requires_grad=True)
        y_true_small = torch.tensor([1e-10 + 1e-12, 1e-15 + 1e-17, 1e-20 + 1e-22], dtype=torch.float)
        
        # Test MSE with extreme values
        mse = MSELoss(reduction='mean')
        loss = mse(y_pred_large, y_true_large)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_large.grad))
        
        # Test MAE with extreme values
        y_pred_large.grad = None  # Reset gradients
        mae = MAELoss(reduction='mean')
        loss = mae(y_pred_large, y_true_large)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_large.grad))
        
        # Test GaussianNLL with extreme values (using reasonable variance)
        y_pred_small.grad = None  # Reset gradients
        var = torch.ones_like(y_pred_small, requires_grad=True)
        gnll = GaussianNLLLoss(reduction='mean')
        loss = gnll(y_pred_small, y_true_small, var)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_small.grad))
        assert torch.all(torch.isfinite(var.grad))
    
    def test_reduction_modes(self):
        """Test different reduction modes for backward pass."""
        from torchregression.losses.gaussian import MSELoss
        
        y_pred = torch.randn(10, 1, requires_grad=True)
        y_true = torch.randn(10, 1)
        
        # Test mean reduction
        mse_mean = MSELoss(reduction='mean')
        loss = mse_mean(y_pred, y_true)
        loss.backward()
        mean_grad = y_pred.grad.clone()
        
        # Test sum reduction
        y_pred.grad = None
        mse_sum = MSELoss(reduction='sum')
        loss = mse_sum(y_pred, y_true)
        loss.backward()
        sum_grad = y_pred.grad.clone()
        
        # Test none reduction
        y_pred.grad = None
        mse_none = MSELoss(reduction='none')
        loss = mse_none(y_pred, y_true)
        loss.mean().backward()
        none_grad = y_pred.grad.clone()
        
        # Mean and sum should give different gradients (sum ~= mean * batch_size)
        assert not torch.allclose(mean_grad, sum_grad)
        # Mean and manual mean over none should be similar
        assert torch.allclose(mean_grad, none_grad)

if __name__ == '__main__':
    unittest.main()