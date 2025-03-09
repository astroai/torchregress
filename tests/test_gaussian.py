import torch
import unittest
from torchregression.gaussian import (
    WeightedMSELoss,
    DiagonalGaussianNLL,
    GaussianNLLWithCovariance,
    AdjustedGaussianNLL,
    HeteroscedasticGaussianNLL,
    LearnedGaussianNLL,
    LowRankGaussianNLL,
    GaussianPoissonMixtureNLL
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

    def test_weighted_mse_loss(self):
        """Test WeightedMSELoss with and without mask, including exact value verification."""
        loss_fn = WeightedMSELoss().to(self.device)
        
        # Test with random data and mask
        weights = torch.rand(self.batch_size, self.n_features_diag, device=self.device)
        loss = loss_fn(self.x, self.x_reconstructed, weights, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test without mask
        loss_no_mask = loss_fn(self.x, self.x_reconstructed, weights)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())
        
        # Test with simple case for exact value verification
        simple_weights = torch.ones(1, 1, device=self.device)
        simple_loss = loss_fn(self.simple_x, self.simple_y, simple_weights)
        self.assertAlmostEqual(simple_loss.item(), 1.0, places=5)  # MSE of (1-2)^2 = 1

    def test_diagonal_gaussian_nll(self):
        """Test DiagonalGaussianNLL with mask and gradient flow verification."""
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

    def test_adjusted_gaussian_nll(self):
        """Test AdjustedGaussianNLL with single/batch covariance matrices and gradient flow."""
        loss_fn = AdjustedGaussianNLL(self.n_features_cov).to(self.device)
        
        # Test with single covariance matrix
        loss = loss_fn(self.x_cov, self.x_cov_reconstructed, self.covariance_matrices[0], self.mask_cov)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test with batch of covariance matrices
        loss = loss_fn(self.x_cov, self.x_cov_reconstructed, self.covariance_matrices, self.mask_cov)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())
        
        # Check gradient flow
        loss.backward()
        self.assertIsNotNone(loss_fn.log_variance_adjustment.grad)

    def test_heteroscedastic_gaussian_nll(self):
        """Test HeteroscedasticGaussianNLL with fixed and non-fixed diagonal variants."""
        # Test without fixed diagonal
        loss_fn = HeteroscedasticGaussianNLL(
            self.n_features_cov, self.n_features_diag, use_fixed_diag_for_other=False).to(self.device)
        loss = loss_fn(
            self.x_cov, self.x_cov_reconstructed, self.covariance_matrices, self.mask_cov,
            self.x, self.x_reconstructed, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test with fixed diagonal
        loss_fn_fixed = HeteroscedasticGaussianNLL(
            self.n_features_cov, self.n_features_diag, use_fixed_diag_for_other=True).to(self.device)
        other_variances = torch.rand(self.batch_size, self.n_features_diag, device=self.device)
        loss_fixed = loss_fn_fixed(
            self.x_cov, self.x_cov_reconstructed, self.covariance_matrices, self.mask_cov,
            self.x, self.x_reconstructed, self.mask, other_variances)
        self.assertTrue(torch.is_tensor(loss_fixed))
        self.assertFalse(torch.isnan(loss_fixed).any())

        # Test for exception when other_variances is not given but required
        with self.assertRaises(ValueError):
            loss_fn_fixed(
                self.x_cov, self.x_cov_reconstructed, self.covariance_matrices, self.mask_cov,
                self.x, self.x_reconstructed, self.mask)

    def test_learned_gaussian_nll(self):
        """Test LearnedGaussianNLL with gradient flow verification."""
        loss_fn = LearnedGaussianNLL(self.n_features_cov).to(self.device)
        
        # Test with mask
        loss = loss_fn(self.x_cov, self.x_cov_reconstructed, self.mask_cov)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())
        
        # Test without mask
        loss_no_mask = loss_fn(self.x_cov, self.x_cov_reconstructed)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())
        
        # Check for gradient flow
        loss.backward()
        self.assertIsNotNone(loss_fn.cholesky_factor.grad)

    def test_low_rank_gaussian_nll(self):
        """Test LowRankGaussianNLL with different ranks and gradient verification."""
        ranks_to_test = [1, 3, self.n_features_cov-1]
        
        for rank in ranks_to_test:
            with self.subTest(rank=rank):
                loss_fn = LowRankGaussianNLL(self.n_features_cov, rank=rank).to(self.device)
                
                # Test with mask
                loss = loss_fn(self.x_cov, self.x_cov_reconstructed, self.mask_cov)
                self.assertTrue(torch.is_tensor(loss))
                self.assertFalse(torch.isnan(loss).any())
                
                # Test without mask
                loss_no_mask = loss_fn(self.x_cov, self.x_cov_reconstructed)
                self.assertTrue(torch.is_tensor(loss_no_mask))
                self.assertFalse(torch.isnan(loss_no_mask).any())
                
                # Check for gradient flow
                loss.backward()
                self.assertIsNotNone(loss_fn.U.grad)
                self.assertIsNotNone(loss_fn.log_variances.grad)

    def test_gaussian_poisson_mixture_nll(self):
        """Test GaussianPoissonMixtureNLL with various configurations."""
        # Test basic functionality
        loss_fn = GaussianPoissonMixtureNLL().to(self.device)
        loss = loss_fn(self.x, self.x_reconstructed, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test without mask
        loss_no_mask = loss_fn(self.x, self.x_reconstructed)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())

        # Test with learned variance
        loss_fn_learn = GaussianPoissonMixtureNLL(learn_gaussian_variance=True).to(self.device)
        loss_learn = loss_fn_learn(self.x, self.x_reconstructed, self.mask)
        self.assertTrue(torch.is_tensor(loss_learn))
        self.assertFalse(torch.isnan(loss_learn).any())
        loss_learn.backward()
        self.assertIsNotNone(loss_fn_learn.log_gaussian_variance.grad)

        # Test log input
        loss_fn_log = GaussianPoissonMixtureNLL(log_input=True).to(self.device)
        loss_log = loss_fn_log(self.x, torch.log(torch.abs(self.x_reconstructed) + 1e-8), self.mask)
        self.assertTrue(torch.is_tensor(loss_log))
        self.assertFalse(torch.isnan(loss_log).any())

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


if __name__ == '__main__':
    unittest.main()