import math
import unittest

import torch

from torchregress.losses.base import (
    WeightedLossWrapper,
)
from torchregress.losses.gaussian import (
    GaussianNLLLoss,
    LowRankGaussianLoss,
    MultivariateGaussianLoss,
    create_gaussian_nll,
    low_rank_output_dim,
    split_low_rank_gaussian_output,
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Data for tests where covariance is NOT provided
        self.x = torch.randn(self.batch_size, self.n_features_diag, device=self.device)
        self.x_reconstructed = torch.randn(
            self.batch_size, self.n_features_diag, device=self.device
        )
        self.mask = torch.randint(
            0, 2, (self.batch_size, self.n_features_diag), device=self.device
        ).bool()

        # Data for tests where covariance IS provided
        self.x_cov = torch.randn(self.batch_size, self.n_features_cov, device=self.device)
        self.x_cov_reconstructed = torch.randn(
            self.batch_size, self.n_features_cov, device=self.device
        )
        self.covariance_matrices = torch.randn(
            self.batch_size, self.n_features_cov, self.n_features_cov, device=self.device
        )
        # Ensure symmetric positive definiteness:
        self.covariance_matrices = self.covariance_matrices @ self.covariance_matrices.transpose(
            -1, -2
        )
        # Add small diagonal term to ensure positive definiteness
        self.covariance_matrices = (
            self.covariance_matrices + torch.eye(self.n_features_cov, device=self.device) * 1e-3
        )
        self.mask_cov = torch.randint(
            0, 2, (self.batch_size, self.n_features_cov), device=self.device
        ).bool()

        # Edge cases
        self.zero_tensor = torch.zeros(self.batch_size, self.n_features_diag, device=self.device)
        self.extreme_values = torch.tensor([1e10, 1e-10, 0.0]).to(self.device)
        self.extreme_tensor = torch.cat(
            [
                self.extreme_values.unsqueeze(0).expand(self.batch_size // 3, self.n_features_diag),
                self.extreme_values.unsqueeze(0).expand(self.batch_size // 3, self.n_features_diag),
                self.extreme_values.unsqueeze(0).expand(
                    self.batch_size - 2 * (self.batch_size // 3), self.n_features_diag
                ),
            ]
        )

    def test_gaussian_nll_loss(self):
        """Test GaussianNLLLoss with mask and various input formats."""
        # Test with tuple input format (mean, logvar)
        loss_fn = GaussianNLLLoss().to(self.device)
        log_var = torch.zeros_like(self.x)

        # Test with mask
        loss = loss_fn((self.x, log_var), self.x_reconstructed, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        # Test without mask
        loss_no_mask = loss_fn((self.x, log_var), self.x_reconstructed)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())

        # Test with concatenated input format [mean, logvar]
        concat_input = torch.cat([self.x, log_var], dim=-1)
        concat_loss = loss_fn(concat_input, self.x_reconstructed)
        self.assertTrue(torch.is_tensor(concat_loss))
        self.assertFalse(torch.isnan(concat_loss).any())

        # Test with fixed variance
        fixed_loss_fn = GaussianNLLLoss(fixed_variance=0.5).to(self.device)
        fixed_loss = fixed_loss_fn(self.x, self.x_reconstructed)
        self.assertTrue(torch.is_tensor(fixed_loss))
        self.assertFalse(torch.isnan(fixed_loss).any())

        # Test with extreme values to check numerical stability
        extreme_loss_fn = GaussianNLLLoss(fixed_variance=1.0).to(self.device)
        extreme_loss = extreme_loss_fn(self.extreme_tensor, self.extreme_tensor * 1.1)
        self.assertFalse(torch.isnan(extreme_loss).any(), "Loss should handle extreme values")
        self.assertFalse(
            torch.isinf(extreme_loss).any(), "Loss should not be infinite with extreme values"
        )

        # Test with weights
        weights = torch.rand(self.batch_size, self.n_features_diag, device=self.device)
        weighted_loss = loss_fn((self.x, log_var), self.x_reconstructed, weights=weights)
        self.assertTrue(torch.is_tensor(weighted_loss))
        self.assertFalse(torch.isnan(weighted_loss).any())

        # Verify reduction modes
        for reduction in ["none", "mean", "sum"]:
            test_loss_fn = GaussianNLLLoss(reduction=reduction).to(self.device)
            red_loss = test_loss_fn((self.x, log_var), self.x_reconstructed)
            if reduction == "none":
                self.assertEqual(red_loss.shape[0], self.batch_size)
            else:
                self.assertEqual(red_loss.dim(), 0)

    def test_multivariate_gaussian_loss(self):
        """Test MultivariateGaussianLoss with and without mask."""
        loss_fn = MultivariateGaussianLoss().to(self.device)

        # Test with mask
        loss = loss_fn(
            self.x_cov, self.x_cov_reconstructed, self.covariance_matrices, self.mask_cov
        )
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
        adj_loss_fn = MultivariateGaussianLoss(
            n_features=self.n_features_cov, learnable_adjustment=True
        ).to(self.device)
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
                jitter_loss_fn = MultivariateGaussianLoss(jitter=jitter).to(self.device)
                jitter_loss = jitter_loss_fn(
                    self.x_cov, self.x_cov_reconstructed, self.covariance_matrices
                )
                self.assertTrue(torch.is_tensor(jitter_loss))
                self.assertFalse(torch.isnan(jitter_loss).any())

    def test_low_rank_gaussian_loss(self):
        """Test LowRankGaussianLoss with and without mask."""
        rank = 2
        cov_factor = torch.randn(self.batch_size, self.n_features_cov, rank, device=self.device)
        cov_diag = (
            torch.abs(torch.randn(self.batch_size, self.n_features_cov, device=self.device)) + 0.1
        )

        loss_fn = LowRankGaussianLoss().to(self.device)

        loss = loss_fn(self.x_cov, self.x_cov_reconstructed, cov_factor, cov_diag, self.mask_cov)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        loss_no_mask = loss_fn(self.x_cov, self.x_cov_reconstructed, cov_factor, cov_diag)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())

        shared_factor = torch.randn(self.n_features_cov, rank, device=self.device)
        shared_diag = torch.ones(self.n_features_cov, device=self.device) * 0.5
        loss_shared = loss_fn(self.x_cov, self.x_cov_reconstructed, shared_factor, shared_diag)
        self.assertTrue(torch.is_tensor(loss_shared))
        self.assertFalse(torch.isnan(loss_shared).any())

        for reduction in ["none", "mean", "sum"]:
            test_loss_fn = LowRankGaussianLoss(reduction=reduction).to(self.device)
            red_loss = test_loss_fn(self.x_cov, self.x_cov_reconstructed, cov_factor, cov_diag)
            if reduction == "none":
                self.assertEqual(red_loss.shape[0], self.batch_size)
            else:
                self.assertEqual(red_loss.dim(), 0)

        # Test with weights
        weights = torch.rand(self.batch_size, device=self.device)
        weighted_loss = loss_fn(
            self.x_cov, self.x_cov_reconstructed, cov_factor, cov_diag, weights=weights
        )
        self.assertTrue(torch.is_tensor(weighted_loss))
        self.assertFalse(torch.isnan(weighted_loss).any())

    def test_low_rank_output_helper(self):
        """Test low-rank output split helper."""
        rank = 2
        out_dim = low_rank_output_dim(self.n_features_cov, rank)
        y_pred = torch.randn(self.batch_size, out_dim, device=self.device)

        mean, cov_factor, cov_diag = split_low_rank_gaussian_output(
            y_pred, self.n_features_cov, rank
        )
        self.assertEqual(mean.shape, (self.batch_size, self.n_features_cov))
        self.assertEqual(cov_factor.shape, (self.batch_size, self.n_features_cov, rank))
        self.assertEqual(cov_diag.shape, (self.batch_size, self.n_features_cov))

        with self.assertRaises(ValueError):
            split_low_rank_gaussian_output(y_pred, self.n_features_cov, rank + 1)

    def test_create_gaussian_nll_factory(self):
        """Test the factory function for creating Gaussian NLL losses."""
        # Test diagonal with model predicting variance (default)
        diag_default = create_gaussian_nll(
            n_features=self.n_features_diag,
            covariance_type="diagonal",
        ).to(self.device)
        self.assertIsInstance(diag_default, GaussianNLLLoss)

        # Test diagonal with fixed variance
        diag_fixed = create_gaussian_nll(
            n_features=self.n_features_diag,
            covariance_type="diagonal",
            model_predicts_variance=False,
            fixed_variance=0.5,
        ).to(self.device)
        self.assertIsInstance(diag_fixed, GaussianNLLLoss)

        # Test full covariance
        full_cov = create_gaussian_nll(
            n_features=self.n_features_cov,
            covariance_type="full",
            jitter=1e-5,
        ).to(self.device)
        self.assertIsInstance(full_cov, MultivariateGaussianLoss)
        self.assertEqual(full_cov.jitter, 1e-5)

        # Test low-rank covariance
        low_rank_cov = create_gaussian_nll(
            n_features=self.n_features_cov,
            covariance_type="low_rank",
            jitter=1e-5,
        ).to(self.device)
        self.assertIsInstance(low_rank_cov, LowRankGaussianLoss)
        self.assertEqual(low_rank_cov.jitter, 1e-5)

        # Test with MSELoss case (simplified diagonal with unit variance)
        mse_case = create_gaussian_nll(
            n_features=self.n_features_diag,
            covariance_type="diagonal",
            model_predicts_variance=False,
            fixed_variance=1.0,
            use_mse_for_unit_variance=True,
        ).to(self.device)
        # WeightedMSELoss is a partial, so check the wrapper type
        self.assertIsInstance(mse_case, WeightedLossWrapper)

        # Test invalid covariance type
        with self.assertRaises(ValueError):
            create_gaussian_nll(n_features=self.n_features_diag, covariance_type="invalid")

    def test_gaussian_nll_numerical_stability(self):
        """Test numerical stability of GaussianNLLLoss."""
        # Test with very small/large variances
        variance_scales = [1e-8, 1e-4, 1.0, 1e4, 1e8]

        for scale in variance_scales:
            with self.subTest(variance_scale=scale):
                # Test fixed variance case
                fixed_var_loss = GaussianNLLLoss(fixed_variance=scale).to(self.device)

                loss = fixed_var_loss(self.x, self.x_reconstructed)
                self.assertFalse(
                    torch.isnan(loss).any(), f"Loss should be stable with variance scale {scale}"
                )
                self.assertFalse(
                    torch.isinf(loss).any(),
                    f"Loss should not be infinite with variance scale {scale}",
                )

                # Test log variance case
                log_var = torch.ones_like(self.x) * math.log(scale)
                var_loss = GaussianNLLLoss().to(self.device)
                loss = var_loss((self.x, log_var), self.x_reconstructed)
                self.assertFalse(
                    torch.isnan(loss).any(),
                    f"Loss should be stable with log variance scale {scale}",
                )
                self.assertFalse(
                    torch.isinf(loss).any(),
                    f"Loss should not be infinite with log variance scale {scale}",
                )

        # Test with very small differences between prediction and target
        small_diff = 1e-10
        close_recon = self.x + small_diff
        log_var = torch.zeros_like(self.x)
        loss_fn = GaussianNLLLoss().to(self.device)
        close_loss = loss_fn((self.x, log_var), close_recon)
        self.assertFalse(
            torch.isnan(close_loss).any(), "Loss should be stable with very small differences"
        )
        self.assertFalse(
            torch.isinf(close_loss).any(), "Loss should not be infinite with very small differences"
        )

    def test_multivariate_gaussian_numerical_stability(self):
        """Test numerical stability of MultivariateGaussianLoss."""
        # Test with covariance matrices of different condition numbers
        for scale in [1e-2, 1.0, 1e2]:
            with self.subTest(condition_number_scale=scale):
                # Create covariance with controlled condition number
                U, _, V = torch.linalg.svd(self.covariance_matrices[0])
                S = torch.diag(
                    torch.tensor([1.0, scale, scale**2, scale**3, scale**4], device=self.device)
                )
                cov = U @ S @ V.T
                cov = cov @ cov.T  # Ensure symmetric positive definite

                cov_loss_fn = MultivariateGaussianLoss().to(self.device)
                cond_loss = cov_loss_fn(self.x_cov[0:1], self.x_cov_reconstructed[0:1], cov)
                self.assertFalse(
                    torch.isnan(cond_loss).any(),
                    f"Loss should be stable with condition number scale {scale}",
                )
                self.assertFalse(
                    torch.isinf(cond_loss).any(),
                    f"Loss should not be infinite with condition number scale {scale}",
                )

    def test_gaussian_losses_with_nans_infs(self):
        """Test handling of NaN and Inf values in inputs."""
        # Test GaussianNLLLoss with NaNs and Infs
        log_var = torch.zeros_like(self.x)
        h_loss_fn = GaussianNLLLoss().to(self.device)

        # Test NaN in inputs with proper masking
        x_nan = self.x.clone()
        x_nan[0, 0] = float("nan")
        mask_nan = self.mask.clone()
        mask_nan[0, 0] = False  # Mask out the NaN
        h_loss = h_loss_fn((x_nan, log_var), self.x_reconstructed, mask_nan)
        self.assertFalse(torch.isnan(h_loss).any(), "Loss should handle masked NaN in inputs")

        # Test infinity in inputs with proper masking
        x_inf = self.x.clone()
        x_inf[0, 0] = float("inf")
        mask_inf = self.mask.clone()
        mask_inf[0, 0] = False  # Mask out the inf
        h_loss = h_loss_fn((x_inf, log_var), self.x_reconstructed, mask_inf)
        self.assertFalse(torch.isinf(h_loss).any(), "Loss should handle masked inf in inputs")

        # Test MultivariateGaussianLoss with NaNs and Infs
        m_loss_fn = MultivariateGaussianLoss().to(self.device)

        # Test NaN in covariance
        cov_nan = self.covariance_matrices.clone()
        cov_nan[0, 0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            _ = m_loss_fn(self.x_cov, self.x_cov_reconstructed, cov_nan)

        # Test infinity in mask (should be handled by casting)
        mask_inf = torch.ones_like(self.mask_cov).float()
        mask_inf[0, 0] = float("inf")
        m_loss = m_loss_fn(self.x_cov, self.x_cov_reconstructed, self.covariance_matrices, mask_inf)
        self.assertFalse(torch.isnan(m_loss).any(), "Loss should handle inf in mask")


if __name__ == "__main__":
    unittest.main()
