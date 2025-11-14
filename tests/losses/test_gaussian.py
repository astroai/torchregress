import unittest

import torch
import torch.nn as nn

from torchregress.distributions import Normal
from torchregress.losses.gaussian import GaussianNLLLoss
from torchregress.models import ProbabilisticModel


class TestGaussianLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = 4
        self.in_features = 10
        self.n_features_diag = 3
        self.n_features_cov = 5

        # Diagonal
        self.x_diag = torch.randn(self.batch_size, self.in_features, device=self.device)
        self.target_diag = torch.randn(self.batch_size, self.n_features_diag, device=self.device)
        self.mask_diag = torch.randint(
            0, 2, (self.batch_size, self.n_features_diag), device=self.device
        ).bool()

        # Full
        self.x_cov = torch.randn(self.batch_size, self.in_features, device=self.device)
        self.target_cov = torch.randn(self.batch_size, self.n_features_cov, device=self.device)
        self.mask_cov = torch.randint(
            0, 2, (self.batch_size, self.n_features_cov), device=self.device
        ).bool()

    def test_diagonal_covariance(self):
        """Test GaussianNLLLoss with diagonal covariance."""
        backbone = nn.Linear(self.in_features, self.in_features).to(self.device)
        dist_head = Normal(self.in_features, self.n_features_diag, covariance_type="diagonal")
        model = ProbabilisticModel(backbone, dist_head).to(self.device)
        loss_fn = GaussianNLLLoss().to(self.device)

        # Forward pass
        dist = model(self.x_diag)
        loss = loss_fn(dist, self.target_diag, mask=self.mask_diag)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

    def test_full_covariance(self):
        """Test GaussianNLLLoss with full covariance."""
        backbone = nn.Linear(self.in_features, self.in_features).to(self.device)
        dist_head = Normal(self.in_features, self.n_features_cov, covariance_type="full")
        model = ProbabilisticModel(backbone, dist_head).to(self.device)
        loss_fn = GaussianNLLLoss().to(self.device)

        # Forward pass
        dist = model(self.x_cov)
        loss = loss_fn(dist, self.target_cov, mask=self.mask_cov)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

    def test_reduction_modes(self):
        """Test reduction modes."""
        backbone = nn.Linear(self.in_features, self.in_features).to(self.device)
        dist_head = Normal(self.in_features, self.n_features_diag, covariance_type="diagonal")
        model = ProbabilisticModel(backbone, dist_head).to(self.device)
        dist = model(self.x_diag)

        for reduction in ["none", "mean", "sum"]:
            loss_fn = GaussianNLLLoss(reduction=reduction).to(self.device)
            loss = loss_fn(dist, self.target_diag)
            if reduction == "none":
                self.assertEqual(loss.shape, self.target_diag.shape)
            else:
                self.assertEqual(loss.dim(), 0)


if __name__ == "__main__":
    unittest.main()
