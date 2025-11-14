import unittest

import torch
import torch.nn as nn

from torchregress.distributions import Poisson
from torchregress.losses.poisson import PoissonNLLLoss
from torchregress.models import ProbabilisticModel


class TestPoissonLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = 4
        self.in_features = 10
        self.n_features = 3

        self.x = torch.randn(self.batch_size, self.in_features, device=self.device)
        self.target = torch.randint(
            0, 10, (self.batch_size, self.n_features), device=self.device
        ).float()
        self.mask = torch.randint(
            0, 2, (self.batch_size, self.n_features), device=self.device
        ).bool()

    def test_poisson_nll_loss(self):
        """Test PoissonNLLLoss."""
        backbone = nn.Linear(self.in_features, self.in_features).to(self.device)
        dist_head = Poisson(self.in_features, self.n_features)
        model = ProbabilisticModel(backbone, dist_head).to(self.device)
        loss_fn = PoissonNLLLoss().to(self.device)

        # Forward pass
        dist = model(self.x)
        loss = loss_fn(dist, self.target, mask=self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

    def test_reduction_modes(self):
        """Test reduction modes."""
        backbone = nn.Linear(self.in_features, self.in_features).to(self.device)
        dist_head = Poisson(self.in_features, self.n_features)
        model = ProbabilisticModel(backbone, dist_head).to(self.device)
        dist = model(self.x)

        for reduction in ["none", "mean", "sum"]:
            loss_fn = PoissonNLLLoss(reduction=reduction).to(self.device)
            loss = loss_fn(dist, self.target)
            if reduction == "none":
                self.assertEqual(loss.shape, self.target.shape)
            else:
                self.assertEqual(loss.dim(), 0)


if __name__ == "__main__":
    unittest.main()
