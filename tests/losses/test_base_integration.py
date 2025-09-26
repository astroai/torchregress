import torch
import torch.nn as nn

from torchregress.losses.base import DistributionLoss, RegressionLoss, WeightedLossWrapper


# Define a simple model for integration testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class SimpleDistributionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_mean = nn.Linear(10, 1)
        self.linear_logvar = nn.Linear(10, 1)

    def forward(self, x):
        mean = self.linear_mean(x)
        logvar = self.linear_logvar(x)
        return torch.cat([mean, logvar], dim=-1)


# Define concrete losses for testing
class L1RegressionLoss(RegressionLoss):
    def forward(self, y_pred, target, mask=None, weights=None):
        self._validate_inputs(y_pred, target, mask)
        loss = torch.abs(y_pred - target)
        return self._reduce_with_mask(loss, mask, weights)


class GaussianNLL(DistributionLoss):
    def _extract_distribution_parameters(self, y_pred):
        mean = y_pred[..., 0:1]
        logvar = y_pred[..., 1:2]
        return {"mean": mean, "logvar": logvar}

    def _calculate_nll(self, y_pred, target, mask=None):
        params = self._extract_distribution_parameters(y_pred)
        mean = params["mean"]
        logvar = params["logvar"]
        nll = 0.5 * (logvar + (target - mean) ** 2 / torch.exp(logvar))
        return nll

    def forward(self, y_pred, target, mask=None, weights=None):
        # Reshape target to match mean dimensionality
        target = target.view(-1, 1)
        self._validate_inputs(y_pred[..., 0:1], target, mask)
        nll = self._calculate_nll(y_pred, target, mask)
        return self._reduce_with_mask(nll, mask, weights)


class TestIntegration:
    def test_regression_model_training(self):
        """Test training a simple model with regression loss."""
        # Create model, loss, and optimizer
        model = SimpleModel()
        loss_fn = L1RegressionLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create fake data
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)

        # Before training
        initial_loss = None
        with torch.no_grad():
            pred = model(X)
            initial_loss = loss_fn(pred, y).item()

        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        # After training
        with torch.no_grad():
            pred = model(X)
            final_loss = loss_fn(pred, y).item()

        # Loss should decrease
        assert final_loss < initial_loss

    def test_masked_regression_training(self):
        """Test training with masked loss."""
        model = SimpleModel()
        loss_fn = L1RegressionLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create fake data with mask
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        mask = torch.rand(100, 1) > 0.3  # 70% of data is valid

        # Train for a few steps
        initial_loss = None
        with torch.no_grad():
            pred = model(X)
            initial_loss = loss_fn(pred, y, mask=mask).item()

        for _ in range(5):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y, mask=mask)
            loss.backward()
            optimizer.step()

        # After training
        with torch.no_grad():
            pred = model(X)
            final_loss = loss_fn(pred, y, mask=mask).item()

        # Loss should decrease
        assert final_loss < initial_loss

    def test_distribution_model_training(self):
        """Test training a distributional model."""
        model = SimpleDistributionModel()
        loss_fn = GaussianNLL()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create fake data
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)

        # Before training
        initial_loss = None
        with torch.no_grad():
            pred = model(X)
            initial_loss = loss_fn(pred, y).item()

        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        # After training
        with torch.no_grad():
            pred = model(X)
            final_loss = loss_fn(pred, y).item()

        # Loss should decrease
        assert final_loss < initial_loss

    def test_weighted_wrapper_integration(self):
        """Test the WeightedLossWrapper with PyTorch model."""
        model = SimpleModel()
        # Wrap PyTorch's MSE loss
        loss_fn = WeightedLossWrapper(nn.MSELoss)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create fake data
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)

        # Train for a few steps
        initial_loss = None
        with torch.no_grad():
            pred = model(X)
            initial_loss = loss_fn(pred, y).item()

        for _ in range(5):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        # Check that loss decreased
        with torch.no_grad():
            pred = model(X)
            final_loss = loss_fn(pred, y).item()

        assert final_loss < initial_loss
