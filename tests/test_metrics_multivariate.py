"""Tests for multivariate metrics."""

import pytest
import torch

from torchregress.metrics.multivariate import (
    MultivariateMAE,
    MultivariateRMSE,
)


class TestMultivariateMetrics:
    @pytest.fixture
    def setup_data(self):
        # Create deterministic dummy data
        y_true = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y_pred = torch.tensor([[1.0, 0.0], [1.0, 2.0], [5.0, 2.0]])
        return y_pred, y_true

    def test_multivariate_rmse_class(self, setup_data):
        y_pred, y_true = setup_data
        metric = MultivariateRMSE()
        metric.update(y_pred, y_true)
        val = metric.compute()
        expected = torch.sqrt(torch.tensor(11.0 / 3.0))
        assert torch.isclose(val, expected)

    def test_multivariate_mae_class(self, setup_data):
        y_pred, y_true = setup_data
        metric = MultivariateMAE()
        metric.update(y_pred, y_true)
        val = metric.compute()
        expected = torch.tensor(5.0 / 3.0)
        assert torch.isclose(val, expected)
