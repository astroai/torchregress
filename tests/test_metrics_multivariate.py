"""Tests for multivariate metrics."""

import pytest
import torch

from torchregress.metrics.multivariate import (
    MultivariateMAE,
    MultivariateRMSE,
    multivariate_mae,
    multivariate_rmse,
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

    def test_multivariate_rmse_functional(self, setup_data):
        y_pred, y_true = setup_data
        val = multivariate_rmse(y_pred, y_true)
        expected = torch.sqrt(torch.tensor(11.0 / 3.0))
        assert torch.isclose(val, expected)

    def test_multivariate_mae_functional(self, setup_data):
        y_pred, y_true = setup_data
        val = multivariate_mae(y_pred, y_true)
        expected = torch.tensor(5.0 / 3.0)
        assert torch.isclose(val, expected)

    def test_numpy_inputs(self, setup_data):
        y_pred, y_true = setup_data
        y_pred_np = y_pred.numpy()
        y_true_np = y_true.numpy()

        val_rmse = multivariate_rmse(y_pred_np, y_true_np)
        expected_rmse = torch.sqrt(torch.tensor(11.0 / 3.0))
        assert torch.isclose(val_rmse, expected_rmse)

        val_mae = multivariate_mae(y_pred_np, y_true_np)
        expected_mae = torch.tensor(5.0 / 3.0)
        assert torch.isclose(val_mae, expected_mae)

    def test_shape_mismatch(self):
        y_pred = torch.randn(3, 2)
        y_true = torch.randn(3, 3)

        with pytest.raises(ValueError):
            multivariate_rmse(y_pred, y_true)

    def test_multivariate_rmse_zero_error(self):
        y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        val = multivariate_rmse(y, y)
        assert torch.isclose(val, torch.tensor(0.0))

    def test_multivariate_mae_zero_error(self):
        y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        val = multivariate_mae(y, y)
        assert torch.isclose(val, torch.tensor(0.0))

    def test_scalar_inputs(self):
        y_pred = torch.tensor(1.0)
        y_true = torch.tensor(1.0)
        with pytest.raises(ValueError, match="Inputs cannot be scalars"):
            multivariate_rmse(y_pred, y_true)

    def test_batch_mismatch(self):
        y_pred = torch.randn(2, 2)
        y_true = torch.randn(3, 2)
        with pytest.raises(ValueError, match="must have same batch size"):
            multivariate_rmse(y_pred, y_true)
