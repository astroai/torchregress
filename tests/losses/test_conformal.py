"""
Tests for the unified conformal prediction module.
"""

import pytest
import torch
from torch.nn import Linear, Module, Sequential

from torchregress.losses.conformal import (
    ConformalLoss,
    MultiDimensionalConformalLoss,
)


class DummyModel(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Sequential(Linear(in_features, out_features))

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize("method", ["cqr", "split", "aci"])
def test_conformal_loss_initialization(method):
    """Test initialization of ConformalLoss for different methods."""
    if method == "aci":
        model = DummyModel(1, 1)
        loss_fn = ConformalLoss(method=method, alpha=0.1, model=model)
    else:
        loss_fn = ConformalLoss(method=method, alpha=0.1)
    assert loss_fn.method == method
    assert loss_fn.alpha == 0.1
    assert loss_fn._predictor is not None
    assert not loss_fn._is_calibrated


def test_conformal_loss_forward_cqr():
    """Test forward pass of ConformalLoss with CQR."""
    loss_fn = ConformalLoss(method="cqr", alpha=0.1)
    batch_size, n_features = 10, 1
    # CQR expects lower and upper quantiles
    y_pred = torch.randn(batch_size, 2 * n_features)
    y_true = torch.randn(batch_size, n_features)
    loss = loss_fn(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


@pytest.mark.parametrize("method", ["split", "aci"])
def test_conformal_loss_forward_split_aci(method):
    """Test forward pass of ConformalLoss with Split and ACI."""
    if method == "aci":
        model = DummyModel(1, 1)
        loss_fn = ConformalLoss(method=method, alpha=0.1, model=model)
    else:
        loss_fn = ConformalLoss(method=method, alpha=0.1)
    batch_size, n_features = 10, 1
    # Split/ACI expect point predictions
    y_pred = torch.randn(batch_size, n_features)
    y_true = torch.randn(batch_size, n_features)
    loss = loss_fn(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


@pytest.mark.parametrize("method", ["cqr", "split", "aci"])
def test_conformal_loss_calibration_and_prediction(method):
    """Test calibration and prediction flow."""
    if method == "aci":
        model = DummyModel(1, 1)
        loss_fn = ConformalLoss(method=method, alpha=0.1, model=model)
    else:
        loss_fn = ConformalLoss(method=method, alpha=0.1)

    batch_size, n_features = 20, 1

    # Prepare data
    if method == "cqr":
        y_pred_cal = torch.randn(batch_size, 2 * n_features)
        y_pred_test = torch.randn(batch_size, 2 * n_features)
    else:
        y_pred_cal = torch.randn(batch_size, n_features)
        y_pred_test = torch.randn(batch_size, n_features)

    y_true_cal = torch.randn(batch_size, n_features)

    # Prediction should fail before calibration
    with pytest.raises(RuntimeError):
        loss_fn.predict_interval(y_pred_test)

    # Calibrate
    loss_fn.calibrate(y_pred_cal, y_true_cal)
    assert loss_fn._is_calibrated

    # Predict intervals
    lower, upper = loss_fn.predict_interval(y_pred_test)
    assert isinstance(lower, torch.Tensor)
    assert isinstance(upper, torch.Tensor)

    expected_shape = (batch_size, n_features)
    assert lower.shape == expected_shape
    assert upper.shape == expected_shape


def test_adaptive_conformal_loss_method():
    """Test that ConformalLoss with method='aci' works correctly."""
    model = DummyModel(1, 1)
    loss_fn = ConformalLoss(method="aci", alpha=0.1, model=model)
    assert isinstance(loss_fn, ConformalLoss)
    assert loss_fn.method == "aci"


def test_conformalized_quantile_loss_method():
    """Test that ConformalLoss with method='cqr' works correctly."""
    loss_fn = ConformalLoss(method="cqr", alpha=0.1)
    assert isinstance(loss_fn, ConformalLoss)
    assert loss_fn.method == "cqr"


def test_multidimensional_conformal_loss_wrapper():
    """Test that MultiDimensionalConformalLoss is a wrapper for ConformalLoss with method='split'."""
    loss_fn = MultiDimensionalConformalLoss(alpha=0.1)
    assert isinstance(loss_fn, ConformalLoss)
    assert loss_fn.method == "split"

    # Test the predict_interval method
    batch_size, n_features = 20, 3
    y_pred_cal = torch.randn(batch_size, n_features)
    y_true_cal = torch.randn(batch_size, n_features)
    y_pred_test = torch.randn(batch_size, n_features)

    loss_fn.calibrate(y_pred_cal, y_true_cal)
    lower, upper = loss_fn.predict_interval(y_pred_test)
    assert lower.shape == y_pred_test.shape
    assert upper.shape == y_pred_test.shape
    assert lower.shape[-1] == n_features


if __name__ == "__main__":
    pytest.main([__file__])
