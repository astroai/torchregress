"""Unit tests for COSA-style residual adapter and weighted conformal regression adapter."""

from __future__ import annotations

import numpy as np
import torch

from torchregress.prediction import PredictiveBatch
from torchregress.test_time.cosa import DelayedLabelResidualAdapter
from torchregress.test_time.ot_conformal import WeightedConformalRegressionAdapter


class SimpleMockModel:
    def __init__(self, const_mean: float = 0.0, const_std: float = 1.0) -> None:
        self.const_mean = const_mean
        self.const_std = const_std

    def predict_distribution(self, X: torch.Tensor) -> PredictiveBatch:
        batch_size = X.shape[0]
        mean = torch.full((batch_size, 1), self.const_mean, device=X.device, dtype=X.dtype)
        std = torch.full((batch_size, 1), self.const_std, device=X.device, dtype=X.dtype)
        quantiles = torch.cat([mean - 1.96 * std, mean + 1.96 * std], dim=-1)
        return PredictiveBatch(
            point=mean,
            mean=mean,
            std=std,
            quantiles=quantiles,
            quantile_levels=torch.tensor([0.025, 0.975], device=X.device),
        )


def test_delayed_label_residual_adapter_point_correction() -> None:
    base_model = SimpleMockModel(const_mean=1.0, const_std=1.0)
    adapter = DelayedLabelResidualAdapter(base_model, ema_beta=1.0)

    # 1. Before any updates, predictions should equal base model predictions
    X = torch.randn(5, 3)
    pred_init = adapter.predict_distribution(X)
    np.testing.assert_allclose(pred_init.mean.numpy(), 1.0)

    # 2. Fit with a shifted target: target is constant 3.0, raw mean is 1.0 (error is +2.0)
    y = torch.full((5, 1), 3.0)
    adapter.partial_fit(X, y)

    # Since ema_beta = 1.0, residual_mean_ should be exactly 2.0
    assert adapter.residual_mean_ is not None
    np.testing.assert_allclose(adapter.residual_mean_.item(), 2.0, rtol=1e-5)

    # 3. Subsequent predictions should be shifted by +2.0 (adapted mean = 3.0)
    pred_adapted = adapter.predict_distribution(X)
    np.testing.assert_allclose(pred_adapted.mean.numpy(), 3.0, rtol=1e-5)
    np.testing.assert_allclose(pred_adapted.point.numpy(), 3.0, rtol=1e-5)

    # 4. Check NumPy input transparency (output is always Tensor)
    X_np = X.numpy()
    pred_np = adapter.predict_distribution(X_np)
    assert isinstance(pred_np.mean, torch.Tensor)
    np.testing.assert_allclose(pred_np.mean.numpy(), 3.0, rtol=1e-5)


def test_delayed_label_residual_adapter_variance_inflation() -> None:
    base_model = SimpleMockModel(const_mean=0.0, const_std=1.0)
    adapter = DelayedLabelResidualAdapter(base_model, ema_beta=1.0, scale_ema_beta=1.0)

    # X, y has raw std of 1.0, but errors are 2.0
    X = torch.randn(10, 3)
    y = torch.tensor(
        [[2.0], [-2.0], [2.0], [-2.0], [2.0], [-2.0], [2.0], [-2.0], [2.0], [-2.0]]
    )  # mean 0, std 2
    adapter.partial_fit(X, y)

    # residual_mean should be 0.0. z_squared should be (y / 1.0)**2 = 4.0.
    # Since scale_ema_beta = 1.0, variance_inflation_ should be 4.0.
    assert adapter.variance_inflation_ is not None
    np.testing.assert_allclose(adapter.variance_inflation_.item(), 4.0, rtol=1e-5)

    # Adapted std should be raw std (1.0) * sqrt(4.0) = 2.0
    pred = adapter.predict_distribution(X)
    np.testing.assert_allclose(pred.std.numpy(), 2.0, rtol=1e-5)

    # Adapted quantiles should be scaled appropriately: mean_adapted + scale * (quantiles - mean_raw)
    # mean_adapted = 0.0, scale = 2.0. Base quantiles: [-1.96, 1.96]. Adapted: [-3.92, 3.92]
    np.testing.assert_allclose(
        pred.quantiles.numpy(), 2.0 * pred_init_quantiles_diff(base_model, X), rtol=1e-5
    )


def pred_init_quantiles_diff(model, X):
    return model.predict_distribution(X).quantiles.numpy()


def test_weighted_conformal_regression_adapter_basics() -> None:
    # Set seed
    torch.manual_seed(0)
    np.random.seed(0)

    X_cal = np.random.normal(0, 1, size=(100, 2))
    X_tgt = np.random.normal(1, 1, size=(80, 2))

    y_cal = np.random.normal(0, 0.5, size=(100, 1))
    y_pred_cal = np.zeros_like(y_cal)

    adapter = WeightedConformalRegressionAdapter(alpha=0.1)
    adapter.calibrate(y_pred_cal, y_cal, X_cal, X_tgt)

    # Verify classifier and weights are estimated
    assert adapter.classifier is not None
    assert adapter.w_cal_ is not None
    assert adapter.w_cal_.shape == (100,)
    assert torch.all(adapter.w_cal_ > 0)

    # Predict intervals for query points
    X_test = np.random.normal(0.5, 1, size=(10, 2))
    y_pred_test = np.zeros((10, 1))

    lower, upper = adapter.predict_interval(y_pred_test, X_test)
    assert isinstance(lower, torch.Tensor)
    assert isinstance(upper, torch.Tensor)
    assert lower.shape == (10, 1)
    assert upper.shape == (10, 1)
    assert torch.all(lower < upper)


def test_weighted_conformal_regression_adapter_difficulty_adaptive() -> None:
    X_cal = torch.randn(50, 2)
    X_tgt = torch.randn(40, 2) + 0.5

    y_cal = torch.randn(50, 1) * 0.2
    # Define predictive batch with varying standard deviations
    std_cal = torch.ones(50, 1) * 0.5
    std_cal[:25] = 2.0  # varying difficulty
    y_pred_cal = PredictiveBatch(mean=torch.zeros(50, 1), std=std_cal)

    adapter = WeightedConformalRegressionAdapter(alpha=0.1)
    adapter.calibrate(y_pred_cal, y_cal, X_cal, X_tgt)

    # Predict with varying std on test set
    x_single = torch.randn(1, 2)
    X_test = x_single.repeat(5, 1)  # Identical features to ensure same density ratios
    std_test = torch.tensor([[0.1], [1.0], [2.0], [0.5], [1.5]])
    y_pred_test = PredictiveBatch(mean=torch.zeros(5, 1), std=std_test)

    lower, upper = adapter.predict_interval(y_pred_test, X_test)
    assert torch.is_tensor(lower)
    assert torch.is_tensor(upper)

    widths = upper - lower
    # Widths should scale exactly with std_test because scores were normalized by std
    np.testing.assert_allclose(
        (widths / std_test).numpy(), (widths[0] / std_test[0]).item(), rtol=1e-3
    )
