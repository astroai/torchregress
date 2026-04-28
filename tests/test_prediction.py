from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.prediction import (
    PredictiveBatch,
    _to_numpy,
    bars_to_density_grid,
    quantiles_to_density_grid,
    samples_to_density_grid,
)


def test_to_numpy():
    # Numpy array
    arr = np.array([1, 2, 3])
    res = _to_numpy(arr)
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, arr)

    # Tensor with requires_grad
    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    res = _to_numpy(t)
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, np.array([1.0, 2.0, 3.0]))


def test_quantiles_to_density_grid_errors():
    quantiles = np.array([[0.1, 0.4, 0.8], [0.2, 0.5, 0.9]])

    # Wrong quantiles shape
    with pytest.raises(ValueError, match="quantiles must have shape"):
        quantiles_to_density_grid(np.array([0.1, 0.4, 0.8]), [0.1, 0.5, 0.9])

    # Shape mismatch
    with pytest.raises(ValueError, match="quantile_levels must match"):
        quantiles_to_density_grid(quantiles, [0.1, 0.5])

    # Too few levels
    with pytest.raises(ValueError, match="at least two quantile levels"):
        quantiles_to_density_grid(np.array([[0.5]]), [0.5])

    # Non-monotonic levels
    with pytest.raises(ValueError, match="strictly increasing"):
        quantiles_to_density_grid(quantiles, [0.5, 0.1, 0.9])


def test_bars_to_density_grid_errors():
    logits = np.array([[1.0, 0.5, -0.2]])
    edges = np.array([0.0, 0.5, 1.0, 1.5])

    # Wrong logits shape
    with pytest.raises(ValueError, match="bar_logits must have shape"):
        bars_to_density_grid(np.array([1.0, 0.5, -0.2]), edges)

    # Shape mismatch
    with pytest.raises(ValueError, match="bin_edges must have shape"):
        bars_to_density_grid(logits, np.array([0.0, 0.5, 1.0]))

    # Edge shape mismatch with 2D edges
    with pytest.raises(ValueError, match="bin_edges must have shape"):
        bars_to_density_grid(logits, np.array([[0.0, 0.5, 1.0]]))


def test_samples_to_density_grid_errors():

    # Wrong shape
    with pytest.raises(ValueError, match="samples must have shape"):
        samples_to_density_grid(np.array([0.1, 0.4, 0.8]))

    # Too few samples
    with pytest.raises(ValueError, match="at least two samples"):
        samples_to_density_grid(np.array([[0.1], [0.2]]))


def test_samples_to_density_grid_3d():
    samples = np.array([[[0.1], [0.4], [0.8]], [[0.2], [0.5], [0.9]]])
    support, density = samples_to_density_grid(samples, n_support=10)
    assert support.shape == (2, 10)
    assert density.shape == (2, 10)
    # Check normalization
    assert np.allclose(np.trapezoid(density, support, axis=1), 1.0, atol=1.0e-3)


def test_predictive_batch_with_density_skips_when_present():
    batch = PredictiveBatch(
        support=np.array([[0.0, 1.0]]),
        density=np.array([[1.0, 1.0]]),
    )
    res = batch.with_density()
    assert res is batch


def test_predictive_batch_with_density_returns_self_when_empty():
    batch = PredictiveBatch()
    res = batch.with_density()
    assert res is batch


def test_predictive_batch_identical_support_collapsing():
    # Helper to test the collapsing logic when support for all items in batch is close

    # Quantiles
    quantiles = np.array([[0.1, 0.5, 0.9], [0.1, 0.5, 0.9]])
    batch = PredictiveBatch(quantiles=quantiles, quantile_levels=[0.1, 0.5, 0.9])
    res = batch.with_density(n_support=5)
    assert res.support.ndim == 1  # Collapsed!
    assert res.support.shape == (5,)

    # Bars
    logits = np.array([[1.0, 0.5], [1.0, 0.5]])
    edges = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    batch2 = PredictiveBatch(bar_logits=logits, bin_edges=edges)
    res2 = batch2.with_density(n_support=5)
    assert res2.support.ndim == 1  # Collapsed!
    assert res2.support.shape == (5,)

    # Samples
    samples = np.array([[0.1, 0.5, 0.9], [0.1, 0.5, 0.9]])
    batch3 = PredictiveBatch(samples=samples)
    res3 = batch3.with_density(n_support=5)
    assert res3.support.ndim == 1  # Collapsed!
    assert res3.support.shape == (5,)
