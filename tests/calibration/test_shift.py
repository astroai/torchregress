from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.calibration.shift import BinnedLabelShiftEstimator


def test_binning_strategies() -> None:
    # Test adaptive (quantile-based) binning
    y_source = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
    pred_source = y_source.copy()
    pred_target = np.array([2.0, 3.0])

    estimator = BinnedLabelShiftEstimator(n_bins=4, binning_strategy="adaptive")
    estimator.fit(y_source, pred_source, pred_target)

    assert estimator.bin_edges_ is not None
    assert len(estimator.bin_edges_) == 5
    assert estimator.bin_edges_[0] == -np.inf
    assert estimator.bin_edges_[-1] == np.inf
    assert estimator.source_prior_ is not None
    assert np.allclose(estimator.source_prior_.sum(), 1.0)

    # Test uniform binning
    estimator_uniform = BinnedLabelShiftEstimator(n_bins=4, binning_strategy="uniform")
    estimator_uniform.fit(y_source, pred_source, pred_target)
    assert len(estimator_uniform.bin_edges_) == 5


def test_bbse_prior_estimation() -> None:
    # 3 bins: we set up source and target datasets with known shifted priors.
    # Bin 0: 0 to 1, Bin 1: 1 to 2, Bin 2: 2 to 3.
    # Source prior: [0.6, 0.3, 0.1]
    # Target prior: [0.2, 0.3, 0.5]
    np.random.seed(42)
    n_source = 3000
    n_target = 3000

    # Source samples
    y_src = np.concatenate(
        [
            np.random.uniform(0.0, 1.0, int(0.6 * n_source)),
            np.random.uniform(1.0, 2.0, int(0.3 * n_source)),
            np.random.uniform(2.0, 3.0, int(0.1 * n_source)),
        ]
    )

    # Target samples
    y_tgt = np.concatenate(
        [
            np.random.uniform(0.0, 1.0, int(0.2 * n_target)),
            np.random.uniform(1.0, 2.0, int(0.3 * n_target)),
            np.random.uniform(2.0, 3.0, int(0.5 * n_target)),
        ]
    )

    # Simulate a proxy predictor by adding noise to true values
    noise_src = np.random.normal(0, 0.1, y_src.shape)
    pred_src = y_src + noise_src

    noise_tgt = np.random.normal(0, 0.1, y_tgt.shape)
    pred_tgt = y_tgt + noise_tgt

    # Fit BBSE
    estimator = BinnedLabelShiftEstimator(n_bins=3, binning_strategy="uniform", method="bbse")
    estimator.fit(y_src, pred_src, pred_tgt)

    # Verify target prior is estimated close to [0.2, 0.3, 0.5]
    true_tgt_prior = np.array([0.2, 0.3, 0.5])
    assert estimator.target_prior_ is not None
    assert np.allclose(estimator.target_prior_, true_tgt_prior, atol=0.08)

    # Verify bin weights are p_target / p_source
    true_weights = true_tgt_prior / estimator.source_prior_
    assert np.allclose(estimator.get_bin_weights(), true_weights, atol=0.08)


def test_em_prior_estimation() -> None:
    np.random.seed(42)
    n_source = 5000
    n_target = 5000

    # Source prior: [0.7, 0.3]
    # Target prior: [0.2, 0.8]
    # Class conditional: x | y=0 ~ N(-0.5, 0.8^2), x | y=1 ~ N(0.5, 0.8^2)
    y_src = np.random.choice([0, 1], size=n_source, p=[0.7, 0.3])
    y_tgt = np.random.choice([0, 1], size=n_target, p=[0.2, 0.8])

    x_src = np.random.normal(loc=y_src - 0.5, scale=0.8)
    x_tgt = np.random.normal(loc=y_tgt - 0.5, scale=0.8)

    def class_conditional(x: np.ndarray, y_val: int) -> np.ndarray:
        return np.exp(-0.5 * ((x - (y_val - 0.5)) / 0.8) ** 2) / (0.8 * np.sqrt(2 * np.pi))

    def source_classifier(x: np.ndarray) -> np.ndarray:
        p_x_y0 = class_conditional(x, 0)
        p_x_y1 = class_conditional(x, 1)
        val0 = p_x_y0 * 0.7
        val1 = p_x_y1 * 0.3
        denom = val0 + val1
        return np.stack([val0 / denom, val1 / denom], axis=1)

    pred_src_probs = source_classifier(x_src)
    pred_tgt_probs = source_classifier(x_tgt)

    # Fit EM
    estimator = BinnedLabelShiftEstimator(n_bins=2, binning_strategy="uniform", method="em")
    estimator.fit(y_src, pred_src_probs, pred_tgt_probs)

    # Verify target prior is estimated close to [0.2, 0.8]
    true_tgt_prior = np.array([0.2, 0.8])
    assert estimator.target_prior_ is not None
    assert np.allclose(estimator.target_prior_, true_tgt_prior, atol=0.08)


def test_sample_weights_numpy_and_torch() -> None:
    y_source = np.array([0.5, 0.5, 1.5, 1.5])
    pred_source = y_source.copy()
    pred_target = np.array([0.5, 1.5, 1.5, 1.5])  # Target prior: 25% bin 0, 75% bin 1

    estimator = BinnedLabelShiftEstimator(n_bins=2, binning_strategy="uniform", method="bbse")
    estimator.fit(y_source, pred_source, pred_target)

    # Bin weights should be:
    # Bin 0: p_tgt=0.25, p_src=0.5 -> weight = 0.5
    # Bin 1: p_tgt=0.75, p_src=0.5 -> weight = 1.5
    assert np.allclose(estimator.get_bin_weights(), [0.5, 1.5])

    # Test sample weights with numpy array
    y_test = np.array([0.2, 1.8, 0.4])
    w_numpy = estimator.sample_weights(y_test)
    assert np.allclose(w_numpy, [0.5, 1.5, 0.5])

    # Test sample weights with torch tensor
    y_tensor = torch.tensor([0.2, 1.8, 0.4], dtype=torch.float32)
    w_tensor = estimator.sample_weights(y_tensor)
    assert isinstance(w_tensor, torch.Tensor)
    assert torch.allclose(w_tensor, torch.tensor([0.5, 1.5, 0.5]))


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        BinnedLabelShiftEstimator(n_bins=1)

    with pytest.raises(ValueError):
        BinnedLabelShiftEstimator(binning_strategy="invalid")

    with pytest.raises(ValueError):
        BinnedLabelShiftEstimator(method="invalid")

    # Mismatched prediction shapes
    estimator = BinnedLabelShiftEstimator(n_bins=3)
    with pytest.raises(ValueError):
        estimator.fit([1.0, 2.0], np.ones((2, 4)), np.ones((2, 3)))

    # Negative probabilities
    with pytest.raises(ValueError):
        estimator.fit([1.0, 2.0], np.array([[-1.0, 2.0, 0.0], [0.0, 1.0, 0.0]]), np.ones((2, 3)))
