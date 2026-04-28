import numpy as np
import pytest
import torch

from torchregress.utils.labels import (
    combine_binary_average,
    combine_binary_weighted_average,
    decode_onehot,
    encode_onehot,
    label_smoothing,
    soft_to_hard_labels,
)


def test_label_smoothing_valid() -> None:
    onehot = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    smoothed = label_smoothing(onehot, alpha=0.1)
    expected = torch.tensor([[0.95, 0.05], [0.05, 0.95]])
    assert torch.allclose(smoothed, expected)


def test_label_smoothing_invalid_alpha() -> None:
    onehot = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="alpha must be in \\[0, 1\\]"):
        label_smoothing(onehot, alpha=-0.1)

    with pytest.raises(ValueError, match="alpha must be in \\[0, 1\\]"):
        label_smoothing(onehot, alpha=1.1)


def test_label_smoothing_numpy() -> None:
    onehot = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    smoothed = label_smoothing(onehot, alpha=0.1)
    expected = torch.tensor([[0.95, 0.05], [0.05, 0.95]])
    assert torch.allclose(smoothed, expected)


def test_encode_onehot() -> None:
    labels = torch.tensor([0, 1, 2])
    onehot = encode_onehot(labels, num_classes=3)
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(onehot, expected)


def test_decode_onehot() -> None:
    onehot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    labels = decode_onehot(onehot)
    expected = torch.tensor([0, 1, 2])
    assert torch.equal(labels, expected)


def test_soft_to_hard_labels() -> None:
    soft = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    hard = soft_to_hard_labels(soft)
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(hard, expected)


def test_combine_binary_average() -> None:
    labels = torch.tensor([[1, 0], [1, 1], [0, 0]])
    avg = combine_binary_average(labels, dim=0)
    expected = torch.tensor([2 / 3, 1 / 3])
    assert torch.allclose(avg, expected)


def test_combine_binary_weighted_average() -> None:
    labels = torch.tensor([[1, 0], [1, 1], [0, 0]])
    weights = torch.tensor([0.5, 0.3, 0.2])
    weighted_avg = combine_binary_weighted_average(labels, weights, dim=0)
    # Weights normalized: [0.5, 0.3, 0.2]
    # Col 0: 1*0.5 + 1*0.3 + 0*0.2 = 0.8
    # Col 1: 0*0.5 + 1*0.3 + 0*0.2 = 0.3
    expected = torch.tensor([0.8, 0.3])
    assert torch.allclose(weighted_avg, expected)
