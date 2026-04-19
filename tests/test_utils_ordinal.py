from __future__ import annotations

import pytest
import torch

from torchregress.utils import (
    class_probs_to_levels,
    cumulative_logits_to_pmf,
    cumulative_probs_to_pmf,
    labels_to_levels,
    normalize_class_probs,
    ordinal_predict,
)
from torchregress.utils.ordinal import _validate_num_classes


def test_validate_num_classes_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="num_classes must be >= 2"):
        _validate_num_classes(-1)
    with pytest.raises(ValueError, match="num_classes must be >= 2"):
        _validate_num_classes(0)
    with pytest.raises(ValueError, match="num_classes must be >= 2"):
        _validate_num_classes(1)

    # Should not raise
    _validate_num_classes(2)
    _validate_num_classes(10)


def test_labels_to_levels_encoding() -> None:
    labels = torch.tensor([0, 1, 3])
    levels = labels_to_levels(labels, num_classes=4)
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    assert torch.allclose(levels, expected)


def test_cumulative_probs_to_pmf_rows_sum_to_one() -> None:
    probs = torch.tensor([[0.9, 0.6, 0.2], [0.7, 0.5, 0.1]], dtype=torch.float32)
    pmf = cumulative_probs_to_pmf(probs)
    assert pmf.shape[-1] == 4
    assert torch.allclose(pmf.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_cumulative_probs_to_pmf_error_dimension_less_than_one() -> None:
    empty_probs = torch.empty((2, 0), dtype=torch.float32)
    with pytest.raises(ValueError, match="cumulative_probs last dimension must be >= 1"):
        cumulative_probs_to_pmf(empty_probs)


def test_ordinal_predict_supports_multiple_encodings() -> None:
    class_logits = torch.tensor([[2.0, 0.5, -0.5], [0.1, 0.3, 1.4]])
    pred_class = ordinal_predict(class_logits, encoding="class_logits")
    assert isinstance(pred_class, torch.Tensor)
    assert pred_class.tolist() == [0, 2]

    cum_logits = torch.tensor([[4.0, 2.0], [-1.0, -2.0]])
    pred_cum = ordinal_predict(cum_logits, encoding="cumulative_logits", strategy="threshold")
    assert isinstance(pred_cum, torch.Tensor)
    assert pred_cum.tolist() == [2, 0]

    pmf = cumulative_logits_to_pmf(cum_logits)
    assert torch.allclose(pmf.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_soft_class_prob_utilities_normalize_and_convert_to_levels() -> None:
    target_probs = torch.tensor(
        [
            [2.0, 1.0, 1.0],
            [0.0, 3.0, 1.0],
        ],
        dtype=torch.float32,
    )
    normalized = normalize_class_probs(target_probs, class_dim=1)
    assert torch.allclose(normalized.sum(dim=1), torch.ones(2), atol=1e-6)

    levels = class_probs_to_levels(normalized, class_dim=1)
    expected = torch.tensor(
        [
            [0.5, 0.25],
            [1.0, 0.25],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(levels, expected, atol=1e-6)


def test_normalize_class_probs_negative_values() -> None:
    target_probs = torch.tensor(
        [
            [2.0, -1.0, 1.0],
            [0.0, 3.0, 1.0],
        ],
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="target_probs must be non-negative"):
        normalize_class_probs(target_probs, class_dim=1)


def test_labels_to_levels_invalid_indices() -> None:
    num_classes = 4

    with pytest.raises(ValueError, match=r"target class indices must be in \[0, num_classes - 1\]"):
        labels_to_levels(torch.tensor([-1, 1, 2]), num_classes=num_classes)

    with pytest.raises(ValueError, match=r"target class indices must be in \[0, num_classes - 1\]"):
        labels_to_levels(torch.tensor([0, 1, 4]), num_classes=num_classes)


def test_ordinal_predict_invalid_strategy() -> None:
    y_pred = torch.tensor([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Unknown strategy: invalid_strategy"):
        ordinal_predict(y_pred, strategy="invalid_strategy")  # type: ignore[arg-type]


def test_ordinal_predict_validation_errors() -> None:
    # Test invalid encoding
    y_pred = torch.tensor([[0.1, 0.9]])
    with pytest.raises(ValueError, match="Unknown encoding: invalid"):
        ordinal_predict(y_pred, encoding="invalid")  # type: ignore[arg-type]

    # Test mismatch between num_classes and prediction shape
    with pytest.raises(ValueError, match="num_classes=5 does not match input shape"):
        ordinal_predict(y_pred, encoding="class_probs", num_classes=5)
