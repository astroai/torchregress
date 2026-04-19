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


def test_ordinal_predict_invalid_strategy() -> None:
    y_pred = torch.tensor([[0.0, 0.0]])
    with pytest.raises(ValueError, match="Unknown strategy: foo"):
        ordinal_predict(y_pred, strategy="foo")  # type: ignore


def test_ordinal_predict_invalid_encoding() -> None:
    y_pred = torch.tensor([[0.0, 0.0]])
    with pytest.raises(ValueError, match="Unknown encoding: foo"):
        ordinal_predict(y_pred, encoding="foo")  # type: ignore


def test_ordinal_predict_threshold_strategy_invalid() -> None:
    y_pred = torch.tensor([[0.0, 0.0]])
    with pytest.raises(
        ValueError, match="strategy='threshold' is only valid for cumulative encodings"
    ):
        ordinal_predict(y_pred, encoding="class_logits", strategy="threshold")
    with pytest.raises(
        ValueError, match="strategy='threshold' is only valid for cumulative encodings"
    ):
        ordinal_predict(y_pred, encoding="class_probs", strategy="threshold")
