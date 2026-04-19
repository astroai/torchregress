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
    class_logits = torch.tensor([[2.0, 0.5, -0.5], [0.1, 0.3, 1.4]])

    with pytest.raises(
        ValueError, match="strategy='threshold' is only valid for cumulative encodings"
    ):
        ordinal_predict(class_logits, encoding="class_logits", strategy="threshold")

    with pytest.raises(
        ValueError, match="strategy='threshold' is only valid for cumulative encodings"
    ):
        ordinal_predict(class_logits, encoding="class_probs", strategy="threshold")

    with pytest.raises(ValueError, match="Unknown strategy: invalid"):
        ordinal_predict(class_logits, strategy="invalid")  # type: ignore


def test_ordinal_predict_other_branches() -> None:
    cum_probs = torch.tensor([[0.9, 0.4], [0.1, 0.05]])

    # Test encoding="cumulative_probs" with strategy="argmax"
    pred1 = ordinal_predict(cum_probs, encoding="cumulative_probs", strategy="argmax")
    assert isinstance(pred1, torch.Tensor)

    # Test encoding="cumulative_probs" with strategy="threshold"
    pred2 = ordinal_predict(cum_probs, encoding="cumulative_probs", strategy="threshold")
    assert isinstance(pred2, torch.Tensor)

    # Test encoding="class_probs" with strategy="argmax"
    class_probs = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    pred3 = ordinal_predict(class_probs, encoding="class_probs", strategy="argmax")
    assert isinstance(pred3, torch.Tensor)

    # Test unknown encoding
    with pytest.raises(ValueError, match="Unknown encoding: invalid"):
        ordinal_predict(class_probs, encoding="invalid")  # type: ignore

    # Test return_pmf
    pred4, pmf = ordinal_predict(
        class_probs, encoding="class_probs", strategy="argmax", return_pmf=True
    )
    assert isinstance(pred4, torch.Tensor)
    assert isinstance(pmf, torch.Tensor)


def test_ordinal_predict_cumulative_logits_argmax() -> None:
    cum_logits = torch.tensor([[4.0, 2.0], [-1.0, -2.0]])
    pred = ordinal_predict(cum_logits, encoding="cumulative_logits", strategy="argmax")
    assert isinstance(pred, torch.Tensor)
