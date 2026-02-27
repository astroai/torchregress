from __future__ import annotations

import torch

from torchregress.utils import (
    cumulative_logits_to_pmf,
    cumulative_probs_to_pmf,
    labels_to_levels,
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
    assert pred_class.tolist() == [0, 2]

    cum_logits = torch.tensor([[4.0, 2.0], [-1.0, -2.0]])
    pred_cum = ordinal_predict(cum_logits, encoding="cumulative_logits", strategy="threshold")
    assert pred_cum.tolist() == [2, 0]

    pmf = cumulative_logits_to_pmf(cum_logits)
    assert torch.allclose(pmf.sum(dim=-1), torch.ones(2), atol=1e-6)
