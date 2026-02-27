from __future__ import annotations

import torch

from torchregress.metrics import (
    mean_absolute_class_error,
    ordinal_accuracy,
    quadratic_weighted_kappa,
)


def test_ordinal_accuracy_from_labels_and_logits() -> None:
    y_true = torch.tensor([0, 1, 2, 1])
    y_pred_labels = torch.tensor([0, 1, 1, 1])
    logits = torch.tensor(
        [
            [4.0, 0.2, -1.0],
            [0.1, 2.0, 0.2],
            [1.0, 1.2, 0.5],
            [0.2, 2.2, 0.1],
        ]
    )

    acc_labels = ordinal_accuracy(y_pred_labels, y_true, encoding="labels")
    acc_logits = ordinal_accuracy(logits, y_true, encoding="class_logits")

    assert torch.isclose(acc_labels, torch.tensor(0.75))
    assert torch.isclose(acc_logits, torch.tensor(0.75))


def test_mean_absolute_class_error() -> None:
    y_true = torch.tensor([0, 1, 2, 3])
    y_pred = torch.tensor([0, 2, 1, 3])
    mae_cls = mean_absolute_class_error(y_pred, y_true, encoding="labels")
    assert torch.isclose(mae_cls, torch.tensor(0.5))


def test_quadratic_weighted_kappa_perfect_and_noisy() -> None:
    y_true = torch.tensor([0, 1, 2, 3, 1, 2])
    kappa_perfect = quadratic_weighted_kappa(y_true, y_true, encoding="labels", num_classes=4)
    assert torch.isclose(kappa_perfect, torch.tensor(1.0))

    y_pred = torch.tensor([3, 2, 1, 0, 0, 1])
    kappa_noisy = quadratic_weighted_kappa(y_pred, y_true, encoding="labels", num_classes=4)
    assert kappa_noisy < 0.5
