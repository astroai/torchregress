from __future__ import annotations

import torch

from torchregress.utils.semisupervised import generate_pseudo_labels, update_ema_teacher_


def test_generate_pseudo_labels_from_log_variance_thresholds_acceptance() -> None:
    mean = torch.tensor([[0.0], [1.0], [2.0]])
    log_var = torch.log(torch.tensor([[0.01], [0.25], [1.0]]))
    pseudo, confidence, accepted = generate_pseudo_labels(
        mean,
        log_variance=log_var,
        max_std=1.0,
        confidence_threshold=0.5,
    )
    assert torch.allclose(pseudo, mean)
    assert accepted.shape == mean.shape
    assert bool(accepted[0].item()) is True
    assert bool(accepted[-1].item()) is False
    assert torch.all((confidence >= 0.0) & (confidence <= 1.0))


def test_update_ema_teacher_moves_toward_student() -> None:
    teacher = torch.nn.Linear(2, 1)
    student = torch.nn.Linear(2, 1)
    with torch.no_grad():
        teacher.weight.fill_(0.0)
        teacher.bias.fill_(0.0)
        student.weight.fill_(1.0)
        student.bias.fill_(2.0)
    update_ema_teacher_(teacher, student, momentum=0.5)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.5))
    assert torch.allclose(teacher.bias, torch.full_like(teacher.bias, 1.0))
