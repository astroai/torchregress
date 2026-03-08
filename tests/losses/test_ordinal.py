from __future__ import annotations

import torch
import torch.nn.functional as F

from torchregress.losses import CORALLoss, CumulativeLinkLoss, OrdinalCrossEntropyLoss
from torchregress.utils.ordinal import labels_to_levels


def test_ordinal_cross_entropy_matches_torch_cross_entropy() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.2, 1.4, 0.3]], dtype=torch.float32)
    target = torch.tensor([0, 1], dtype=torch.long)

    loss_fn = OrdinalCrossEntropyLoss(reduction="mean")
    actual = loss_fn(logits, target)
    expected = F.cross_entropy(logits, target, reduction="mean")
    assert torch.allclose(actual, expected)


def test_ordinal_cross_entropy_accepts_soft_targets() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.2, 1.4, 0.3]], dtype=torch.float32)
    hard_target = torch.tensor([0, 1], dtype=torch.long)
    soft_target = F.one_hot(hard_target, num_classes=3).to(dtype=torch.float32)

    loss_fn = OrdinalCrossEntropyLoss(reduction="mean")
    actual = loss_fn(logits, soft_target)
    expected = F.cross_entropy(logits, hard_target, reduction="mean")
    assert torch.allclose(actual, expected)


def test_cumulative_link_loss_respects_mask_and_weights() -> None:
    logits = torch.tensor([[3.0, 2.5], [0.1, -0.2], [2.0, 1.5]], dtype=torch.float32)
    target = torch.tensor([2, 1, 2], dtype=torch.long)
    mask = torch.tensor([True, False, True])
    weights = torch.tensor([1.0, 1.0, 2.0])

    loss_fn = CumulativeLinkLoss(reduction="mean")
    loss = loss_fn(logits, target, mask=mask, weights=weights)

    levels = labels_to_levels(target, num_classes=3)
    per_level = F.binary_cross_entropy_with_logits(logits, levels, reduction="none")
    per_sample = per_level.mean(dim=-1)
    expected = (per_sample[mask] * weights[mask]).sum() / weights[mask].sum()
    assert torch.allclose(loss, expected)


def test_cumulative_link_accepts_soft_class_prob_targets() -> None:
    logits = torch.tensor([[3.0, 2.5], [0.1, -0.2], [2.0, 1.5]], dtype=torch.float32)
    target = torch.tensor([2, 1, 2], dtype=torch.long)
    soft_target = F.one_hot(target, num_classes=3).to(dtype=torch.float32)

    loss_fn = CumulativeLinkLoss(reduction="mean")
    actual = loss_fn(logits, soft_target)
    expected = loss_fn(logits, target)
    assert torch.allclose(actual, expected)


def test_coral_loss_forward_and_backward() -> None:
    logits = torch.randn(8, 4, requires_grad=True)
    target = torch.randint(0, 5, (8,))

    loss_fn = CORALLoss()
    loss = loss_fn(logits, target)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


def test_cumulative_link_accepts_spatial_targets() -> None:
    logits = torch.randn(4, 3, 2)
    target = torch.randint(0, 4, (4, 2))

    loss_fn = CumulativeLinkLoss()
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss)


def test_ordinal_losses_validate_shapes() -> None:
    ce = OrdinalCrossEntropyLoss()
    logits = torch.randn(4, 3)
    target_bad = torch.randint(0, 3, (4, 1))

    try:
        ce(logits, target_bad)
    except ValueError as exc:
        assert "target shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid target shape")

    cum = CumulativeLinkLoss()
    soft_target_bad = torch.rand(4, 5)
    try:
        cum(logits, soft_target_bad)
    except ValueError as exc:
        assert "soft target shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid soft ordinal target shape")
