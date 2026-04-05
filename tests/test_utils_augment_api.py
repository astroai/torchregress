from __future__ import annotations

import torch
import torch.nn as nn

from torchregress.utils.augment import Adversarial, GaussianNoise


def test_augmentation_augment_and_apply_compatibility() -> None:
    aug = GaussianNoise(std=0.0, probability=1.0)
    x = torch.randn(4, 3)
    y = torch.randn(4, 1)

    x1, y1 = aug.augment(x, y)
    x2, y2 = aug.apply(x, y)
    x3, y3 = aug(x, y)

    assert x1 is not None and x2 is not None and x3 is not None
    assert y1 is not None and y2 is not None and y3 is not None

    assert torch.equal(x1, x2)
    assert torch.equal(x2, x3)
    assert torch.equal(y1, y2)
    assert torch.equal(y2, y3)


def test_augmentation_probability_zero_skips_transform() -> None:
    aug = GaussianNoise(std=10.0, probability=0.0)
    x = torch.randn(2, 2)
    y = torch.randn(2, 1)

    x_out, y_out = aug(x, y)
    assert x_out is not None and y_out is not None
    assert torch.equal(x_out, x)
    assert torch.equal(y_out, y)


def test_adversarial_augmentation_stays_in_epsilon_ball_and_avoids_param_grads() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
    aug = Adversarial(
        model=model,
        loss_fn=nn.MSELoss(),
        epsilon=0.05,
        steps=2,
        alpha=0.025,
        probability=1.0,
        random_start=True,
    )
    x = torch.randn(4, 3)
    y = torch.randn(4, 1)

    x_adv, y_adv = aug(x, y)

    assert y_adv is not None
    assert torch.all((x_adv - x).abs() <= 0.050001)
    assert any(not torch.equal(x_adv[i], x[i]) for i in range(x.shape[0]))
    for param in model.parameters():
        assert param.grad is None
