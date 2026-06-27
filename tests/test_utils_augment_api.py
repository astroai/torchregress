from __future__ import annotations

import torch
import torch.nn as nn

from torchregress.utils.augment import Adversarial


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
