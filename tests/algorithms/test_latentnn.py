from __future__ import annotations

import torch
import torch.nn as nn

from torchregress.algorithms import LatentNN


def _linear_factory() -> nn.Module:
    return nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))


def _slope(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    centered_true = y_true - y_true.mean()
    centered_pred = y_pred - y_pred.mean()
    denom = centered_true.pow(2).sum().clamp_min(1.0e-8)
    return float((centered_true * centered_pred).sum() / denom)


def test_latentnn_fit_predict_shape() -> None:
    torch.manual_seed(0)
    x_true = torch.linspace(-2.0, 2.0, 64).unsqueeze(-1)
    y = 2.0 * x_true + 0.1 * torch.randn_like(x_true)
    x_obs = x_true + 0.4 * torch.randn_like(x_true)

    model = LatentNN(
        model_factory=_linear_factory,
        sigma_x=0.4,
        sigma_y=0.1,
        epochs=20,
        lr=1.0e-2,
        batch_size=32,
    )
    model.fit(x_obs, y)
    pred = model.predict(x_obs)

    assert pred.shape == y.shape
    assert model.x_latent_ is not None
    assert torch.isfinite(pred).all()


def test_latentnn_reduces_attenuation_against_naive_model() -> None:
    torch.manual_seed(1)
    x_true = torch.linspace(-3.0, 3.0, 160).unsqueeze(-1)
    y_true = 3.0 * x_true
    y_obs = y_true + 0.05 * torch.randn_like(y_true)
    x_obs = x_true + 0.8 * torch.randn_like(x_true)

    naive = _linear_factory()
    opt = torch.optim.Adam(naive.parameters(), lr=1.0e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = nn.functional.mse_loss(naive(x_obs), y_obs)
        loss.backward()
        opt.step()
    with torch.no_grad():
        naive_pred = naive(x_true)

    latent = LatentNN(
        model_factory=_linear_factory,
        sigma_x=0.8,
        sigma_y=0.05,
        epochs=250,
        lr=5.0e-3,
        latent_lr=1.0e-2,
        batch_size=64,
        weight_decay=1.0e-4,
        latent_penalty_weight=1.0,
    )
    latent.fit(x_obs, y_obs)
    latent_pred = latent.predict(x_true)

    naive_slope = _slope(y_true.view(-1), naive_pred.view(-1))
    latent_slope = _slope(y_true.view(-1), latent_pred.view(-1))

    assert abs(latent_slope - 1.0) < abs(naive_slope - 1.0)
