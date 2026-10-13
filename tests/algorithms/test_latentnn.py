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


def test_latentnn_full_batch_matches_dense_adam() -> None:
    """Full batch: the lazy latent Adam equals torch.optim.Adam on a dense parameter."""
    torch.manual_seed(3)
    x_obs = torch.randn(40, 2)
    y = x_obs.sum(-1, keepdim=True) + 0.1 * torch.randn(40, 1)

    def factory() -> nn.Module:
        return nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))

    torch.manual_seed(7)
    fitted = LatentNN(model_factory=factory, sigma_x=0.3, sigma_y=0.2, epochs=15, lr=1e-2)
    fitted.fit(x_obs, y)

    torch.manual_seed(7)
    model = factory()
    latent = nn.Parameter(x_obs.clone())
    opt = torch.optim.Adam(
        [{"params": model.parameters(), "lr": 1e-2}, {"params": [latent], "lr": 0.1}]
    )
    for _ in range(15):
        opt.zero_grad()
        model_loss = ((model(latent) - y) / 0.2).pow(2).sum(1).mean()
        latent_loss = ((latent - x_obs) / 0.3).pow(2).sum(1).mean()
        (model_loss + latent_loss).backward()
        opt.step()

    assert fitted.x_latent_ is not None
    torch.testing.assert_close(fitted.x_latent_, latent.detach(), atol=1e-5, rtol=1e-4)


def test_latentnn_minibatch_step_cost_does_not_grow_with_n() -> None:
    """One epoch's cost is linear in N (the dense optimiser was O(N^2 D / B))."""
    import time

    def factory() -> nn.Module:
        return nn.Linear(16, 1)

    def epoch_time(n: int) -> float:
        x = torch.randn(n, 16)
        y = torch.randn(n, 1)
        est = LatentNN(model_factory=factory, sigma_x=0.2, epochs=1, batch_size=64)
        t0 = time.perf_counter()
        est.fit(x, y)
        return time.perf_counter() - t0

    epoch_time(2_000)  # warm-up
    small, large = epoch_time(8_000), epoch_time(64_000)
    # 8x the rows: linear scaling gives ~8x; the old quadratic loop gave ~64x.
    assert large / small < 20, (small, large)


def test_latentnn_chunked_mc_predict_shape() -> None:
    torch.manual_seed(0)
    x = torch.randn(300, 4)
    est = LatentNN(model_factory=lambda: nn.Linear(4, 2), epochs=1, batch_size=50)
    est.fit(x, torch.randn(300, 2))
    pred = est.predict(torch.randn(5000, 4), sigma_x=0.3, n_samples=64)
    assert pred.shape == (5000, 2) and torch.isfinite(pred).all()
