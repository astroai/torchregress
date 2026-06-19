"""Tests for WarmupMCTrainer — MSE warmup → FunctionalEIV_MC with shared optimizer."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from torchregress.algorithms import WarmupMCTrainer
from torchregress.losses import FunctionalEIVLoss

# ── helpers ────────────────────────────────────────────────────────────


def _make_linear_data(
    n_samples: int = 200,
    n_features: int = 4,
    sigma_x: float = 0.5,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic linear EIV data for fast unit tests."""
    rng = torch.Generator().manual_seed(seed)
    X_true = torch.randn(n_samples, n_features, generator=rng)
    beta = torch.randn(n_features, 1, generator=rng)
    X_obs = X_true + sigma_x * torch.randn(n_samples, n_features, generator=rng)
    y = X_true @ beta + 0.1 * torch.randn(n_samples, 1, generator=rng)
    return X_obs, y


def _make_mlp(in_dim: int = 4, hidden: int = 16) -> nn.Module:
    """Return a freshly initialized MLP model."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def _get_initial_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot all parameter values (detached clones)."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


# ── basic API tests ────────────────────────────────────────────────────


def test_warmup_mc_fit_predict_shape() -> None:
    """fit() returns self, predict() returns correct shape."""
    torch.manual_seed(0)
    X, y = _make_linear_data(n_samples=100, sigma_x=0.5)

    trainer = WarmupMCTrainer(
        model_factory=lambda: _make_mlp(4),
        sigma_x=0.5,
        total_epochs=10,
        warmup_epochs=3,
        lr=1e-2,
        batch_size=32,
    )
    result = trainer.fit(X, y)
    assert result is trainer  # chaining

    preds = trainer.predict(X)
    assert preds.shape == (100, 1)
    assert torch.isfinite(preds).all()


def test_warmup_mc_model_parameters_change_during_training() -> None:
    """Model parameters are different after fit() vs before."""
    torch.manual_seed(1)
    X, y = _make_linear_data(n_samples=100, sigma_x=0.5)

    model = _make_mlp(4)
    params_before = _get_initial_params(model)

    trainer = WarmupMCTrainer(
        model_factory=lambda: model,
        sigma_x=0.5,
        total_epochs=20,
        warmup_epochs=5,
        lr=1e-2,
        batch_size=32,
    )
    trainer.fit(X, y)

    for name, p_after in model.named_parameters():
        p_before = params_before[name]
        assert not torch.equal(p_before, p_after), f"Parameter {name} did not change after training"


def test_warmup_mc_zero_warmup_still_trains() -> None:
    """warmup_epochs=0 skips MSE phase but still runs MC phase."""
    torch.manual_seed(2)
    X, y = _make_linear_data(n_samples=100, sigma_x=0.5)

    trainer = WarmupMCTrainer(
        model_factory=lambda: _make_mlp(4),
        sigma_x=0.5,
        total_epochs=10,
        warmup_epochs=0,
        lr=1e-2,
        batch_size=32,
    )
    trainer.fit(X, y)
    preds = trainer.predict(X)
    assert torch.isfinite(preds).all()


def test_warmup_mc_predict_before_fit_raises() -> None:
    """predict() before fit() raises RuntimeError."""
    trainer = WarmupMCTrainer(
        model_factory=lambda: _make_mlp(4),
        sigma_x=0.5,
        total_epochs=10,
        warmup_epochs=3,
    )
    X = torch.randn(10, 4)
    with pytest.raises(RuntimeError, match="must be fit before calling predict"):
        trainer.predict(X)


# ── validation tests ───────────────────────────────────────────────────


def test_warmup_mc_warmup_gte_total_raises() -> None:
    """warmup_epochs >= total_epochs raises ValueError."""
    with pytest.raises(ValueError, match="must be less than total_epochs"):
        WarmupMCTrainer(
            model_factory=lambda: _make_mlp(4),
            sigma_x=0.5,
            total_epochs=10,
            warmup_epochs=10,
        )

    with pytest.raises(ValueError, match="must be less than total_epochs"):
        WarmupMCTrainer(
            model_factory=lambda: _make_mlp(4),
            sigma_x=0.5,
            total_epochs=10,
            warmup_epochs=15,
        )


def test_warmup_mc_invalid_X_shape_raises() -> None:
    """Non-2D X raises ValueError in fit()."""
    trainer = WarmupMCTrainer(
        model_factory=lambda: _make_mlp(4),
        sigma_x=0.5,
        total_epochs=5,
        warmup_epochs=2,
    )
    with pytest.raises(ValueError, match="X must be a 2D tensor"):
        trainer.fit(torch.randn(64), torch.randn(64, 1))


# ── shared optimizer momentum carryover tests ─────────────────────────


def test_warmup_mc_shared_optimizer_single_adam() -> None:
    """fit() creates exactly one Adam optimizer shared across both phases.

    Momentum from MSE warmup carries into the MC phase because the same
    optimizer object is reused.  We verify by patching the Adam import
    that WarmupMCTrainer uses (via `from torch.optim import Adam`).
    """
    torch.manual_seed(3)
    X, y = _make_linear_data(n_samples=100, sigma_x=0.5)

    # Patch the module-level Adam binding that WarmupMCTrainer actually uses.
    # Patching torch.optim.Adam doesn't work because WarmupMCTrainer imports
    # Adam directly via `from torch.optim import Adam`.
    with patch(
        "torchregress.algorithms.warmup_mc.Adam",
        wraps=torch.optim.Adam,
    ) as mock_adam:
        trainer = WarmupMCTrainer(
            model_factory=lambda: _make_mlp(4),
            sigma_x=0.5,
            total_epochs=10,
            warmup_epochs=3,
            lr=1e-2,
            batch_size=32,
        )
        trainer.fit(X, y)

    assert mock_adam.call_count == 1, (
        f"Expected 1 Adam creation (shared optimizer), got {mock_adam.call_count}. "
        "If > 1, momentum does not carry over from MSE to MC phase."
    )


def test_warmup_mc_momentum_carries_mse_to_mc_different_from_separate() -> None:
    """Shared-optimizer warmup produces different results vs separate optimizers.

    Train two models on same data/setup:
    - Model A: WarmupMCTrainer (warmup=5, total=15 — shared optimizer)
    - Model B: MSE for 5 epochs + new optimizer + MC for 10 epochs

    The shared optimizer's momentum carryover should make model A's
    parameters differ from model B's.
    """
    torch.manual_seed(42)
    X, y = _make_linear_data(n_samples=150, sigma_x=0.5)
    n_features = X.shape[1]

    # ── Model A: shared optimizer warmup ──
    torch.manual_seed(42)
    model_a = _make_mlp(n_features)
    trainer = WarmupMCTrainer(
        model_factory=lambda: model_a,
        sigma_x=0.5,
        total_epochs=15,
        warmup_epochs=5,
        lr=1e-2,
        batch_size=32,
        n_mc_samples=20,
    )
    trainer.fit(X, y)
    params_a = _get_initial_params(model_a)

    # ── Model B: separate optimizers ──
    # Re-seed before each step to match model A's RNG state (model A's
    # DataLoader is created inside fit() after model_factory() which only
    # returns the pre-built model — advancing RNG 0 times).
    torch.manual_seed(42)
    model_b = _make_mlp(n_features)

    # Phase 1: MSE with its own Adam
    opt_mse = torch.optim.Adam(model_b.parameters(), lr=1e-2)
    torch.manual_seed(42)  # match RNG state before DataLoader shuffle init
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model_b.train()
    for _ in range(5):  # warmup epochs
        for Xb, yb in loader:
            opt_mse.zero_grad(set_to_none=True)
            loss = ((model_b(Xb) - yb) ** 2).mean()
            loss.backward()
            opt_mse.step()

    # Phase 2: MC with a fresh Adam (no momentum carryover)
    opt_mc = torch.optim.Adam(model_b.parameters(), lr=1e-2)
    mc_loss = FunctionalEIVLoss(
        model=model_b,
        sigma_x=0.5,
        mode="mc",
        n_samples=20,
    )
    for _ in range(10):  # MC epochs
        for Xb, yb in loader:
            opt_mc.zero_grad(set_to_none=True)
            loss = mc_loss(Xb, yb)
            loss.backward()
            opt_mc.step()

    params_b = _get_initial_params(model_b)

    # ── Assert: parameters differ (momentum carryover matters) ──
    any_differ = False
    for name in params_a:
        if not torch.allclose(params_a[name], params_b[name], atol=1e-5):
            any_differ = True
            break
    assert any_differ, (
        "Shared-optimizer warmup produced identical parameters to "
        "separate-optimizer training — momentum carryover had no effect"
    )


def test_warmup_mc_converged_fit_sets_model_attribute() -> None:
    """After fit(), trainer.model is a trained nn.Module."""
    torch.manual_seed(4)
    X, y = _make_linear_data(n_samples=80, sigma_x=0.5)

    trainer = WarmupMCTrainer(
        model_factory=lambda: _make_mlp(4),
        sigma_x=0.5,
        total_epochs=10,
        warmup_epochs=3,
        lr=1e-2,
        batch_size=32,
    )
    trainer.fit(X, y)

    # Verify fit() completes and the model attribute is set
    assert trainer.model is not None
    assert isinstance(trainer.model, nn.Module)
