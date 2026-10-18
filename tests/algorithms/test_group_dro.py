"""Tests for continuous-target Group DRO (binned minimax risk)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from torchregress.algorithms.group_dro import ContinuousGroupDRO, GroupDROTrainer

# ── helpers ────────────────────────────────────────────────────────────


def _set_weights(module: ContinuousGroupDRO, weights: list[float]) -> None:
    """Manually set the adversarial group weights (test hook)."""
    with torch.no_grad():
        module.log_q.copy_(torch.log(torch.tensor(weights, dtype=torch.float64)))


def _softmax_update(log_q: torch.Tensor, eta: float, losses: list[float]) -> torch.Tensor:
    """Reference exponentiated-gradient step: softmax(log_q + eta * L)."""
    delta = torch.tensor(losses, dtype=torch.float64)
    return torch.softmax(log_q + eta * delta, dim=0)


class _FourthPowerLoss(nn.Module):
    """Custom element-wise base loss used to check the module path."""

    reduction = "none"

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (y_pred - target) ** 4


_STD_EDGES = [0.0, 1.0, 2.0, 3.0]


# ── closed-form updates ────────────────────────────────────────────────


def test_update_matches_closed_form_softmax() -> None:
    """One training forward reproduces softmax(log_q + eta * L) exactly."""
    torch.manual_seed(0)
    dro = ContinuousGroupDRO(_STD_EDGES, eta=0.1)
    _set_weights(dro, [0.2, 0.5, 0.3])
    log_q = dro.log_q.detach().clone()

    y_pred = torch.zeros(3, 1)
    target = torch.tensor([[0.5], [1.5], [2.5]])  # one sample per group
    losses = [0.25, 2.25, 6.25]  # (pred - target)^2 per group

    loss = dro(y_pred, target)

    expected_q = _softmax_update(log_q, 0.1, losses)
    assert torch.allclose(dro.group_weights(), expected_q, atol=1e-9)
    assert torch.isclose(dro.group_weights().sum(), torch.tensor(1.0, dtype=torch.float64))
    expected_loss = torch.dot(expected_q, torch.tensor(losses, dtype=torch.float64))
    assert torch.isclose(loss.double(), expected_loss, atol=1e-5)
    assert int(dro.n_updates) == 1


def test_absent_groups_keep_previous_log_weight() -> None:
    """Groups without batch samples only move through the final normalisation."""
    torch.manual_seed(0)
    dro = ContinuousGroupDRO(_STD_EDGES, eta=0.5)
    _set_weights(dro, [0.2, 0.5, 0.3])
    log_q = dro.log_q.detach().clone()

    # Only groups 0 and 2 are present; group 1 must receive a zero log-update.
    y_pred = torch.zeros(2, 1)
    target = torch.tensor([[0.5], [2.5]])
    dro(y_pred, target)

    expected_q = _softmax_update(log_q, 0.5, [0.25, 0.0, 6.25])
    assert torch.allclose(dro.group_weights(), expected_q, atol=1e-9)

    dro.eval()
    losses = dro.group_losses(y_pred, target)
    assert torch.isclose(losses[0], torch.tensor(0.25, dtype=losses.dtype))
    assert torch.isclose(losses[2], torch.tensor(6.25, dtype=losses.dtype))
    assert torch.isnan(losses[1])


def test_update_is_log_domain_and_stable_for_large_losses() -> None:
    """Huge finite losses saturate q without producing NaN/Inf."""
    dro = ContinuousGroupDRO([0.0, 1.0, 2.0], eta=1.0)
    y_pred = torch.zeros(2, 1)
    target = torch.tensor([[1e6], [1e6]])

    for _ in range(5):
        loss = dro(y_pred, target)
        assert torch.isfinite(loss)
    weights = dro.group_weights()
    assert torch.isfinite(weights).all()
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=weights.dtype))


# ── train/eval behaviour and schedule ─────────────────────────────────


def test_eval_mode_freezes_weights() -> None:
    """Eval mode does not update q and repeated calls are idempotent."""
    torch.manual_seed(0)
    dro = ContinuousGroupDRO(_STD_EDGES, eta=0.5)
    y_pred = torch.zeros(3, 1)
    target = torch.tensor([[0.5], [1.5], [2.5]])

    dro.train()
    dro(y_pred, target)
    dro.eval()
    weights_before = dro.group_weights().clone()
    counter_before = int(dro.n_updates)

    first = dro(y_pred, target)
    second = dro(y_pred, target)

    assert torch.equal(first, second)
    assert torch.equal(dro.group_weights(), weights_before)
    assert int(dro.n_updates) == counter_before

    dro.train()
    dro(y_pred, target)
    assert not torch.equal(dro.group_weights(), weights_before)
    assert int(dro.n_updates) == counter_before + 1


def test_update_every_is_honoured() -> None:
    """With update_every=2 the weights refresh on the 1st, 3rd, ... call."""
    dro = ContinuousGroupDRO([0.0, 1.0, 2.0], eta=0.1, update_every=2)
    log_q = dro.log_q.detach().clone()
    y_pred = torch.zeros(2, 1)
    target = torch.tensor([[0.5], [1.5]])
    losses = [0.25, 2.25]

    dro(y_pred, target)
    after_first = _softmax_update(log_q, 0.1, losses)
    assert torch.allclose(dro.group_weights(), after_first, atol=1e-9)

    dro(y_pred, target)  # skipped: n_updates == 1
    assert torch.allclose(dro.group_weights(), after_first, atol=1e-9)

    dro(y_pred, target)  # applies a second update
    after_third = torch.softmax(log_q + 2 * 0.1 * torch.tensor(losses, dtype=torch.float64), dim=0)
    assert torch.allclose(dro.group_weights(), after_third, atol=1e-9)


def test_reset_restores_uniform_weights() -> None:
    dro = ContinuousGroupDRO(_STD_EDGES, eta=1.0)
    dro(torch.zeros(3, 1), torch.tensor([[0.5], [1.5], [2.5]]))
    assert not torch.allclose(dro.group_weights(), torch.full((3,), 1 / 3, dtype=torch.float64))

    dro.reset()
    assert torch.allclose(dro.group_weights(), torch.full((3,), 1 / 3, dtype=torch.float64))
    assert int(dro.n_updates) == 0


# ── binning ───────────────────────────────────────────────────────────


def test_binning_boundaries_and_clamping() -> None:
    """Bins are half-open [edge_g, edge_{g+1}); out-of-range values clamp."""
    dro = ContinuousGroupDRO([-1.0, 0.0, 1.0])
    target = torch.tensor([-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0])
    expected = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    assert torch.equal(dro.group_indices(target), expected)
    assert torch.equal(dro.group_counts(target), torch.tensor([3, 4]))

    counts = dro.group_counts(torch.linspace(-1.0, 1.0, 10))
    assert int(counts.sum()) == 10


def test_integer_group_edges_are_inferred_from_data() -> None:
    dro = ContinuousGroupDRO(4)
    assert dro.edges.numel() == 0
    assert dro.n_groups == 4

    target = torch.linspace(-2.0, 2.0, 9).unsqueeze(1)
    dro(torch.zeros(9, 1), target)
    assert dro.edges.numel() == 5
    assert torch.allclose(dro.edges, torch.linspace(-2.0, 2.0, 5, dtype=torch.float64))
    counts = dro.group_counts(target)
    assert int(counts.sum()) == 9
    assert dro.group_weights().shape == (4,)


def test_fit_group_edges_explicit_and_noop_for_fixed_edges() -> None:
    inferred = ContinuousGroupDRO(3)
    inferred.fit_group_edges(torch.tensor([0.0, 3.0]))
    assert torch.allclose(inferred.edges, torch.linspace(0.0, 3.0, 4, dtype=torch.float64))

    fixed = ContinuousGroupDRO([0.0, 1.0, 2.0])
    fixed.fit_group_edges(torch.tensor([-100.0, 100.0]))  # no-op: edges stay as given
    assert torch.allclose(fixed.edges, torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64))

    # Degenerate (constant) targets still produce valid, increasing edges.
    degenerate = ContinuousGroupDRO(2)
    degenerate.fit_group_edges(torch.full((5,), 3.0))
    assert bool((degenerate.edges[1:] > degenerate.edges[:-1]).all())


def test_2d_targets_bin_by_mean() -> None:
    dro = ContinuousGroupDRO([-1.0, 0.0, 1.0])
    target = torch.tensor([[-0.5, -0.5], [0.5, 0.5], [1.0, 2.0]])
    assert torch.equal(dro.group_indices(target), torch.tensor([0, 1, 1]))


# ── base losses, mask and weights ─────────────────────────────────────


def test_base_loss_mse_mae_and_custom_module() -> None:
    y_pred = torch.zeros(2, 1)
    target = torch.tensor([[0.5], [1.5]])

    mse = ContinuousGroupDRO([0.0, 1.0, 2.0], base_loss="mse").eval()
    assert torch.allclose(mse.group_losses(y_pred, target), torch.tensor([0.25, 2.25]))

    mae = ContinuousGroupDRO([0.0, 1.0, 2.0], base_loss="mae").eval()
    assert torch.allclose(mae.group_losses(y_pred, target), torch.tensor([0.5, 1.5]))

    custom = ContinuousGroupDRO([0.0, 1.0, 2.0], base_loss=_FourthPowerLoss()).eval()
    assert torch.allclose(custom.group_losses(y_pred, target), torch.tensor([0.5**4, 1.5**4]))

    with pytest.raises(ValueError, match="reduction='none'"):
        ContinuousGroupDRO([0.0, 1.0, 2.0], base_loss=nn.MSELoss())


def test_mask_and_zero_weights_exclude_samples() -> None:
    dro = ContinuousGroupDRO([0.0, 1.0, 2.0]).eval()
    y_pred = torch.tensor([[100.0], [1.0], [1.0], [1.0]])
    target = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    mask = torch.tensor([False, True, True, True])
    weights = torch.tensor([1.0, 1.0, 2.0, 2.0])

    losses = dro.group_losses(y_pred, target, mask=mask, weights=weights)
    # Group 0 only sees the masked-in sample with loss 1.0; group 1 has zero error.
    assert torch.allclose(losses, torch.tensor([1.0, 0.0]), atol=1e-6)

    # Excluding via mask or weight 0 gives the same group means.
    via_mask = dro.group_losses(y_pred, target, mask=mask)
    via_weights = dro.group_losses(y_pred, target, weights=torch.tensor([0.0, 1.0, 1.0, 1.0]))
    assert torch.allclose(via_mask, via_weights, atol=1e-6)

    # With uniform q the reduced loss is the mean of the group means.
    loss = dro(y_pred, target, mask=mask, weights=weights)
    assert torch.isclose(loss, torch.tensor(0.5), atol=1e-6)


def test_group_losses_is_stateless() -> None:
    dro = ContinuousGroupDRO(_STD_EDGES, eta=1.0)
    y_pred = torch.zeros(3, 1)
    target = torch.tensor([[0.5], [1.5], [2.5]])
    weights_before = dro.group_weights().clone()
    counter_before = int(dro.n_updates)

    losses = dro.group_losses(y_pred, target)
    assert losses.shape == (3,)
    assert torch.equal(dro.group_weights(), weights_before)
    assert int(dro.n_updates) == counter_before


# ── floor, single group, validation ──────────────────────────────────


def test_min_group_weight_floor_is_respected() -> None:
    dro = ContinuousGroupDRO([0.0, 1.0, 2.0, 3.0], eta=1.0, min_group_weight=0.1)
    # Group 0 has a huge loss and would otherwise absorb the whole simplex.
    for _ in range(20):
        dro(torch.zeros(3, 1), torch.tensor([[1e3], [1.5], [2.5]]))
    weights = dro.group_weights()
    assert bool((weights >= 0.1 - 1e-9).all())
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=weights.dtype))

    with pytest.raises(ValueError, match="min_group_weight"):
        ContinuousGroupDRO([0.0, 1.0, 2.0], min_group_weight=0.6)


def test_single_group_reduces_to_plain_loss() -> None:
    torch.manual_seed(0)
    y_pred = torch.randn(11, 1)
    target = torch.randn(11, 1)
    weights = torch.rand(11)
    manual = ((y_pred - target) ** 2).mean()

    dro = ContinuousGroupDRO([-10.0, 10.0])
    assert dro.group_weights().shape == (1,)
    assert torch.isclose(dro.group_weights()[0], torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(dro(y_pred, target), manual)

    weight_col = weights.reshape(-1, 1)  # keep (N, 1), avoid (N, N) broadcast
    weighted = ((y_pred - target) ** 2 * weight_col).sum() / weight_col.sum()
    assert torch.isclose(dro(y_pred, target, weights=weights), weighted)

    loss = dro(y_pred, target)
    assert torch.isclose(loss, manual)  # repeated calls never change q
    assert torch.isclose(dro.group_weights()[0], torch.tensor(1.0, dtype=torch.float64))


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ContinuousGroupDRO([0.0, 1.0, 0.5])
    with pytest.raises(ValueError, match="at least 2 edges"):
        ContinuousGroupDRO([0.0])
    with pytest.raises(ValueError, match="integer group_edges"):
        ContinuousGroupDRO(0)
    with pytest.raises(TypeError, match="group_edges"):
        ContinuousGroupDRO(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="eta"):
        ContinuousGroupDRO([0.0, 1.0], eta=0.0)
    with pytest.raises(ValueError, match="eta"):
        ContinuousGroupDRO([0.0, 1.0], eta=float("nan"))
    with pytest.raises(ValueError, match="update_every"):
        ContinuousGroupDRO([0.0, 1.0], update_every=0)
    with pytest.raises(ValueError, match="min_group_weight"):
        ContinuousGroupDRO([0.0, 1.0], min_group_weight=-0.1)
    with pytest.raises(ValueError, match="reduction"):
        ContinuousGroupDRO([0.0, 1.0], reduction="median")
    with pytest.raises(ValueError, match="base_loss"):
        ContinuousGroupDRO([0.0, 1.0], base_loss="huber")
    with pytest.raises(TypeError, match="base_loss"):
        ContinuousGroupDRO([0.0, 1.0], base_loss=object())  # type: ignore[arg-type]

    dro = ContinuousGroupDRO([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="batch dimension"):
        dro(torch.zeros(3, 1), torch.zeros(4, 1))
    with pytest.raises(ValueError, match="non-empty batch"):
        dro(torch.zeros(0, 1), torch.zeros(0, 1))
    with pytest.raises(ValueError, match="non-negative"):
        dro(torch.zeros(2, 1), torch.tensor([[0.5], [1.5]]), weights=torch.tensor([1.0, -1.0]))


# ── trainer ───────────────────────────────────────────────────────────


def _tiny_data(n_samples: int = 60, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, 2, generator=generator)
    y = (X[:, :1] + 0.5 * torch.randn(n_samples, 1, generator=generator)).clamp(-3.0, 3.0)
    return X, y


def _make_mlp(in_dim: int = 2, hidden: int = 8) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def test_trainer_fit_predict_and_attributes() -> None:
    torch.manual_seed(0)
    X, y = _tiny_data()
    trainer = GroupDROTrainer(
        model_factory=_make_mlp,
        group_edges=[-3.0, 0.0, 3.0],
        epochs=5,
        lr=1e-2,
        batch_size=16,
        seed=7,
    )
    assert trainer.fit(X, y) is trainer
    assert trainer.model is not None
    assert trainer.predict(X).shape == (X.shape[0], 1)
    assert torch.isfinite(trainer.predict(X)).all()

    assert trainer.group_weights_ is not None
    assert trainer.group_weights_.shape == (2,)
    assert torch.isclose(trainer.group_weights_.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert trainer.group_losses_ is not None
    assert trainer.group_losses_.shape == (2,)
    assert torch.isfinite(trainer.group_losses_).all()
    assert len(trainer.history_) == 5
    assert {"loss", "q"} <= set(trainer.history_[-1])
    assert all(math.isfinite(entry["loss"]) for entry in trainer.history_)
    assert len(trainer.history_[-1]["q"]) == 2


def test_trainer_is_deterministic_with_seed() -> None:
    torch.manual_seed(0)
    X, y = _tiny_data(seed=3)
    kwargs = dict(
        model_factory=_make_mlp,
        group_edges=2,  # inferred equal-width bins
        epochs=4,
        lr=1e-2,
        batch_size=16,
        seed=11,
    )
    first = GroupDROTrainer(**kwargs).fit(X, y)
    second = GroupDROTrainer(**kwargs).fit(X, y)
    assert first.history_ == second.history_
    assert torch.equal(first.group_weights_, second.group_weights_)
    assert torch.equal(first.predict(X), second.predict(X))


def test_trainer_validation_errors() -> None:
    trainer = GroupDROTrainer(model_factory=_make_mlp, group_edges=[-1.0, 0.0, 1.0], epochs=1)
    with pytest.raises(RuntimeError, match="fit"):
        trainer.predict(torch.randn(3, 2))

    X, y = _tiny_data()
    with pytest.raises(ValueError, match="X must be a 2-D"):
        trainer.fit(X[:, 0], y)
    with pytest.raises(ValueError, match="same number of samples"):
        trainer.fit(X, y[:-1])
    with pytest.raises(ValueError, match="epochs"):
        GroupDROTrainer(model_factory=_make_mlp, group_edges=[-1.0, 1.0], epochs=0)


# ── end-to-end controlled experiment ─────────────────────────────────


def _two_regime_data(
    n_major: int = 240, n_minor: int = 36, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic two-regime regression problem with a rare, shifted regime.

    Regime A (majority): ``y = +1 + 0.3 x0 + 0.05 noise``.
    Regime B (minority): ``y = -1 + 0.3 x0 + 0.20 noise``.

    ``x1`` is pure noise, so the only way to reduce the worst-regime risk is to
    fit the *bias* shared by the two target levels.  Plain MSE weights regimes
    by prevalence and lands near the majority level ``+1``; Group DRO equalises
    the two regimes and lands near ``0``.
    """
    generator = torch.Generator().manual_seed(seed)
    n = n_major + n_minor
    X = torch.randn(n, 2, generator=generator)
    y = torch.empty(n, 1)
    y[:n_major] = 1.0 + 0.3 * X[:n_major, 0:1] + 0.05 * torch.randn(n_major, 1, generator=generator)
    y[n_major:] = (
        -1.0 + 0.3 * X[n_major:, 0:1] + 0.20 * torch.randn(n_minor, 1, generator=generator)
    )
    return X, y


def _plain_mse_fit(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    lr: float = 1e-2,
    batch_size: int = 64,
) -> nn.Module:
    """Plain-MSE counterpart of ``GroupDROTrainer`` with identical shuffling."""
    torch.manual_seed(seed)
    model = _make_mlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(X.shape[0], generator=generator)
        for start in range(0, X.shape[0], batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(X[batch]) - y[batch]) ** 2).mean()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _worst_regime_mse(model: nn.Module, X: torch.Tensor, y: torch.Tensor, n_major: int) -> float:
    with torch.no_grad():
        per_sample = (model(X) - y).squeeze(1) ** 2
    return float(max(per_sample[:n_major].mean(), per_sample[n_major:].mean()))


def test_group_dro_improves_worst_group_holdout_mse() -> None:
    """Controlled comparison: same MLP / seed / data order, only the loss differs.

    The numbers are deterministic for seed 0 and the gap between the two
    worst-regime MSEs is large, so the tolerances below are deliberately
    generous and cannot flake on platform-level floating-point differences.
    """
    n_major, n_minor = 240, 36
    epochs, seed = 40, 0
    X_train, y_train = _two_regime_data(n_major, n_minor, seed=seed)
    X_holdout, y_holdout = _two_regime_data(n_major, n_minor, seed=99)

    torch.manual_seed(seed)
    plain_model = _plain_mse_fit(X_train, y_train, seed=seed, epochs=epochs)
    plain_worst = _worst_regime_mse(plain_model, X_holdout, y_holdout, n_major)

    torch.manual_seed(seed)  # identical model initialisation for both methods
    trainer = GroupDROTrainer(
        model_factory=_make_mlp,
        group_edges=[-3.0, 0.0, 3.0],  # separates the two target regimes
        eta=0.03,
        epochs=epochs,
        lr=1e-2,
        batch_size=64,
        seed=seed,
    ).fit(X_train, y_train)
    dro_worst = _worst_regime_mse(trainer.model, X_holdout, y_holdout, n_major)

    # Sanity: the imbalance really is severe for plain MSE (worst regime ~3).
    assert plain_worst > 1.5
    # Group DRO must not increase the worst-group holdout risk ...
    assert dro_worst <= plain_worst + 0.25
    # ... and in this controlled setup it lowers it substantially.
    assert dro_worst < plain_worst
