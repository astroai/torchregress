"""Tests for SnapshotEnsemble (F3) and FullNetworkLaplace (F4)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.ensemble import FullNetworkLaplace, SnapshotEnsemble


def _make_data(n: int = 64, seed: int = 0) -> tuple[TensorDataset, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(n, 3)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5]]) + 0.05 * torch.randn(n, 1)
    return TensorDataset(x, y), x[:8]


def _tiny_net() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 1))


# ---------------------------------------------------------------------------
# F3: SnapshotEnsemble
# ---------------------------------------------------------------------------


def test_snapshot_ensemble_collects_one_state_dict_per_cycle():
    data, x = _make_data()
    model = _tiny_net()
    ens = SnapshotEnsemble(model, n_snapshots=3, cycle_epochs=2, lr_max=1e-2, lr_min=1e-4)
    ens.fit(DataLoader(data, batch_size=16), epochs=6)
    assert len(ens.snapshots) == 3
    # Snapshots are full state_dict copies...
    ref_keys = set(model.state_dict().keys())
    for snap in ens.snapshots:
        assert set(snap.keys()) == ref_keys
    # ...and training progressed between cycle minima.
    w0 = ens.snapshots[0]["2.weight"]
    w2 = ens.snapshots[2]["2.weight"]
    assert not torch.allclose(w0, w2)


def test_snapshot_ensemble_cosine_lr_hits_endpoints():
    ens = SnapshotEnsemble(nn.Linear(1, 1), n_snapshots=2, cycle_epochs=4, lr_max=1.0, lr_min=0.0)
    assert ens._cosine_lr(0) == pytest.approx(1.0)
    assert ens._cosine_lr(2) == pytest.approx(0.5)
    assert ens._cosine_lr(4) == pytest.approx(0.0)


def test_snapshot_ensemble_predict_trio_mirrors_mc_dropout_api():
    data, x = _make_data()
    ens = SnapshotEnsemble(_tiny_net(), n_snapshots=2, cycle_epochs=2, lr_max=1e-2, lr_min=1e-4)
    with pytest.raises(RuntimeError, match="fit"):
        ens.mc_forward(x)
    ens.fit(DataLoader(data, batch_size=16), epochs=4)

    stacked = ens.mc_forward(x)
    assert stacked.shape == (2, x.shape[0], 1)

    mean, std = ens.predict_with_uncertainty(x)
    torch.testing.assert_close(mean, stacked.mean(dim=0))
    torch.testing.assert_close(std, stacked.std(dim=0))

    lower, upper = ens.predict_interval(x, confidence=0.95)
    assert torch.all(lower <= upper)


# ---------------------------------------------------------------------------
# F4: FullNetworkLaplace
# ---------------------------------------------------------------------------


def test_full_network_laplace_fisher_type_validated():
    with pytest.raises(ValueError, match="empirical"):
        FullNetworkLaplace(_tiny_net(), fisher_type="kroncker")


def test_full_network_laplace_requires_per_sample_loss():
    data, _ = _make_data()
    lap = FullNetworkLaplace(_tiny_net())
    with pytest.raises(ValueError, match="per-sample"):
        lap.fit(DataLoader(data, batch_size=16), nn.MSELoss())


def test_full_network_laplace_predict_with_uncertainty():
    data, x = _make_data()
    lap = FullNetworkLaplace(_tiny_net(), damping=1e-3, n_samples=50)
    lap.fit(DataLoader(data, batch_size=16), nn.MSELoss(reduction="none"))
    assert lap.is_fitted
    # Every parameter received a diagonal Fisher entry.
    all_params = {n for n, p in lap.model.named_parameters() if p.requires_grad}
    assert set(lap.fisher_diag.keys()) == all_params

    mean, std = lap.predict_with_uncertainty(x, n_samples=40)
    assert mean.shape == std.shape == (x.shape[0], 1)
    assert torch.all(std > 0)
    lower, upper = lap.predict_interval(x, confidence=0.9, n_samples=40)
    assert torch.all(lower <= upper)


def test_full_network_laplace_damping_shrinks_posterior_std():
    data, _ = _make_data()
    small = FullNetworkLaplace(_tiny_net(), damping=1e-6)
    small.fit(DataLoader(data, batch_size=32), nn.MSELoss(reduction="none"))
    large = FullNetworkLaplace(_tiny_net(), damping=1e6)
    large.fit(DataLoader(data, batch_size=32), nn.MSELoss(reduction="none"))
    max_small = max(v.max() for v in small.posterior_std.values())
    max_large = max(v.max() for v in large.posterior_std.values())
    assert max_small > max_large


def test_full_network_laplace_unfitted_raises():
    lap = FullNetworkLaplace(_tiny_net())
    with pytest.raises(RuntimeError, match="not fitted"):
        lap.predict_with_uncertainty(torch.randn(4, 3))
