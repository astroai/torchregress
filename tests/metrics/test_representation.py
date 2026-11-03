"""Tests for representation-geometry and readout diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from torchregress.metrics.representation import (
    frozen_readout_metrics,
    neighbour_target_gap,
    nmad,
    representation_geometry_report,
    target_ordering,
)


def _blobs(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    features = y[:, None] * 2.0 + rng.normal(scale=0.3, size=(n, 2))
    return features, y


def test_gap_and_ordering() -> None:
    f, y = _blobs()
    gap = neighbour_target_gap(f, y, k=10)
    order = target_ordering(f, y)
    assert 0.0 <= gap < 1.0
    assert 0.5 < order <= 1.0
    # Random features should have a larger neighbourhood gap and worse ordering.
    rng = np.random.default_rng(1)
    fr = rng.normal(size=f.shape)
    assert neighbour_target_gap(fr, y, k=10) > gap
    assert target_ordering(fr, y) < order


def test_nmad() -> None:
    y = np.array([0.0, 1.0, 2.0])
    assert nmad(y, y) == 0.0
    assert nmad(y + 0.1, y) > 0.0


def test_frozen_readouts_and_report() -> None:
    f, y = _blobs(n=200)
    metrics = frozen_readout_metrics(f, y, seed=0)
    assert "ols_RMSE" in metrics and "knn_RMSE" in metrics
    assert 0.0 < metrics["ols_RMSE"] < 1.0
    report = representation_geometry_report(f, y, k=10, seed=0)
    assert {"neighbour_target_gap", "target_ordering", "ols_RMSE"} <= set(report)


def _brute_force_gap(features: np.ndarray, y: np.ndarray, k: int) -> float:
    f = (features - features.mean(0)) / features.std(0, ddof=1)
    d = np.sqrt(((f[:, None, :] - f[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    per_point = np.abs(y[idx] - y[:, None]).mean(axis=1)
    return float(np.median(per_point) / y.std(ddof=1))


def test_neighbour_target_gap_uses_numpy_median_for_even_counts() -> None:
    f, y = _blobs(n=200, seed=4)  # even N: the median averages the two middle values
    assert neighbour_target_gap(f, y, k=5) == pytest.approx(_brute_force_gap(f, y, 5), rel=1e-12)


def test_chunked_distances_match_single_block(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchregress.metrics import representation

    f, y = _blobs(n=401, seed=2)
    full = (
        neighbour_target_gap(f, y, k=7),
        target_ordering(f, y),
        target_ordering(f, y, max_pairs=3000, seed=5),
        frozen_readout_metrics(f, y, k=9, seed=1),
    )
    monkeypatch.setattr(representation, "_DISTANCE_CHUNK_CELLS", 997)
    chunked = (
        neighbour_target_gap(f, y, k=7),
        target_ordering(f, y),
        target_ordering(f, y, max_pairs=3000, seed=5),
        frozen_readout_metrics(f, y, k=9, seed=1),
    )
    assert chunked == full


def test_frozen_readout_nmad_is_textbook() -> None:
    f, y = _blobs(n=200, seed=6)  # 40 test rows: even count
    metrics = frozen_readout_metrics(f, y, k=5, seed=3)
    rng = np.random.default_rng(3)
    order = rng.permutation(200)
    test, train = order[:40], order[40:]
    ft, fe = f[train], f[test]
    mu, sd = ft.mean(0), ft.std(0, ddof=1)
    ft, fe = (ft - mu) / sd, (fe - mu) / sd
    d = np.sqrt(((fe[:, None, :] - ft[None, :, :]) ** 2).sum(-1))
    idx = np.argsort(d, axis=1, kind="stable")[:, :5]
    err = y[train][idx].mean(axis=1) - y[test]
    assert metrics["knn_NMAD"] == pytest.approx(
        1.482602218505602 * np.median(np.abs(err)), rel=1e-12
    )
    assert metrics["knn_MAE"] == pytest.approx(np.abs(err).mean(), rel=1e-12)


def test_representation_metrics_accept_tensors_with_grad() -> None:
    import torch

    f, y = _blobs(n=120, seed=8)
    ft = torch.as_tensor(f).requires_grad_(True)
    yt = torch.as_tensor(y)
    assert representation_geometry_report(ft, yt, k=5, seed=0) == representation_geometry_report(
        f, y, k=5, seed=0
    )
