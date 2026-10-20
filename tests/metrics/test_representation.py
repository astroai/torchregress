"""Tests for representation-geometry and readout diagnostics."""

from __future__ import annotations

import numpy as np

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
