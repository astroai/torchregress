"""Representation-geometry and readout diagnostics for regression."

These metrics implement the companion methods-paper comparison: neighbourhood
target homogeneity, how well feature distances order target distances, and
simple frozen-feature readouts (OLS and k-NN).  They are descriptive summaries,
not a single ranking criterion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


def _standardize(features: Tensor) -> Tensor:
    mu = features.mean(dim=0, keepdim=True)
    sd = features.std(dim=0, keepdim=True).clamp_min(1e-12)
    return (features - mu) / sd


def neighbour_target_gap(features: Any, target: Any, *, k: int = 10) -> float:
    """Median absolute target difference among each point's k nearest features.

    Features are standardized and Euclidean k-NN is used.  The result is divided
    by the target standard deviation, so lower is more target-homogeneous.
    """
    f = torch.as_tensor(features, dtype=torch.float64)
    y = torch.as_tensor(target, dtype=torch.float64).reshape(-1)
    if f.dim() != 2 or f.shape[0] != y.numel():
        raise ValueError("features must be [N, D] with N matching target")
    if k < 1 or k >= f.shape[0]:
        raise ValueError("k must be in [1, N-1]")
    f = _standardize(f)
    d = torch.cdist(f, f)
    d.fill_diagonal_(float("inf"))
    _, idx = torch.topk(d, k, dim=1, largest=False)
    gaps = (y[idx] - y[:, None]).abs().mean(dim=1)
    return float((gaps.median() / y.std().clamp_min(1e-12)).item())


def target_ordering(
    features: Any, target: Any, *, max_pairs: int = 2_000_000, seed: int = 0
) -> float:
    """Spearman correlation between feature distance and target distance.

    A high positive value means nearby features also have nearby targets.  For
    large samples only ``max_pairs`` random pairs are used.
    """
    from scipy.stats import spearmanr

    f = torch.as_tensor(features, dtype=torch.float64)
    y = torch.as_tensor(target, dtype=torch.float64).reshape(-1)
    if f.dim() != 2 or f.shape[0] != y.numel():
        raise ValueError("features must be [N, D] with N matching target")
    f = _standardize(f)
    n = f.shape[0]
    if n * (n - 1) // 2 <= max_pairs:
        d = torch.cdist(f, f)
        dy = (y[:, None] - y[None, :]).abs()
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        fd = d[mask].numpy()
        td = dy[mask].numpy()
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
        fd = torch.linalg.vector_norm(f[i] - f[j], dim=1).numpy()
        td = (y[i] - y[j]).abs().numpy()
    if len(fd) < 2:
        return float("nan")
    return float(spearmanr(fd, td).statistic)


def nmad(y_pred: Any, y_true: Any) -> float:
    """Normalized median absolute deviation of residuals (1.4826 * median |r|)."""
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if pred.shape != true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    return float(1.4826 * np.median(np.abs(pred - true)))


def frozen_readout_metrics(
    features: Any,
    target: Any,
    *,
    k: int = 25,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, float]:
    """RMSE/MAE/NMAD for OLS and k-NN readouts on frozen features.

    A deterministic split is used; this is a quick diagnostic, not a substitute
    for a task-specific validation protocol.
    """
    f = torch.as_tensor(features, dtype=torch.float64)
    y = torch.as_tensor(target, dtype=torch.float64).reshape(-1)
    n = f.shape[0]
    if n < 10:
        raise ValueError("need at least 10 samples for a readout split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    test, train = order[:n_test], order[n_test:]
    ft, yt = f[train], y[train]
    fe, ye = f[test], y[test]
    # Standardize features on train, add intercept.
    mu = ft.mean(dim=0, keepdim=True)
    sd = ft.std(dim=0, keepdim=True).clamp_min(1e-12)
    ft = (ft - mu) / sd
    fe = (fe - mu) / sd
    design = torch.cat([torch.ones(ft.shape[0], 1, dtype=ft.dtype), ft], dim=1)
    coef = torch.linalg.lstsq(design, yt.unsqueeze(-1)).solution
    design_e = torch.cat([torch.ones(fe.shape[0], 1, dtype=fe.dtype), fe], dim=1)
    ols_pred = (design_e @ coef).reshape(-1)
    d = torch.cdist(fe, ft)
    _, idx = torch.topk(d, min(k, ft.shape[0]), dim=1, largest=False)
    knn_pred = yt[idx].mean(dim=1)
    out: dict[str, float] = {}
    for name, pred in (("ols", ols_pred), ("knn", knn_pred)):
        err = pred - ye
        out[f"{name}_RMSE"] = float(torch.sqrt(torch.mean(err**2)))
        out[f"{name}_MAE"] = float(torch.mean(err.abs()))
        out[f"{name}_NMAD"] = float(1.4826 * torch.median(err.abs()))
    return out


def representation_geometry_report(
    features: Any,
    target: Any,
    *,
    k: int = 10,
    readouts: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    """Aggregate the paper's neighbourhood/ordering/readout diagnostics."""
    out = {
        "neighbour_target_gap": neighbour_target_gap(features, target, k=k),
        "target_ordering": target_ordering(features, target, seed=seed),
    }
    if readouts:
        out.update(frozen_readout_metrics(features, target, seed=seed))
    return out
