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

from ..utils.tensor_ops import _MAD_TO_SIGMA, _median

# Distance rows are computed in chunks of about this many (row, column) cells, so
# peak memory is O(chunk * N) rather than O(N^2) (2**24 float64 cells = 128 MiB).
_DISTANCE_CHUNK_CELLS = 1 << 24


def _as_float64(x: Any) -> Tensor:
    """Float64 tensor on the input's device (CPU for NumPy/array-like input)."""
    return torch.as_tensor(x, dtype=torch.float64).detach()


def _row_chunk(n_columns: int) -> int:
    return max(1, _DISTANCE_CHUNK_CELLS // max(1, n_columns))


def _standardize(features: Tensor) -> Tensor:
    mu = features.mean(dim=0, keepdim=True)
    sd = features.std(dim=0, keepdim=True).clamp_min(1e-12)
    return (features - mu) / sd


def _knn_indices(queries: Tensor, reference: Tensor, k: int, *, exclude_self: bool) -> Tensor:
    """Indices of the ``k`` nearest ``reference`` rows for each query row.

    Row chunks of :func:`torch.cdist` keep memory at O(chunk * N_ref).  With
    ``exclude_self`` the queries are ``reference`` itself and each row's own
    index is excluded.
    """
    n = queries.shape[0]
    chunk = _row_chunk(reference.shape[0])
    out = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = torch.cdist(queries[start:stop], reference)
        if exclude_self:
            rows = torch.arange(stop - start, device=d.device)
            d[rows, rows + start] = float("inf")
        out.append(torch.topk(d, k, dim=1, largest=False).indices)
    return torch.cat(out)


def neighbour_target_gap(features: Any, target: Any, *, k: int = 10) -> float:
    """Median over points of the mean absolute target difference to their k nearest features.

    For each point, the absolute target differences to its ``k`` nearest
    neighbours (Euclidean, standardized features, the point itself excluded)
    are averaged; the median of these per-point means (even counts average the
    two middle values) is divided by the target standard deviation, so lower is
    more target-homogeneous.  Distances are computed in row chunks on the
    input's device, with O(chunk * N) memory.
    """
    f = _as_float64(features)
    y = _as_float64(target).reshape(-1).to(f.device)
    if f.dim() != 2 or f.shape[0] != y.numel():
        raise ValueError("features must be [N, D] with N matching target")
    if k < 1 or k >= f.shape[0]:
        raise ValueError("k must be in [1, N-1]")
    f = _standardize(f)
    idx = _knn_indices(f, f, k, exclude_self=True)
    gaps = (y[idx] - y[:, None]).abs().mean(dim=1)
    return float((_median(gaps) / y.std().clamp_min(1e-12)).item())


def target_ordering(
    features: Any, target: Any, *, max_pairs: int = 2_000_000, seed: int = 0
) -> float:
    """Spearman correlation between feature distance and target distance.

    A high positive value means nearby features also have nearby targets.  All
    ``N (N - 1) / 2`` pairs are used when that is at most ``max_pairs``;
    otherwise ``max_pairs`` random pairs (drawn with ``seed``, self-pairs
    dropped) are used.  Distances are computed in chunks on the input's device;
    only the pair distances are moved to the CPU for the rank correlation.
    """
    from scipy.stats import spearmanr

    f = _as_float64(features)
    y = _as_float64(target).reshape(-1).to(f.device)
    if f.dim() != 2 or f.shape[0] != y.numel():
        raise ValueError("features must be [N, D] with N matching target")
    f = _standardize(f)
    n = f.shape[0]
    fd_parts: list[Tensor] = []
    td_parts: list[Tensor] = []
    if n * (n - 1) // 2 <= max_pairs:
        chunk = _row_chunk(n)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            d = torch.cdist(f[start:stop], f)
            dy = (y[start:stop, None] - y[None, :]).abs()
            rows = torch.arange(start, stop, device=f.device)
            upper = rows[:, None] < torch.arange(n, device=f.device)[None, :]
            fd_parts.append(d[upper])
            td_parts.append(dy[upper])
    else:
        rng = np.random.default_rng(seed)
        i_np = rng.integers(0, n, size=max_pairs)
        j_np = rng.integers(0, n, size=max_pairs)
        keep = i_np != j_np
        i = torch.as_tensor(i_np[keep], device=f.device)
        j = torch.as_tensor(j_np[keep], device=f.device)
        chunk = _row_chunk(f.shape[1])
        for start in range(0, i.numel(), chunk):
            ii, jj = i[start : start + chunk], j[start : start + chunk]
            fd_parts.append(torch.linalg.vector_norm(f[ii] - f[jj], dim=1))
            td_parts.append((y[ii] - y[jj]).abs())
    fd = torch.cat(fd_parts).cpu().numpy() if fd_parts else np.empty(0)
    td = torch.cat(td_parts).cpu().numpy() if td_parts else np.empty(0)
    if len(fd) < 2:
        return float("nan")
    return float(spearmanr(fd, td).statistic)


def nmad(y_pred: Any, y_true: Any) -> float:
    """Normalized median absolute deviation of residuals (1.482602218505602 * median |r|)."""
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if pred.shape != true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    return float(_MAD_TO_SIGMA * np.median(np.abs(pred - true)))


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
    for a task-specific validation protocol.  NMAD is
    ``1.482602218505602 * median |r|`` (even counts average the two middle
    values).  Work stays on the input's device; the k-NN search runs in test-row
    chunks with O(chunk * N_train) memory.
    """
    f = _as_float64(features)
    y = _as_float64(target).reshape(-1).to(f.device)
    n = f.shape[0]
    if n < 10:
        raise ValueError("need at least 10 samples for a readout split")
    rng = np.random.default_rng(seed)
    order = torch.as_tensor(rng.permutation(n), device=f.device)
    n_test = max(1, int(round(n * test_fraction)))
    test, train = order[:n_test], order[n_test:]
    ft, yt = f[train], y[train]
    fe, ye = f[test], y[test]
    # Standardize features on train, add intercept.
    mu = ft.mean(dim=0, keepdim=True)
    sd = ft.std(dim=0, keepdim=True).clamp_min(1e-12)
    ft = (ft - mu) / sd
    fe = (fe - mu) / sd
    design = torch.cat([torch.ones(ft.shape[0], 1, dtype=ft.dtype, device=ft.device), ft], dim=1)
    coef = torch.linalg.lstsq(design, yt.unsqueeze(-1)).solution
    design_e = torch.cat([torch.ones(fe.shape[0], 1, dtype=fe.dtype, device=fe.device), fe], dim=1)
    ols_pred = (design_e @ coef).reshape(-1)
    idx = _knn_indices(fe, ft, min(k, ft.shape[0]), exclude_self=False)
    knn_pred = yt[idx].mean(dim=1)
    out: dict[str, float] = {}
    for name, pred in (("ols", ols_pred), ("knn", knn_pred)):
        err = pred - ye
        out[f"{name}_RMSE"] = float(torch.sqrt(torch.mean(err**2)))
        out[f"{name}_MAE"] = float(torch.mean(err.abs()))
        out[f"{name}_NMAD"] = float(_MAD_TO_SIGMA * _median(err.abs()))
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
