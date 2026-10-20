"""Photo-z style PDF diagnostics from Cheng et al. (2026) and related work.

The functions here implement the metric conventions used in the companion
astronomy papers: PIT Cramér--von Mises, exact CRPS for a piecewise-linear
binned CDF, local (photometric-neighbour) PIT Wasserstein distance, the
normalized central-68% PDF width, and paired bootstrap intervals.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor


def _as_1d(x: Any, name: str) -> Tensor:
    t = torch.as_tensor(x, dtype=torch.float64)
    if t.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {tuple(t.shape)}")
    return t


def pit_cvm(pit: Any, *, normalized: bool = False) -> float:
    """Cramér--von Mises distance of PIT values from Uniform(0, 1).

    ``CvM = ∫_0^1 (F_hat(u) - u)^2 du`` evaluated exactly from the sorted PIT
    values.  The expected value for ``N`` independent uniform PIT values is
    ``1 / (6 N)``; set ``normalized=True`` to divide by that reference.
    """
    u = _as_1d(pit, "pit")
    if u.numel() == 0:
        raise ValueError("pit must contain at least one value")
    u = u.clamp(0.0, 1.0)
    u, _ = torch.sort(u)
    n = u.numel()
    # Piecewise-constant empirical CDF: F=i/n on [u_i, u_{i+1}], with F=0 on
    # [0, u_0] and F=1 on [u_{n-1}, 1].
    boundaries = torch.cat([torch.zeros(1, dtype=u.dtype), u, torch.ones(1, dtype=u.dtype)])
    f_values = torch.cat(
        [torch.zeros(1, dtype=u.dtype), torch.arange(1, n + 1, dtype=u.dtype) / n]
    )  # F on [boundary_i, boundary_{i+1}], length n+1
    widths = boundaries[1:] - boundaries[:-1]
    integral_f2 = (f_values**2 * widths).sum()
    # ∫ u F(u) du: F is constant over each interval.
    integral_uf = (f_values * (boundaries[1:] ** 2 - boundaries[:-1] ** 2) / 2.0).sum()
    cvm = float(integral_f2 - 2.0 * integral_uf + 1.0 / 3.0)
    cvm = max(cvm, 0.0)
    if normalized:
        return cvm / (1.0 / (6.0 * n)) if n > 0 else float("nan")
    return cvm


def _integral_sq_linear(fa: Tensor, fb: Tensor, width: Tensor) -> Tensor:
    """∫_a^b L(x)^2 dx for a line from (a, fa) to (b, fb)."""
    return width * (fa**2 + fa * fb + fb**2) / 3.0


def crps_from_binned_cdf(
    bin_edges: Any,
    cdf: Any,
    target: Any,
    *,
    reduction: str = "mean",
) -> Any:
    """Exact CRPS for a piecewise-linear CDF defined on fixed bin edges.

    The CDF rises linearly between adjacent edges.  Each sample's integral is
    computed in closed form, including the intervals below and above the
    reference redshift; no grid approximation is used.  Inputs with bin masses
    instead of CDF values can be converted with :func:`binned_cdf`.
    """
    edges = _as_1d(bin_edges, "bin_edges")
    if edges.numel() < 2:
        raise ValueError("bin_edges needs at least two values")
    c = torch.as_tensor(cdf, dtype=torch.float64)
    y = _as_1d(target, "target")
    if c.ndim == 1:
        c = c.unsqueeze(0)
    if c.shape != (y.numel(), edges.numel()):
        raise ValueError(
            f"cdf must have shape [{y.numel()}, {edges.numel()}], got {tuple(c.shape)}"
        )
    if not bool(torch.all(c[:, 1:] >= c[:, :-1] - 1e-12)):
        raise ValueError("cdf must be non-decreasing along the bin axis")
    if not bool(torch.allclose(c[:, 0], torch.zeros_like(c[:, 0]), atol=1e-8)) or not bool(
        torch.allclose(c[:, -1], torch.ones_like(c[:, -1]), atol=1e-8)
    ):
        raise ValueError("cdf must start at 0 and end at 1")
    widths = edges[1:] - edges[:-1]
    fa = c[:, :-1]
    fb = c[:, 1:]
    n = y.numel()
    values = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        yi = float(y[i])
        if yi <= float(edges[0]):
            # At every support z >= y, the step CDF is 1.
            values[i] = _integral_sq_linear(1.0 - fa[i], 1.0 - fb[i], widths).sum()
            continue
        if yi >= float(edges[-1]):
            # At every support z < y, the step CDF is 0.
            values[i] = _integral_sq_linear(fa[i], fb[i], widths).sum()
            continue
        total = torch.zeros((), dtype=torch.float64)
        for j in range(len(widths)):
            a, b = float(edges[j]), float(edges[j + 1])
            if b <= yi:
                total = total + _integral_sq_linear(fa[i, j], fb[i, j], widths[j])
            elif a >= yi:
                total = total + _integral_sq_linear(1.0 - fa[i, j], 1.0 - fb[i, j], widths[j])
            else:
                frac = (yi - a) / (b - a)
                fy = fa[i, j] + frac * (fb[i, j] - fa[i, j])
                left_w = yi - a
                right_w = b - yi
                total = total + _integral_sq_linear(
                    fa[i, j], fy, torch.as_tensor(left_w, dtype=torch.float64)
                )
                total = total + _integral_sq_linear(
                    1.0 - fy,
                    1.0 - fb[i, j],
                    torch.as_tensor(right_w, dtype=torch.float64),
                )
        values[i] = total
    if reduction == "mean":
        return float(values.mean())
    if reduction == "sum":
        return float(values.sum())
    if reduction == "none":
        return values.numpy()
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def binned_cdf(probabilities: Any) -> np.ndarray:
    """CDF values at bin edges from bin masses.

    Parameters
    ----------
    probabilities : array-like, shape ``[N, K]``
        Normalized probability masses per bin (rows are normalized internally).
    """
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(p, axis=1)], axis=1)


def pdf_width_68(samples: Any, redshift: Any) -> float:
    """Median central-68% PDF width divided by ``1 + z``.

    ``samples`` is ``[N, S]`` predictive samples; ``redshift`` is ``[N]``.
    This is the paper's ``W68`` diagnostic.
    """
    s = torch.as_tensor(samples, dtype=torch.float64)
    z = _as_1d(redshift, "redshift")
    if s.ndim != 2 or s.shape[0] != z.numel():
        raise ValueError("samples must be [N, S] with N matching redshift")
    lo = torch.quantile(s, 0.16, dim=1)
    hi = torch.quantile(s, 0.84, dim=1)
    return float(torch.median((hi - lo) / (1.0 + z)))


def pit_local_w1(features: Any, pit: Any, *, k: int = 100, chunk: int = 2048) -> float:
    """Median local PIT 1-Wasserstein distance over photometric neighbours.

    For each object, take its ``k`` nearest neighbours in standardized feature
    space (excluding itself), compute the 1-Wasserstein distance between the
    neighbours' PIT values and Uniform(0,1), and return the median over objects.
    This is the paper's ``local-W1``.  The finite-sample reference is ~0.028 at
    ``k=100``; it is not a strict lower bound.

    The distance is computed from the sorted neighbour PIT values using a
    dense uniform grid, which is deterministic and adequate at k~100.
    """
    f = torch.as_tensor(features, dtype=torch.float64)
    u = _as_1d(pit, "pit")
    if f.ndim != 2 or f.shape[0] != u.numel():
        raise ValueError("features must be [N, D] with N matching pit")
    if k < 2:
        raise ValueError("k must be at least 2")
    n = f.shape[0]
    if n <= k:
        raise ValueError("k must be smaller than the number of objects")
    mu = f.mean(dim=0, keepdim=True)
    sd = f.std(dim=0, keepdim=True).clamp_min(1e-12)
    f = (f - mu) / sd
    grid = torch.linspace(0.0, 1.0, 2001, dtype=torch.float64)
    w1 = torch.empty(n, dtype=torch.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = torch.cdist(f[start:stop], f)  # [chunk, N]
        # Exclude self by setting its distance to +inf before top-k.
        rows = torch.arange(start, stop)
        d[torch.arange(stop - start), rows] = float("inf")
        _, idx = torch.topk(d, k, dim=1, largest=False)
        neigh = u[idx]  # [chunk, k]
        # Empirical CDF on the uniform grid; integrate |F_n - u|.
        cdf = (neigh[:, :, None] <= grid[None, None, :]).to(torch.float64).mean(dim=1)
        w1[start:stop] = torch.trapezoid(torch.abs(cdf - grid[None, :]), grid, dim=1)
    return float(torch.median(w1))


def paired_bootstrap_ci(
    a: Any,
    b: Any,
    *,
    n_boot: int = 2000,
    alpha: float = 0.32,
    seed: Optional[int] = None,
) -> tuple[float, float, float]:
    """Paired percentile bootstrap interval for ``mean(a - b)``.

    Returns ``(difference, lower, upper)``.  The default 68% interval matches
    the paper's use of the 16th--84th percentiles.  This describes test-sample
    uncertainty only; it does not include retraining variation.
    """
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.shape != bb.shape:
        raise ValueError("a and b must have the same shape")
    rng = np.random.default_rng(seed)
    diffs = aa - bb
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        boot[i] = diffs[idx].mean()
    lower, upper = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(diffs.mean()), float(lower), float(upper)
