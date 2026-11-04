"""Generic calibration and binned-distribution diagnostics.

These functions are not astronomy-specific; they are used by general
conditional-density and survival/ordinal workflows.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor


def _as_1d(x: Any, name: str) -> Tensor:
    t = torch.as_tensor(x, dtype=torch.float64).detach().cpu()
    if t.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {tuple(t.shape)}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} must contain finite values")
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
    if not bool(((u >= 0) & (u <= 1)).all()):
        raise ValueError("pit values must be between 0 and 1")
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


# Rows per chunk are chosen so one chunk holds about this many (row, bin) cells;
# peak memory is a few float64 [chunk, K] temporaries regardless of N.
_CRPS_CHUNK_CELLS = 1 << 22


def _input_device(*values: Any) -> torch.device:
    for v in values:
        if isinstance(v, Tensor):
            return v.device
    return torch.device("cpu")


def _as_1d_on(x: Any, name: str, device: torch.device) -> Tensor:
    t = torch.as_tensor(x, dtype=torch.float64, device=device).detach()
    if t.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {tuple(t.shape)}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} must contain finite values")
    return t


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
    reference target; no grid approximation is used.  Inputs with bin masses
    instead of CDF values can be converted with :func:`binned_cdf`.

    For a target ``y`` and bin ``[a, b]`` let ``t = clip(y, a, b)``; the bin
    contributes ``∫_a^t F^2 + ∫_t^b (1 - F)^2`` (both closed-form for a linear
    ``F``), and targets outside ``[e_0, e_K]`` add ``max(e_0 - y, 0) +
    max(y - e_K, 0)``.  The computation is batched over rows and bins in float64
    on the device of the first tensor argument (CPU for NumPy input), in row
    chunks of bounded memory.  ``reduction="mean"``/``"sum"`` return a Python
    float and ``"none"`` a NumPy array, whatever the input type.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")
    device = _input_device(cdf, target, bin_edges)
    edges = _as_1d_on(bin_edges, "bin_edges", device)
    if edges.numel() < 2:
        raise ValueError("bin_edges needs at least two values")
    if not bool(torch.all(edges[1:] > edges[:-1])):
        raise ValueError("bin_edges must be strictly increasing")
    c = torch.as_tensor(cdf, dtype=torch.float64, device=device).detach()
    if not bool(torch.isfinite(c).all()):
        raise ValueError("cdf must contain finite values")
    y = _as_1d_on(target, "target", device)
    if y.numel() == 0:
        raise ValueError("target must contain at least one value")
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
    lo = edges[:-1]
    hi = edges[1:]
    widths = hi - lo
    n = y.numel()
    chunk = max(1, _CRPS_CHUNK_CELLS // widths.numel())
    parts = []
    for start in range(0, n, chunk):
        yc = y[start : start + chunk, None]
        fa = c[start : start + chunk, :-1]
        fb = c[start : start + chunk, 1:]
        # Split point of each bin: the target clipped to the bin.
        t = torch.minimum(torch.maximum(yc, lo), hi)
        left_w = t - lo
        right_w = hi - t
        fy = fa + (left_w / widths) * (fb - fa)
        below = _integral_sq_linear(fa, fy, left_w)
        above = _integral_sq_linear(1.0 - fy, 1.0 - fb, right_w)
        outside = (edges[0] - yc[:, 0]).clamp_min(0.0) + (yc[:, 0] - edges[-1]).clamp_min(0.0)
        parts.append((below + above).sum(dim=1) + outside)
    values = torch.cat(parts)
    if reduction == "mean":
        return float(values.mean())
    if reduction == "sum":
        return float(values.sum())
    return values.cpu().numpy()


def binned_cdf(probabilities: Any) -> np.ndarray:
    """CDF values at bin edges from bin masses.

    Parameters
    ----------
    probabilities : array-like, shape ``[N, K]``
        Normalized probability masses per bin (rows are normalized internally).
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError("probabilities must be a nonempty [N, K] matrix")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p.sum(axis=1) <= 0):
        raise ValueError("probabilities must be finite, nonnegative, with positive row sums")
    p = p / p.sum(axis=1, keepdims=True)
    return np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(p, axis=1)], axis=1)


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
    if not aa.size or not np.isfinite(aa).all() or not np.isfinite(bb).all():
        raise ValueError("paired samples must be nonempty and finite")
    if not isinstance(n_boot, int) or n_boot < 1 or not 0 < alpha < 1:
        raise ValueError("n_boot must be positive and alpha must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    diffs = aa - bb
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        boot[i] = diffs[idx].mean()
    lower, upper = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(diffs.mean()), float(lower), float(upper)
