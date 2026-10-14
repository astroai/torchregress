"""
Heuristic **neighbourhood target covariance** pseudo-labels for multivariate regression.

Estimates a per-sample SPD covariance of targets from weighted neighbours in input space,
following the self-supervision sketch in the Wasserstein-supervision research plan. This is
experimental: quality depends on representation geometry and local linearity.

See ``docs/losses/gaussian_wasserstein.md`` for pairing with ``GaussianWassersteinBoundLoss``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..utils.tensor_ops import float_dtype

Metric = Literal["mahalanobis", "euclidean"]
Weighting = Literal["softmax"]


@dataclass(frozen=True)
class NeighborhoodCovarianceConfig:
    """
    Configuration for neighbourhood covariance pseudo-labeling.

    Attributes:
        n_neighbors: Neighbour count per row (excluding self).
        metric: ``"mahalanobis"`` uses pooled feature precision; ``"euclidean"`` uses squared
            Euclidean distances in ``x`` space.
        weighting: Only ``"softmax"`` is supported in v1 (negative distance / temperature).
        regularization: Diagonal jitter on feature precision and on output covariances.
        temperature: Softmax temperature on negative squared distances.
    """

    n_neighbors: int = 32
    metric: Metric = "mahalanobis"
    weighting: Weighting = "softmax"
    regularization: float = 1e-5
    temperature: float = 1.0


_BLOCK_ELEMS = 1 << 24


def _as_tensor_2d(x: torch.Tensor | Any, *, name: str) -> torch.Tensor:
    # Keep a floating input dtype (float64 stays float64); integers/bools and python
    # sequences use the default dtype.
    t = torch.as_tensor(x)
    if not t.is_floating_point():
        t = t.to(torch.get_default_dtype())
    if t.dim() != 2:
        raise ValueError(f"{name} must be 2D [n, dim], got shape {tuple(t.shape)}")
    return t


def _precision_from_features(x: torch.Tensor, *, reg: float) -> torch.Tensor:
    """Inverse covariance of features with diagonal jitter (symmetric)."""
    n, p = x.shape
    if n < 2:
        raise ValueError("Need at least 2 reference rows to build a feature covariance.")
    x_c = x - x.mean(dim=0, keepdim=True)
    cov = (x_c.T @ x_c) / max(n - 1, 1)
    eye = torch.eye(p, device=x.device, dtype=x.dtype)
    return torch.linalg.inv(cov + reg * eye)


def _pairwise_quad_form(diff: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """diff [a, b, p], m [p, p] symmetric -> v^T M v for each pair, shape [a, b]."""
    mv = torch.matmul(diff, m.T)
    return (diff * mv).sum(dim=-1)


def _spdize(cov: torch.Tensor, *, jitter: float) -> torch.Tensor:
    sym = 0.5 * (cov + cov.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(sym)
    evals = torch.clamp(evals, min=jitter)
    # Coverage invariants (TOR003): chain .to() on torch.diag_embed because
    # torch.diag_embed does not accept device=/dtype= kwargs natively. Pin
    # to the input covariance's device/dtype so that mixed-precision callers
    # do not silently promote the diagonal into float64.
    return (
        evecs
        @ torch.diag_embed(evals).to(device=cov.device, dtype=cov.dtype)
        @ evecs.transpose(-1, -2)
    )


class NeighborhoodCovariancePseudoLabeler:
    """
    Per-sample weighted covariance of neighbour targets in ``(x, y)`` space.

    For each index ``i``, neighbours are the ``n_neighbors`` smallest metric distances to
    ``j != i``. Weights are a softmax over ``-distance / temperature``. The pseudo covariance
    is the weighted covariance of neighbour ``y`` values plus diagonal ``regularization`` on
    the target dimensions.

    This is **not** a statistically calibrated estimator; use for auxiliary supervision only.
    """

    def __init__(
        self,
        config: NeighborhoodCovarianceConfig | None = None,
    ) -> None:
        if config is None:
            config = NeighborhoodCovarianceConfig()
        if config.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if config.weighting != "softmax":
            raise ValueError("weighting must be 'softmax' in v1")
        if config.regularization <= 0:
            raise ValueError("regularization must be positive")
        if config.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.config = config

    @torch.no_grad()
    def fit_predict(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x0 = _as_tensor_2d(x, name="x")
        y0 = _as_tensor_2d(y, name="y")
        if x0.shape[0] != y0.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        dtype = float_dtype(x0, y0)
        x0, y0 = x0.to(dtype), y0.to(dtype)
        n = x0.shape[0]
        if n < self.config.n_neighbors + 1:
            raise ValueError(
                f"Need at least n_neighbors + 1 = {self.config.n_neighbors + 1} rows, got {n}"
            )
        return self._pseudo_cov_batch(x0, x0, y0, exclude_self=True)

    @torch.no_grad()
    def predict_for_query(
        self,
        x_query: torch.Tensor,
        *,
        x_reference: torch.Tensor,
        y_reference: torch.Tensor,
    ) -> torch.Tensor:
        """Pseudo covariances for query rows using neighbour sets in ``x_reference`` space."""
        xq = _as_tensor_2d(x_query, name="x_query")
        xr = _as_tensor_2d(x_reference, name="x_reference")
        yr = _as_tensor_2d(y_reference, name="y_reference")
        if xr.shape[0] != yr.shape[0]:
            raise ValueError("x_reference and y_reference must have the same number of rows")
        dtype = float_dtype(xq, xr, yr)
        xq, xr, yr = xq.to(dtype), xr.to(dtype), yr.to(dtype)
        n_ref = xr.shape[0]
        if n_ref < self.config.n_neighbors:
            raise ValueError(
                f"Reference set must have at least n_neighbors={self.config.n_neighbors} "
                f"rows, got {n_ref}"
            )
        return self._pseudo_cov_batch(xq, xr, yr, exclude_self=False)

    def _pseudo_cov_batch(
        self,
        x_query: torch.Tensor,
        x_ref: torch.Tensor,
        y_ref: torch.Tensor,
        *,
        exclude_self: bool,
    ) -> torch.Tensor:
        n = x_ref.shape[0]
        d = y_ref.shape[1]
        device = x_ref.device
        dtype = x_ref.dtype
        reg = self.config.regularization
        if exclude_self:
            k = min(self.config.n_neighbors, n - 1)
        else:
            k = min(self.config.n_neighbors, n)

        if self.config.metric == "mahalanobis":
            prec = _precision_from_features(x_ref, reg=reg)
        elif self.config.metric == "euclidean":
            prec = None
        else:
            raise ValueError(f"Unknown metric {self.config.metric!r}")

        # Process query rows in blocks so the (block, Nref, D) difference tensor stays
        # bounded by ~_BLOCK_ELEMS elements.
        n_ref, p = x_ref.shape
        n_query = x_query.shape[0]
        block = max(1, _BLOCK_ELEMS // max(1, n_ref * p))
        vals_blocks: list[torch.Tensor] = []
        idx_blocks: list[torch.Tensor] = []
        for start in range(0, max(n_query, 1), block):
            xq = x_query[start : start + block]
            diff = xq.unsqueeze(1) - x_ref.unsqueeze(0)
            if prec is not None:
                dist = _pairwise_quad_form(diff, prec)
            else:
                dist = (diff * diff).sum(dim=-1)
            if exclude_self:
                # Global query row i has its self-match in reference column i.
                rows = torch.arange(start, start + xq.shape[0], device=device)
                dist = dist.clone()
                dist[torch.arange(xq.shape[0], device=device), rows] = float("inf")
            v, i = dist.topk(k, largest=False)
            vals_blocks.append(v)
            idx_blocks.append(i)
        vals = torch.cat(vals_blocks, dim=0)
        idx = torch.cat(idx_blocks, dim=0)

        logits = -vals / self.config.temperature
        w = torch.softmax(logits, dim=-1)

        neigh_y = y_ref[idx]
        wmean = (w.unsqueeze(-1) * neigh_y).sum(dim=1)
        centered = neigh_y - wmean.unsqueeze(1)
        outer = centered.unsqueeze(-1) @ centered.unsqueeze(-2)
        cov = (w.unsqueeze(-1).unsqueeze(-1) * outer).sum(dim=1)
        eye_d = torch.eye(d, device=device, dtype=dtype)
        cov = cov + reg * eye_d
        return _spdize(cov, jitter=reg)
