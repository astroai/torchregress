"""
Heuristic **neighbourhood target covariance** pseudo-labels for multivariate regression.

Estimates a per-sample SPD covariance of targets from weighted neighbours in input space,
following the self-supervision sketch in the Wasserstein-supervision research plan. This is
experimental: quality depends on representation geometry and local linearity.

See ``docs/losses/gaussian_wasserstein.md`` for pairing with ``GaussianWassersteinBoundLoss``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

import torch

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



def _as_tensor_2d(x: Union[torch.Tensor, Any], *, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    t = x.float()
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
    return evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)


@torch.no_grad()
def mahalanobis_covariance_pseudo_labels(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    config: NeighborhoodCovarianceConfig | None = None,
) -> torch.Tensor:
    """
    Functional API for :class:`NeighborhoodCovariancePseudoLabeler.fit_predict`.

    Args:
        x: Reference inputs ``[n, p]``.
        y: Reference targets ``[n, d]`` (use ``y.unsqueeze(-1)`` for scalar outputs).
        config: Configuration for the pseudo-labeling. If None, uses default configuration.

    Returns:
        Tensor ``[n, d, d]`` SPD-ish per-row target covariances (symmetric, eigenvalues floored).
    """
    if config is None:
        config = NeighborhoodCovarianceConfig()
    labeler = NeighborhoodCovariancePseudoLabeler(config=config)
    return labeler.fit_predict(x, y)


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
        x: Union[torch.Tensor, "torch.Tensor"],
        y: Union[torch.Tensor, "torch.Tensor"],
    ) -> torch.Tensor:
        x0 = _as_tensor_2d(x, name="x")
        y0 = _as_tensor_2d(y, name="y")
        if x0.shape[0] != y0.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        n = x0.shape[0]
        if n < self.config.n_neighbors + 1:
            raise ValueError(
                f"Need at least n_neighbors + 1 = {self.config.n_neighbors + 1} rows, got {n}"
            )
        return self._pseudo_cov_batch(x0, x0, y0, exclude_self=True)

    @torch.no_grad()
    def predict_for_query(
        self,
        x_query: Union[torch.Tensor, "torch.Tensor"],
        *,
        x_reference: Union[torch.Tensor, "torch.Tensor"],
        y_reference: Union[torch.Tensor, "torch.Tensor"],
    ) -> torch.Tensor:
        """Pseudo covariances for query rows using neighbour sets in ``x_reference`` space."""
        xq = _as_tensor_2d(x_query, name="x_query")
        xr = _as_tensor_2d(x_reference, name="x_reference")
        yr = _as_tensor_2d(y_reference, name="y_reference")
        if xr.shape[0] != yr.shape[0]:
            raise ValueError("x_reference and y_reference must have the same number of rows")
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
            diff = x_query.unsqueeze(1) - x_ref.unsqueeze(0)
            dist = _pairwise_quad_form(diff, prec)
        elif self.config.metric == "euclidean":
            diff = x_query.unsqueeze(1) - x_ref.unsqueeze(0)
            dist = (diff * diff).sum(dim=-1)
        else:
            raise ValueError(f"Unknown metric {self.config.metric!r}")

        if exclude_self:
            mask_self = torch.eye(n, device=device, dtype=torch.bool)
            dist_sq = dist.clone()
            dist_sq.masked_fill_(mask_self, float("inf"))
            vals, idx = dist_sq.topk(k, largest=False)
        else:
            vals, idx = dist.topk(k, largest=False)

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
