"""Subspace and feature-stat alignment tools for regression test-time adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _feature_significance(X: np.ndarray, y: np.ndarray | None, eps: float) -> np.ndarray:
    if y is None:
        return np.clip(X.std(axis=0), eps, None)
    y_centered = y - y.mean()
    x_centered = X - X.mean(axis=0, keepdims=True)
    numer = np.abs((x_centered * y_centered[:, None]).mean(axis=0))
    denom = np.clip(x_centered.std(axis=0) * y_centered.std(), eps, None)
    return np.clip(numer / denom, eps, None)


@dataclass(frozen=True)
class SubspaceAlignmentState:
    source_mean: np.ndarray
    target_mean: np.ndarray
    source_scale: np.ndarray
    target_scale: np.ndarray
    components: np.ndarray
    feature_weights: np.ndarray
    rank: int


class SignificantSubspaceAligner:
    """SSA-style low-rank alignment focused on significant regression structure."""

    def __init__(
        self,
        *,
        rank: int | None = None,
        variance_threshold: float = 0.95,
        eps: float = 1.0e-6,
    ) -> None:
        self.rank = rank
        self.variance_threshold = float(variance_threshold)
        self.eps = float(eps)
        self.state_: SubspaceAlignmentState | None = None

    def fit(
        self, X_source: np.ndarray, y_source: np.ndarray | None = None
    ) -> "SignificantSubspaceAligner":
        X = np.asarray(X_source, dtype=float)
        y = None if y_source is None else np.asarray(y_source, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X_source must be 2D")
        if y is not None and y.shape[0] != X.shape[0]:
            raise ValueError("y_source must match X_source rows")

        weights = _feature_significance(X, y, self.eps)
        x_mean = X.mean(axis=0)
        Xw = (X - x_mean[None, :]) * np.sqrt(weights)[None, :]
        _, s, vt = np.linalg.svd(Xw, full_matrices=False)
        if self.rank is None:
            explained = np.cumsum(s**2) / np.clip(np.sum(s**2), self.eps, None)
            rank = int(np.searchsorted(explained, self.variance_threshold) + 1)
        else:
            rank = int(self.rank)
        rank = max(1, min(rank, vt.shape[0]))
        comps = vt[:rank]
        proj = Xw @ comps.T
        src_mean = proj.mean(axis=0)
        src_scale = np.clip(proj.std(axis=0), self.eps, None)
        self.state_ = SubspaceAlignmentState(
            source_mean=src_mean,
            target_mean=np.zeros_like(src_mean),
            source_scale=src_scale,
            target_scale=np.ones_like(src_scale),
            components=comps,
            feature_weights=weights,
            rank=rank,
        )
        self._x_source_mean = x_mean
        return self

    def transform(self, X_target: np.ndarray) -> np.ndarray:
        if self.state_ is None:
            raise RuntimeError("call fit() before transform()")
        X = np.asarray(X_target, dtype=float)
        weights = self.state_.feature_weights
        Xw = (X - self._x_source_mean[None, :]) * np.sqrt(weights)[None, :]
        proj = Xw @ self.state_.components.T
        tgt_mean = proj.mean(axis=0)
        tgt_scale = np.clip(proj.std(axis=0), self.eps, None)
        aligned_proj = (proj - tgt_mean[None, :]) * (
            self.state_.source_scale[None, :] / tgt_scale[None, :]
        )
        aligned_proj = aligned_proj + self.state_.source_mean[None, :]
        residual = Xw - (proj @ self.state_.components)
        Xw_aligned = residual + (aligned_proj @ self.state_.components)
        X_aligned = Xw_aligned / np.sqrt(weights)[None, :] + self._x_source_mean[None, :]
        self.state_ = SubspaceAlignmentState(
            source_mean=self.state_.source_mean,
            target_mean=tgt_mean,
            source_scale=self.state_.source_scale,
            target_scale=tgt_scale,
            components=self.state_.components,
            feature_weights=self.state_.feature_weights,
            rank=self.state_.rank,
        )
        return X_aligned

    def fit_transform(
        self, X_source: np.ndarray, X_target: np.ndarray, y_source: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(X_source, y_source=y_source).transform(X_target)


class FeatureStatNormalizer:
    """Low-risk test-time feature-stat alignment."""

    def __init__(self, *, eps: float = 1.0e-6) -> None:
        self.eps = float(eps)
        self.source_mean_: np.ndarray | None = None
        self.source_std_: np.ndarray | None = None

    def fit(self, X_source: np.ndarray) -> "FeatureStatNormalizer":
        X = np.asarray(X_source, dtype=float)
        self.source_mean_ = X.mean(axis=0)
        self.source_std_ = np.clip(X.std(axis=0), self.eps, None)
        return self

    def transform(self, X_target: np.ndarray) -> np.ndarray:
        if self.source_mean_ is None or self.source_std_ is None:
            raise RuntimeError("call fit() before transform()")
        X = np.asarray(X_target, dtype=float)
        tgt_mean = X.mean(axis=0)
        tgt_std = np.clip(X.std(axis=0), self.eps, None)
        return (X - tgt_mean[None, :]) * (
            self.source_std_[None, :] / tgt_std[None, :]
        ) + self.source_mean_[None, :]


__all__ = [
    "FeatureStatNormalizer",
    "SignificantSubspaceAligner",
    "SubspaceAlignmentState",
]
