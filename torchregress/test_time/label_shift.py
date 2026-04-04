"""Test-time label-shift correction utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize_rows(probabilities: np.ndarray, eps: float) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=float), eps, None)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), eps, None)


@dataclass(frozen=True)
class LabelShiftEstimate:
    source_prior: np.ndarray
    target_prior: np.ndarray
    iterations: int
    converged: bool


def apply_label_shift_correction(
    probabilities: np.ndarray,
    *,
    source_prior: np.ndarray,
    target_prior: np.ndarray,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """Apply posterior correction under label shift using prior ratios."""
    probs = _normalize_rows(probabilities, eps)
    src = np.clip(np.asarray(source_prior, dtype=float), eps, None)
    tgt = np.clip(np.asarray(target_prior, dtype=float), eps, None)
    if probs.shape[1] != src.shape[0] or src.shape != tgt.shape:
        raise ValueError("prior shapes must match probability columns")
    corrected = probs * (tgt / src)[None, :]
    return corrected / np.clip(corrected.sum(axis=1, keepdims=True), eps, None)


def estimate_target_prior_em(
    probabilities: np.ndarray,
    *,
    source_prior: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    eps: float = 1.0e-8,
) -> LabelShiftEstimate:
    """Estimate target priors from unlabeled predictions via EM."""
    probs = _normalize_rows(probabilities, eps)
    n_classes = probs.shape[1]
    if source_prior is None:
        src = probs.mean(axis=0)
    else:
        src = np.asarray(source_prior, dtype=float)
    if src.shape != (n_classes,):
        raise ValueError("source_prior must have shape [n_classes]")
    src = np.clip(src, eps, None)
    src = src / src.sum()

    tgt = src.copy()
    converged = False
    for step in range(1, max_iter + 1):
        corrected = apply_label_shift_correction(probs, source_prior=src, target_prior=tgt, eps=eps)
        new_tgt = corrected.mean(axis=0)
        new_tgt = np.clip(new_tgt, eps, None)
        new_tgt = new_tgt / new_tgt.sum()
        if np.max(np.abs(new_tgt - tgt)) < tol:
            tgt = new_tgt
            converged = True
            return LabelShiftEstimate(src, tgt, step, converged)
        tgt = new_tgt
    return LabelShiftEstimate(src, tgt, max_iter, converged)


class PosteriorLabelShiftAdapter:
    """Reusable label-shift adapter for batch predictions."""

    def __init__(
        self,
        *,
        source_prior: np.ndarray | None = None,
        max_iter: int = 100,
        tol: float = 1.0e-6,
        eps: float = 1.0e-8,
    ) -> None:
        self.source_prior = None if source_prior is None else np.asarray(source_prior, dtype=float)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.eps = float(eps)
        self.last_estimate: LabelShiftEstimate | None = None

    def estimate(self, probabilities: np.ndarray) -> LabelShiftEstimate:
        estimate = estimate_target_prior_em(
            probabilities,
            source_prior=self.source_prior,
            max_iter=self.max_iter,
            tol=self.tol,
            eps=self.eps,
        )
        self.last_estimate = estimate
        if self.source_prior is None:
            self.source_prior = estimate.source_prior
        return estimate

    def transform(
        self, probabilities: np.ndarray, *, target_prior: np.ndarray | None = None
    ) -> np.ndarray:
        if target_prior is None:
            if self.last_estimate is None:
                self.estimate(probabilities)
            assert self.last_estimate is not None
            target_prior = self.last_estimate.target_prior
        if self.source_prior is None:
            raise RuntimeError(
                "source_prior is unavailable; call estimate() first or pass source_prior"
            )
        return apply_label_shift_correction(
            probabilities,
            source_prior=self.source_prior,
            target_prior=np.asarray(target_prior, dtype=float),
            eps=self.eps,
        )

    def fit_transform(self, probabilities: np.ndarray) -> tuple[np.ndarray, LabelShiftEstimate]:
        estimate = self.estimate(probabilities)
        corrected = self.transform(probabilities, target_prior=estimate.target_prior)
        return corrected, estimate


__all__ = [
    "LabelShiftEstimate",
    "PosteriorLabelShiftAdapter",
    "apply_label_shift_correction",
    "estimate_target_prior_em",
]
