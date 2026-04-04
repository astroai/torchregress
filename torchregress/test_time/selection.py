"""Confidence and local-consistency utilities for test-time adaptation."""

from __future__ import annotations

import numpy as np


def entropy_scores(probabilities: np.ndarray, *, eps: float = 1.0e-8) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    probs = np.clip(probs, eps, None)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), eps, None)
    return -np.sum(probs * np.log(probs), axis=1)


def confidence_scores(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    return probs.max(axis=1)


def pseudo_label_targets(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=float)
    labels = probs.argmax(axis=1)
    weights = probs[np.arange(probs.shape[0]), labels]
    return labels, weights


def select_high_confidence(
    probabilities: np.ndarray,
    *,
    min_confidence: float | None = None,
    max_entropy: float | None = None,
    top_fraction: float | None = None,
    min_count: int = 1,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    mask = np.ones(probs.shape[0], dtype=bool)
    if min_confidence is not None:
        mask &= confidence_scores(probs) >= float(min_confidence)
    if max_entropy is not None:
        mask &= entropy_scores(probs) <= float(max_entropy)
    if top_fraction is not None:
        frac = float(top_fraction)
        if not 0.0 < frac <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        scores = confidence_scores(probs)
        k = max(int(np.ceil(frac * probs.shape[0])), int(min_count))
        top_idx = np.argsort(scores)[-k:]
        top_mask = np.zeros(probs.shape[0], dtype=bool)
        top_mask[top_idx] = True
        mask &= top_mask
    if mask.sum() < min_count:
        top_idx = np.argsort(confidence_scores(probs))[-int(min_count) :]
        mask = np.zeros(probs.shape[0], dtype=bool)
        mask[top_idx] = True
    return mask


def local_consistency_weights(
    features: np.ndarray,
    probabilities: np.ndarray,
    *,
    k: int = 5,
    temperature: float = 1.0,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """Compute FTAT-style neighborhood consistency weights from feature space."""
    x = np.asarray(features, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    if x.ndim != 2 or probs.ndim != 2 or x.shape[0] != probs.shape[0]:
        raise ValueError("features and probabilities must have matching batch dimensions")
    if x.shape[0] == 1:
        return np.ones(1, dtype=float)
    k = max(1, min(int(k), x.shape[0] - 1))
    dists = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(dists, np.inf)
    nbr_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
    neighbor_probs = probs[nbr_idx].mean(axis=1)
    agreement = np.sum(np.sqrt(np.clip(probs, eps, None) * np.clip(neighbor_probs, eps, None)), axis=1)
    weights = np.exp((agreement - 1.0) / max(float(temperature), eps))
    return weights / np.clip(weights.mean(), eps, None)


__all__ = [
    "confidence_scores",
    "entropy_scores",
    "local_consistency_weights",
    "pseudo_label_targets",
    "select_high_confidence",
]
