"""Confidence and local-consistency utilities for test-time adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sample_reference_indices(
    n_rows: int, reference_size: int | None, *, random_state: int | None
) -> np.ndarray:
    if reference_size is None or reference_size <= 0 or reference_size >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n_rows, size=int(reference_size), replace=False)
    return np.sort(idx.astype(np.int64, copy=False))


def entropy_scores(probabilities: np.ndarray, *, eps: float = 1.0e-8) -> np.ndarray:
    """
    Compute Shannon entropy scores over class probabilities.

    Parameters
    ----------
    probabilities : np.ndarray
        Array of probabilities of shape [batch, n_classes].
    eps : float
        Small positive constant for numerical stability.

    Returns
    -------
    np.ndarray
        1D array of Shannon entropy values per sample.
    """
    probs = np.asarray(probabilities, dtype=float)
    probs = np.clip(probs, eps, None)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), eps, None)
    return -np.sum(probs * np.log(probs), axis=1)


def confidence_scores(probabilities: np.ndarray) -> np.ndarray:
    """
    Extract the maximum probability score as confidence score.

    Parameters
    ----------
    probabilities : np.ndarray
        Array of probabilities of shape [batch, n_classes].

    Returns
    -------
    np.ndarray
        1D array of maximum probabilities per sample.
    """
    probs = np.asarray(probabilities, dtype=float)
    return probs.max(axis=1)


def pseudo_label_targets(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pseudo-labels and corresponding confidence weights.

    Parameters
    ----------
    probabilities : np.ndarray
        Array of probabilities of shape [batch, n_classes].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of (predicted_class_labels, max_probability_weights).
    """
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
    """
    Filter predictions keeping only high-confidence or low-entropy instances.

    Parameters
    ----------
    probabilities : np.ndarray
        Array of probabilities of shape [batch, n_classes].
    min_confidence : Optional[float]
        Minimum confidence threshold.
    max_entropy : Optional[float]
        Maximum entropy threshold.
    top_fraction : Optional[float]
        Keep only this fraction of top-confidence instances.
    min_count : int
        Ensure at least this many instances are selected.

    Returns
    -------
    np.ndarray
        A boolean mask indicating selected instances.
    """
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
    safe_min_count = min(int(min_count), probs.shape[0])
    if mask.sum() < safe_min_count:
        top_idx = np.argsort(confidence_scores(probs))[-safe_min_count:]
        mask = np.zeros(probs.shape[0], dtype=bool)
        mask[top_idx] = True
    return mask


@dataclass(frozen=True)
class LocalConsistencyConfig:
    """
    Configuration options for neighborhood local feature consistency weights.

    Parameters
    ----------
    k : int
        Number of nearest neighbors.
    temperature : float
        Softmax temperature scale.
    reference_size : Optional[int]
        Subsample reference size for consistency search.
    max_exact_rows : int
        Maximum size before performing subsampled reference indexing.
    query_chunk_size : Optional[int]
        Batch chunk size to compute pairwise distances.
    random_state : Optional[int]
        Random seed.
    eps : float
        Small positive constant.
    """

    k: int = 5
    temperature: float = 1.0
    reference_size: int | None = None
    max_exact_rows: int = 4096
    query_chunk_size: int | None = 2048
    random_state: int | None = 0
    eps: float = 1.0e-8


def local_consistency_weights(
    features: np.ndarray,
    probabilities: np.ndarray,
    config: LocalConsistencyConfig | None = None,
) -> np.ndarray:
    """Compute FTAT-style neighborhood consistency weights from feature space."""
    if config is None:
        config = LocalConsistencyConfig()
    x = np.asarray(features, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    if x.ndim != 2 or probs.ndim != 2 or x.shape[0] != probs.shape[0]:
        raise ValueError("features and probabilities must have matching batch dimensions")
    if x.shape[0] == 1:
        return np.ones(1, dtype=float)
    reference_idx = (
        np.arange(x.shape[0], dtype=np.int64)
        if x.shape[0] <= int(config.max_exact_rows) and config.reference_size is None
        else _sample_reference_indices(
            x.shape[0],
            config.reference_size or config.max_exact_rows,
            random_state=config.random_state,
        )
    )
    ref_x = x[reference_idx]
    ref_probs = probs[reference_idx]
    k = max(1, min(int(config.k), ref_x.shape[0]))
    neighbor_probs = np.empty_like(probs, dtype=float)
    chunk_size = (
        x.shape[0]
        if config.query_chunk_size is None or config.query_chunk_size <= 0
        else int(config.query_chunk_size)
    )
    exact_self_reference = reference_idx.shape[0] == x.shape[0] and np.array_equal(
        reference_idx, np.arange(x.shape[0], dtype=np.int64)
    )
    if exact_self_reference:
        k = max(1, min(int(config.k), x.shape[0] - 1))
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        dists = np.sum((x[start:stop, None, :] - ref_x[None, :, :]) ** 2, axis=-1)
        if exact_self_reference:
            row_idx = np.arange(start, stop, dtype=np.int64)
            dists[np.arange(stop - start), row_idx] = np.inf
        nbr_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        neighbor_probs[start:stop] = ref_probs[nbr_idx].mean(axis=1)
    sqrt_prod = np.sqrt(
        np.clip(probs, config.eps, None) * np.clip(neighbor_probs, config.eps, None)
    )
    agreement = np.sum(sqrt_prod, axis=1)
    weights = np.exp((agreement - 1.0) / max(float(config.temperature), config.eps))
    return weights / np.clip(weights.mean(), config.eps, None)


__all__ = [
    "confidence_scores",
    "entropy_scores",
    "local_consistency_weights",
    "pseudo_label_targets",
    "select_high_confidence",
    "LocalConsistencyConfig",
]
