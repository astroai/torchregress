"""Test-time label-shift correction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf

import numpy as np

from .selection import LocalConsistencyConfig, local_consistency_weights, select_high_confidence


def _normalize_rows(probabilities: np.ndarray, eps: float) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=float), eps, None)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), eps, None)


def _weighted_average(values: np.ndarray, weights: np.ndarray | None, *, eps: float) -> np.ndarray:
    if weights is None:
        return values.mean(axis=0)
    w = np.clip(np.asarray(weights, dtype=float).reshape(-1), eps, None)
    if w.shape[0] != values.shape[0]:
        raise ValueError("sample_weights must match probability rows")
    w = w / np.clip(w.sum(), eps, None)
    return np.sum(values * w[:, None], axis=0)


def _subsample_probabilities(
    probabilities: np.ndarray,
    sample_weights: np.ndarray | None,
    sample_size: int | None,
    *,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if sample_size is None or sample_size <= 0 or probabilities.shape[0] <= int(sample_size):
        return probabilities, sample_weights
    rng = np.random.default_rng(random_state)
    idx = rng.choice(probabilities.shape[0], size=int(sample_size), replace=False)
    idx = np.sort(idx.astype(np.int64, copy=False))
    if sample_weights is None:
        return probabilities[idx], None
    return probabilities[idx], np.asarray(sample_weights, dtype=float).reshape(-1)[idx]


@dataclass(frozen=True)
class LabelShiftEMConfig:
    """
    Configuration options for EM-based target prior estimation.

    Parameters
    ----------
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance for prior differences.
    eps : float
        Small constant for numerical stability.
    """

    max_iter: int = 100
    tol: float = 1.0e-6
    eps: float = 1.0e-8


@dataclass(frozen=True)
class LabelShiftEstimate:
    """
    Results container for EM label shift estimation.

    Parameters
    ----------
    source_prior : np.ndarray
        Estimated or provided source label prior distribution.
    target_prior : np.ndarray
        Estimated target label prior distribution.
    iterations : int
        Number of EM iterations executed.
    converged : bool
        Whether the estimation converged within tolerance.
    """

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
    source_prior: np.ndarray,
    sample_weights: np.ndarray | None = None,
    sample_size: int | None = None,
    random_state: int | None = 0,
    config: LabelShiftEMConfig | None = None,
) -> LabelShiftEstimate:
    """Estimate target priors from unlabeled predictions via EM."""
    if source_prior is None:
        raise ValueError("source_prior must be explicitly provided for EM label-shift correction.")
    cfg = config or LabelShiftEMConfig()
    probs = _normalize_rows(probabilities, cfg.eps)
    probs, weights = _subsample_probabilities(
        probs,
        sample_weights,
        sample_size,
        random_state=random_state,
    )
    n_classes = probs.shape[1]
    src = np.asarray(source_prior, dtype=float)
    if src.shape != (n_classes,):
        raise ValueError("source_prior must have shape [n_classes]")
    src = np.clip(src, cfg.eps, None)
    src = src / src.sum()

    tgt = src.copy()
    converged = False
    for step in range(1, cfg.max_iter + 1):
        corrected = apply_label_shift_correction(
            probs, source_prior=src, target_prior=tgt, eps=cfg.eps
        )
        new_tgt = _weighted_average(corrected, weights, eps=cfg.eps)
        new_tgt = np.clip(new_tgt, cfg.eps, None)
        new_tgt = new_tgt / new_tgt.sum()
        if np.max(np.abs(new_tgt - tgt)) < cfg.tol:
            tgt = new_tgt
            converged = True
            return LabelShiftEstimate(src, tgt, step, converged)
        tgt = new_tgt
    return LabelShiftEstimate(src, tgt, cfg.max_iter, converged)


class PosteriorLabelShiftAdapter:
    """
    Reusable label-shift adapter for batch predictions.

    References
    ----------
    .. [1] Lipton, Z. C., Wang, Y. X., & Smola, A. J. (2018). Detecting and Correcting
       for Label Shift with Black Box Predictors. In *ICML 2018*.
       https://arxiv.org/abs/1802.03916
    """

    def __init__(
        self,
        *,
        source_prior: np.ndarray,
        sample_size: int | None = None,
        random_state: int | None = 0,
        config: LabelShiftEMConfig | None = None,
    ) -> None:
        if source_prior is None:
            raise ValueError("source_prior must be explicitly provided.")
        self.source_prior = np.asarray(source_prior, dtype=float)
        self.sample_size = sample_size
        self.random_state = random_state
        self.config = config or LabelShiftEMConfig()
        self.last_estimate: LabelShiftEstimate | None = None

    def estimate(
        self, probabilities: np.ndarray, *, sample_weights: np.ndarray | None = None
    ) -> LabelShiftEstimate:
        estimate = estimate_target_prior_em(
            probabilities,
            source_prior=self.source_prior,
            sample_weights=sample_weights,
            sample_size=self.sample_size,
            random_state=self.random_state,
            config=self.config,
        )
        self.last_estimate = estimate
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
            eps=self.config.eps,
        )

    def fit_transform(
        self,
        probabilities: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, LabelShiftEstimate]:
        estimate = self.estimate(probabilities, sample_weights=sample_weights)
        corrected = self.transform(probabilities, target_prior=estimate.target_prior)
        return corrected, estimate


def gaussian_bin_edges_from_targets(targets: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Compute bin edges for continuous target discretization.

    Parameters
    ----------
    targets : np.ndarray
        Array of continuous target values.
    n_bins : int
        Desired number of bins.

    Returns
    -------
    np.ndarray
        Sorted array of unique bin edges.
    """
    values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("targets must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("targets must contain only finite values")
    quantiles = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        lo = float(np.min(values))
        hi = float(np.max(values))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, max(2, int(n_bins)) + 1, dtype=float)
    return edges.astype(np.float64, copy=False)


def gaussian_bin_probabilities(
    mean: np.ndarray,
    std: np.ndarray,
    bin_edges: np.ndarray,
    *,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """
    Calculate probability mass in each bin for Gaussian predictions.

    Parameters
    ----------
    mean : np.ndarray
        Array of predicted Gaussian means.
    std : np.ndarray
        Array of predicted Gaussian standard deviations.
    bin_edges : np.ndarray
        Sorted array of bin edges.
    eps : float
        Small positive constant for numerical stability.

    Returns
    -------
    np.ndarray
        Bin probabilities of shape [batch, n_bins].
    """
    mu = np.asarray(mean, dtype=np.float64).reshape(-1, 1)
    sigma = np.clip(np.asarray(std, dtype=np.float64).reshape(-1, 1), eps, None)
    z = (bin_edges[None, :] - mu) / sigma
    cdf = 0.5 * (1.0 + np.vectorize(erf, otypes=[np.float64])(z / np.sqrt(2.0)))
    probs = np.diff(cdf, axis=1)
    probs = np.clip(probs, eps, None)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), eps, None)


def gaussian_moments_from_binned_probabilities(
    probabilities: np.ndarray,
    bin_edges: np.ndarray,
    *,
    eps: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct Gaussian mean and standard deviation from discrete bin probabilities.

    Parameters
    ----------
    probabilities : np.ndarray
        Bin probabilities of shape [batch, n_bins].
    bin_edges : np.ndarray
        Sorted array of bin edges.
    eps : float
        Small positive constant for numerical stability.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Reconstructed mean and standard deviation.
    """
    probs = _normalize_rows(probabilities, eps)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean = probs @ centers
    second = probs @ (centers**2)
    var = np.clip(second - mean**2, eps, None)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


@dataclass(frozen=True)
class GaussianLabelShiftConfig:
    """
    Configuration options for continuous Gaussian label shift correction.

    Parameters
    ----------
    n_bins : int
        Number of bins for discretization.
    estimation_rows : Optional[int]
        Number of rows to subsample for prior estimation.
    top_fraction : Optional[float]
        Fraction of high-confidence predictions to select.
    reference_size : Optional[int]
        Reference sample size for local consistency calculation.
    seed : Optional[int]
        Random seed for reproducibility.
    eps : float
        Small positive constant.
    """

    n_bins: int = 32
    estimation_rows: int | None = None
    top_fraction: float | None = 0.5
    reference_size: int | None = 2048
    seed: int | None = 0
    eps: float = 1.0e-8


def correct_gaussian_predictions_for_label_shift(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    source_targets: np.ndarray,
    features: np.ndarray | None = None,
    config: GaussianLabelShiftConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """
    Correct continuous Gaussian predictions for test-time label shift.

    Discretizes the continuous targets into quantiles, runs EM target-prior
    estimation on binned probabilities, and maps corrected probabilities
    back to Gaussian mean and standard deviation.

    Parameters
    ----------
    mean : np.ndarray
        Predicted Gaussian means.
    std : np.ndarray
        Predicted Gaussian standard deviations.
    source_targets : np.ndarray
        Target labels from the source domain.
    features : Optional[np.ndarray]
        Input features for consistency weighting.
    config : Optional[GaussianLabelShiftConfig]
        Configuration options.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, object]]
        Corrected mean, corrected standard deviation, and metadata dictionary.
    """
    cfg = config or GaussianLabelShiftConfig()
    bin_edges = gaussian_bin_edges_from_targets(source_targets, cfg.n_bins)
    src_t = np.asarray(source_targets, dtype=np.float64)
    source_prior = np.histogram(src_t, bins=bin_edges)[0].astype(np.float64)
    source_prior = np.clip(source_prior, cfg.eps, None)
    source_prior = source_prior / source_prior.sum()
    probs = gaussian_bin_probabilities(mean, std, bin_edges, eps=cfg.eps)
    weights = None
    if features is not None and len(features) == len(probs):
        weights = local_consistency_weights(
            features,
            probs,
            config=LocalConsistencyConfig(
                k=min(5, max(1, len(probs) - 1)),
                reference_size=cfg.reference_size,
                random_state=cfg.seed,
            ),
        )
    mask = (
        select_high_confidence(
            probs,
            top_fraction=cfg.top_fraction,
            min_count=min(max(16, probs.shape[1] * 2), probs.shape[0]),
        )
        if cfg.top_fraction is not None
        else np.ones(probs.shape[0], dtype=bool)
    )
    adapter = PosteriorLabelShiftAdapter(
        source_prior=source_prior,
        sample_size=cfg.estimation_rows,
        random_state=cfg.seed,
        config=LabelShiftEMConfig(eps=cfg.eps),
    )
    estimate = adapter.estimate(
        probs[mask],
        sample_weights=weights[mask] if weights is not None else None,
    )
    corrected = adapter.transform(probs, target_prior=estimate.target_prior)
    corrected_mean, corrected_std = gaussian_moments_from_binned_probabilities(
        corrected, bin_edges, eps=cfg.eps
    )
    metadata: dict[str, object] = {
        "target_prior": estimate.target_prior.tolist(),
        "source_prior": estimate.source_prior.tolist(),
        "estimate_converged": bool(estimate.converged),
        "estimate_iterations": int(estimate.iterations),
        "selected_rows": int(mask.sum()),
    }
    return corrected_mean, corrected_std, metadata


__all__ = [
    "GaussianLabelShiftConfig",
    "LabelShiftEMConfig",
    "LabelShiftEstimate",
    "PosteriorLabelShiftAdapter",
    "apply_label_shift_correction",
    "correct_gaussian_predictions_for_label_shift",
    "estimate_target_prior_em",
    "gaussian_bin_edges_from_targets",
    "gaussian_bin_probabilities",
    "gaussian_moments_from_binned_probabilities",
]
