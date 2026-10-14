"""Test-time label-shift correction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf

import numpy as np
import torch

from .selection import LocalConsistencyConfig, local_consistency_weights, select_high_confidence


def _normalize_rows(probabilities: np.ndarray, eps: float) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=float), eps, None)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), eps, None)


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
        Small constant for numerical stability. Classes (bins) whose source
        prior is at most ``eps`` are excluded from EM and get target prior 0.
    loglik_tol : float, optional
        If set, EM also stops (``converged=True``) once the mean target
        log-likelihood ``mean_i log sum_k p_ik q_k / pi_k`` improves by less
        than ``loglik_tol`` nats per row in one iteration. Useful for many
        overlapping classes (fine regression bins), where the prior iterate
        drifts too slowly for the ``tol`` criterion while the fit has stopped
        improving. ``None`` (default) keeps the ``tol`` criterion only.
    """

    max_iter: int = 100
    tol: float = 1.0e-6
    eps: float = 1.0e-8
    loglik_tol: float | None = None

    def __post_init__(self) -> None:
        if self.loglik_tol is not None and not self.loglik_tol > 0.0:
            raise ValueError(f"loglik_tol must be positive when set, got {self.loglik_tol}")


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
    """
    Estimate the target label prior from unlabeled predictions by EM
    (Saerens, Latinne & Decaestecker 2002).

    Each iteration reweights the source posteriors by ``q / pi`` (target over
    source prior), renormalises them per row and sets ``q`` to their mean.

    Parameters
    ----------
    probabilities : np.ndarray
        Target predicted probabilities ``[n, k]`` (rows are renormalised).
    source_prior : np.ndarray
        Source prior ``[k]``. Classes with source prior ``<= config.eps``
        are excluded (target prior 0): under label shift the target support
        lies inside the source support, and their ratio ``q / pi`` is
        otherwise unbounded.
    sample_weights : np.ndarray, optional
        Row weights for the M-step average (and the log-likelihood).
    sample_size : int, optional
        Subsample this many rows (without replacement) first.
    random_state : int, optional
        Seed for the subsample.
    config : LabelShiftEMConfig, optional
        Iteration limits and tolerances.

    Returns
    -------
    LabelShiftEstimate
        Source prior (clipped at ``eps``, normalised), target prior,
        iterations and convergence flag.

    Notes
    -----
    **Calibrated posteriors are required.** EM is the maximum-likelihood
    estimate of ``q`` only when ``probabilities`` are calibrated *source
    posteriors* ``p_s(y = k | x)`` whose source marginal is ``source_prior``
    (``E_s[p_s(y = k | x)] = pi_k``), and label shift holds at the level of
    the classes or bins (``p(x | y = k)`` shared by source and target).
    Binned Gaussians qualify only if ``N(mean, std)`` is a calibrated
    predictive ``p_s(y | x)``; for a prediction that is a noisy copy of the
    label with ``std`` the residual spread they are the likelihood
    ``p(pred | y)`` (no shrinkage toward the prior). With such miscalibrated
    probabilities EM converges to a biased fixed point. Calibrate first, or
    pass the mean source posterior as ``source_prior``;
    :func:`estimate_target_prior_bbse` needs only a confusion matrix and stays
    consistent without calibration. On many overlapping bins the maximum
    likelihood prior is an ill-posed deconvolution: iterating to the ``tol``
    criterion fits sampling noise, so prefer ``loglik_tol`` there.

    References
    ----------
    .. [1] Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the
       outputs of a classifier to new a priori probabilities: a simple
       procedure. *Neural Computation*, 14(1), 21-41.
    """
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
    src_raw = np.asarray(source_prior, dtype=float)
    if src_raw.shape != (n_classes,):
        raise ValueError("source_prior must have shape [n_classes]")
    src = np.clip(src_raw, cfg.eps, None)
    src = src / src.sum()

    # EM runs on the classes with source mass only.
    mass = np.clip(src_raw, 0.0, None)
    active = mass > cfg.eps * max(float(mass.sum()), cfg.eps)
    if active.all() or not active.any():
        active = np.ones(n_classes, dtype=bool)
        probs_a, src_a = probs, src
    else:
        probs_a = _normalize_rows(probs[:, active], cfg.eps)
        src_a = src[active] / src[active].sum()

    def _expand(prior_a: np.ndarray) -> np.ndarray:
        if active.all():
            return prior_a
        full = np.zeros(n_classes, dtype=float)
        full[active] = prior_a
        return full

    tgt = src_a.copy()
    prev_loglik: float | None = None
    for step in range(1, cfg.max_iter + 1):
        if cfg.loglik_tol is not None:
            evidence = (probs_a * (np.clip(tgt, cfg.eps, None) / src_a)[None, :]).sum(axis=1)
            log_ev = np.log(np.clip(evidence, cfg.eps, None))
            loglik = float(np.average(log_ev, weights=weights))
            if prev_loglik is not None and loglik - prev_loglik < cfg.loglik_tol:
                return LabelShiftEstimate(src, _expand(tgt), step - 1, True)
            prev_loglik = loglik
        corrected = apply_label_shift_correction(
            probs_a, source_prior=src_a, target_prior=tgt, eps=cfg.eps
        )
        new_tgt = (
            np.average(corrected, axis=0, weights=weights)
            if weights is not None
            else corrected.mean(axis=0)
        )
        new_tgt = np.clip(new_tgt, cfg.eps, None)
        new_tgt = new_tgt / new_tgt.sum()
        if np.max(np.abs(new_tgt - tgt)) < cfg.tol:
            return LabelShiftEstimate(src, _expand(new_tgt), step, True)
        tgt = new_tgt
    return LabelShiftEstimate(src, _expand(tgt), cfg.max_iter, False)


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
        Reconstructed mean and standard deviation, in the promoted dtype of
        ``probabilities`` and ``bin_edges`` (float64 for float64 edges; never
        down-cast to float32, which loses sub-0.03 resolution at offsets ~3e5).
    """
    probs = _normalize_rows(probabilities, eps)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean = probs @ centers
    second = probs @ (centers**2)
    var = np.clip(second - mean**2, eps, None)
    return mean, np.sqrt(var)


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
    # Return in the dtype of the supplied predictions (float64 stays float64).
    out_dtype = np.result_type(np.asarray(mean).dtype, np.asarray(std).dtype)
    if np.issubdtype(out_dtype, np.floating):
        corrected_mean = corrected_mean.astype(out_dtype, copy=False)
        corrected_std = corrected_std.astype(out_dtype, copy=False)
    metadata: dict[str, object] = {
        "target_prior": estimate.target_prior.tolist(),
        "source_prior": estimate.source_prior.tolist(),
        "estimate_converged": bool(estimate.converged),
        "estimate_iterations": int(estimate.iterations),
        "selected_rows": int(mask.sum()),
    }
    return corrected_mean, corrected_std, metadata


def estimate_target_prior_bbse(
    probabilities_source: np.ndarray,
    labels_source: np.ndarray,
    probabilities_target: np.ndarray,
    *,
    cond_threshold: float = 1e8,
) -> np.ndarray:
    """
    Estimate target priors under label shift with BBSE (Black-Box Shift
    Estimation, Lipton, Wang & Smola 2018): invert the source confusion matrix
    against the target mean predicted-label distribution.

    Args:
        probabilities_source: Source predicted probabilities [n_source, k].
        labels_source: True source labels [n_source] (integers in [0, k)).
        probabilities_target: Target predicted probabilities [n_target, k].
        cond_threshold: Raise if the confusion-matrix condition number
            (2-norm) exceeds this value.

    Returns:
        Estimated target prior ``w`` [k], the solution of ``C^T w = mu`` where
        ``C[i, j] = P(pred=j | y=i)`` and ``mu`` is the target distribution of
        predicted labels. Entries may be negative or exceed 1 — BBSE is an
        unbiased but unconstrained moment estimate.
    """
    probs_src = _normalize_rows(probabilities_source, eps=1e-12)
    probs_tgt = _normalize_rows(probabilities_target, eps=1e-12)
    labels = np.asarray(labels_source).astype(int).ravel()
    n_classes = probs_src.shape[1]
    if probs_tgt.shape[1] != n_classes:
        raise ValueError(
            f"source probabilities have {n_classes} classes but target has {probs_tgt.shape[1]}"
        )
    if labels.size != probs_src.shape[0]:
        raise ValueError("labels_source must align row-wise with probabilities_source")
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError(f"labels_source values must be integers in [0, {n_classes})")

    counts = np.zeros((n_classes, n_classes), dtype=np.float64)
    for true_cls, pred_cls in zip(labels, probs_src.argmax(axis=1), strict=True):
        counts[true_cls, pred_cls] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    confusion = counts / np.clip(row_sums, 1e-12, None)

    mu_counts = np.bincount(probs_tgt.argmax(axis=1), minlength=n_classes).astype(np.float64)
    mu = mu_counts / max(mu_counts.sum(), 1.0)

    confusion_t = torch.from_numpy(confusion.T)  # solve C^T w = mu
    cond = torch.linalg.cond(confusion_t)
    if not bool(torch.isfinite(cond)) or float(cond) > float(cond_threshold):
        raise ValueError(f"confusion matrix ill-conditioned (cond={float(cond):.2e})")

    weights = torch.linalg.solve(confusion_t, torch.from_numpy(mu))
    return weights.numpy().astype(np.float64)


__all__ = [
    "GaussianLabelShiftConfig",
    "LabelShiftEMConfig",
    "LabelShiftEstimate",
    "PosteriorLabelShiftAdapter",
    "apply_label_shift_correction",
    "correct_gaussian_predictions_for_label_shift",
    "estimate_target_prior_em",
    "estimate_target_prior_bbse",
    "gaussian_bin_edges_from_targets",
    "gaussian_bin_probabilities",
    "gaussian_moments_from_binned_probabilities",
]
