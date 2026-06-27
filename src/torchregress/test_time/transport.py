"""Shift-factored predictive transport orchestration for regression test-time adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
import torch

from torchregress.calibration.shift import RepresentationShiftInflator
from torchregress.inference import PPIConfig, ppi_mean_ci, ppi_ols_ci, ppi_quantile_ci
from torchregress.prediction import PredictiveBatch

from .base import SupportsPredictiveBatch
from .label_shift import LabelShiftEMConfig, PosteriorLabelShiftAdapter
from .selection import LocalConsistencyConfig, local_consistency_weights, select_high_confidence
from .subspace import WeightedSubspaceMomentAligner


def _to_numpy(x: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_1d(x: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    return _to_numpy(x).reshape(-1)


def _as_2d(x: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    arr = _to_numpy(x)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _support_grid_from_targets(
    targets: np.ndarray,
    *,
    n_support: int,
    support_margin: float,
) -> np.ndarray:
    values = np.asarray(targets, dtype=float).reshape(-1)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("source_targets must be finite")
    if hi <= lo:
        hi = lo + 1.0
    width = max(hi - lo, 1.0e-6)
    return np.linspace(
        lo - support_margin * width,
        hi + support_margin * width,
        max(16, int(n_support)),
        dtype=float,
    )


def _support_edges(support: np.ndarray) -> np.ndarray:
    supp = np.asarray(support, dtype=float).reshape(-1)
    if supp.size < 2:
        raise ValueError("support must contain at least two points")
    mids = 0.5 * (supp[:-1] + supp[1:])
    left = supp[0] - 0.5 * (supp[1] - supp[0])
    right = supp[-1] + 0.5 * (supp[-1] - supp[-2])
    return np.concatenate([[left], mids, [right]])


def _uniform_dx(support: np.ndarray) -> float:
    supp = np.asarray(support, dtype=float).reshape(-1)
    return float(max(np.mean(np.diff(supp)), 1.0e-8))


def _normalize_density(support: np.ndarray, density: np.ndarray, eps: float) -> np.ndarray:
    dens = np.clip(np.asarray(density, dtype=float), 0.0, None)
    integral = np.trapezoid(dens, support, axis=1)
    integral = np.clip(integral, eps, None)
    return dens / integral[:, None]


def _density_to_probabilities(support: np.ndarray, density: np.ndarray, eps: float) -> np.ndarray:
    dx = _uniform_dx(support)
    probs = np.clip(np.asarray(density, dtype=float), 0.0, None) * dx
    row_sum = probs.sum(axis=1, keepdims=True)
    # Fallback to uniform for degenerate (all-zero) rows to avoid NaN propagation
    degenerate = row_sum.ravel() < eps
    if np.any(degenerate):
        probs[degenerate] = 1.0 / probs.shape[1]
        row_sum[degenerate] = 1.0
    return probs / np.clip(row_sum, eps, None)


def _probabilities_to_density(
    support: np.ndarray,
    probabilities: np.ndarray,
    eps: float,
) -> np.ndarray:
    dx = _uniform_dx(support)
    density = np.asarray(probabilities, dtype=float) / max(dx, eps)
    return _normalize_density(support, density, eps)


def _gaussian_density_on_support(
    mean: np.ndarray,
    std: np.ndarray,
    support: np.ndarray,
    eps: float,
) -> np.ndarray:
    mu = np.asarray(mean, dtype=float).reshape(-1, 1)
    sigma = np.clip(np.asarray(std, dtype=float).reshape(-1, 1), eps, None)
    supp = np.asarray(support, dtype=float).reshape(1, -1)
    z = (supp - mu) / sigma
    dens = np.exp(-0.5 * z**2) / (sigma * np.sqrt(2.0 * np.pi))
    return _normalize_density(support, dens, eps)


def _resample_density(
    support_in: np.ndarray,
    density_in: np.ndarray,
    support_out: np.ndarray,
    eps: float,
) -> np.ndarray:
    supp_out = np.asarray(support_out, dtype=float).reshape(-1)
    supp_in = np.asarray(support_in, dtype=float)
    dens_in = np.asarray(density_in, dtype=float)
    out = np.empty((dens_in.shape[0], supp_out.size), dtype=float)
    if supp_in.ndim == 1:
        for idx in range(dens_in.shape[0]):
            out[idx] = np.interp(supp_out, supp_in, dens_in[idx], left=0.0, right=0.0)
    else:
        for idx in range(dens_in.shape[0]):
            out[idx] = np.interp(supp_out, supp_in[idx], dens_in[idx], left=0.0, right=0.0)
    return _normalize_density(supp_out, out, eps)


def _probability_moments(
    support: np.ndarray,
    probabilities: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    supp = np.asarray(support, dtype=float).reshape(-1)
    probs = np.asarray(probabilities, dtype=float)
    mean = probs @ supp
    second = probs @ (supp**2)
    var = np.clip(second - mean**2, eps, None)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _probability_quantiles(
    support: np.ndarray,
    probabilities: np.ndarray,
    quantile_levels: Sequence[float],
) -> np.ndarray:
    supp = np.asarray(support, dtype=float).reshape(-1)
    probs = np.asarray(probabilities, dtype=float)
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    levels = np.asarray(list(quantile_levels), dtype=float)
    return np.stack([np.interp(levels, cdf[idx], supp) for idx in range(probs.shape[0])], axis=0)


def _point_density_on_support(point: np.ndarray, support: np.ndarray, eps: float) -> np.ndarray:
    pts = np.asarray(point, dtype=float).reshape(-1)
    dx = _uniform_dx(support)
    std = np.full_like(pts, fill_value=max(2.0 * dx, eps), dtype=float)
    return _gaussian_density_on_support(pts, std, support, eps)


def _batch_to_support_density(
    batch: PredictiveBatch,
    support: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    hinted_family = None
    if batch.extra is not None:
        raw_family = batch.extra.get("family")
        if isinstance(raw_family, str):
            hinted_family = raw_family
    if batch.support is not None and batch.density is not None:
        density = _resample_density(
            _to_numpy(batch.support),
            _to_numpy(batch.density),
            support,
            eps,
        )
        metadata["family"] = hinted_family or "density"
        return density, metadata
    if batch.bar_logits is not None and batch.bin_edges is not None:
        converted = batch.with_density(n_support=len(support))
        assert converted.support is not None
        assert converted.density is not None
        density = _resample_density(
            _to_numpy(converted.support),
            _to_numpy(converted.density),
            support,
            eps,
        )
        metadata["family"] = "bar"
        return density, metadata
    if batch.quantiles is not None and batch.quantile_levels is not None:
        converted = batch.with_density(n_support=len(support))
        assert converted.support is not None
        assert converted.density is not None
        density = _resample_density(
            _to_numpy(converted.support),
            _to_numpy(converted.density),
            support,
            eps,
        )
        metadata["family"] = "quantile"
        metadata["quantile_levels"] = list(batch.quantile_levels)
        return density, metadata
    if batch.samples is not None:
        converted = batch.with_density(n_support=len(support))
        assert converted.support is not None
        assert converted.density is not None
        density = _resample_density(
            _to_numpy(converted.support),
            _to_numpy(converted.density),
            support,
            eps,
        )
        metadata["family"] = hinted_family or "samples"
        return density, metadata
    if batch.mean is not None and batch.std is not None:
        density = _gaussian_density_on_support(
            _to_numpy(batch.mean),
            _to_numpy(batch.std),
            support,
            eps,
        )
        metadata["family"] = "gaussian"
        return density, metadata
    if batch.point is not None:
        density = _point_density_on_support(_to_numpy(batch.point), support, eps)
        metadata["family"] = "point"
        return density, metadata
    raise ValueError("Unsupported PredictiveBatch representation for transport")


def _log_density_at_targets(
    support: np.ndarray,
    density: np.ndarray,
    targets: np.ndarray,
    eps: float,
) -> np.ndarray:
    targ = np.asarray(targets, dtype=float).reshape(-1)
    supp = np.asarray(support, dtype=float).reshape(-1)
    dens = np.asarray(density, dtype=float)
    values = np.empty(targ.shape[0], dtype=float)
    for idx, target in enumerate(targ):
        values[idx] = np.interp(target, supp, dens[idx], left=eps, right=eps)
    return np.log(np.clip(values, eps, None))


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    n = int(scores.shape[0])
    q = min(ceil((n + 1) * (1.0 - alpha)) / max(n, 1), 1.0)
    return float(np.quantile(scores, q, method="higher"))


def _native_interval(
    batch: PredictiveBatch,
    *,
    alpha: float,
    eps: float,
    family_hint: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    levels = (alpha / 2.0, 1.0 - alpha / 2.0)
    family = family_hint
    if family is None and batch.extra is not None:
        raw_family = batch.extra.get("family")
        if isinstance(raw_family, str):
            family = raw_family
    if batch.quantiles is not None and batch.quantile_levels is not None:
        values = _to_numpy(batch.quantiles)
        known = np.asarray(batch.quantile_levels, dtype=float)
        lower = np.stack(
            [np.interp(levels[0], known, values[i]) for i in range(values.shape[0])],
            axis=0,
        )
        upper = np.stack(
            [np.interp(levels[1], known, values[i]) for i in range(values.shape[0])],
            axis=0,
        )
        return lower, upper

    z = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))

    if family in {"gaussian", "point"}:
        mean = _as_1d(batch.mean if batch.mean is not None else batch.point)
        std = np.clip(
            _as_1d(batch.std) if batch.std is not None else np.full_like(mean, eps),
            eps,
            None,
        )
        return mean - z * std, mean + z * std
    if batch.support is not None and batch.density is not None:
        support = _to_numpy(batch.support).reshape(-1)
        density = _to_numpy(batch.density)
        probs = _density_to_probabilities(support, density, eps)
        quantiles = _probability_quantiles(support, probs, levels)
        return quantiles[:, 0], quantiles[:, 1]
    mean = _as_1d(batch.mean if batch.mean is not None else batch.point)
    std = np.clip(
        _as_1d(batch.std) if batch.std is not None else np.full_like(mean, eps),
        eps,
        None,
    )
    return mean - z * std, mean + z * std


def _discrete_tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance for discrete distributions on the same support."""
    return 0.5 * float(np.sum(np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float))))


def _stabilize_target_prior(
    *,
    source_prior: np.ndarray,
    target_prior: np.ndarray,
    selected_probabilities: np.ndarray,
    converged: bool,
    config: "ShiftFactoredTransportConfig",
    evidence_scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    src = np.asarray(source_prior, dtype=float)
    tgt = np.asarray(target_prior, dtype=float)
    probs = np.asarray(selected_probabilities, dtype=float)
    n_classes = max(src.shape[0], 1)
    confidence_floor = 1.0 / float(n_classes)
    if probs.size == 0:
        mean_confidence = confidence_floor
    else:
        mean_confidence = float(np.mean(np.max(probs, axis=1)))
    confidence_scale = (mean_confidence - confidence_floor) / max(
        1.0 - confidence_floor,
        config.eps,
    )
    confidence_scale = float(np.clip(confidence_scale, 0.0, 1.0))
    converged_scale = 1.0
    if config.prior_transport_requires_convergence and not converged:
        converged_scale = 0.0
    elif not converged:
        converged_scale = 0.5
    ev = float(np.clip(evidence_scale, 0.0, 1.0))
    shrink_weight = float(
        np.clip(
            config.prior_transport_strength * confidence_scale * converged_scale * ev,
            0.0,
            1.0,
        )
    )
    shrunk = (1.0 - shrink_weight) * src + shrink_weight * tgt
    ratio = shrunk / np.clip(src, config.eps, None)
    clip = max(float(config.prior_ratio_clip), 1.0)
    ratio = np.clip(ratio, 1.0 / clip, clip)
    stabilized = src * ratio
    stabilized = stabilized / np.clip(stabilized.sum(), config.eps, None)
    metadata = {
        "prior_mean_confidence": mean_confidence,
        "prior_confidence_scale": confidence_scale,
        "prior_shrink_weight": shrink_weight,
        "prior_transport_skipped": float(converged_scale == 0.0),
        "prior_evidence_scale": ev,
    }
    return stabilized, metadata


@dataclass(frozen=True)
class ShiftFactoredTransportConfig:
    """
    Configuration options for shift-factored predictive transport adaptation.

    Parameters
    ----------
    n_support : int
        Number of support grid points to discretize targets.
    support_margin : float
        Fractional boundary margin beyond targets range to pad support grid.
    alpha : float
        Nominal conformal coverage failure rate (1 - coverage probability).
    top_fraction : float
        Fraction of high-confidence predictions selected for EM target prior estimation.
    min_selection_count : int
        Minimum row count to select for EM estimation to prevent sample starvation.
    local_consistency_k : int
        Number of neighbors for local feature consistency weight updates.
    prior_estimation_rows : Optional[int]
        Max target rows to sample for EM prior iterations.
    prior_transport_strength : float
        Shrinkage weight in [0, 1] applied to target prior estimates toward source prior.
    prior_ratio_clip : float
        Maximum allowed density ratio multiplier between target and source priors.
    prior_transport_requires_convergence : bool
        If True, only applies target prior shift adjustments if EM converged.
    prior_transport_min_selected_fraction : Optional[float]
        Prior adjustment is skipped if selected rows ratio is below this floor.
    prior_transport_max_prior_tv : Optional[float]
        Prior adjustment is skipped if TV distance between priors exceeds this bound.
    random_state : Optional[int]
        Random seed.
    enable_alignment : bool
        If True, performs test-time feature/subspace alignment.
    allow_input_alignment_rerun : bool
        If True, re-evaluates student predictions on aligned feature space.
    enable_uncertainty_inflation : bool
        If True, applies temperature-scaling calibration to predicted std dev/variance.
    uncertainty_base_temperature : float
        Base temperature for variance scaling.
    uncertainty_slope : float
        Sensitivity slope mapping feature shifts to temperature adjustments.
    uncertainty_max_temperature : float
        Clipping ceiling for temperature adjustments.
    uncertainty_clip_quantile : Optional[float]
        Outlier Winsorization quantile threshold for variance calculations.
    gaussian_conformal_uses_native_interval : bool
        If True, wraps prediction intervals natively rather than running full CTI.
    eps : float
        Small positive constant.
    """

    n_support: int = 256
    support_margin: float = 0.05
    alpha: float = 0.1
    top_fraction: float = 0.5
    min_selection_count: int = 16
    local_consistency_k: int = 5
    prior_estimation_rows: int | None = None
    prior_transport_strength: float = 0.5
    prior_ratio_clip: float = 2.0
    prior_transport_requires_convergence: bool = True
    prior_transport_min_selected_fraction: float | None = None
    prior_transport_max_prior_tv: float | None = None
    random_state: int | None = 0
    enable_alignment: bool = True
    allow_input_alignment_rerun: bool = False
    enable_uncertainty_inflation: bool = True
    uncertainty_base_temperature: float = 1.0
    uncertainty_slope: float = 0.2
    uncertainty_max_temperature: float = 2.0
    uncertainty_clip_quantile: float | None = 0.05
    gaussian_conformal_uses_native_interval: bool = True
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.n_support < 16:
            raise ValueError("n_support must be at least 16")
        if self.support_margin < 0.0:
            raise ValueError("support_margin must be non-negative")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must lie in (0, 1]")
        if self.min_selection_count < 1:
            raise ValueError("min_selection_count must be positive")
        if self.local_consistency_k < 1:
            raise ValueError("local_consistency_k must be positive")
        if self.prior_estimation_rows is not None and self.prior_estimation_rows < 1:
            raise ValueError("prior_estimation_rows must be positive when set")
        if not 0.0 <= self.prior_transport_strength <= 1.0:
            raise ValueError("prior_transport_strength must lie in [0, 1]")
        if self.prior_ratio_clip < 1.0:
            raise ValueError("prior_ratio_clip must be at least 1")
        if (
            self.prior_transport_min_selected_fraction is not None
            and self.prior_transport_min_selected_fraction < 0.0
        ):
            raise ValueError("prior_transport_min_selected_fraction must be non-negative")
        if (
            self.prior_transport_max_prior_tv is not None
            and not 0.0 <= self.prior_transport_max_prior_tv <= 1.0
        ):
            raise ValueError("prior_transport_max_prior_tv must lie in [0, 1]")
        if self.uncertainty_base_temperature <= 0.0:
            raise ValueError("uncertainty_base_temperature must be positive")
        if self.uncertainty_slope < 0.0:
            raise ValueError("uncertainty_slope must be non-negative")
        if self.uncertainty_max_temperature < self.uncertainty_base_temperature:
            raise ValueError(
                "uncertainty_max_temperature must be at least uncertainty_base_temperature"
            )
        if (
            self.uncertainty_clip_quantile is not None
            and not 0.0 <= self.uncertainty_clip_quantile < 0.5
        ):
            raise ValueError("uncertainty_clip_quantile must lie in [0, 0.5)")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True)
class ShiftFactoredTransportState:
    source_support: np.ndarray
    source_prior: np.ndarray
    source_targets: np.ndarray
    source_inputs: np.ndarray | None = None
    source_representations: np.ndarray | None = None
    last_target_prior: np.ndarray | None = None
    conformal_method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ShiftFactoredPredictiveTransport:
    """
    Model-agnostic transport of predictive laws under target shift.

    References
    ----------
    .. [1] Shift-factored predictive transport (SPT-Reg) manuscript and benchmarks:
       ``torchregress-research`` (see ``neurips_spt_reg/``). Library API only; not a
       battle-tested production default.
    """

    def __init__(self, config: ShiftFactoredTransportConfig | None = None) -> None:
        self.config = config or ShiftFactoredTransportConfig()
        self.state_: ShiftFactoredTransportState | None = None
        self._subspace_aligner: WeightedSubspaceMomentAligner | None = None
        self._shift_calibrator: RepresentationShiftInflator | None = None
        self._conformal_state: dict[str, Any] | None = None

    def fit_source(
        self,
        source_predictions: PredictiveBatch,
        source_targets: np.ndarray | torch.Tensor | Sequence[float],
        *,
        source_inputs: np.ndarray | torch.Tensor | None = None,
        source_representations: np.ndarray | torch.Tensor | None = None,
    ) -> "ShiftFactoredPredictiveTransport":
        del source_predictions  # transport uses source targets and optional source-side features.
        cfg = self.config
        targets = _as_1d(source_targets)
        support = _support_grid_from_targets(
            targets,
            n_support=cfg.n_support,
            support_margin=cfg.support_margin,
        )
        edges = _support_edges(support)
        source_prior = np.histogram(targets, bins=edges)[0].astype(float)
        source_prior = np.clip(source_prior, cfg.eps, None)
        source_prior = source_prior / source_prior.sum()

        source_inputs_np = None if source_inputs is None else _as_2d(source_inputs)
        source_repr_np = None if source_representations is None else _as_2d(source_representations)
        source_features = source_repr_np if source_repr_np is not None else source_inputs_np

        if cfg.enable_alignment and source_features is not None:
            self._subspace_aligner = WeightedSubspaceMomentAligner(
                random_state=cfg.random_state,
            ).fit(source_features, targets)
        if cfg.enable_uncertainty_inflation and source_features is not None:
            self._shift_calibrator = RepresentationShiftInflator(
                base_temperature=cfg.uncertainty_base_temperature,
                slope=cfg.uncertainty_slope,
                max_temperature=cfg.uncertainty_max_temperature,
                random_state=cfg.random_state,
                clip_quantile=cfg.uncertainty_clip_quantile,
            ).fit(source_features)

        self.state_ = ShiftFactoredTransportState(
            source_support=support,
            source_prior=source_prior,
            source_targets=targets,
            source_inputs=source_inputs_np,
            source_representations=source_repr_np,
            metadata={"n_support": int(support.size)},
        )
        return self

    def adapt_unlabeled_target(
        self,
        *,
        target_predictions: PredictiveBatch | None = None,
        target_inputs: np.ndarray | torch.Tensor | None = None,
        target_representations: np.ndarray | torch.Tensor | None = None,
        predictor: SupportsPredictiveBatch | None = None,
    ) -> PredictiveBatch:
        if self.state_ is None:
            raise RuntimeError("call fit_source() before adapt_unlabeled_target()")

        cfg = self.config
        target_inputs_np = None if target_inputs is None else _as_2d(target_inputs)
        target_repr_np = None if target_representations is None else _as_2d(target_representations)

        alignment_applied = False
        effective_predictions = target_predictions
        if (
            predictor is not None
            and self._subspace_aligner is not None
            and self.state_.source_inputs is not None
            and target_inputs_np is not None
            and cfg.enable_alignment
            and cfg.allow_input_alignment_rerun
        ):
            aligned_inputs = self._subspace_aligner.transform(target_inputs_np)
            effective_predictions = predictor.predict_distribution(aligned_inputs)
            alignment_applied = True
        elif (
            predictor is not None and target_inputs_np is not None and effective_predictions is None
        ):
            effective_predictions = predictor.predict_distribution(target_inputs_np)

        if effective_predictions is None:
            raise ValueError("target_predictions are required unless predictor can recompute them")

        density, metadata = _batch_to_support_density(
            effective_predictions,
            self.state_.source_support,
            cfg.eps,
        )
        probabilities = _density_to_probabilities(self.state_.source_support, density, cfg.eps)

        target_features = target_repr_np if target_repr_np is not None else target_inputs_np
        sample_weights = None
        if target_features is not None and target_features.shape[0] == probabilities.shape[0]:
            sample_weights = local_consistency_weights(
                target_features,
                probabilities,
                config=LocalConsistencyConfig(
                    k=min(cfg.local_consistency_k, max(1, probabilities.shape[0] - 1)),
                    random_state=cfg.random_state,
                    eps=cfg.eps,
                ),
            )

        min_count = min(
            max(cfg.min_selection_count, probabilities.shape[1] * 2),
            probabilities.shape[0],
        )
        mask = select_high_confidence(
            probabilities,
            top_fraction=cfg.top_fraction,
            min_count=min_count,
        )

        adapter = PosteriorLabelShiftAdapter(
            source_prior=self.state_.source_prior,
            sample_size=cfg.prior_estimation_rows,
            random_state=cfg.random_state,
            config=LabelShiftEMConfig(eps=cfg.eps),
        )
        estimate = adapter.estimate(
            probabilities[mask],
            sample_weights=sample_weights[mask] if sample_weights is not None else None,
        )
        selected_fraction = float(mask.mean()) if mask.size else 0.0
        prior_tv = _discrete_tv(estimate.source_prior, estimate.target_prior)
        evidence_scale = 1.0
        skip_reason: str | None = None
        if cfg.prior_transport_min_selected_fraction is not None:
            if selected_fraction + cfg.eps < float(cfg.prior_transport_min_selected_fraction):
                evidence_scale = 0.0
                skip_reason = "low_selection_fraction"
        if cfg.prior_transport_max_prior_tv is not None:
            if prior_tv > float(cfg.prior_transport_max_prior_tv):
                evidence_scale = 0.0
                skip_reason = "high_prior_tv"
        stabilized_prior, prior_meta = _stabilize_target_prior(
            source_prior=estimate.source_prior,
            target_prior=estimate.target_prior,
            selected_probabilities=probabilities[mask],
            converged=bool(estimate.converged),
            config=cfg,
            evidence_scale=evidence_scale,
        )
        prior_meta = {
            **prior_meta,
            "prior_selection_fraction": selected_fraction,
            "prior_source_target_tv": prior_tv,
        }
        if skip_reason is not None:
            prior_meta["prior_transport_skip_reason"] = skip_reason
        transport_applied = not np.allclose(
            stabilized_prior,
            estimate.source_prior,
            atol=max(cfg.eps * 10.0, 1.0e-6),
            rtol=0.0,
        )
        transported = adapter.transform(probabilities, target_prior=stabilized_prior)

        if (
            metadata.get("family") == "gaussian"
            and not transport_applied
            and effective_predictions.mean is not None
            and effective_predictions.std is not None
        ):
            mean = _as_1d(effective_predictions.mean).astype(np.float32)
            std = np.clip(_as_1d(effective_predictions.std), cfg.eps, None).astype(np.float32)
            if (
                self._shift_calibrator is not None
                and target_features is not None
                and cfg.enable_uncertainty_inflation
            ):
                std = self._shift_calibrator.calibrate_std(std, target_features).astype(np.float32)
            adapted_density = _gaussian_density_on_support(
                mean,
                std,
                self.state_.source_support,
                cfg.eps,
            )
            transported = _density_to_probabilities(
                self.state_.source_support,
                adapted_density,
                cfg.eps,
            )
        else:
            mean, std = _probability_moments(self.state_.source_support, transported, cfg.eps)
            if (
                self._shift_calibrator is not None
                and target_features is not None
                and cfg.enable_uncertainty_inflation
            ):
                if metadata.get("family") == "gaussian":
                    std = self._shift_calibrator.calibrate_std(std, target_features).astype(
                        np.float32
                    )
                    adapted_density = _gaussian_density_on_support(
                        mean,
                        std,
                        self.state_.source_support,
                        cfg.eps,
                    )
                    transported = _density_to_probabilities(
                        self.state_.source_support,
                        adapted_density,
                        cfg.eps,
                    )
                else:
                    transported = self._shift_calibrator.calibrate_probabilities(
                        transported,
                        target_features,
                    )
                    adapted_density = _probabilities_to_density(
                        self.state_.source_support,
                        transported,
                        cfg.eps,
                    )
                    mean, std = _probability_moments(
                        self.state_.source_support, transported, cfg.eps
                    )
            else:
                adapted_density = _probabilities_to_density(
                    self.state_.source_support,
                    transported,
                    cfg.eps,
                )

        quantile_levels = metadata.get("quantile_levels")
        quantiles = None
        if quantile_levels is not None:
            quantiles = _probability_quantiles(
                self.state_.source_support,
                transported,
                quantile_levels,
            ).astype(np.float32)

        extra = {
            "family": metadata.get("family"),
            "alignment_applied": alignment_applied,
            "transport_applied": transport_applied,
            "selected_rows": int(mask.sum()),
            "target_prior": stabilized_prior.tolist(),
            "target_prior_raw": estimate.target_prior.tolist(),
            "source_prior": estimate.source_prior.tolist(),
            "estimate_converged": bool(estimate.converged),
            "estimate_iterations": int(estimate.iterations),
            **prior_meta,
        }

        self.state_ = replace(
            self.state_,
            last_target_prior=stabilized_prior,
            metadata={**self.state_.metadata, "last_alignment_applied": alignment_applied},
        )
        output_support = self.state_.source_support.astype(np.float32)
        output_density = adapted_density.astype(np.float32)
        if metadata.get("family") == "gaussian":
            output_support = None
            output_density = None
        return PredictiveBatch(
            point=mean,
            mean=mean,
            std=std,
            quantiles=quantiles,
            quantile_levels=quantile_levels,
            support=output_support,
            density=output_density,
            extra=extra,
        )

    def calibrate_target(
        self,
        calibration_predictions: PredictiveBatch,
        calibration_targets: np.ndarray | torch.Tensor | Sequence[float],
        *,
        method: str | None = None,
    ) -> "ShiftFactoredPredictiveTransport":
        if self.state_ is None:
            raise RuntimeError("call fit_source() before calibrate_target()")
        cfg = self.config
        targets = _as_1d(calibration_targets)
        density, metadata = _batch_to_support_density(
            calibration_predictions,
            self.state_.source_support,
            cfg.eps,
        )
        probabilities = _density_to_probabilities(self.state_.source_support, density, cfg.eps)
        mean, std = _probability_moments(self.state_.source_support, probabilities, cfg.eps)

        chosen_method = method
        if chosen_method is None:
            if metadata.get("family") == "quantile":
                chosen_method = "cqr"
            elif metadata.get("family") in {"density", "bar", "samples", "mdn"}:
                chosen_method = "cti"
            elif (
                metadata.get("family") == "gaussian" and cfg.gaussian_conformal_uses_native_interval
            ):
                chosen_method = "interval"
            else:
                chosen_method = "split"

        if chosen_method == "cqr":
            quantiles = calibration_predictions.quantiles
            if quantiles is None:
                quantiles = _probability_quantiles(
                    self.state_.source_support,
                    probabilities,
                    [cfg.alpha / 2.0, 1.0 - cfg.alpha / 2.0],
                )
            q = _to_numpy(quantiles)
            lower = q[:, 0]
            upper = q[:, -1]
            scores = np.maximum(np.maximum(lower - targets, targets - upper), 0.0)
        elif chosen_method == "cti":
            scores = -_log_density_at_targets(
                self.state_.source_support,
                density,
                targets,
                cfg.eps,
            )
        elif chosen_method == "split":
            scores = np.abs(targets - mean) / np.clip(std, cfg.eps, None)
        elif chosen_method == "interval":
            lower, upper = _native_interval(
                calibration_predictions,
                alpha=cfg.alpha,
                eps=cfg.eps,
                family_hint=metadata.get("family"),
            )
            scores = np.maximum(np.maximum(lower - targets, targets - upper), 0.0)
        else:
            raise ValueError(f"Unsupported conformal method: {chosen_method}")

        q_hat = _finite_sample_quantile(np.asarray(scores, dtype=float), cfg.alpha)
        self._conformal_state = {
            "method": chosen_method,
            "alpha": cfg.alpha,
            "q_hat": q_hat,
        }
        self.state_ = replace(self.state_, conformal_method=chosen_method)
        return self

    def predict(
        self,
        *,
        target_predictions: PredictiveBatch | None = None,
        target_inputs: np.ndarray | torch.Tensor | None = None,
        target_representations: np.ndarray | torch.Tensor | None = None,
        predictor: SupportsPredictiveBatch | None = None,
        apply_conformal: bool = True,
    ) -> PredictiveBatch:
        batch = self.adapt_unlabeled_target(
            target_predictions=target_predictions,
            target_inputs=target_inputs,
            target_representations=target_representations,
            predictor=predictor,
        )
        return self.apply_conformal(batch) if apply_conformal else batch

    def apply_conformal(self, batch: PredictiveBatch) -> PredictiveBatch:
        if self._conformal_state is None:
            return batch

        method = str(self._conformal_state["method"])
        q_hat = float(self._conformal_state["q_hat"])
        extra = dict(batch.extra or {})
        if method == "split":
            scale = np.clip(_to_numpy(batch.std), self.config.eps, None)
            center = _to_numpy(batch.mean)
            lower = center - q_hat * scale
            upper = center + q_hat * scale
        elif method == "interval":
            lower, upper = _native_interval(
                batch,
                alpha=self.config.alpha,
                eps=self.config.eps,
                family_hint=extra.get("family") if isinstance(extra.get("family"), str) else None,
            )
            lower = lower - q_hat
            upper = upper + q_hat
        elif method == "cqr":
            quantiles = batch.quantiles
            if quantiles is None:
                probs = _density_to_probabilities(
                    _to_numpy(batch.support),
                    _to_numpy(batch.density),
                    self.config.eps,
                )
                quantiles = _probability_quantiles(
                    _to_numpy(batch.support),
                    probs,
                    [self.config.alpha / 2.0, 1.0 - self.config.alpha / 2.0],
                )
            q = _to_numpy(quantiles)
            lower = q[:, 0] - q_hat
            upper = q[:, -1] + q_hat
        elif method == "cti":
            density = _to_numpy(batch.density)
            support_arr = _to_numpy(batch.support)
            lower = np.empty(density.shape[0], dtype=float)
            upper = np.empty(density.shape[0], dtype=float)
            if support_arr.ndim == 1:
                support = support_arr.reshape(-1)
                neg_log = -np.log(np.clip(density, self.config.eps, None))
                mask = neg_log <= q_hat
                lower.fill(support[0])
                upper.fill(support[-1])
                for idx in range(density.shape[0]):
                    if np.any(mask[idx]):
                        valid = support[mask[idx]]
                        lower[idx] = valid[0]
                        upper[idx] = valid[-1]
            else:
                for idx in range(density.shape[0]):
                    sup = support_arr[idx]
                    neg_log = -np.log(np.clip(density[idx], self.config.eps, None))
                    mask = neg_log <= q_hat
                    if np.any(mask):
                        valid = sup[mask]
                        lower[idx] = valid[0]
                        upper[idx] = valid[-1]
                    else:
                        lower[idx] = sup[0]
                        upper[idx] = sup[-1]
        else:
            raise ValueError(f"Unsupported conformal method: {method}")

        extra["interval_lower"] = lower.tolist()
        extra["interval_upper"] = upper.tolist()
        extra["conformal_method"] = method
        extra["conformal_q_hat"] = q_hat
        return replace(batch, extra=extra)

    def ppi_target_ci(
        self,
        estimand: str,
        labeled_targets: np.ndarray | torch.Tensor | Sequence[float],
        labeled_predictions: PredictiveBatch | np.ndarray | torch.Tensor | Sequence[float],
        unlabeled_predictions: PredictiveBatch | np.ndarray | torch.Tensor | Sequence[float],
        *,
        x_labeled: np.ndarray | torch.Tensor | None = None,
        x_unlabeled: np.ndarray | torch.Tensor | None = None,
        q: float | None = None,
        alpha: float = 0.1,
        n_boot: int = 2000,
        seed: int | None = None,
    ) -> dict[str, Any]:
        y_l = torch.as_tensor(_as_1d(labeled_targets), dtype=torch.float32)
        pred_l = torch.as_tensor(
            self._ppi_vector(labeled_predictions, q=q),
            dtype=torch.float32,
        )
        pred_u = torch.as_tensor(
            self._ppi_vector(unlabeled_predictions, q=q),
            dtype=torch.float32,
        )
        config = PPIConfig(alpha=alpha, n_boot=n_boot, seed=seed)
        if estimand == "mean":
            return ppi_mean_ci(y_l, pred_l, pred_u, config=config)
        if estimand == "quantile":
            if q is None:
                raise ValueError("q is required for quantile PPI")
            return ppi_quantile_ci(
                y_l,
                pred_l,
                pred_u,
                q=q,
                config=config,
            )
        if estimand == "ols":
            if x_labeled is None or x_unlabeled is None:
                raise ValueError("x_labeled and x_unlabeled are required for OLS PPI")
            return ppi_ols_ci(
                torch.as_tensor(_as_2d(x_labeled), dtype=torch.float32),
                y_l,
                torch.as_tensor(_as_2d(x_unlabeled), dtype=torch.float32),
                pred_l,
                pred_u,
                config=PPIConfig(alpha=alpha, n_boot=max(1000, n_boot // 2), seed=seed),
            )
        raise ValueError(f"Unsupported estimand: {estimand}")

    def _ppi_vector(
        self,
        predictions: PredictiveBatch | np.ndarray | torch.Tensor | Sequence[float],
        *,
        q: float | None,
    ) -> np.ndarray:
        if isinstance(predictions, PredictiveBatch):
            if q is None:
                if predictions.mean is not None:
                    return _as_1d(predictions.mean)
                if predictions.point is not None:
                    return _as_1d(predictions.point)
            else:
                if predictions.quantiles is not None and predictions.quantile_levels is not None:
                    levels = np.asarray(predictions.quantile_levels, dtype=float)
                    values = _to_numpy(predictions.quantiles)
                    return np.stack(
                        [np.interp(q, levels, values[idx]) for idx in range(values.shape[0])],
                        axis=0,
                    ).reshape(-1)
                if predictions.support is not None and predictions.density is not None:
                    probs = _density_to_probabilities(
                        _to_numpy(predictions.support),
                        _to_numpy(predictions.density),
                        self.config.eps,
                    )
                    return _probability_quantiles(
                        _to_numpy(predictions.support),
                        probs,
                        [q],
                    ).reshape(-1)
                if predictions.mean is not None and predictions.std is not None:
                    z = float(NormalDist().inv_cdf(float(q)))
                    return _as_1d(predictions.mean) + z * np.clip(
                        _as_1d(predictions.std),
                        self.config.eps,
                        None,
                    )
            if predictions.support is not None and predictions.density is not None:
                probs = _density_to_probabilities(
                    _to_numpy(predictions.support),
                    _to_numpy(predictions.density),
                    self.config.eps,
                )
                mean, _ = _probability_moments(
                    _to_numpy(predictions.support),
                    probs,
                    self.config.eps,
                )
                return mean.reshape(-1)
            raise ValueError("Unsupported PredictiveBatch for PPI summary extraction")
        return _as_1d(predictions)


__all__ = [
    "ShiftFactoredPredictiveTransport",
    "ShiftFactoredTransportConfig",
    "ShiftFactoredTransportState",
]
