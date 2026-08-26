"""Shift-factored predictive transport orchestration for regression test-time adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil
from statistics import NormalDist
from typing import Any, Sequence, cast

import numpy as np
import torch
from torch import Tensor

from torchregress.calibration.shift import RepresentationShiftInflator
from torchregress.inference import PPIConfig, ppi_mean_ci, ppi_ols_ci, ppi_quantile_ci
from torchregress.prediction import PredictiveBatch

from .label_shift import LabelShiftEMConfig, PosteriorLabelShiftAdapter
from .selection import LocalConsistencyConfig, local_consistency_weights, select_high_confidence
from .subspace import WeightedSubspaceMomentAligner

ArrayLike = np.ndarray | Tensor | Sequence[float]


def _to_tensor(x: ArrayLike) -> Tensor:
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)


def _to_numpy(x: ArrayLike) -> "np.ndarray":
    import numpy as np

    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_1d(x: ArrayLike) -> Tensor:
    return _to_tensor(x).reshape(-1)


def _as_2d(x: ArrayLike) -> Tensor:
    arr = _to_tensor(x)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _interp_np(
    x: Tensor,
    xp: Tensor,
    fp: Tensor,
    left: float | Tensor | None = None,
    right: float | Tensor | None = None,
) -> Tensor:
    """torch equivalent of np.interp(x, xp, fp, left, right)."""
    if left is None:
        left = float(fp[0].item()) if fp.numel() > 0 else 0.0
    if right is None:
        right = float(fp[-1].item()) if fp.numel() > 0 else 0.0
    idx = torch.searchsorted(xp, x)
    idx = idx.clamp(1, len(xp) - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    t = ((x - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
    result = y0 + t * (y1 - y0)
    result = torch.where(x < xp[0], torch.tensor(left, dtype=result.dtype), result)
    result = torch.where(x > xp[-1], torch.tensor(right, dtype=result.dtype), result)
    return result


def _to_numpy_if_inputs_were(result: Tensor, *inputs: Any) -> Tensor | np.ndarray:
    if any(isinstance(x, np.ndarray) for x in inputs):
        return result.numpy()
    return result


def _support_grid_from_targets(
    targets: Tensor,
    *,
    n_support: int,
    support_margin: float,
) -> Tensor:
    values = _to_tensor(targets).double().reshape(-1)
    lo = values.min()
    hi = values.max()
    if not torch.isfinite(lo) or not torch.isfinite(hi):
        raise ValueError("source_targets must be finite")
    if hi <= lo:
        hi = lo + 1.0
    width = max(float(hi - lo), 1.0e-6)
    return torch.linspace(
        float(lo) - support_margin * width,
        float(hi) + support_margin * width,
        max(16, int(n_support)),
        dtype=torch.float64,
    )


def _support_edges(support: Tensor) -> Tensor:
    supp = _to_tensor(support).double().reshape(-1)
    if supp.numel() < 2:
        raise ValueError("support must contain at least two points")
    mids = 0.5 * (supp[:-1] + supp[1:])
    left = supp[0] - 0.5 * (supp[1] - supp[0])
    right = supp[-1] + 0.5 * (supp[-1] - supp[-2])
    return torch.cat([left.unsqueeze(0), mids, right.unsqueeze(0)])


def _uniform_dx(support: Tensor) -> float:
    supp = _to_tensor(support).double().reshape(-1)
    return float(max(float(supp.diff().mean()), 1.0e-8))


def _normalize_density(support: Tensor, density: Tensor, eps: float) -> Tensor:
    dens = _to_tensor(density).double().clamp(0.0, None)
    integral = torch.trapezoid(dens, _to_tensor(support).double(), dim=-1)
    integral = integral.clamp(eps, None)
    return dens / integral.unsqueeze(-1)


def _density_to_probabilities(support: Tensor, density: Tensor, eps: float) -> Tensor:
    dx = _uniform_dx(support)
    probs = _to_tensor(density).double().clamp(0.0, None) * dx
    row_sum = probs.sum(dim=-1, keepdim=True)
    degenerate = row_sum.reshape(-1) < eps
    n_cols = probs.shape[-1]
    if degenerate.any():
        probs[degenerate] = 1.0 / n_cols
        row_sum[degenerate] = 1.0
    result = probs / row_sum.clamp(eps, None)
    return cast(Tensor, _to_numpy_if_inputs_were(result, support, density))


def _probabilities_to_density(support: Tensor, probabilities: ArrayLike, eps: float) -> Tensor:
    dx = _uniform_dx(support)
    density = _to_tensor(probabilities).double() / max(dx, eps)
    return _normalize_density(support, density, eps)


def _gaussian_density_on_support(mean: Tensor, std: Tensor, support: Tensor, eps: float) -> Tensor:
    mu = _to_tensor(mean).double().reshape(-1, 1)
    sigma = _to_tensor(std).double().reshape(-1, 1).clamp(eps, None)
    supp = _to_tensor(support).double().reshape(1, -1)
    z = (supp - mu) / sigma
    dens = torch.exp(-0.5 * z**2) / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))
    result = _normalize_density(support, dens, eps)
    return cast(Tensor, _to_numpy_if_inputs_were(result, mean, std, support))


def _resample_density(
    support_in: Tensor, density_in: Tensor, support_out: Tensor, eps: float
) -> Tensor:
    supp_out = _to_tensor(support_out).double().reshape(-1)
    supp_in = _to_tensor(support_in).double()
    dens_in = _to_tensor(density_in).double()
    out = torch.empty(dens_in.shape[0], supp_out.numel(), dtype=torch.float64)
    if supp_in.ndim == 1:
        for idx in range(dens_in.shape[0]):
            out[idx] = _interp_np(supp_out, supp_in, dens_in[idx], left=0.0, right=0.0)
    else:
        for idx in range(dens_in.shape[0]):
            out[idx] = _interp_np(supp_out, supp_in[idx], dens_in[idx], left=0.0, right=0.0)
    result = _normalize_density(supp_out, out, eps)
    return cast(Tensor, _to_numpy_if_inputs_were(result, density_in))


def _probability_moments(
    support: Tensor, probabilities: ArrayLike, eps: float
) -> tuple[Tensor, Tensor]:
    supp = _to_tensor(support).double().reshape(-1)
    probs = _to_tensor(probabilities).double()
    mean = probs @ supp
    second = probs @ (supp**2)
    var = (second - mean**2).clamp(eps, None)
    return mean.float(), var.sqrt().float()


def _probability_quantiles(
    support: Tensor, probabilities: ArrayLike, quantile_levels: Sequence[float]
) -> Tensor:
    supp = _to_tensor(support).double().reshape(-1)
    probs = _to_tensor(probabilities).double()
    cdf = probs.cumsum(dim=-1)
    cdf[:, -1] = 1.0
    levels_t = torch.tensor(list(quantile_levels), dtype=torch.float64)
    out = torch.empty(probs.shape[0], len(quantile_levels), dtype=torch.float64)
    for idx in range(probs.shape[0]):
        out[idx] = _interp_np(levels_t, cdf[idx], supp, left=supp[0], right=supp[-1])
    return out


def _point_density_on_support(point: Tensor, support: Tensor, eps: float) -> Tensor:
    pts = _to_tensor(point).double().reshape(-1)
    dx = _uniform_dx(support)
    std = torch.full_like(pts, fill_value=max(2.0 * dx, eps))
    result = _gaussian_density_on_support(pts, std, support, eps)
    return cast(Tensor, _to_numpy_if_inputs_were(result, point, support))


def _batch_to_support_density(
    batch: PredictiveBatch,
    support: Tensor,
    eps: float,
) -> tuple[Tensor, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    hinted_family = None
    if batch.extra is not None:
        raw_family = batch.extra.get("family")
        if isinstance(raw_family, str):
            hinted_family = raw_family
    if batch.support is not None and batch.density is not None:
        density = _resample_density(
            _to_tensor(batch.support),
            _to_tensor(batch.density),
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
            _to_tensor(converted.support),
            _to_tensor(converted.density),
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
            _to_tensor(converted.support),
            _to_tensor(converted.density),
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
            _to_tensor(converted.support),
            _to_tensor(converted.density),
            support,
            eps,
        )
        metadata["family"] = hinted_family or "samples"
        return density, metadata
    if batch.mean is not None and batch.std is not None:
        density = _gaussian_density_on_support(
            _to_tensor(batch.mean),
            _to_tensor(batch.std),
            support,
            eps,
        )
        metadata["family"] = "gaussian"
        return density, metadata
    if batch.point is not None:
        density = _point_density_on_support(_to_tensor(batch.point), support, eps)
        metadata["family"] = "point"
        return density, metadata
    raise ValueError("Unsupported PredictiveBatch representation for transport")


def _log_density_at_targets(
    support: Tensor, density: Tensor, targets: Tensor, eps: float
) -> Tensor:
    targ = _to_tensor(targets).double().reshape(-1)
    supp = _to_tensor(support).double().reshape(-1)
    dens = _to_tensor(density).double()
    values = torch.empty(targ.shape[0], dtype=torch.float64)
    for idx, target in enumerate(targ):
        values[idx] = _interp_np(target.unsqueeze(0), supp, dens[idx], left=eps, right=eps)
    log_dens = values.clamp(eps, None).log()
    return cast(Tensor, _to_numpy_if_inputs_were(log_dens, support, density, targets))


def _finite_sample_quantile(scores: Tensor, alpha: float) -> float:
    s = _to_tensor(scores).double()
    n = int(s.shape[0])
    q = min(ceil((n + 1) * (1.0 - alpha)) / max(n, 1), 1.0)
    sorted_scores = s.sort().values
    idx = max(0, min(int(ceil(q * (n - 1))), n - 1))
    return float(sorted_scores[idx].item())


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
        values = _to_tensor(batch.quantiles)
        known = torch.tensor(batch.quantile_levels, dtype=torch.float64)
        lower = torch.stack(
            [
                _interp_np(torch.tensor([levels[0]]), known, values[i])[0]
                for i in range(values.shape[0])
            ],
            dim=0,
        )
        upper = torch.stack(
            [
                _interp_np(torch.tensor([levels[1]]), known, values[i])[0]
                for i in range(values.shape[0])
            ],
            dim=0,
        )
        return lower.numpy(), upper.numpy()

    z = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))

    if family in {"gaussian", "point"}:
        mean = _as_1d(cast(ArrayLike, batch.mean if batch.mean is not None else batch.point))
        std = (
            _as_1d(cast(ArrayLike, batch.std)).clamp(eps, None)
            if batch.std is not None
            else torch.full_like(mean, eps)
        )
        lower, upper = mean - z * std, mean + z * std
        return lower.numpy(), upper.numpy()
    if batch.support is not None and batch.density is not None:
        support = _to_tensor(batch.support).reshape(-1)
        density = _to_tensor(batch.density)
        probs = _density_to_probabilities(support, density, eps)
        quantiles = _probability_quantiles(support, probs, levels)
        return quantiles[:, 0].numpy(), quantiles[:, 1].numpy()
    mean = _as_1d(cast(ArrayLike, batch.mean if batch.mean is not None else batch.point))
    std = (
        _as_1d(cast(ArrayLike, batch.std)).clamp(eps, None)
        if batch.std is not None
        else torch.full_like(mean, eps)
    )
    lower, upper = mean - z * std, mean + z * std
    return lower.numpy(), upper.numpy()


def _discrete_tv(p: ArrayLike, q: ArrayLike) -> float:
    return 0.5 * float(torch.abs(_to_tensor(p).double() - _to_tensor(q).double()).sum().item())


def _stabilize_target_prior(
    *,
    source_prior: ArrayLike,
    target_prior: ArrayLike,
    selected_probabilities: Tensor,
    converged: bool,
    config: "ShiftFactoredTransportConfig",
    evidence_scale: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    src = _to_tensor(source_prior).double()
    tgt = _to_tensor(target_prior).double()
    probs = _to_tensor(selected_probabilities).double()
    n_classes = max(src.shape[0], 1)
    confidence_floor = 1.0 / float(n_classes)
    if probs.numel() == 0:
        mean_confidence = confidence_floor
    else:
        mean_confidence = float(probs.max(dim=1).values.mean().item())
    confidence_scale = (mean_confidence - confidence_floor) / max(
        1.0 - confidence_floor,
        config.eps,
    )
    confidence_scale = float(torch.clamp(torch.tensor(confidence_scale), 0.0, 1.0))
    converged_scale = 1.0
    if config.prior_transport_requires_convergence and not converged:
        converged_scale = 0.0
    elif not converged:
        converged_scale = 0.5
    ev = float(torch.clamp(torch.tensor(evidence_scale), 0.0, 1.0))
    shrink_weight = float(
        torch.clamp(
            torch.tensor(config.prior_transport_strength * confidence_scale * converged_scale * ev),
            0.0,
            1.0,
        )
    )
    shrunk = (1.0 - shrink_weight) * src + shrink_weight * tgt
    ratio = shrunk / src.clamp(config.eps, None)
    clip = max(float(config.prior_ratio_clip), 1.0)
    ratio = ratio.clamp(1.0 / clip, clip)
    stabilized = src * ratio
    stabilized = stabilized / stabilized.sum().clamp(config.eps, None)
    metadata: dict[str, float] = {
        "prior_mean_confidence": mean_confidence,
        "prior_confidence_scale": confidence_scale,
        "prior_shrink_weight": shrink_weight,
        "prior_transport_skipped": float(converged_scale == 0.0),
        "prior_evidence_scale": ev,
    }
    return stabilized, metadata


@dataclass(frozen=True)
class ShiftFactoredTransportConfig:
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
    source_support: Tensor
    source_prior: np.ndarray
    source_targets: Tensor
    source_inputs: Tensor | None = None
    source_representations: Tensor | None = None
    last_target_prior: Tensor | None = None
    conformal_method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ShiftFactoredPredictiveTransport:
    def __init__(self, config: ShiftFactoredTransportConfig | None = None) -> None:
        self.config = config or ShiftFactoredTransportConfig()
        self.state_: ShiftFactoredTransportState | None = None
        self._subspace_aligner: WeightedSubspaceMomentAligner | None = None
        self._shift_calibrator: RepresentationShiftInflator | None = None
        self._conformal_state: dict[str, Any] | None = None

    def fit_source(
        self,
        source_predictions: PredictiveBatch,
        source_targets: Tensor | Sequence[float],
        *,
        source_inputs: Tensor | None = None,
        source_representations: Tensor | None = None,
    ) -> "ShiftFactoredPredictiveTransport":
        del source_predictions
        cfg = self.config
        targets = _as_1d(source_targets)
        support = _support_grid_from_targets(
            targets,
            n_support=cfg.n_support,
            support_margin=cfg.support_margin,
        )
        edges = _support_edges(support)
        edges_cpu = edges.detach().cpu().to(dtype=targets.dtype)
        source_prior = torch.histogram(targets, bins=edges_cpu)[0].double()
        source_prior = source_prior.clamp(cfg.eps, None)
        source_prior = source_prior / source_prior.sum()

        source_inputs_t = None if source_inputs is None else _as_2d(source_inputs)
        source_repr_t = None if source_representations is None else _as_2d(source_representations)
        source_features = source_repr_t if source_repr_t is not None else source_inputs_t

        if cfg.enable_alignment and source_features is not None:
            self._subspace_aligner = WeightedSubspaceMomentAligner(
                random_state=cfg.random_state,
            ).fit(cast(np.ndarray, source_features), cast(np.ndarray, targets))
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
            source_prior=source_prior.numpy(),
            source_targets=targets,
            source_inputs=source_inputs_t,
            source_representations=source_repr_t,
            metadata={"n_support": int(support.numel())},
        )
        return self

    def adapt_unlabeled_target(
        self,
        *,
        target_predictions: PredictiveBatch | None = None,
        target_inputs: Tensor | None = None,
        target_representations: Tensor | None = None,
        predictor: Any | None = None,
    ) -> PredictiveBatch:
        if self.state_ is None:
            raise RuntimeError("call fit_source() before adapt_unlabeled_target()")

        cfg = self.config
        target_inputs_t = None if target_inputs is None else _as_2d(target_inputs)
        target_repr_t = None if target_representations is None else _as_2d(target_representations)

        alignment_applied = False
        effective_predictions = target_predictions
        if (
            predictor is not None
            and self._subspace_aligner is not None
            and self.state_.source_inputs is not None
            and target_inputs_t is not None
            and cfg.enable_alignment
            and cfg.allow_input_alignment_rerun
        ):
            aligned_inputs = self._subspace_aligner.transform(cast(np.ndarray, target_inputs_t))
            effective_predictions = predictor.predict_distribution(aligned_inputs)
            alignment_applied = True
        elif (
            predictor is not None and target_inputs_t is not None and effective_predictions is None
        ):
            effective_predictions = predictor.predict_distribution(target_inputs_t)

        if effective_predictions is None:
            raise ValueError("target_predictions are required unless predictor can recompute them")

        density, metadata = _batch_to_support_density(
            effective_predictions,
            self.state_.source_support,
            cfg.eps,
        )
        probabilities = _density_to_probabilities(self.state_.source_support, density, cfg.eps)

        target_features = target_repr_t if target_repr_t is not None else target_inputs_t
        sample_weights = None
        if target_features is not None and target_features.shape[0] == probabilities.shape[0]:
            sample_weights = local_consistency_weights(
                cast(np.ndarray, target_features),
                cast(np.ndarray, probabilities),
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
        mask = torch.as_tensor(
            select_high_confidence(
                cast(np.ndarray, probabilities),
                top_fraction=cfg.top_fraction,
                min_count=min_count,
            )
        )

        adapter = PosteriorLabelShiftAdapter(
            source_prior=self.state_.source_prior,
            sample_size=cfg.prior_estimation_rows,
            random_state=cfg.random_state,
            config=LabelShiftEMConfig(eps=cfg.eps),
        )
        estimate = adapter.estimate(
            cast(np.ndarray, probabilities[mask]),
            sample_weights=sample_weights[mask] if sample_weights is not None else None,
        )
        selected_fraction = float(mask.float().mean().item()) if mask.numel() else 0.0
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
        transport_applied = not torch.allclose(
            stabilized_prior.double(),
            _to_tensor(estimate.source_prior).double(),
            atol=max(cfg.eps * 10.0, 1.0e-6),
            rtol=0.0,
        )
        transported = adapter.transform(
            cast(np.ndarray, probabilities),
            target_prior=cast(np.ndarray, stabilized_prior),
        )

        if (
            metadata.get("family") == "gaussian"
            and not transport_applied
            and effective_predictions.mean is not None
            and effective_predictions.std is not None
        ):
            mean = _as_1d(effective_predictions.mean).float()
            std = _as_1d(effective_predictions.std).clamp(cfg.eps, None).float()
            if (
                self._shift_calibrator is not None
                and target_features is not None
                and cfg.enable_uncertainty_inflation
            ):
                std = cast(
                    Tensor, self._shift_calibrator.calibrate_std(std, target_features)
                ).float()
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
                    std = cast(
                        Tensor, self._shift_calibrator.calibrate_std(std, target_features)
                    ).float()
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
            ).float()

        extra: dict[str, Any] = {
            "family": metadata.get("family"),
            "alignment_applied": alignment_applied,
            "transport_applied": transport_applied,
            "selected_rows": int(mask.sum().item()),
            "target_prior": stabilized_prior.tolist(),
            "target_prior_raw": _to_tensor(estimate.target_prior).tolist(),
            "source_prior": _to_tensor(estimate.source_prior).tolist(),
            "estimate_converged": bool(estimate.converged),
            "estimate_iterations": int(estimate.iterations),
            **prior_meta,
        }

        self.state_ = replace(
            self.state_,
            last_target_prior=stabilized_prior,
            metadata={**self.state_.metadata, "last_alignment_applied": alignment_applied},
        )
        output_support = self.state_.source_support.float()
        output_density = adapted_density.float()
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
        calibration_targets: Tensor | Sequence[float],
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
            q = _to_tensor(quantiles)
            lower = q[:, 0]
            upper = q[:, -1]
            scores = torch.maximum(
                torch.maximum(lower - targets, targets - upper), torch.tensor(0.0)
            )
        elif chosen_method == "cti":
            scores = -_log_density_at_targets(
                self.state_.source_support,
                density,
                targets,
                cfg.eps,
            )
        elif chosen_method == "split":
            scores = (targets - mean).abs() / std.clamp(cfg.eps, None)
        elif chosen_method == "interval":
            lower, upper = _native_interval(
                calibration_predictions,
                alpha=cfg.alpha,
                eps=cfg.eps,
                family_hint=metadata.get("family"),
            )
            lower, upper = _to_tensor(lower), _to_tensor(upper)
            scores = torch.maximum(
                torch.maximum(lower - targets, targets - upper), torch.tensor(0.0)
            )
        else:
            raise ValueError(f"Unsupported conformal method: {chosen_method}")

        q_hat = _finite_sample_quantile(scores.double(), cfg.alpha)
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
        target_inputs: Tensor | None = None,
        target_representations: Tensor | None = None,
        predictor: Any | None = None,
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
            scale = _to_tensor(cast(ArrayLike, batch.std)).clamp(self.config.eps, None)
            center = _to_tensor(cast(ArrayLike, batch.mean))
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
                    _to_tensor(cast(ArrayLike, batch.support)),
                    _to_tensor(cast(ArrayLike, batch.density)),
                    self.config.eps,
                )
                quantiles = _probability_quantiles(
                    _to_tensor(cast(ArrayLike, batch.support)),
                    probs,
                    [self.config.alpha / 2.0, 1.0 - self.config.alpha / 2.0],
                )
            q = _to_tensor(quantiles)
            lower = q[:, 0] - q_hat
            upper = q[:, -1] + q_hat
        elif method == "cti":
            density = _to_tensor(cast(ArrayLike, batch.density))
            support_arr = _to_tensor(cast(ArrayLike, batch.support))
            lower = torch.empty(density.shape[0], dtype=torch.float64)
            upper = torch.empty(density.shape[0], dtype=torch.float64)
            if support_arr.ndim == 1:
                support = support_arr.reshape(-1)
                neg_log = -(density.clamp(self.config.eps, None)).log()
                mask = neg_log <= q_hat
                lower.fill_(support[0].item())
                upper.fill_(support[-1].item())
                for idx in range(density.shape[0]):
                    if mask[idx].any():
                        valid = support[mask[idx]]
                        lo, hi = valid[0], valid[-1]
                        lower[idx] = lo
                        upper[idx] = hi
            else:
                for idx in range(density.shape[0]):
                    sup = support_arr[idx]
                    neg_log = -(density[idx].clamp(self.config.eps, None)).log()
                    mask = neg_log <= q_hat
                    if mask.any():
                        valid = sup[mask]
                        lower[idx] = valid[0]
                        upper[idx] = valid[-1]
                    else:
                        lower[idx] = sup[0]
                        upper[idx] = sup[-1]
        else:
            raise ValueError(f"Unsupported conformal method: {method}")

        extra["interval_lower"] = [float(v) for v in lower]
        extra["interval_upper"] = [float(v) for v in upper]
        extra["conformal_method"] = method
        extra["conformal_q_hat"] = q_hat
        return replace(batch, extra=extra)

    def ppi_target_ci(
        self,
        estimand: str,
        labeled_targets: Tensor | Sequence[float],
        labeled_predictions: PredictiveBatch | Tensor | Sequence[float],
        unlabeled_predictions: PredictiveBatch | Tensor | Sequence[float],
        *,
        x_labeled: Tensor | None = None,
        x_unlabeled: Tensor | None = None,
        q: float | None = None,
        alpha: float = 0.1,
        n_boot: int = 2000,
        seed: int | None = None,
    ) -> dict[str, Any]:
        y_l = _as_1d(labeled_targets).float()
        pred_l = _as_1d(self._ppi_vector(labeled_predictions, q=q)).float()
        pred_u = _as_1d(self._ppi_vector(unlabeled_predictions, q=q)).float()
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
                _as_2d(x_labeled).float(),
                y_l,
                _as_2d(x_unlabeled).float(),
                pred_l,
                pred_u,
                config=PPIConfig(alpha=alpha, n_boot=max(1000, n_boot // 2), seed=seed),
            )
        raise ValueError(f"Unsupported estimand: {estimand}")

    def _ppi_vector(
        self,
        predictions: PredictiveBatch | Tensor | Sequence[float],
        *,
        q: float | None,
    ) -> Tensor:
        if isinstance(predictions, PredictiveBatch):
            if q is None:
                if predictions.mean is not None:
                    return _as_1d(predictions.mean)
                if predictions.point is not None:
                    return _as_1d(predictions.point)
            else:
                if predictions.quantiles is not None and predictions.quantile_levels is not None:
                    levels = torch.tensor(predictions.quantile_levels, dtype=torch.float64)
                    values = _to_tensor(predictions.quantiles)
                    return torch.stack(
                        [
                            _interp_np(torch.tensor([q]), levels, values[idx])[0]
                            for idx in range(values.shape[0])
                        ],
                        dim=0,
                    ).reshape(-1)
                if predictions.support is not None and predictions.density is not None:
                    probs = _density_to_probabilities(
                        _to_tensor(predictions.support),
                        _to_tensor(predictions.density),
                        self.config.eps,
                    )
                    return _probability_quantiles(
                        _to_tensor(predictions.support),
                        probs,
                        [q],
                    ).reshape(-1)
                if predictions.mean is not None and predictions.std is not None:
                    z = float(NormalDist().inv_cdf(float(q)))
                    return _as_1d(predictions.mean) + z * _as_1d(predictions.std).clamp(
                        self.config.eps, None
                    )
            if predictions.support is not None and predictions.density is not None:
                probs = _density_to_probabilities(
                    _to_tensor(predictions.support),
                    _to_tensor(predictions.density),
                    self.config.eps,
                )
                mean, _ = _probability_moments(
                    _to_tensor(predictions.support),
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
