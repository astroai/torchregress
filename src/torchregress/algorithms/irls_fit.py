"""Solver-agnostic iteratively reweighted fitting."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Generic, Literal, Protocol, TypeVar, cast, runtime_checkable

import torch

from ..losses.robust import _barron_elementwise

StateT = TypeVar("StateT")
RobustWeightName = Literal["huber", "pseudo_huber", "tukey", "cauchy", "barron"]
IRLSTermination = Literal["converged", "max_iterations", "empty", "breakdown", "non_finite"]
ScaleEstimator = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
WeightFunction = Callable[..., torch.Tensor]


class IRLSBreakdownError(RuntimeError):
    """Signal that a caller-owned weighted solve cannot produce a valid refit."""


@runtime_checkable
class IRLSProblem(Protocol[StateT]):
    """Minimal callback protocol for solver-agnostic IRLS orchestration."""

    def residuals(self, state: StateT) -> torch.Tensor:
        """Return residuals with samples on the first axis."""

    def weighted_fit(self, state: StateT, precision: torch.Tensor) -> StateT:
        """Return a refitted state using the supplied effective precision."""


@dataclass(frozen=True)
class IRLSCallbacks(Generic[StateT]):
    """Functional adapter for :class:`IRLSProblem` callbacks."""

    residuals_fn: Callable[[StateT], torch.Tensor]
    weighted_fit_fn: Callable[[StateT, torch.Tensor], StateT]
    covariance_fn: Callable[[StateT], torch.Tensor | None] | None = None
    whiten_fn: Callable[[StateT, torch.Tensor], torch.Tensor] | None = None
    state_step_fn: Callable[[StateT, StateT], float | torch.Tensor] | None = None

    def residuals(self, state: StateT) -> torch.Tensor:
        return self.residuals_fn(state)

    def weighted_fit(self, state: StateT, precision: torch.Tensor) -> StateT:
        return self.weighted_fit_fn(state, precision)


@dataclass(frozen=True)
class IRLSContinuationStage:
    """One robust-loss stage in a continuation schedule."""

    max_iter: int
    weight: RobustWeightName | WeightFunction = "huber"
    weight_params: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("IRLSContinuationStage.max_iter must be positive")
        object.__setattr__(self, "weight_params", MappingProxyType(dict(self.weight_params)))


@dataclass(frozen=True)
class IRLSFitConfig:
    """Configuration for generic refit-capable IRLS."""

    max_iter: int = 50
    weight: RobustWeightName | WeightFunction = "huber"
    weight_params: Mapping[str, float] = field(default_factory=dict)
    scale: Literal["mad", "std", "rms", "none"] | float | torch.Tensor | ScaleEstimator = "mad"
    min_scale: float = 1e-8
    base_precision: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    continuation: tuple[IRLSContinuationStage, ...] = ()
    objective_tol: float | None = 1e-8
    weight_tol: float | None = 1e-6
    parameter_tol: float | None = None
    min_iter: int = 2
    min_effective_samples: float = 1.0
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.min_iter <= 0:
            raise ValueError("min_iter must be positive")
        if self.min_scale <= 0 or self.eps <= 0:
            raise ValueError("min_scale and eps must be positive")
        if self.min_effective_samples < 0:
            raise ValueError("min_effective_samples must be non-negative")
        for name, value in (
            ("objective_tol", self.objective_tol),
            ("weight_tol", self.weight_tol),
            ("parameter_tol", self.parameter_tol),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")
        object.__setattr__(self, "weight_params", MappingProxyType(dict(self.weight_params)))
        object.__setattr__(self, "continuation", tuple(self.continuation))


@dataclass(frozen=True)
class IRLSIteration:
    """Immutable diagnostics for one robust weight update and refit."""

    iteration: int
    stage: int
    weight: str
    objective: float
    objective_change: float
    weight_change: float
    parameter_step: float | None
    effective_sample_size: float
    active_samples: int
    scale: torch.Tensor
    min_weight: float
    max_weight: float
    residuals_finite: bool
    weights_finite: bool
    objective_finite: bool
    state_finite: bool
    parameter_step_finite: bool

    @property
    def finite(self) -> bool:
        return (
            self.residuals_finite
            and self.weights_finite
            and self.objective_finite
            and self.state_finite
            and self.parameter_step_finite
        )


@dataclass(frozen=True)
class IRLSFitResult(Generic[StateT]):
    """Immutable generic IRLS result and convergence history."""

    state: StateT
    weights: torch.Tensor
    precision: torch.Tensor
    residuals: torch.Tensor
    scale: torch.Tensor
    objective: float
    effective_sample_size: float
    converged: bool
    termination: IRLSTermination
    iterations: int
    history: tuple[IRLSIteration, ...]
    message: str


@dataclass(frozen=True)
class _WeightUpdate:
    weights: torch.Tensor
    precision: torch.Tensor
    scale: torch.Tensor
    objective: torch.Tensor
    effective_sample_size: torch.Tensor
    active_samples: int


def robust_weights(
    standardized_residuals: torch.Tensor,
    weight: RobustWeightName | WeightFunction = "huber",
    weight_params: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Return IRLS multipliers for standardized scalar residual magnitudes."""
    params = dict(weight_params or {})
    value = standardized_residuals.abs()
    if callable(weight):
        result = weight(value, **params)
    elif weight == "huber":
        delta = _positive_parameter(params, "delta", 1.345)
        result = torch.where(value <= delta, torch.ones_like(value), delta / value.clamp_min(1e-30))
    elif weight == "pseudo_huber":
        delta = _positive_parameter(params, "delta", 1.345)
        result = torch.rsqrt(1.0 + (value / delta).square())
    elif weight == "tukey":
        c = _positive_parameter(params, "c", 4.685)
        inside = (1.0 - (value / c).square()).square()
        result = torch.where(value <= c, inside, torch.zeros_like(value))
    elif weight == "cauchy":
        c = _positive_parameter(params, "c", 2.3849)
        result = torch.reciprocal(1.0 + (value / c).square())
    elif weight == "barron":
        alpha = float(params.get("alpha", 0.0))
        if not math.isfinite(alpha) or alpha > 2.0:
            raise ValueError("Barron IRLS weights require finite alpha <= 2")
        scale = _positive_parameter(params, "scale", 1.0)
        alpha_tensor = torch.as_tensor(alpha, dtype=value.dtype, device=value.device)
        beta = torch.sqrt((alpha_tensor - 2.0).square() + 1e-8)
        result = (1.0 + (value / scale).square() / beta).pow(alpha_tensor / 2.0 - 1.0)
    else:
        raise ValueError(f"Unknown robust weight function: {weight!r}")
    if result.shape != value.shape:
        raise ValueError(
            f"Robust weight function returned shape {result.shape}; expected {value.shape}"
        )
    if result.device != value.device or result.dtype != value.dtype:
        raise ValueError("Robust weight functions must preserve dtype and device")
    return result


def iteratively_reweighted_fit(
    problem: IRLSProblem[StateT] | IRLSCallbacks[StateT],
    initial_state: StateT,
    config: IRLSFitConfig | None = None,
) -> IRLSFitResult[StateT]:
    """Run solver-agnostic IRLS and refit after every robust weight update."""
    cfg = config or IRLSFitConfig()
    state = initial_state
    residuals = _residuals(problem, state)
    if residuals.ndim == 0:
        raise ValueError("residuals must have a sample axis")
    mask = _validated_mask(cfg.mask, residuals)
    if residuals.shape[0] == 0 or not bool(_sample_mask(mask, residuals).any().item()):
        return _empty_result(state, residuals, cfg, mask)
    stages = cfg.continuation or (
        IRLSContinuationStage(cfg.max_iter, cfg.weight, cfg.weight_params),
    )
    history: list[IRLSIteration] = []
    last_weights = torch.ones(residuals.shape[0], dtype=residuals.dtype, device=residuals.device)
    last_scale = torch.ones((), dtype=residuals.dtype, device=residuals.device)
    last_precision = _initial_precision(residuals, cfg)
    last_objective = math.inf
    last_ess = 0.0
    global_iteration = 0

    for stage_index, stage in enumerate(stages):
        previous_weights: torch.Tensor | None = None
        previous_objective: float | None = None
        for stage_iteration in range(stage.max_iter):
            if not bool(torch.isfinite(residuals).all().item()):
                return _result(
                    state,
                    last_weights,
                    last_precision,
                    residuals,
                    last_scale,
                    float("nan"),
                    last_ess,
                    False,
                    "non_finite",
                    history,
                    "Residuals are non-finite",
                )
            try:
                update = _weight_update(problem, state, residuals, cfg, mask, stage)
            except IRLSBreakdownError as error:
                return _result(
                    state,
                    last_weights,
                    last_precision,
                    residuals,
                    last_scale,
                    last_objective,
                    last_ess,
                    False,
                    "breakdown",
                    history,
                    str(error),
                )
            update_finite = (
                bool(torch.isfinite(update.weights).all().item())
                and bool(torch.isfinite(update.precision).all().item())
                and bool(torch.isfinite(update.scale).all().item())
                and bool(torch.isfinite(update.objective).item())
                and bool(torch.isfinite(update.effective_sample_size).item())
            )
            if not update_finite:
                return _result(
                    state,
                    update.weights,
                    update.precision,
                    residuals,
                    update.scale,
                    float("nan"),
                    float("nan"),
                    False,
                    "non_finite",
                    history,
                    "Robust weights, precision, scale, objective, or ESS are non-finite",
                )
            if bool((update.weights < 0).any().item()):
                raise ValueError("Robust weights must be non-negative")
            ess = float(update.effective_sample_size.item())
            if ess < cfg.min_effective_samples:
                return _result(
                    state,
                    update.weights,
                    update.precision,
                    residuals,
                    update.scale,
                    float(update.objective.item()),
                    ess,
                    False,
                    "breakdown",
                    history,
                    "Effective sample size fell below min_effective_samples",
                )
            try:
                next_state = problem.weighted_fit(state, update.precision.detach())
            except IRLSBreakdownError as error:
                return _result(
                    state,
                    update.weights,
                    update.precision,
                    residuals,
                    update.scale,
                    float(update.objective.item()),
                    ess,
                    False,
                    "breakdown",
                    history,
                    str(error),
                )
            next_residuals = _residuals(problem, next_state)
            if next_residuals.shape != residuals.shape:
                raise ValueError(
                    "residual shape changed after refit: "
                    f"{residuals.shape} -> {next_residuals.shape}"
                )
            state_finite = _state_is_finite(next_state)
            residuals_finite = bool(torch.isfinite(next_residuals).all().item())
            parameter_step = _parameter_step(problem, state, next_state, cfg.eps)
            try:
                fitted_update = _weight_update(
                    problem, next_state, next_residuals, cfg, mask, stage
                )
                objective = float(fitted_update.objective.item())
                objective_finite = math.isfinite(objective)
            except IRLSBreakdownError:
                objective = float("nan")
                objective_finite = False
            weight_change = _relative_change(update.weights, previous_weights, cfg.eps)
            objective_change = _scalar_relative_change(objective, previous_objective, cfg.eps)
            weight_name = stage.weight if isinstance(stage.weight, str) else "callable"
            record = IRLSIteration(
                iteration=global_iteration,
                stage=stage_index,
                weight=weight_name,
                objective=objective,
                objective_change=objective_change,
                weight_change=weight_change,
                parameter_step=parameter_step,
                effective_sample_size=ess,
                active_samples=update.active_samples,
                scale=update.scale.detach().clone(),
                min_weight=float(update.weights.min().item()),
                max_weight=float(update.weights.max().item()),
                residuals_finite=residuals_finite,
                weights_finite=True,
                objective_finite=objective_finite,
                state_finite=state_finite,
                parameter_step_finite=parameter_step is None or math.isfinite(parameter_step),
            )
            history.append(record)
            state = next_state
            residuals = next_residuals
            last_weights = update.weights
            last_precision = update.precision
            last_scale = update.scale
            last_objective = objective
            last_ess = ess
            global_iteration += 1
            if not record.finite:
                return _result(
                    state,
                    last_weights,
                    last_precision,
                    residuals,
                    last_scale,
                    last_objective,
                    last_ess,
                    False,
                    "non_finite",
                    history,
                    "Non-finite residual, objective, or state detected after refit",
                )
            converged = _converged(
                cfg,
                stage_iteration + 1,
                objective_change,
                weight_change,
                parameter_step,
            )
            previous_weights = update.weights
            previous_objective = objective
            if converged:
                if stage_index == len(stages) - 1:
                    return _result(
                        state,
                        last_weights,
                        last_precision,
                        residuals,
                        last_scale,
                        last_objective,
                        last_ess,
                        True,
                        "converged",
                        history,
                        "All enabled convergence criteria were met",
                    )
                break

    return _result(
        state,
        last_weights,
        last_precision,
        residuals,
        last_scale,
        last_objective,
        last_ess,
        False,
        "max_iterations",
        history,
        "Continuation schedule exhausted before convergence",
    )


def _positive_parameter(params: Mapping[str, float], name: str, default: float) -> float:
    value = float(params.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _robust_objective(
    distance: torch.Tensor,
    weight: RobustWeightName | WeightFunction,
    params: Mapping[str, float],
) -> torch.Tensor:
    if callable(weight):
        return 0.5 * distance.square() * robust_weights(distance, weight, params)
    if weight == "huber":
        delta = _positive_parameter(params, "delta", 1.345)
        return torch.where(
            distance <= delta,
            0.5 * distance.square(),
            delta * (distance - 0.5 * delta),
        )
    if weight == "pseudo_huber":
        delta = _positive_parameter(params, "delta", 1.345)
        return delta**2 * (torch.sqrt(1.0 + (distance / delta).square()) - 1.0)
    if weight == "tukey":
        c = _positive_parameter(params, "c", 4.685)
        inside = c**2 / 6.0 * (1.0 - (1.0 - (distance / c).square()).pow(3))
        return torch.where(distance <= c, inside, torch.full_like(distance, c**2 / 6.0))
    if weight == "cauchy":
        c = _positive_parameter(params, "c", 2.3849)
        return 0.5 * c**2 * torch.log1p((distance / c).square())
    alpha = float(params.get("alpha", 0.0))
    if not math.isfinite(alpha) or alpha > 2.0:
        raise ValueError("Barron IRLS objective requires finite alpha <= 2")
    scale = _positive_parameter(params, "scale", 1.0)
    return _barron_elementwise(distance, alpha, scale)


def _weight_update(
    problem: IRLSProblem[StateT] | IRLSCallbacks[StateT],
    state: StateT,
    residuals: torch.Tensor,
    config: IRLSFitConfig,
    mask: torch.Tensor | None,
    stage: IRLSContinuationStage,
) -> _WeightUpdate:
    standardized, scale, covariance_precision = _standardize(
        problem, state, residuals, config, mask
    )
    if standardized.ndim == 1:
        distance = standardized.abs()
    else:
        dims = tuple(range(1, standardized.ndim))
        distance = torch.linalg.vector_norm(standardized, dim=dims)
    active = _sample_mask(mask, residuals)
    weights = robust_weights(distance, stage.weight, stage.weight_params)
    weights = torch.where(active, weights, torch.zeros_like(weights))
    base_precision = _base_precision(config.base_precision, covariance_precision, residuals)
    precision = base_precision * _sample_broadcast(weights, base_precision)
    precision = _apply_mask_to_precision(precision, mask, residuals)
    base_sample = _sample_precision(base_precision, residuals)
    diagnostic_weights = torch.where(active, base_sample * weights, torch.zeros_like(weights))
    total = diagnostic_weights.sum()
    ess = total.square() / diagnostic_weights.square().sum().clamp_min(config.eps)
    rho = _robust_objective(distance, stage.weight, stage.weight_params)
    denominator = torch.where(active, base_sample, torch.zeros_like(base_sample)).sum()
    objective = (rho * torch.where(active, base_sample, torch.zeros_like(base_sample))).sum()
    objective = objective / denominator.clamp_min(config.eps)
    return _WeightUpdate(
        weights.detach(),
        precision.detach(),
        scale.detach(),
        objective.detach(),
        ess.detach(),
        int(active.sum().item()),
    )


def _standardize(
    problem: IRLSProblem[StateT] | IRLSCallbacks[StateT],
    state: StateT,
    residuals: torch.Tensor,
    config: IRLSFitConfig,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    covariance = _covariance(problem, state)
    whiten = _whiten_callback(problem)
    if covariance is not None and whiten is not None:
        raise ValueError("Provide either residual covariance or whitening, not both")
    if (covariance is not None or whiten is not None) and _has_partial_block_mask(mask, residuals):
        raise ValueError("Block covariance/whitening requires a sample-level mask")
    covariance_precision: torch.Tensor | None = None
    if covariance is not None:
        if not isinstance(covariance, torch.Tensor):
            raise TypeError("residual_covariance must return a torch.Tensor or None")
        if residuals.ndim != 2:
            raise ValueError("Block covariance requires residuals with shape (samples, dimensions)")
        expected = (residuals.shape[0], residuals.shape[1], residuals.shape[1])
        if covariance.shape != expected:
            raise ValueError(f"covariance shape {covariance.shape} must be {expected}")
        covariance = covariance.to(dtype=residuals.dtype, device=residuals.device)
        if not bool(torch.isfinite(covariance).all().item()):
            raise IRLSBreakdownError("Residual covariance is non-finite")
        chol, info = torch.linalg.cholesky_ex(covariance)
        if bool((info != 0).any().item()):
            raise IRLSBreakdownError("Residual covariance is not positive definite")
        standardized = torch.linalg.solve_triangular(
            chol, residuals.unsqueeze(-1), upper=False
        ).squeeze(-1)
        covariance_precision = torch.cholesky_inverse(chol)
    elif whiten is not None:
        whitened = whiten(state, residuals)
        if not isinstance(whitened, torch.Tensor):
            raise TypeError("whiten_residuals must return a torch.Tensor")
        standardized = whitened.detach()
        if standardized.shape != residuals.shape:
            raise ValueError("whiten_residuals must preserve residual shape")
    else:
        standardized = residuals
    scale = _estimate_scale(standardized, mask, config.scale, config.min_scale)
    standardized = standardized / scale
    if mask is not None:
        standardized = torch.where(_expanded_mask(mask, residuals), standardized, 0.0)
    return standardized, scale, covariance_precision


def _estimate_scale(
    residuals: torch.Tensor,
    mask: torch.Tensor | None,
    scale_spec: Literal["mad", "std", "rms", "none"] | float | torch.Tensor | ScaleEstimator,
    min_scale: float,
) -> torch.Tensor:
    if isinstance(scale_spec, torch.Tensor):
        scale = scale_spec.to(dtype=residuals.dtype, device=residuals.device)
    elif isinstance(scale_spec, (int, float)):
        scale = torch.as_tensor(scale_spec, dtype=residuals.dtype, device=residuals.device)
    elif callable(scale_spec):
        scale = cast(ScaleEstimator, scale_spec)(residuals, mask)
    elif scale_spec == "none":
        scale = torch.ones(
            residuals.shape[1:] or (), dtype=residuals.dtype, device=residuals.device
        )
    else:
        valid = (
            _expanded_mask(mask, residuals)
            if mask is not None
            else torch.ones_like(residuals, dtype=torch.bool)
        )
        count = valid.sum(dim=0).clamp_min(1)
        if scale_spec == "mad":
            center = _masked_median(residuals, valid)
            scale = 1.4826 * _masked_median((residuals - center).abs(), valid)
        elif scale_spec == "std":
            mean = torch.where(valid, residuals, 0.0).sum(dim=0) / count
            variance = torch.where(valid, (residuals - mean).square(), 0.0).sum(dim=0)
            scale = torch.sqrt(variance / count)
        elif scale_spec == "rms":
            scale = torch.sqrt(torch.where(valid, residuals.square(), 0.0).sum(dim=0) / count)
        else:
            raise ValueError(f"Unknown robust scale option: {scale_spec!r}")
    if not bool(torch.isfinite(scale).all().item()):
        raise IRLSBreakdownError("Robust scale is non-finite")
    if bool((scale < 0).any().item()):
        raise ValueError("Robust scale must be non-negative")
    return scale.clamp_min(min_scale)


def _masked_median(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    flat_values = values.reshape(values.shape[0], -1)
    flat_valid = valid.reshape(valid.shape[0], -1)
    ordered = torch.where(flat_valid, flat_values, torch.inf).sort(dim=0).values
    indices = ((flat_valid.sum(dim=0) - 1).clamp_min(0) // 2).unsqueeze(0)
    median = ordered.gather(0, indices).squeeze(0)
    return median.reshape(values.shape[1:])


def _residuals(problem: IRLSProblem[StateT] | IRLSCallbacks[StateT], state: StateT) -> torch.Tensor:
    residuals = problem.residuals(state)
    if not isinstance(residuals, torch.Tensor):
        raise TypeError("problem.residuals must return a torch.Tensor")
    if not (residuals.is_floating_point() or residuals.is_complex()):
        raise TypeError("residuals must use a floating-point dtype")
    if residuals.is_complex():
        raise TypeError("complex residuals are not supported")
    return residuals.detach()


def _covariance(
    problem: IRLSProblem[StateT] | IRLSCallbacks[StateT], state: StateT
) -> torch.Tensor | None:
    if isinstance(problem, IRLSCallbacks):
        return None if problem.covariance_fn is None else problem.covariance_fn(state)
    callback = getattr(problem, "residual_covariance", None)
    return (
        None if callback is None else cast(Callable[[StateT], torch.Tensor | None], callback)(state)
    )


def _whiten_callback(
    problem: IRLSProblem[StateT] | IRLSCallbacks[StateT],
) -> Callable[[StateT, torch.Tensor], torch.Tensor] | None:
    if isinstance(problem, IRLSCallbacks):
        return problem.whiten_fn
    callback = getattr(problem, "whiten_residuals", None)
    return cast(Callable[[StateT, torch.Tensor], torch.Tensor] | None, callback)


def _parameter_step(
    problem: IRLSProblem[StateT] | IRLSCallbacks[StateT],
    previous: StateT,
    current: StateT,
    eps: float,
) -> float | None:
    if isinstance(problem, IRLSCallbacks):
        callback = problem.state_step_fn
    else:
        callback = cast(
            Callable[[StateT, StateT], float | torch.Tensor] | None,
            getattr(problem, "state_step", None),
        )
    if callback is not None:
        value = callback(previous, current)
        return float(value.item() if isinstance(value, torch.Tensor) else value)
    if isinstance(previous, torch.Tensor) and isinstance(current, torch.Tensor):
        numerator = torch.linalg.vector_norm((current - previous).detach())
        denominator = torch.linalg.vector_norm(previous.detach()).clamp_min(eps)
        return float((numerator / denominator).item())
    return None


def _state_is_finite(state: Any) -> bool:
    if isinstance(state, torch.Tensor):
        return bool(torch.isfinite(state).all().item())
    if isinstance(state, float):
        return math.isfinite(state)
    if is_dataclass(state) and not isinstance(state, type):
        return all(_state_is_finite(getattr(state, item.name)) for item in fields(state))
    if isinstance(state, Mapping):
        return all(_state_is_finite(value) for value in state.values())
    if isinstance(state, (tuple, list)):
        return all(_state_is_finite(value) for value in state)
    return True


def _validated_mask(mask: torch.Tensor | None, residuals: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        raise TypeError("mask must have dtype torch.bool")
    mask = mask.to(device=residuals.device)
    try:
        torch.broadcast_shapes(mask.shape, residuals.shape)
    except RuntimeError as error:
        if mask.shape != (residuals.shape[0],):
            raise ValueError(
                f"mask shape {mask.shape} is not sample-level or broadcastable to {residuals.shape}"
            ) from error
    return mask


def _expanded_mask(mask: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    if mask.shape == (residuals.shape[0],):
        return mask.reshape((residuals.shape[0],) + (1,) * (residuals.ndim - 1)).expand_as(
            residuals
        )
    return torch.broadcast_to(mask, residuals.shape)


def _sample_mask(mask: torch.Tensor | None, residuals: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(residuals.shape[0], dtype=torch.bool, device=residuals.device)
    expanded = _expanded_mask(mask, residuals)
    if residuals.ndim == 1:
        return expanded
    return expanded.reshape(residuals.shape[0], -1).any(dim=1)


def _has_partial_block_mask(mask: torch.Tensor | None, residuals: torch.Tensor) -> bool:
    if mask is None or residuals.ndim == 1 or mask.shape == (residuals.shape[0],):
        return False
    expanded = _expanded_mask(mask, residuals).reshape(residuals.shape[0], -1)
    return bool((expanded.any(dim=1) != expanded.all(dim=1)).any().item())


def _base_precision(
    configured: torch.Tensor | None,
    covariance_precision: torch.Tensor | None,
    residuals: torch.Tensor,
) -> torch.Tensor:
    if configured is None:
        if covariance_precision is not None:
            return covariance_precision
        return torch.ones(residuals.shape[0], dtype=residuals.dtype, device=residuals.device)
    precision = configured.to(dtype=residuals.dtype, device=residuals.device)
    if not bool(torch.isfinite(precision).all().item()):
        raise ValueError("base_precision must be finite")
    valid_shapes = {(residuals.shape[0],), residuals.shape}
    if residuals.ndim == 2:
        valid_shapes.add((residuals.shape[0], residuals.shape[1], residuals.shape[1]))
    if precision.shape not in valid_shapes:
        raise ValueError(
            "base_precision shape "
            f"{precision.shape} is incompatible with residuals {residuals.shape}"
        )
    if precision.ndim == residuals.ndim + 1:
        if not torch.allclose(precision, precision.transpose(-1, -2)):
            raise ValueError("block base_precision must be symmetric")
        eigenvalues = torch.linalg.eigvalsh(precision)
        tolerance = 10.0 * torch.finfo(precision.dtype).eps * precision.abs().amax().clamp_min(1.0)
        if bool((eigenvalues < -tolerance).any().item()):
            raise ValueError("block base_precision must be positive semidefinite")
    elif bool((precision < 0).any().item()):
        raise ValueError("base_precision must be non-negative")
    return precision


def _apply_mask_to_precision(
    precision: torch.Tensor,
    mask: torch.Tensor | None,
    residuals: torch.Tensor,
) -> torch.Tensor:
    if mask is None:
        return precision
    expanded = _expanded_mask(mask, residuals)
    partial = _has_partial_block_mask(mask, residuals)
    if precision.ndim == residuals.ndim + 1:
        if partial:
            raise ValueError("Partial element masks are not supported with block precision")
        return precision
    if precision.ndim == 1 and residuals.ndim > 1 and partial:
        precision = precision.reshape((precision.shape[0],) + (1,) * (residuals.ndim - 1))
        precision = precision.expand_as(residuals)
    if precision.ndim == residuals.ndim:
        return torch.where(expanded, precision, torch.zeros_like(precision))
    return precision


def _initial_precision(residuals: torch.Tensor, config: IRLSFitConfig) -> torch.Tensor:
    return _base_precision(config.base_precision, None, residuals)


def _sample_broadcast(weights: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
    return weights.reshape((weights.shape[0],) + (1,) * (precision.ndim - 1))


def _sample_precision(precision: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    if precision.ndim == 1:
        return precision
    if precision.ndim == residuals.ndim:
        return precision.reshape(precision.shape[0], -1).mean(dim=1)
    return torch.diagonal(precision, dim1=-2, dim2=-1).mean(dim=-1)


def _relative_change(current: torch.Tensor, previous: torch.Tensor | None, eps: float) -> float:
    if previous is None:
        return math.inf
    numerator = torch.linalg.vector_norm(current - previous)
    denominator = torch.linalg.vector_norm(previous).clamp_min(eps)
    return float((numerator / denominator).item())


def _scalar_relative_change(current: float, previous: float | None, eps: float) -> float:
    if previous is None:
        return math.inf
    return abs(current - previous) / max(abs(previous), eps)


def _converged(
    config: IRLSFitConfig,
    stage_iterations: int,
    objective_change: float,
    weight_change: float,
    parameter_step: float | None,
) -> bool:
    if stage_iterations < config.min_iter:
        return False
    checks: list[bool] = []
    if config.objective_tol is not None:
        checks.append(objective_change <= config.objective_tol)
    if config.weight_tol is not None:
        checks.append(weight_change <= config.weight_tol)
    if config.parameter_tol is not None:
        checks.append(parameter_step is not None and parameter_step <= config.parameter_tol)
    return bool(checks) and all(checks)


def _empty_result(
    state: StateT,
    residuals: torch.Tensor,
    config: IRLSFitConfig,
    mask: torch.Tensor | None,
) -> IRLSFitResult[StateT]:
    weights = torch.zeros(residuals.shape[0], dtype=residuals.dtype, device=residuals.device)
    precision = _base_precision(config.base_precision, None, residuals)
    precision = precision * _sample_broadcast(weights, precision)
    scale = torch.ones(residuals.shape[1:] or (), dtype=residuals.dtype, device=residuals.device)
    return IRLSFitResult(
        state=state,
        weights=weights,
        precision=precision,
        residuals=residuals.detach().clone(),
        scale=scale,
        objective=0.0,
        effective_sample_size=0.0,
        converged=False,
        termination="empty",
        iterations=0,
        history=(),
        message="No active samples",
    )


def _result(
    state: StateT,
    weights: torch.Tensor,
    precision: torch.Tensor,
    residuals: torch.Tensor,
    scale: torch.Tensor,
    objective: float,
    effective_sample_size: float,
    converged: bool,
    termination: IRLSTermination,
    history: list[IRLSIteration],
    message: str,
) -> IRLSFitResult[StateT]:
    return IRLSFitResult(
        state=state,
        weights=weights.detach().clone(),
        precision=precision.detach().clone(),
        residuals=residuals.detach().clone(),
        scale=scale.detach().clone(),
        objective=objective,
        effective_sample_size=effective_sample_size,
        converged=converged,
        termination=termination,
        iterations=len(history),
        history=tuple(history),
        message=message,
    )
