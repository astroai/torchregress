"""
Ensemble forecasting metrics and uncertainty decomposition.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions import Normal
from torchmetrics import Metric

from .interval import IntervalScore, PredictionIntervalCoverageProbability
from .utils import convert_to_tensor, metric_state_tensor, prepare_functional_metric


def _standard_normal_icdf(prob: float, like: torch.Tensor) -> torch.Tensor:
    """Standard-normal quantile at ``prob`` in the dtype/device of ``like``."""
    dist = Normal(
        torch.zeros((), device=like.device, dtype=like.dtype),
        torch.ones((), device=like.device, dtype=like.dtype),
    )
    return dist.icdf(torch.tensor(prob, device=like.device, dtype=like.dtype))


def _variance_floor(var: torch.Tensor, min_variance: Optional[float]) -> torch.Tensor:
    """Clamp ``var`` below at ``min_variance`` (default: ``finfo(dtype).tiny``).

    An absolute floor such as ``1e-6`` silently rescales targets whose natural
    variance is below it (M-MET-010); the dtype ``tiny`` only prevents division
    by zero.
    """
    floor = torch.finfo(var.dtype).tiny if min_variance is None else float(min_variance)
    return torch.clamp(var, min=floor)


def ensemble_statistics(
    predictions: Union[torch.Tensor, np.ndarray],
    dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ensemble mean and variance across dimension `dim`.
    """
    preds = convert_to_tensor(predictions)
    mean = torch.mean(preds, dim=dim)
    var = torch.var(preds, dim=dim, unbiased=False)
    return mean, var


def ensemble_mean(predictions: Union[torch.Tensor, np.ndarray], dim: int = 0) -> torch.Tensor:
    """
    Alias for ensemble mean across dimension `dim`.
    """
    mean, _ = ensemble_statistics(predictions, dim=dim)
    return mean


def ensemble_std(predictions: Union[torch.Tensor, np.ndarray], dim: int = 0) -> torch.Tensor:
    """
    Alias for ensemble standard deviation across dimension `dim`.
    """
    _, var = ensemble_statistics(predictions, dim=dim)
    return torch.sqrt(var)


def ensemble_variance_decomposition(
    means: Union[torch.Tensor, np.ndarray],
    variances: Union[torch.Tensor, np.ndarray],
    dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Alias returning (epistemic, aleatoric) uncertainty.
    """
    stats = uncertainty_decomposition(means, variances, dim=dim)
    return stats["epistemic_uncertainty"], stats["aleatoric_uncertainty"]


def uncertainty_decomposition(
    means: Union[torch.Tensor, np.ndarray],
    variances: Union[torch.Tensor, np.ndarray],
    dim: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Decompose uncertainty into epistemic, aleatoric, and total.
    """
    means_t = convert_to_tensor(means)
    vars_t = convert_to_tensor(variances)
    mean, _ = ensemble_statistics(means_t, dim)
    epistemic = torch.var(means_t, dim=dim, unbiased=False)
    aleatoric = torch.mean(vars_t, dim=dim)
    total = epistemic + aleatoric
    return {
        "mean": mean,
        "epistemic_uncertainty": epistemic,
        "aleatoric_uncertainty": aleatoric,
        "total_uncertainty": total,
    }


class GaussianNLLEnsemble(Metric):
    """Gaussian negative log-likelihood for ensemble forecasts.

    The ensemble is moment-matched to ``N(mean_m mu_m, var_m mu_m + mean_m var_m)``
    (law of total variance) and scored with the Gaussian NLL.

    Parameters
    ----------
    dim : int
        Ensemble-member dimension.
    min_variance : float, optional
        Lower bound on the total predictive variance, in the target's squared
        units. ``None`` (default) uses ``torch.finfo(dtype).tiny`` so small-scale
        targets are scored exactly; pass e.g. ``1e-6`` (the pre-0.3 floor) to
        regularise degenerate members.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, dim: int = 0, min_variance: Optional[float] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.min_variance = min_variance
        self.add_state("nll_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        means: Union[torch.Tensor, np.ndarray],
        variances: Union[torch.Tensor, np.ndarray],
        y_true: torch.Tensor,
    ) -> None:
        """Update state with predictions and targets."""
        means_t = convert_to_tensor(means)
        vars_t = convert_to_tensor(variances)
        if means_t.dim() != vars_t.dim():
            raise ValueError(
                f"means and variances must have the same number of dimensions; "
                f"got {means_t.dim()} vs {vars_t.dim()}"
            )
        if torch.isnan(vars_t).any() or torch.isinf(vars_t).any():
            raise ValueError("variances contain NaN or infinite values")

        y = convert_to_tensor(y_true)
        stats = uncertainty_decomposition(means_t, vars_t, dim=self.dim)
        mean = stats["mean"]
        total_var = _variance_floor(stats["total_uncertainty"], self.min_variance)
        diff2 = (y - mean) ** 2
        nll = 0.5 * (torch.log(2 * math.pi * total_var) + diff2 / total_var)
        metric_state_tensor(self.nll_sum).add_(torch.sum(nll))
        metric_state_tensor(self.total).add_(torch.as_tensor(y.numel(), device=y.device))

    def compute(self) -> torch.Tensor:
        """Compute NLL."""
        return metric_state_tensor(self.nll_sum) / metric_state_tensor(self.total)


class EnsembleIntervalMetrics(Metric):
    """Interval score and coverage for ensemble predictions.

    Parameters
    ----------
    alpha : float
        Miscoverage level of the central Gaussian interval.
    min_variance : float, optional
        Lower bound on the total predictive variance (default
        ``torch.finfo(dtype).tiny``; see :class:`GaussianNLLEnsemble`).
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self, alpha: float = 0.1, min_variance: Optional[float] = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.min_variance = min_variance
        self.interval_score = IntervalScore(alpha=alpha)
        self.picp = PredictionIntervalCoverageProbability()

    def reset(self) -> None:
        """Reset the state, including the child interval-score and PICP metrics."""
        super().reset()
        # M-MET-009: torchmetrics does not reset metrics held as attributes.
        self.interval_score.reset()
        self.picp.reset()

    def update(self, means: torch.Tensor, variances: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        lower, upper = self.ensemble_interval_bounds(means, variances)
        # torchmetrics ``Metric.update``/``compute`` overrides confuse ty's
        # union resolution (the base ``update(*_, **__)`` gets unioned in), so
        # the calls below carry per-line suppressions.
        self.interval_score.update(lower, upper, y_true)  # ty: ignore[invalid-argument-type]
        self.picp.update(lower, upper, y_true)  # ty: ignore[invalid-argument-type]

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute metrics."""
        return {
            "interval_score": self.interval_score.compute(),  # ty: ignore[missing-argument]
            "picp": self.picp.compute(),  # ty: ignore[missing-argument]
        }

    def ensemble_interval_bounds(
        self, means: torch.Tensor, variances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute symmetric Gaussian prediction intervals from ensemble.
        """
        stats = uncertainty_decomposition(means, variances)
        mean = stats["mean"]
        total_var = _variance_floor(stats["total_uncertainty"], self.min_variance)
        sd = torch.sqrt(total_var)
        z = _standard_normal_icdf(1 - self.alpha / 2, mean)
        lower = mean - z * sd
        upper = mean + z * sd
        return lower, upper


def gaussian_nll_ensemble(
    means: Union[torch.Tensor, np.ndarray],
    variances: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    dim: int = 0,
    min_variance: Optional[float] = None,
) -> torch.Tensor:
    """
    Functional Gaussian NLL for ensemble mean/variance predictions.

    ``min_variance`` bounds the total variance below (default
    ``torch.finfo(dtype).tiny``; see :class:`GaussianNLLEnsemble`).
    """
    means_t = convert_to_tensor(means)
    variances_t = convert_to_tensor(variances)
    y_true_t = convert_to_tensor(y_true)
    metric = prepare_functional_metric(
        GaussianNLLEnsemble(dim=dim, min_variance=min_variance), means_t, variances_t, y_true_t
    )
    metric.update(means_t, variances_t, y_true_t)  # ty: ignore[invalid-argument-type]  # torchmetrics update/compute overrides confuse ty
    return metric.compute()  # ty: ignore[missing-argument]  # torchmetrics update/compute overrides confuse ty


def ensemble_interval_bounds(
    means: Union[torch.Tensor, np.ndarray],
    variances: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.1,
    dim: int = 0,
    min_variance: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functional symmetric Gaussian prediction intervals from ensemble statistics.

    ``min_variance`` bounds the total variance below (default
    ``torch.finfo(dtype).tiny``; see :class:`GaussianNLLEnsemble`).
    """
    stats = uncertainty_decomposition(means, variances, dim=dim)
    mean = stats["mean"]
    total_var = _variance_floor(stats["total_uncertainty"], min_variance)
    sd = torch.sqrt(total_var)
    z = _standard_normal_icdf(1 - alpha / 2, mean)
    return mean - z * sd, mean + z * sd


def ensemble_interval_metrics(
    means: Union[torch.Tensor, np.ndarray],
    variances: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.1,
    min_variance: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Functional interval score + coverage for ensemble predictions.
    """
    means_t = convert_to_tensor(means)
    variances_t = convert_to_tensor(variances)
    y_true_t = convert_to_tensor(y_true)
    metric = prepare_functional_metric(
        EnsembleIntervalMetrics(alpha=alpha, min_variance=min_variance),
        means_t,
        variances_t,
        y_true_t,
    )
    metric.update(means_t, variances_t, y_true_t)  # ty: ignore[invalid-argument-type]  # torchmetrics update/compute overrides confuse ty
    return metric.compute()  # ty: ignore[missing-argument]  # torchmetrics update/compute overrides confuse ty
