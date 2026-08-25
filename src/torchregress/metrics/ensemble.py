"""
Ensemble forecasting metrics and uncertainty decomposition.
"""

import math
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from torch.distributions import Normal
from torchmetrics import Metric

from .interval import IntervalScore, PredictionIntervalCoverageProbability
from .utils import convert_to_tensor, metric_state_tensor


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
    """
    Compute Gaussian negative log-likelihood for ensemble forecasts.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, dim: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dim = dim
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
        total_var = stats["total_uncertainty"]
        total_var = torch.clamp(total_var, min=1e-6)
        diff2 = (y - mean) ** 2
        nll = 0.5 * (torch.log(2 * math.pi * total_var) + diff2 / total_var)
        metric_state_tensor(self.nll_sum).add_(torch.sum(nll))
        metric_state_tensor(self.total).add_(torch.as_tensor(y.numel(), device=y.device))

    def compute(self) -> torch.Tensor:
        """Compute NLL."""
        return metric_state_tensor(self.nll_sum) / metric_state_tensor(self.total)


class EnsembleIntervalMetrics(Metric):
    """
    Interval score and coverage for ensemble predictions.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, alpha: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.interval_score = IntervalScore(alpha=alpha)
        self.picp = PredictionIntervalCoverageProbability()

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
        total_var = torch.clamp(stats["total_uncertainty"], min=1e-6)
        sd = torch.sqrt(total_var)
        z = Normal(0, 1).icdf(torch.tensor(1 - self.alpha / 2, device=mean.device))
        lower = mean - z * sd
        upper = mean + z * sd
        return lower, upper


def gaussian_nll_ensemble(
    means: Union[torch.Tensor, np.ndarray],
    variances: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    dim: int = 0,
) -> torch.Tensor:
    """
    Functional Gaussian NLL for ensemble mean/variance predictions.
    """
    metric = GaussianNLLEnsemble(dim=dim)
    metric.update(convert_to_tensor(means), convert_to_tensor(variances), convert_to_tensor(y_true))  # ty: ignore[invalid-argument-type]  # torchmetrics update/compute overrides confuse ty
    return metric.compute()  # ty: ignore[missing-argument]  # torchmetrics update/compute overrides confuse ty


def ensemble_interval_bounds(
    means: Union[torch.Tensor, np.ndarray],
    variances: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.1,
    dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functional symmetric Gaussian prediction intervals from ensemble statistics.
    """
    stats = uncertainty_decomposition(means, variances, dim=dim)
    mean = stats["mean"]
    total_var = torch.clamp(stats["total_uncertainty"], min=1e-6)
    sd = torch.sqrt(total_var)
    z = Normal(0, 1).icdf(torch.tensor(1 - alpha / 2, device=mean.device))
    return mean - z * sd, mean + z * sd


def ensemble_interval_metrics(
    means: Union[torch.Tensor, np.ndarray],
    variances: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Functional interval score + coverage for ensemble predictions.
    """
    metric = EnsembleIntervalMetrics(alpha=alpha)
    metric.update(convert_to_tensor(means), convert_to_tensor(variances), convert_to_tensor(y_true))  # ty: ignore[invalid-argument-type]  # torchmetrics update/compute overrides confuse ty
    return metric.compute()  # ty: ignore[missing-argument]  # torchmetrics update/compute overrides confuse ty
