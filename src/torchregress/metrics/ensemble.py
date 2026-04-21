"""
Ensemble forecasting metrics and uncertainty decomposition.
"""

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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("nll_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, means: torch.Tensor, variances: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        y = convert_to_tensor(y_true)
        stats = uncertainty_decomposition(means, variances)
        mean = stats["mean"]
        total_var = stats["total_uncertainty"]
        total_var = torch.clamp(total_var, min=1e-6)
        diff2 = (y - mean) ** 2
        nll = 0.5 * (torch.log(2 * np.pi * total_var) + diff2 / total_var)
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
        self.interval_score.update(lower, upper, y_true)
        self.picp.update(lower, upper, y_true)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute metrics."""
        return {
            "interval_score": self.interval_score.compute(),
            "picp": self.picp.compute(),
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
) -> torch.Tensor:
    """
    Functional Gaussian NLL for ensemble mean/variance predictions.
    """
    metric = GaussianNLLEnsemble()
    metric.update(convert_to_tensor(means), convert_to_tensor(variances), convert_to_tensor(y_true))
    return metric.compute()


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
    metric.update(convert_to_tensor(means), convert_to_tensor(variances), convert_to_tensor(y_true))
    return metric.compute()
