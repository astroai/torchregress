"""
Interval metrics for evaluating prediction intervals in regression.
"""

from typing import Any, Dict, Union

import numpy as np
import torch
from torchmetrics import Metric

from torchregress.metrics.utils import convert_to_tensor, validate_inputs


class IntervalScore(Metric):
    """
    Calculate prediction interval score (Winkler score).
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, alpha: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        y_true: torch.Tensor,
    ) -> None:
        """Update state with predictions and targets."""
        lower_bound = convert_to_tensor(lower_bound)
        upper_bound = convert_to_tensor(upper_bound)
        y_true = convert_to_tensor(y_true)
        validate_inputs(lower_bound, y_true)
        validate_inputs(upper_bound, y_true)

        interval_width = upper_bound - lower_bound
        if torch.any(interval_width < 0):
            raise ValueError("Upper bounds must be greater than or equal to lower bounds")

        below_lower = torch.clamp(lower_bound - y_true, min=0)
        above_upper = torch.clamp(y_true - upper_bound, min=0)

        score = interval_width + (2 / self.alpha) * (below_lower + above_upper)

        self.score += torch.sum(score)
        self.total += y_true.numel()

    def compute(self) -> torch.Tensor:
        """Compute interval score."""
        return self.score / self.total


class PredictionIntervalCoverageProbability(Metric):
    """
    Calculate Prediction Interval Coverage Probability (PICP).
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("covered", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        y_true: torch.Tensor,
    ) -> None:
        """Update state with predictions and targets."""
        lower_bound = convert_to_tensor(lower_bound)
        upper_bound = convert_to_tensor(upper_bound)
        y_true = convert_to_tensor(y_true)

        coverage = ((y_true >= lower_bound) & (y_true <= upper_bound)).float()
        self.covered += torch.sum(coverage)
        self.total += y_true.numel()

    def compute(self) -> torch.Tensor:
        """Compute PICP."""
        return self.covered / self.total


def interval_score(
    lower_bound: Union[torch.Tensor, np.ndarray],
    upper_bound: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.1,
    reduction: str = "mean",
) -> Union[torch.Tensor, float, Dict[str, torch.Tensor]]:
    """
    Functional interval score (Winkler score).
    """
    lower_t = convert_to_tensor(lower_bound)
    upper_t = convert_to_tensor(upper_bound)
    y_true_t = convert_to_tensor(y_true)

    validate_inputs(lower_t, y_true_t)
    validate_inputs(upper_t, y_true_t)

    interval_width = upper_t - lower_t
    if torch.any(interval_width < 0):
        raise ValueError("Upper bounds must be greater than or equal to lower bounds")

    below_lower = torch.clamp(lower_t - y_true_t, min=0)
    above_upper = torch.clamp(y_true_t - upper_t, min=0)
    score = interval_width + (2 / alpha) * (below_lower + above_upper)

    coverage = torch.mean(((y_true_t >= lower_t) & (y_true_t <= upper_t)).float())
    expected_coverage = 1.0 - alpha
    mean_width = torch.mean(interval_width)

    if reduction == "full":
        return {
            "score": torch.mean(score),
            "mean_width": mean_width,
            "mean_coverage": coverage,
            "expected_coverage": torch.tensor(expected_coverage, device=score.device),
            "coverage_error": coverage - expected_coverage,
        }

    if reduction == "none":
        return score
    if reduction == "sum":
        return torch.sum(score)
    return torch.mean(score)


def prediction_interval_coverage_probability(
    lower_bound: Union[torch.Tensor, np.ndarray],
    upper_bound: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.1,
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, float, Dict[str, torch.Tensor]]:
    """
    Functional Prediction Interval Coverage Probability (PICP).
    """
    lower_t = convert_to_tensor(lower_bound)
    upper_t = convert_to_tensor(upper_bound)
    y_true_t = convert_to_tensor(y_true)

    coverage_mask = (y_true_t >= lower_t) & (y_true_t <= upper_t)
    picp = torch.mean(coverage_mask.float())
    mpiw = torch.mean(upper_t - lower_t)

    miss_rate_low = torch.mean((y_true_t < lower_t).float())
    miss_rate_high = torch.mean((y_true_t > upper_t).float())
    expected_coverage = 1.0 - alpha

    if not return_diagnostics:
        return picp

    return {
        "picp": picp,
        "expected_coverage": torch.tensor(expected_coverage, device=picp.device),
        "coverage_error": picp - expected_coverage,
        "mpiw": mpiw,
        "miss_rate_low": miss_rate_low,
        "miss_rate_high": miss_rate_high,
    }


def interval_metrics_report(
    predictions: Dict[str, Dict[str, Union[torch.Tensor, np.ndarray]]],
    y_true: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.1,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Generate interval metrics report for multiple models.
    """
    report: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, bounds in predictions.items():
        lower = bounds["lower"]
        upper = bounds["upper"]
        score = interval_score(lower, upper, y_true, alpha=alpha)
        picp = prediction_interval_coverage_probability(
            lower, upper, y_true, alpha=alpha, return_diagnostics=False
        )
        mpiw = torch.mean(convert_to_tensor(upper) - convert_to_tensor(lower))
        report[name] = {"score": score, "picp": picp, "mpiw": mpiw}
    return report
