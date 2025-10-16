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

    def update(self, lower_bound: torch.Tensor, upper_bound: torch.Tensor, y_true: torch.Tensor) -> None:
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

    def update(self, lower_bound: torch.Tensor, upper_bound: torch.Tensor, y_true: torch.Tensor) -> None:
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