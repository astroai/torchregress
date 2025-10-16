"""
Distribution metrics for evaluating probabilistic regression models.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torchmetrics import Metric

from torchregress.metrics.utils import convert_to_tensor, validate_inputs


class ContinuousRankedProbabilityScore(Metric):
    """
    Calculate Continuous Ranked Probability Score (CRPS) for probabilistic forecasts.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("crps_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, y_pred_quantiles: Dict[float, torch.Tensor], y_true: torch.Tensor
    ) -> None:
        """Update state with predictions and targets."""
        y_true = convert_to_tensor(y_true)

        quantiles = sorted(y_pred_quantiles.keys())
        if len(quantiles) < 2:
            raise ValueError("At least 2 quantile levels are required for CRPS calculation")

        forecasts = []
        for q in quantiles:
            q_pred = convert_to_tensor(y_pred_quantiles[q])
            validate_inputs(q_pred, y_true)
            forecasts.append(q_pred)

        forecast_tensor = torch.stack(forecasts)
        quantile_tensor = torch.tensor(quantiles, device=forecast_tensor.device)

        zero_tensor = torch.tensor([0.0], device=quantile_tensor.device)
        one_tensor = torch.tensor([1.0], device=quantile_tensor.device)
        weights = torch.diff(torch.cat([zero_tensor, quantile_tensor, one_tensor]))

        crps_values = torch.zeros_like(y_true, device=y_true.device)
        for i, q in enumerate(quantiles):
            diff = y_true - forecast_tensor[i]
            q_tensor = torch.tensor(q, device=diff.device)
            quantile_loss = torch.max(q_tensor * diff, (q_tensor - 1) * diff)
            crps_values = crps_values + weights[i + 1] * quantile_loss

        self.crps_sum += torch.sum(crps_values)
        self.total += y_true.numel()

    def compute(self) -> torch.Tensor:
        """Compute CRPS."""
        return self.crps_sum / self.total


class EnergyScore(Metric):
    """
    Calculate Energy Score for multivariate probabilistic forecasts.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, beta: float = 1.0, max_pairs: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.max_pairs = max_pairs
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_samples: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        y_true = convert_to_tensor(y_true)
        y_samples = convert_to_tensor(y_samples)

        n_samples = y_samples.shape[0]
        batch_size = y_true.shape[0]

        if self.max_pairs is not None and n_samples > self.max_pairs:
            indices = torch.randperm(n_samples)[: self.max_pairs]
            y_samples = y_samples[indices]
            n_samples = self.max_pairs

        norms = torch.zeros(batch_size, n_samples, device=y_true.device)
        for i in range(n_samples):
            diff = y_samples[i] - y_true
            if self.beta == 1.0:
                norms[:, i] = torch.norm(diff, dim=1)
            else:
                norms[:, i] = torch.pow(torch.sum(torch.pow(torch.abs(diff), self.beta), dim=1), 1 / self.beta)

        term1 = torch.mean(norms, dim=1)

        term2 = torch.zeros(batch_size, device=y_true.device)
        n_pairs = 0
        for i in range(n_samples):
            sample_i = y_samples[i]
            for j in range(i + 1, n_samples):
                sample_j = y_samples[j]
                term2 += torch.norm(sample_i - sample_j, dim=1) ** self.beta
                n_pairs += 1

        term2 /= 2.0 * n_pairs

        energy_scores = term1 - term2
        self.score_sum += torch.sum(energy_scores)
        self.total += batch_size

    def compute(self) -> torch.Tensor:
        """Compute energy score."""
        return self.score_sum / self.total