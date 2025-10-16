"""
Calibration metrics for evaluating probabilistic regression models.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torchmetrics import Metric

from torchregress.metrics.utils import convert_to_tensor, validate_inputs


class ExpectedCalibrationError(Metric):
    """
    Calculate Expected Calibration Error (ECE) for quantile regression.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, n_bins: int = 10, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.add_state("y_pred_quantiles", default=[], dist_reduce_fx=None)
        self.add_state("y_true", default=[], dist_reduce_fx=None)

    def update(
        self, y_pred_quantiles: Dict[float, torch.Tensor], y_true: torch.Tensor
    ) -> None:
        """Update state with predictions and targets."""
        self.y_pred_quantiles.append(y_pred_quantiles)
        self.y_true.append(y_true)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute ECE."""
        y_true = torch.cat([convert_to_tensor(y) for y in self.y_true])

        y_pred_quantiles = {}
        for q in self.y_pred_quantiles[0].keys():
            y_pred_quantiles[q] = torch.cat(
                [convert_to_tensor(d[q]) for d in self.y_pred_quantiles]
            )

        quantiles = sorted(y_pred_quantiles.keys())
        expected_proportions = torch.tensor(quantiles, device=y_true.device)
        actual_proportions = []

        for q in quantiles:
            q_pred = y_pred_quantiles[q]
            validate_inputs(q_pred, y_true)
            proportion_below = torch.mean((y_true <= q_pred).float())
            actual_proportions.append(proportion_below)

        actual_proportions = torch.stack(actual_proportions)

        abs_errors = torch.abs(actual_proportions - expected_proportions)
        mace = torch.mean(abs_errors)
        rmsce = torch.sqrt(torch.mean((actual_proportions - expected_proportions) ** 2))
        max_ce = torch.max(abs_errors)

        return {
            "mean_absolute_calibration_error": mace,
            "root_mean_squared_calibration_error": rmsce,
            "maximum_calibration_error": max_ce,
        }


class MarginalCalibrationError(Metric):
    """
    Calculate Marginal Calibration Error (MCE) for probabilistic regression.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, n_bins: int = 20, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.add_state("y_pred_samples", default=[], dist_reduce_fx=None)
        self.add_state("y_true", default=[], dist_reduce_fx=None)

    def update(self, y_pred_samples: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        self.y_pred_samples.append(y_pred_samples)
        self.y_true.append(y_true)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute MCE."""
        y_true = torch.cat([convert_to_tensor(y) for y in self.y_true])
        y_pred_samples = torch.cat([convert_to_tensor(y) for y in self.y_pred_samples], dim=1)

        all_values = torch.cat([y_true, y_pred_samples.view(-1)])
        min_val = torch.min(all_values)
        max_val = torch.max(all_values)

        if torch.isclose(min_val, max_val, rtol=1e-5):
            min_val = min_val - 1e-5
            max_val = max_val + 1e-5

        bin_edges = torch.linspace(min_val, max_val, self.n_bins + 1, device="cpu")

        y_true_cpu = y_true.cpu()

        obs_hist = torch.histogram(y_true_cpu, bin_edges)[0]
        obs_cdf = torch.cumsum(obs_hist, dim=0) / max(1, len(y_true))

        pred_cdfs = []
        for i in range(y_pred_samples.shape[0]):
            pred_samples_cpu = y_pred_samples[i].cpu()
            pred_hist = torch.histogram(pred_samples_cpu, bin_edges)[0]
            pred_cdf = torch.cumsum(pred_hist, dim=0) / max(1, len(y_pred_samples[i]))
            pred_cdfs.append(pred_cdf)

        pred_cdf_mean = torch.stack(pred_cdfs).mean(dim=0)

        abs_errors = torch.abs(obs_cdf - pred_cdf_mean)
        mce = torch.mean(abs_errors)
        rmsce = torch.sqrt(torch.mean((obs_cdf - pred_cdf_mean) ** 2))
        max_mce = torch.max(abs_errors)

        device = y_true.device
        if device.type != "cpu":
            mce = mce.to(device)
            rmsce = rmsce.to(device)
            max_mce = max_mce.to(device)

        return {
            "marginal_calibration_error": mce,
            "root_mean_squared_mce": rmsce,
            "maximum_marginal_calibration_error": max_mce,
        }