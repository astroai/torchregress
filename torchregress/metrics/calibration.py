"""
Calibration metrics for evaluating probabilistic regression models.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torchmetrics import Metric

from torchregress.metrics.utils import convert_to_tensor, create_metric_result, validate_inputs


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


def expected_calibration_error(
    y_pred_quantiles: Dict[float, Union[torch.Tensor, np.ndarray]],
    y_true: Union[torch.Tensor, np.ndarray],
    return_diagnostics: bool = False,
    as_numpy: bool = False,
) -> Dict[str, Union[torch.Tensor, float, np.ndarray]]:
    """
    Functional Expected Calibration Error (ECE) for quantile regression.
    """
    y_true_t = convert_to_tensor(y_true)
    device = y_true_t.device

    quantiles = sorted(y_pred_quantiles.keys())
    expected_proportions = torch.tensor(quantiles, device=device)
    actual_proportions = []

    for q in quantiles:
        q_pred = convert_to_tensor(y_pred_quantiles[q]).to(device)
        validate_inputs(q_pred, y_true_t)
        actual_proportions.append(torch.mean((y_true_t <= q_pred).float()))

    actual_proportions = torch.stack(actual_proportions)
    abs_errors = torch.abs(actual_proportions - expected_proportions)

    result = {
        "mean_absolute_calibration_error": torch.mean(abs_errors),
        "root_mean_squared_calibration_error": torch.sqrt(
            torch.mean((actual_proportions - expected_proportions) ** 2)
        ),
        "maximum_calibration_error": torch.max(abs_errors),
    }

    if return_diagnostics:
        result.update(
            {
                "bin_errors": abs_errors,
                "expected_proportions": expected_proportions,
                "actual_proportions": actual_proportions,
            }
        )

    if as_numpy or isinstance(y_true, np.ndarray):
        return create_metric_result(result, as_numpy=True)
    return create_metric_result(result, as_numpy=False)


def marginal_calibration_error(
    y_pred_samples: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    n_bins: int = 20,
    return_diagnostics: bool = False,
    as_numpy: bool = False,
) -> Dict[str, Union[torch.Tensor, float, np.ndarray]]:
    """
    Functional Marginal Calibration Error (MCE) for probabilistic regression.
    """
    y_true_t = convert_to_tensor(y_true)
    y_pred_samples_t = convert_to_tensor(y_pred_samples)

    if y_pred_samples_t.dim() < 2:
        raise ValueError("y_pred_samples must have shape [n_samples, batch, ...]")

    if y_true_t.dim() > 1:
        y_true_t = y_true_t.view(y_true_t.shape[0], -1)
        y_true_flat = y_true_t.reshape(-1)
    else:
        y_true_flat = y_true_t

    samples_flat = y_pred_samples_t.reshape(y_pred_samples_t.shape[0], -1)

    all_values = torch.cat([y_true_flat, samples_flat.view(-1)])
    min_val = torch.min(all_values)
    max_val = torch.max(all_values)

    if torch.isclose(min_val, max_val, rtol=1e-5):
        min_val = min_val - 1e-5
        max_val = max_val + 1e-5

    bin_edges = torch.linspace(min_val, max_val, n_bins + 1, device="cpu")
    y_true_cpu = y_true_flat.detach().cpu()

    obs_hist = torch.histogram(y_true_cpu, bin_edges)[0]
    obs_cdf = torch.cumsum(obs_hist, dim=0) / max(1, len(y_true_flat))

    pred_cdfs = []
    for i in range(samples_flat.shape[0]):
        pred_samples_cpu = samples_flat[i].detach().cpu()
        pred_hist = torch.histogram(pred_samples_cpu, bin_edges)[0]
        pred_cdf = torch.cumsum(pred_hist, dim=0) / max(1, pred_samples_cpu.numel())
        pred_cdfs.append(pred_cdf)

    pred_cdf_mean = torch.stack(pred_cdfs).mean(dim=0)
    abs_errors = torch.abs(obs_cdf - pred_cdf_mean)

    mce = torch.mean(abs_errors)
    rmsce = torch.sqrt(torch.mean((obs_cdf - pred_cdf_mean) ** 2))
    max_mce = torch.max(abs_errors)

    device = y_true_t.device
    mce = mce.to(device)
    rmsce = rmsce.to(device)
    max_mce = max_mce.to(device)

    result = {
        "marginal_calibration_error": mce,
        "root_mean_squared_mce": rmsce,
        "maximum_marginal_calibration_error": max_mce,
    }

    if return_diagnostics:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        result.update(
            {
                "bin_centers": bin_centers.to(device),
                "observed_cdf": obs_cdf.to(device),
                "predicted_cdf": pred_cdf_mean.to(device),
                "abs_errors": abs_errors.to(device),
            }
        )

    if as_numpy or isinstance(y_true, np.ndarray):
        return create_metric_result(result, as_numpy=True)
    return create_metric_result(result, as_numpy=False)


def bias(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    as_numpy: bool = False,
) -> Union[torch.Tensor, float, np.ndarray]:
    """
    Compute mean prediction bias (mean of y_pred - y_true).
    """
    y_pred_t = convert_to_tensor(y_pred)
    y_true_t = convert_to_tensor(y_true)
    validate_inputs(y_pred_t, y_true_t)
    result = torch.mean(y_pred_t - y_true_t)

    if as_numpy or isinstance(y_pred, np.ndarray) or isinstance(y_true, np.ndarray):
        return create_metric_result(result, as_numpy=True)
    return create_metric_result(result, as_numpy=False)


def calibration_metrics_report(
    dist_or_samples: Optional[Union[torch.distributions.Distribution, torch.Tensor, np.ndarray, Dict[str, Any]]],
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred_quantiles: Optional[Dict[float, Union[torch.Tensor, np.ndarray]]] = None,
    n_bins: int = 20,
    n_samples: int = 100,
) -> Dict[str, Union[torch.Tensor, float, np.ndarray]]:
    """
    Generate a calibration report from distributions, samples, or quantiles.
    """
    results: Dict[str, Union[torch.Tensor, float, np.ndarray]] = {}

    if y_pred_quantiles is not None:
        results.update(
            expected_calibration_error(
                y_pred_quantiles, y_true, return_diagnostics=False, as_numpy=False
            )
        )

    if dist_or_samples is not None:
        if isinstance(dist_or_samples, dict):
            loc = convert_to_tensor(dist_or_samples.get("loc", dist_or_samples.get("mean")))
            scale = convert_to_tensor(dist_or_samples.get("scale", dist_or_samples.get("std")))
            dist_or_samples = torch.distributions.Normal(loc, scale)

        if isinstance(dist_or_samples, torch.distributions.Distribution):
            samples = dist_or_samples.sample((n_samples,))
        else:
            samples = dist_or_samples

        results.update(
            marginal_calibration_error(samples, y_true, n_bins=n_bins, return_diagnostics=False)
        )

    return results
