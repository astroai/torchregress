"""
Calibration metrics for evaluating probabilistic regression models.
"""

from typing import Any, Dict, Optional, Union, cast

import numpy as np
import torch
from torchmetrics import Metric

from torchregress.utils.tensor_ops import convert_to_tensor
from torchregress.utils.validation import validate_metric_inputs as validate_inputs


def _compute_histograms(samples: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """
    Compute histograms for each row in samples using bin_edges.

    Args:
        samples: Tensor of shape (S, N) containing S samples of N predictions.
        bin_edges: Tensor of shape (B+1,) defining B bins.

    Returns:
        Tensor of shape (S, B) containing counts.
    """
    S, N = samples.shape
    n_bins = bin_edges.shape[0] - 1

    indices = torch.bucketize(samples, bin_edges, right=True)
    indices = indices - 1
    indices.clamp_(0, n_bins - 1)

    offset = torch.arange(S, device=samples.device).unsqueeze(1) * n_bins
    flat_indices = (indices + offset).view(-1)

    counts_flat = torch.bincount(flat_indices, minlength=S * n_bins)
    return counts_flat.view(S, n_bins).float()


class ExpectedCalibrationError(Metric):
    """
    Calculate Expected Calibration Error (ECE) for quantile regression.

    References
    ----------
    .. [1] Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining Well-Calibrated
       Probabilities Using Bayesian Binning. In *AAAI 2015*.
       https://ojs.aaai.org/index.php/AAAI/article/view/9602
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        # TR-MET-13: the dead n_bins kwarg was removed; ECE here is
        # quantile-proportion based and never bins.
        super().__init__(**kwargs)
        self.add_state("y_pred_quantiles", default=[], dist_reduce_fx=None)
        self.add_state("y_true", default=[], dist_reduce_fx=None)

    def update(self, y_pred_quantiles: Dict[float, torch.Tensor], y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        from torchregress.metrics.utils import metric_state_list

        metric_state_list[dict[float, torch.Tensor]](self.y_pred_quantiles).append(y_pred_quantiles)
        metric_state_list[torch.Tensor](self.y_true).append(y_true)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute ECE."""
        from torchregress.metrics.utils import metric_state_list

        y_true_state = metric_state_list[torch.Tensor](self.y_true)
        y_true = torch.cat([convert_to_tensor(y) for y in y_true_state])

        y_pred_state = metric_state_list[dict[float, torch.Tensor]](self.y_pred_quantiles)
        y_pred_lists: Dict[float, list[torch.Tensor]] = {q: [] for q in y_pred_state[0]}

        for d in y_pred_state:
            for q, v in d.items():
                y_pred_lists[q].append(convert_to_tensor(v))

        y_pred_quantiles: Dict[float, torch.Tensor] = {
            q: torch.cat(tensors) for q, tensors in y_pred_lists.items()
        }

        device = y_true.device
        quantiles = sorted(y_pred_quantiles.keys())
        expected_proportions = torch.tensor(quantiles, device=device)

        preds = torch.stack([convert_to_tensor(y_pred_quantiles[q]).to(device) for q in quantiles])

        if len(preds) > 0:
            validate_inputs(preds[0], y_true)

        actual_proportions = (y_true.unsqueeze(0) <= preds).float().flatten(1).mean(dim=1)
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
        from torchregress.metrics.utils import metric_state_list

        metric_state_list[torch.Tensor](self.y_pred_samples).append(y_pred_samples)
        metric_state_list[torch.Tensor](self.y_true).append(y_true)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute MCE."""
        from torchregress.metrics.utils import metric_state_list

        with torch.no_grad():
            y_true = torch.cat(
                [convert_to_tensor(y) for y in metric_state_list[torch.Tensor](self.y_true)]
            )
            y_pred_samples = torch.cat(
                [
                    convert_to_tensor(y)
                    for y in metric_state_list[torch.Tensor](self.y_pred_samples)
                ],
                dim=1,
            )

            all_values = torch.cat([y_true, y_pred_samples.view(-1)])
            min_val = torch.min(all_values)
            max_val = torch.max(all_values)

            if torch.isclose(min_val, max_val, rtol=1e-5):
                min_val = min_val - 1e-5
                max_val = max_val + 1e-5

            device = y_true.device
            bin_edges = torch.linspace(min_val, max_val, self.n_bins + 1, device=device)

            n_samples_per_point = y_pred_samples.shape[1]
            obs_hist = torch.histogram(y_true.float(), bin_edges)[0]
            obs_cdf = torch.cumsum(obs_hist, dim=0) / max(1, len(y_true))
            pred_hists = _compute_histograms(y_pred_samples, bin_edges)
            # Normalize each row of histograms to a proper CDF *before*
            # averaging across MC samples: the previous version divided by
            # ``y_pred_samples.shape[1]`` which is the per-row sample count
            # only when the sample axis is 1.  Use the row count explicitly
            # so the average remains an average of CDFs (in [0, 1]).
            row_sums = pred_hists.sum(dim=1, keepdim=True).clamp(min=1.0)
            pred_cdfs = torch.cumsum(pred_hists, dim=1) / row_sums
            pred_cdf_mean = pred_cdfs.mean(dim=0)

            abs_errors = torch.abs(obs_cdf - pred_cdf_mean)
            mce = torch.mean(abs_errors)
            rmsce = torch.sqrt(torch.mean((obs_cdf - pred_cdf_mean) ** 2))
            max_mce = torch.max(abs_errors)
            del n_samples_per_point  # kept for clarity; not needed for the new formula

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

    preds = torch.stack([convert_to_tensor(y_pred_quantiles[q]).to(device) for q in quantiles])

    if len(preds) > 0:
        validate_inputs(preds[0], y_true_t)

    actual_proportions = (y_true_t.unsqueeze(0) <= preds).float().flatten(1).mean(dim=1)
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
        from torchregress.metrics.utils import create_metric_result

        return cast(
            Dict[str, Union[torch.Tensor, float, np.ndarray]],
            create_metric_result(result, as_numpy=True),
        )
    from torchregress.metrics.utils import create_metric_result

    return cast(
        Dict[str, Union[torch.Tensor, float, np.ndarray]],
        create_metric_result(result, as_numpy=False),
    )


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

    device = y_true_t.device
    bin_edges = torch.linspace(min_val, max_val, n_bins + 1, device=device)

    obs_hist = torch.histogram(y_true_flat.float(), bin_edges)[0]
    obs_cdf = torch.cumsum(obs_hist, dim=0) / max(1, len(y_true_flat))

    pred_hists = _compute_histograms(samples_flat, bin_edges)
    # Normalize each row of histograms to a proper CDF *before* averaging
    # across MC samples: dividing by ``samples_flat.shape[1]`` is only
    # correct when the sample axis is 1, which is not always the case for
    # multi-dimensional ``y_true`` with leading sample axis.  Use the
    # row-wise histogram sum to convert counts → probabilities.
    row_sums = pred_hists.sum(dim=1, keepdim=True).clamp(min=1.0)
    pred_cdfs = torch.cumsum(pred_hists, dim=1) / row_sums
    pred_cdf_mean = pred_cdfs.mean(dim=0)
    abs_errors = torch.abs(obs_cdf - pred_cdf_mean)
    mce = torch.mean(abs_errors)
    rmsce = torch.sqrt(torch.mean((obs_cdf - pred_cdf_mean) ** 2))
    max_mce = torch.max(abs_errors)

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
        from torchregress.metrics.utils import create_metric_result

        return cast(
            Dict[str, Union[torch.Tensor, float, np.ndarray]],
            create_metric_result(result, as_numpy=True),
        )
    from torchregress.metrics.utils import create_metric_result

    return cast(
        Dict[str, Union[torch.Tensor, float, np.ndarray]],
        create_metric_result(result, as_numpy=False),
    )


def calibration_score(
    y_true: Union[torch.Tensor, np.ndarray],
    pred_mean: Union[torch.Tensor, np.ndarray],
    pred_std: Union[torch.Tensor, np.ndarray],
    n_levels: int = 19,
    as_numpy: bool = False,
) -> Dict[str, Union[torch.Tensor, float, np.ndarray]]:
    """
    Convenience calibration score for Gaussian predictive outputs.

    Builds quantile predictions from ``pred_mean`` and ``pred_std`` and then computes
    quantile calibration errors via :func:`expected_calibration_error`.
    """
    mean_t = convert_to_tensor(pred_mean)
    std_t = convert_to_tensor(pred_std).to(mean_t.device).clamp(min=1e-8)
    levels = torch.linspace(0.05, 0.95, n_levels, device=mean_t.device)
    standard = torch.distributions.Normal(
        torch.tensor(0.0, device=mean_t.device),
        torch.tensor(1.0, device=mean_t.device),
    )
    quantiles = {}
    for q in levels:
        z = standard.icdf(q)
        quantiles[float(q.item())] = mean_t + z * std_t

    return expected_calibration_error(quantiles, y_true, as_numpy=as_numpy)


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
        from torchregress.metrics.utils import create_metric_result

        return cast(
            Union[torch.Tensor, float, np.ndarray],
            create_metric_result(result, as_numpy=True),
        )
    from torchregress.metrics.utils import create_metric_result

    return cast(
        Union[torch.Tensor, float, np.ndarray],
        create_metric_result(result, as_numpy=False),
    )


def calibration_metrics_report(
    dist_or_samples: Optional[
        Union[torch.distributions.Distribution, torch.Tensor, np.ndarray, Dict[str, Any]]
    ],
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
            loc_val: Any = dist_or_samples.get("loc")
            if loc_val is None:
                loc_val = dist_or_samples.get("mean")
            scale_val: Any = dist_or_samples.get("scale")
            if scale_val is None:
                scale_val = dist_or_samples.get("std")
            if loc_val is None or scale_val is None:
                raise ValueError("dist_or_samples dict must contain loc/mean and scale/std")
            loc = convert_to_tensor(cast(Any, loc_val))
            scale = convert_to_tensor(cast(Any, scale_val))
            dist_or_samples = torch.distributions.Normal(loc, scale)

        samples: Union[torch.Tensor, np.ndarray]
        if isinstance(dist_or_samples, torch.distributions.Distribution):
            samples = dist_or_samples.sample((n_samples,))
        else:
            samples = cast(Union[torch.Tensor, np.ndarray], dist_or_samples)

        results.update(
            marginal_calibration_error(samples, y_true, n_bins=n_bins, return_diagnostics=False)
        )

    return results


__all__ = [
    "ExpectedCalibrationError",
    "MarginalCalibrationError",
    "bias",
    "calibration_metrics_report",
    "calibration_score",
    "expected_calibration_error",
    "marginal_calibration_error",
]
