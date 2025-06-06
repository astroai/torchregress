"""
Ensemble forecasting metrics and uncertainty decomposition.
"""
import torch
import numpy as np
from typing import Union, Dict, Tuple
from torch.distributions import Normal
from .utils import convert_to_tensor, apply_reduction
from .interval import interval_score, prediction_interval_coverage_probability


def ensemble_statistics(
    predictions: Union[torch.Tensor, np.ndarray],
    dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ensemble mean and variance across dimension `dim`.

    Args:
        predictions: Tensor or array of shape [n_models, ...]
        dim: dimension to aggregate

    Returns:
        mean, variance tensors matching shape without `dim`
    """
    preds = convert_to_tensor(predictions)
    mean = torch.mean(preds, dim=dim)
    var = torch.var(preds, dim=dim, unbiased=False)
    return mean, var


def uncertainty_decomposition(
    means: Union[torch.Tensor, np.ndarray],
    variances: Union[torch.Tensor, np.ndarray],
    dim: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Decompose uncertainty into epistemic, aleatoric, and total.

    Args:
        means: predicted means [n_models, ...]
        variances: predicted aleatoric variances [n_models, ...]
        dim: ensemble dimension

    Returns:
        dict with keys 'mean', 'epistemic_uncertainty', 'aleatoric_uncertainty', 'total_uncertainty'
    """
    means_t = convert_to_tensor(means)
    vars_t = convert_to_tensor(variances)
    mean, _ = ensemble_statistics(means_t, dim)
    epistemic = torch.var(means_t, dim=dim, unbiased=False)
    aleatoric = torch.mean(vars_t, dim=dim)
    total = epistemic + aleatoric
    return {
        'mean': mean,
        'epistemic_uncertainty': epistemic,
        'aleatoric_uncertainty': aleatoric,
        'total_uncertainty': total,
    }


def gaussian_nll_ensemble(
    means: Union[torch.Tensor, np.ndarray],
    variances: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    dim: int = 0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute Gaussian negative log-likelihood for ensemble forecasts.

    Args:
        means: predicted means [n_models, ...]
        variances: predicted aleatoric variances [n_models, ...]
        y_true: ground truth
        dim: ensemble dimension
        reduction: 'none', 'mean', or 'sum'

    Returns:
        NLL score
    """
    y = convert_to_tensor(y_true)
    stats = uncertainty_decomposition(means, variances, dim)
    mean = stats['mean']
    total_var = stats['total_uncertainty']
    total_var = torch.clamp(total_var, min=1e-6)
    diff2 = (y - mean) ** 2
    nll = 0.5 * (torch.log(2 * np.pi * total_var) + diff2 / total_var)
    return apply_reduction(nll, reduction)


def ensemble_interval_bounds(
    means: Union[torch.Tensor, np.ndarray],
    variances: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.1,
    dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute symmetric Gaussian prediction intervals from ensemble.

    Returns lower and upper bounds at level 1-alpha.
    """
    stats = uncertainty_decomposition(means, variances, dim)
    mean = stats['mean']
    total_var = torch.clamp(stats['total_uncertainty'], min=1e-6)
    sd = torch.sqrt(total_var)
    z = Normal(0, 1).icdf(torch.tensor(1 - alpha / 2, device=mean.device))
    lower = mean - z * sd
    upper = mean + z * sd
    return lower, upper


def ensemble_interval_metrics(
    means: Union[torch.Tensor, np.ndarray],
    variances: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Interval score and coverage for ensemble predictions.
    """
    lower, upper = ensemble_interval_bounds(means, variances, alpha)
    y = convert_to_tensor(y_true)
    score = interval_score(lower, upper, y, alpha, reduction='mean')
    picp = prediction_interval_coverage_probability(lower, upper, y, expected_coverage=1 - alpha)
    return {'interval_score': score, 'picp': picp}
