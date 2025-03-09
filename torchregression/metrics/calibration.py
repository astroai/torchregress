"""
Calibration metrics for evaluating probabilistic regression models.
"""

import torch
import numpy as np
from typing import Union, Optional, Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from torchregression.metrics.utils import convert_to_tensor, apply_reduction, create_metric_result

def expected_calibration_error(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred_quantiles: Dict[float, Union[torch.Tensor, np.ndarray]],
    n_bins: int = 10,
    return_diagnostics: bool = False
) -> Dict[str, Union[float, List]]:
    """
    Calculate Expected Calibration Error (ECE) for quantile regression.
    
    Args:
        y_true: Ground truth values
        y_pred_quantiles: Dictionary mapping quantile levels to predictions
        n_bins: Number of bins for discretizing predictions
        return_diagnostics: Whether to return additional diagnostic information
        
    Returns:
        Dictionary with calibration metrics
    """
    y_true = convert_to_tensor(y_true)
    
    quantiles = sorted(y_pred_quantiles.keys())
    expected_proportions = torch.tensor(quantiles, device=y_true.device)
    actual_proportions = []
    
    for q in quantiles:
        q_pred = convert_to_tensor(y_pred_quantiles[q])
        proportion_below = torch.mean((y_true <= q_pred).float())
        actual_proportions.append(proportion_below)
    
    actual_proportions = torch.stack(actual_proportions)
    
    # Calculate calibration errors
    abs_errors = torch.abs(actual_proportions - expected_proportions)
    mace = torch.mean(abs_errors)  # Mean Absolute Calibration Error
    rmsce = torch.sqrt(torch.mean((actual_proportions - expected_proportions)**2))  # Root Mean Square Calibration Error
    
    # Calculate maximum calibration error
    max_ce = torch.max(abs_errors)
    
    result = {
        'mean_absolute_calibration_error': mace.item(),
        'root_mean_squared_calibration_error': rmsce.item(),
        'maximum_calibration_error': max_ce.item(),
    }
    
    if return_diagnostics:
        # Bin the errors
        bin_edges = torch.linspace(0, 1, n_bins + 1)
        bin_indices = torch.bucketize(expected_proportions, bin_edges) - 1
        
        bin_errors = []
        for i in range(n_bins):
            mask = bin_indices == i
            if torch.any(mask):
                bin_error = torch.mean(abs_errors[mask])
                bin_errors.append(bin_error.item())
            else:
                bin_errors.append(0.0)
        
        # Add diagnostic information
        result.update({
            'bin_errors': bin_errors,
            'expected_proportions': expected_proportions.tolist(),
            'actual_proportions': actual_proportions.tolist()
        })
    
    return result

def calibration_error(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred_quantiles: Dict[float, Union[torch.Tensor, np.ndarray]],
    n_bins: int = 10
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Wrapper around expected_calibration_error for backward compatibility.
    """
    return expected_calibration_error(y_true, y_pred_quantiles, n_bins, return_diagnostics=True)

def plot_reliability_diagram(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred_quantiles: Dict[float, Union[torch.Tensor, np.ndarray]],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Reliability Diagram",
    show_diagonal: bool = True,
    show_grid: bool = True,
    color: str = "blue",
    marker: str = "o",
    markersize: int = 8,
    return_figure: bool = False
) -> Optional[Figure]:
    """
    Plot a reliability diagram for quantile predictions.
    
    A reliability diagram plots the predicted quantiles against the 
    empirical proportions of observations below each quantile.
    
    Args:
        y_true: Ground truth values
        y_pred_quantiles: Dictionary mapping quantile levels to predictions
        figsize: Figure size (width, height)
        title: Plot title
        show_diagonal: Whether to show diagonal line (perfect calibration)
        show_grid: Whether to show grid
        color: Line color
        marker: Marker style
        markersize: Size of markers
        return_figure: If True, return figure object instead of displaying
        
    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    # Calculate calibration metrics
    calibration_metrics = expected_calibration_error(y_true, y_pred_quantiles, return_diagnostics=True)
    
    # Extract data for plotting
    expected = calibration_metrics['expected_proportions']
    actual = calibration_metrics['actual_proportions']
    mace = calibration_metrics['mean_absolute_calibration_error']
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data points
    ax.plot(expected, actual, marker=marker, markersize=markersize, 
            linestyle='-', color=color, label=f'MACE: {mace:.4f}')
    
    # Show diagonal line for perfect calibration
    if show_diagonal:
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    ax.set_xlabel('Expected proportion')
    ax.set_ylabel('Observed proportion')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title(title)
    ax.legend(loc='best')
    
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    if return_figure:
        return fig
    else:
        plt.tight_layout()
        plt.show()
        return None

def marginal_calibration_error(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred_samples: Union[torch.Tensor, np.ndarray],
    n_bins: int = 20,
    return_diagnostics: bool = False
) -> Dict[str, Union[float, List]]:
    """
    Calculate Marginal Calibration Error (MCE) for probabilistic regression.
    
    MCE measures how well the predictive distribution's marginals match
    the empirical distribution of observations.
    
    Args:
        y_true: Ground truth values [batch_size]
        y_pred_samples: Samples from predictive distribution [n_samples, batch_size]
        n_bins: Number of bins for histogram
        return_diagnostics: Whether to return additional diagnostic information
        
    Returns:
        Dictionary with calibration metrics
    """
    y_true = convert_to_tensor(y_true)
    y_pred_samples = convert_to_tensor(y_pred_samples)
    
    # Calculate empirical CDFs
    # First, determine range from combined predictions and observations
    all_values = torch.cat([y_true, y_pred_samples.view(-1)])
    min_val = torch.min(all_values)
    max_val = torch.max(all_values)
    
    # Create bin edges
    bin_edges = torch.linspace(min_val, max_val, n_bins + 1)
    
    # Calculate empirical CDF for observations
    obs_hist = torch.histogram(y_true, bin_edges)[0]
    obs_cdf = torch.cumsum(obs_hist, dim=0) / len(y_true)
    
    # Calculate empirical CDF for predictions (average across samples)
    pred_cdfs = []
    for i in range(y_pred_samples.shape[0]):  # Loop through samples
        pred_hist = torch.histogram(y_pred_samples[i], bin_edges)[0]
        pred_cdf = torch.cumsum(pred_hist, dim=0) / len(y_pred_samples[i])
        pred_cdfs.append(pred_cdf)
    
    pred_cdf_mean = torch.stack(pred_cdfs).mean(dim=0)
    
    # Calculate calibration errors
    abs_errors = torch.abs(obs_cdf - pred_cdf_mean)
    mce = torch.mean(abs_errors)  # Mean Marginal Calibration Error
    rmsce = torch.sqrt(torch.mean((obs_cdf - pred_cdf_mean)**2))  # Root Mean Square MCE
    max_mce = torch.max(abs_errors)  # Maximum MCE
    
    result = {
        'marginal_calibration_error': mce.item(),
        'root_mean_squared_mce': rmsce.item(),
        'maximum_marginal_calibration_error': max_mce.item(),
    }
    
    if return_diagnostics:
        # Add diagnostic information
        result.update({
            'bin_centers': (bin_edges[:-1] + bin_edges[1:]).tolist(),
            'observed_cdf': obs_cdf.tolist(),
            'predicted_cdf': pred_cdf_mean.tolist(),
            'abs_errors': abs_errors.tolist(),
        })
    
    return result

def calibration_metrics_report(
    y_true: Union[torch.Tensor, np.ndarray],
    distribution=None,  # Optional torch distribution
    y_pred_quantiles: Optional[Dict[float, Union[torch.Tensor, np.ndarray]]] = None,
    y_pred_samples: Optional[Union[torch.Tensor, np.ndarray]] = None,
    quantile_levels: List[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
) -> Dict[str, float]:
    """
    Generate a comprehensive report on calibration quality.
    
    This function calculates multiple calibration metrics depending on available inputs.
    
    Args:
        y_true: Ground truth values
        distribution: PyTorch distribution object (optional)
        y_pred_quantiles: Dictionary mapping quantile levels to predictions (optional)
        y_pred_samples: Samples from predictive distribution (optional)
        quantile_levels: Quantile levels to check if using distribution or samples
        
    Returns:
        Dictionary with calibration metrics
    """
    y_true = convert_to_tensor(y_true)
    metrics = {}
    
    # Generate quantiles from distribution if available
    if y_pred_quantiles is None and distribution is not None:
        y_pred_quantiles = {}
        for q in quantile_levels:
            y_pred_quantiles[q] = distribution.icdf(torch.tensor(q))
    
    # Generate quantiles from samples if available
    if y_pred_quantiles is None and y_pred_samples is not None:
        y_pred_samples = convert_to_tensor(y_pred_samples)
        y_pred_quantiles = {}
        for q in quantile_levels:
            y_pred_quantiles[q] = torch.quantile(y_pred_samples, q, dim=0)
    
    # Calculate ECE if quantiles are available
    if y_pred_quantiles is not None:
        ece_metrics = expected_calibration_error(y_true, y_pred_quantiles)
        metrics.update(ece_metrics)
    
    # Calculate Marginal Calibration Error if samples are available
    if y_pred_samples is not None:
        mce_metrics = marginal_calibration_error(y_true, y_pred_samples)
        metrics.update(mce_metrics)
    
    return metrics
