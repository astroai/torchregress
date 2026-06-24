"""
Utility functions for ensemble models.

This module provides common utility functions used across different
ensemble implementations, including tools for prediction aggregation
and uncertainty estimation.
"""

from typing import Callable, Dict, List, Union

import torch
import torch.nn as nn

from torchregress.utils.gaussian_output import parse_heteroscedastic_output, variance_from_logvar

__all__ = [
    "generate_prediction_samples",
    "parse_heteroscedastic_output",
    "run_ensemble_model",
    "run_heteroscedastic_ensemble_model",
]


def run_ensemble_model(
    model: Callable,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    return_individual: bool = False,
    correction: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Run a model on multiple input variations and aggregate results.

    Args:
        model: Model function to run
        inputs: List of input tensors or batched tensor [n_samples, batch_size, ...]
        return_individual: Whether to return individual predictions
        correction: Bessel's correction setting. Default: 0

    Returns:
        Dictionary with aggregated predictions:
        - mean: Mean prediction across samples
        - variance: Variance of predictions (epistemic uncertainty)
        - individual_preds: Individual predictions (if return_individual=True)
    """
    if isinstance(inputs, list):
        inputs_stacked = torch.stack(inputs)
    else:
        inputs_stacked = inputs

    n_samples = inputs_stacked.shape[0]
    batch_size = inputs_stacked.shape[1]
    inputs_flat = inputs_stacked.reshape(-1, *inputs_stacked.shape[2:])

    with torch.no_grad():
        outputs_flat = model(inputs_flat)

        if isinstance(outputs_flat, tuple):
            outputs_flat = outputs_flat[0]

        if outputs_flat.ndim == 1:
            n_outputs = 1
            outputs = outputs_flat.reshape(n_samples, batch_size, 1)
        else:
            n_outputs = outputs_flat.shape[1]
            outputs = outputs_flat.reshape(n_samples, batch_size, n_outputs)

    mean_pred = torch.mean(outputs, dim=0)
    variance = torch.var(outputs, dim=0, correction=correction)

    result = {"mean": mean_pred, "variance": variance}

    if return_individual:
        result["individual_preds"] = outputs

    return result


def run_heteroscedastic_ensemble_model(
    model: Callable,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    correction: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Run a heteroscedastic model on multiple input variations and aggregate results.

    This assumes the model outputs both mean and variance predictions.
    """
    if isinstance(inputs, list):
        inputs_stacked = torch.stack(inputs)
    else:
        inputs_stacked = inputs

    n_samples = inputs_stacked.shape[0]
    batch_size = inputs_stacked.shape[1]
    inputs_flat = inputs_stacked.reshape(-1, *inputs_stacked.shape[2:])

    with torch.no_grad():
        outputs_flat = model(inputs_flat)
        means_flat, log_vars_flat = parse_heteroscedastic_output(outputs_flat)
        variances_flat = variance_from_logvar(log_vars_flat)

        n_outputs = means_flat.shape[1]
        means = means_flat.reshape(n_samples, batch_size, n_outputs)
        variances = variances_flat.reshape(n_samples, batch_size, n_outputs)

    ensemble_mean = torch.mean(means, dim=0)
    epistemic_var = torch.var(means, dim=0, correction=correction)
    aleatoric_var = torch.mean(variances, dim=0)
    total_var = epistemic_var + aleatoric_var

    return {
        "mean": ensemble_mean,
        "variance": total_var,
        "epistemic_variance": epistemic_var,
        "aleatoric_variance": aleatoric_var,
    }


def generate_prediction_samples(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 10,
    return_samples: bool = False,
    correction: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Generate multiple predictions using dropout at inference time (MC Dropout).
    """
    model.train()

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(x)
            samples.append(outputs)

    stacked_samples = torch.stack(samples)
    mean_pred = torch.mean(stacked_samples, dim=0)
    variance = torch.var(stacked_samples, dim=0, correction=correction)

    result = {"mean": mean_pred, "variance": variance}

    if return_samples:
        result["samples"] = stacked_samples

    return result
