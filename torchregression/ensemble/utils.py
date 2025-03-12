"""
Utility functions for ensemble models.

This module provides common utility functions used across different
ensemble implementations, including tools for prediction aggregation
and uncertainty estimation.
"""

import torch
from typing import Callable, List, Dict, Union


def run_ensemble_model(
    model: Callable,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    return_individual: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Run a model on multiple input variations and aggregate results.

    Args:
        model: Model function to run
        inputs: List of input tensors or batched tensor [n_samples, batch_size, ...]
        return_individual: Whether to return individual predictions

    Returns:
        Dictionary with aggregated predictions:
        - mean: Mean prediction across samples
        - variance: Variance of predictions (epistemic uncertainty)
        - individual_preds: Individual predictions (if return_individual=True)
    """
    # Convert list to stacked tensor if needed
    if isinstance(inputs, list):
        inputs_stacked = torch.stack(inputs)
    else:
        inputs_stacked = inputs

    n_samples = inputs_stacked.shape[0]
    batch_size = inputs_stacked.shape[1]

    # Reshape for batch processing
    inputs_flat = inputs_stacked.reshape(-1, *inputs_stacked.shape[2:])

    # Run model on all inputs
    with torch.no_grad():
        outputs_flat = model(inputs_flat)

        # Handle different output types
        if isinstance(outputs_flat, tuple):
            # Take first output if model returns multiple outputs
            outputs_flat = outputs_flat[0]

        # Get output feature dimension
        if outputs_flat.ndim == 1:
            # Handle scalar output case
            n_outputs = 1
            outputs = outputs_flat.reshape(n_samples, batch_size, 1)
        else:
            # Normal tensor output case
            n_outputs = outputs_flat.shape[1]
            outputs = outputs_flat.reshape(n_samples, batch_size, n_outputs)

    # Calculate mean and variance across samples
    mean_pred = torch.mean(outputs, dim=0)  # [batch_size, n_outputs]
    variance = torch.var(outputs, dim=0, unbiased=True)  # [batch_size, n_outputs]

    # Prepare results
    result = {"mean": mean_pred, "variance": variance}

    if return_individual:
        result["individual_preds"] = outputs

    return result


def run_heteroscedastic_ensemble_model(
    model: Callable, inputs: Union[torch.Tensor, List[torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Run a heteroscedastic model on multiple input variations and aggregate results.

    This assumes the model outputs both mean and variance predictions.

    Args:
        model: Heteroscedastic model function
        inputs: List of input tensors or batched tensor [n_samples, batch_size, ...]

    Returns:
        Dictionary with:
        - mean: Mean prediction across samples
        - variance: Total variance
        - epistemic_variance: Variance of means
        - aleatoric_variance: Mean of variances
    """
    # Convert list to stacked tensor if needed
    if isinstance(inputs, list):
        inputs_stacked = torch.stack(inputs)
    else:
        inputs_stacked = inputs

    n_samples = inputs_stacked.shape[0]
    batch_size = inputs_stacked.shape[1]

    # Reshape for batch processing
    inputs_flat = inputs_stacked.reshape(-1, *inputs_stacked.shape[2:])

    # Run model on all inputs
    with torch.no_grad():
        outputs_flat = model(inputs_flat)

        # Extract means and variances based on model output format
        if isinstance(outputs_flat, tuple) and len(outputs_flat) == 2:
            # (mean, log_var) tuple format
            means_flat, log_vars_flat = outputs_flat
            variances_flat = torch.exp(log_vars_flat)
        elif (
            isinstance(outputs_flat, dict)
            and "means" in outputs_flat
            and "log_vars" in outputs_flat
        ):
            # Dictionary format from BatchEnsemble
            means_flat = outputs_flat["means"]
            variances_flat = torch.exp(outputs_flat["log_vars"])
        elif outputs_flat.shape[1] == 2 * outputs_flat.shape[1] // 2:
            # Concatenated [mean, log_var] format
            n_dims = outputs_flat.shape[1] // 2
            means_flat = outputs_flat[:, :n_dims]
            log_vars_flat = outputs_flat[:, n_dims:]
            variances_flat = torch.exp(log_vars_flat)
        else:
            raise ValueError(
                "Model output format not recognized for heteroscedastic uncertainty. "
                "Expected tuple (mean, log_var) or tensor with concatenated [mean, log_var]."
            )

        # Reshape to [n_samples, batch_size, n_outputs]
        n_outputs = means_flat.shape[1]
        means = means_flat.reshape(n_samples, batch_size, n_outputs)
        variances = variances_flat.reshape(n_samples, batch_size, n_outputs)

    # Calculate ensemble statistics
    # Mean prediction (average of means)
    ensemble_mean = torch.mean(means, dim=0)  # [batch_size, n_outputs]

    # Epistemic uncertainty (variance of means)
    epistemic_var = torch.var(means, dim=0, unbiased=True)  # [batch_size, n_outputs]

    # Aleatoric uncertainty (average of variances)
    aleatoric_var = torch.mean(variances, dim=0)  # [batch_size, n_outputs]

    # Total predictive variance
    total_var = epistemic_var + aleatoric_var  # [batch_size, n_outputs]

    return {
        "mean": ensemble_mean,
        "variance": total_var,
        "epistemic_variance": epistemic_var,
        "aleatoric_variance": aleatoric_var,
    }


def generate_prediction_samples(
    model: Callable, x: torch.Tensor, n_samples: int = 10, return_samples: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Generate multiple predictions using dropout at inference time (MC Dropout).

    Args:
        model: Model with dropout layers
        x: Input tensor [batch_size, ...]
        n_samples: Number of samples to generate
        return_samples: Whether to return individual samples

    Returns:
        Dictionary with:
        - mean: Mean prediction across samples
        - variance: Variance of predictions
        - samples: Individual predictions (if return_samples=True)
    """
    # Ensure model is in training mode to activate dropout
    model.train()

    # Generate predictions
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(x)
            samples.append(outputs)

    # Stack predictions [n_samples, batch_size, n_outputs]
    stacked_samples = torch.stack(samples)

    # Calculate statistics
    mean_pred = torch.mean(stacked_samples, dim=0)
    variance = torch.var(stacked_samples, dim=0, unbiased=True)

    # Prepare output
    result = {"mean": mean_pred, "variance": variance}

    if return_samples:
        result["samples"] = stacked_samples

    # Switch model back to evaluation mode
    model.eval()

    return result
