"""
Initialization for the torchregress.losses.eiv module.

This module provides various Error-in-Variables (EIV) loss functions
for regression tasks with uncertain inputs.
"""

# Import EIV utility functions
from .eiv_utils import (
    prepare_param,
    prepare_sigma,
    prepare_covariance,
    prepare_cross_covariance,
    compute_model_gradients,
    calculate_gaussian_nll,
    prepare_model_input_for_gradients,
    calculate_propagated_variance,
    generate_perturbed_samples,
)

# Import EIV standard losses
from .eiv_standard import (
    BaseEIVLoss,
    FunctionalEIVLoss,
    StructuralEIVLoss,
    OrthogonalDistanceRegressionLoss,
    EnsembleEIVLoss,
    # Aliases for backward compatibility
    GeneralErrorInVariablesLoss,
    CorrelatedEIVLoss,
)

# Import EIV Chamfer-based losses
from .eiv_chamfer import chamfer_distance, ChamferEIVLoss, HybridEIVChamferLoss

# Import EIV MDN-based losses and models
from .eiv_mdn import MDNEIVLoss, MDNEIVModel

# Import EIV robust sampling-based losses
from .eiv_rfit import (
    RobustEIVLoss,
    gaussian_variation,
    uniform_variation,
    bootstrap_variation,
    structured_variation,
    adversarial_variation,
)

# Import EIV quantile-based losses
from .eiv_quantile import QuantileEIVLoss, MultiQuantileEIVLoss

# Export everything
__all__ = [
    # Utility functions
    "prepare_param",
    "prepare_sigma",
    "prepare_covariance",
    "prepare_cross_covariance",
    "compute_model_gradients",
    "calculate_gaussian_nll",
    "prepare_model_input_for_gradients",
    "calculate_propagated_variance",
    "generate_perturbed_samples",
    # Standard EIV losses
    "BaseEIVLoss",
    "FunctionalEIVLoss",
    "StructuralEIVLoss",
    "OrthogonalDistanceRegressionLoss",
    "EnsembleEIVLoss",
    "GeneralErrorInVariablesLoss",  # Alias
    "CorrelatedEIVLoss",  # Alias
    # Chamfer-based losses
    "chamfer_distance",
    "ChamferEIVLoss",
    "HybridEIVChamferLoss",
    # MDN-based losses and models
    "MDNEIVLoss",
    "MDNEIVModel",
    # Robust sampling-based losses
    "RobustEIVLoss",
    "gaussian_variation",
    "uniform_variation",
    "bootstrap_variation",
    "structured_variation",
    "adversarial_variation",
    # Quantile-based losses
    "QuantileEIVLoss",
    "MultiQuantileEIVLoss",
]

# Version info
__version__ = "0.1.0"
