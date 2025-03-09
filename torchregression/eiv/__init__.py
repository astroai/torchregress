"""
Error-in-Variables (EIV) models and utilities.

This module provides implementations for regression with errors in both 
dependent and independent variables. It includes utilities for uncertainty 
propagation, robust fitting methods, and specialized EIV losses.
"""

# Import EIV utilities
from .eiv_utils import (
    prepare_model_input_for_gradients,
    compute_model_gradients,
    calculate_propagated_variance,
    prepare_sigma,
    prepare_param,
    prepare_covariance,
    prepare_cross_covariance,
    calculate_gaussian_nll
)

# Import Chamfer distance EIV
from .eiv_chamfer import (
    chamfer_distance,
    ChamferEIVLoss,
    HybridEIVChamferLoss
)

# Import Mixture Density Network EIV
from .eiv_mdn import (
    MDNEIVLoss,
    MDNEIVModel
)

# Import robust fitting methods
from .eiv_rfit import (
    gaussian_variation,
    uniform_variation,
    bootstrap_variation,
    RobustEIVLoss
)

__all__ = [
    # EIV utilities
    'prepare_model_input_for_gradients',
    'compute_model_gradients',
    'calculate_propagated_variance',
    'prepare_sigma',
    'prepare_param',
    'prepare_covariance',
    'prepare_cross_covariance',
    'calculate_gaussian_nll',
    
    # Chamfer EIV
    'chamfer_distance',
    'ChamferEIVLoss',
    'HybridEIVChamferLoss',
    
    # MDN EIV
    'MDNEIVLoss',
    'MDNEIVModel',
    
    # Robust fitting
    'gaussian_variation',
    'uniform_variation',
    'bootstrap_variation',
    'RobustEIVLoss'
]
