"""
Initialization for the torchregression.losses.eiv module.

This module provides various Error-in-Variables (EIV) loss functions
for regression tasks with uncertain inputs.
"""

# Import EIV loss functions
from .eiv_utils import (
    prepare_param, 
    prepare_sigma, 
    prepare_covariance,
    prepare_cross_covariance,
    compute_model_gradients, 
    calculate_gaussian_nll,
    prepare_model_input_for_gradients,
    calculate_propagated_variance
)
from .eiv_chamfer import ChamferEIVLoss, HybridEIVChamferLoss
from .eiv_mdn import MDNEIVLoss
from .eiv_rfit import RobustEIVLoss

__all__ = [
    'prepare_param', 
    'prepare_sigma', 
    'prepare_covariance',
    'prepare_cross_covariance',
    'compute_model_gradients', 
    'calculate_gaussian_nll',
    'prepare_model_input_for_gradients',
    'calculate_propagated_variance',
    'ChamferEIVLoss', 
    'HybridEIVChamferLoss',
    'MDNEIVLoss',
    'RobustEIVLoss'
]

# Version info
__version__ = '0.1.0'
