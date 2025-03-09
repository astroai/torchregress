"""
TorchRegression: PyTorch-based regression modeling toolkit.

This package provides utilities, models, and loss functions for 
various regression tasks in PyTorch, with a focus on uncertainty
estimation and robust regression techniques.
"""

__version__ = "0.1.0"

# Import main submodules
from . import losses
from . import metrics
from . import algorithms
from . import ensemble
from . import utils

# Export key components for easy access
from .losses import register_loss, get_loss_function, list_available_losses
from .losses.base import MaskedLoss, RegressionLoss, DistributionLoss
from .algorithms import IRLS, iteratively_reweighted_least_squares

__all__ = [
    # Main modules
    'losses',
    'metrics',
    'algorithms',
    'ensemble',
    'utils',
    
    # Key components
    'register_loss',
    'get_loss_function',
    'list_available_losses',
    'MaskedLoss',
    'RegressionLoss',
    'DistributionLoss',
    'IRLS',
    'iteratively_reweighted_least_squares',
    '__version__',
]