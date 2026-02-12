"""
torchregress: PyTorch-based regression modeling toolkit.

This package provides utilities, models, and loss functions for
various regression tasks, with a focus on uncertainty
estimation and robust regression techniques.
"""

__version__ = "0.1.0"

# Import main submodules
from . import algorithms, ensemble, losses, metrics, utils, viz
from .algorithms import iteratively_reweighted_least_squares

# Export key components for easy access
from .losses import (
    BaseLoss,
    DistributionLoss,
    RegressionLoss,
)

__all__ = [
    # Main modules
    "losses",
    "metrics",
    "algorithms",
    "ensemble",
    "viz",
    "utils",
    # Core classes
    "BaseLoss",
    "RegressionLoss",
    "DistributionLoss",
    # Core functions
    "iteratively_reweighted_least_squares",
    "__version__",
]
