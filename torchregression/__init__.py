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
from . import wrappers  # Add import for wrappers module

# Export key components for easy access
from .losses import register_loss, get_loss_function, list_available_losses
from .losses.base import MaskedLoss, RegressionLoss, DistributionLoss
from .algorithms import IRLS, iteratively_reweighted_least_squares

# Add key wrapper functions
from .wrappers import (
    create_gaussian_regression,
    create_robust_regression,
    create_quantile_regression,
    create_histogram_regression,
    create_mdn_model,
    create_deep_ensemble,
    wrap_pytorch_loss,
)

__all__ = [
    # Main modules
    "losses",
    "metrics",
    "algorithms",
    "ensemble",
    "utils",
    "wrappers",  # Add wrappers module
    # Key components
    "register_loss",
    "get_loss_function",
    "list_available_losses",
    "MaskedLoss",
    "RegressionLoss",
    "DistributionLoss",
    "IRLS",
    "iteratively_reweighted_least_squares",
    # Add wrapper exports
    "create_gaussian_regression",
    "create_robust_regression",
    "create_quantile_regression",
    "create_histogram_regression",
    "create_mdn_model",
    "create_deep_ensemble",
    "wrap_pytorch_loss",
    "__version__",
]
