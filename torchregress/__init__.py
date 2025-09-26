"""
TorchRegress: PyTorch-based regression modeling toolkit.

This package provides utilities, models, and loss functions for
various regression tasks in PyTorch, with a focus on uncertainty
estimation and robust regression techniques.
"""

__version__ = "0.1.0"

# Import main submodules
from . import algorithms, ensemble, losses, metrics, utils, wrappers
from .algorithms import IRLS, iteratively_reweighted_least_squares

# Export key components for easy access
from .losses.base import DistributionLoss, MaskedLoss, RegressionLoss

# Add key wrapper functions
from .wrappers import (
    create_deep_ensemble,
    create_mdn_model,
    wrap_pytorch_loss,
)
from .wrappers import (
    create_gaussian_model as create_gaussian_regression,
)
from .wrappers import (
    create_quantile_model as create_quantile_regression,
)
from .wrappers import (
    create_robust_model as create_robust_regression,
)

__all__ = [
    # Main modules
    "losses",
    "metrics",
    "algorithms",
    "ensemble",
    "utils",
    "wrappers",
    "MaskedLoss",
    "RegressionLoss",
    "DistributionLoss",
    "IRLS",
    "iteratively_reweighted_least_squares",
    "create_gaussian_regression",
    "create_robust_regression",
    "create_quantile_regression",
    "create_mdn_model",
    "create_deep_ensemble",
    "wrap_pytorch_loss",
    "__version__",
]
