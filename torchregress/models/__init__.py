"""
Models package for torchregress.
"""

# Heteroscedastic regression utilities
from .heteroscedastic import DualHeadRegressionModel, create_dual_head_regression

__all__ = [
    "DualHeadRegressionModel",
    "create_dual_head_regression",
]
