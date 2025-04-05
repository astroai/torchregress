"""
Specialized algorithms for regression tasks.

This module provides implementations of algorithms for various
regression tasks, including IRLS, Gaussian processes, and others.
"""

# Import IRLS algorithm
from .irls import (
    iteratively_reweighted_least_squares,
    IRLS,
)

# Import weight functions from utils
from ..utils.irls_utils import (
    huber_weights as _huber_weights,
    tukey_weights as _tukey_weights,
    power_weights as _power_weights,
)

# Convenience aliases
IRLS_Algorithm = IRLS

__all__ = [
    # IRLS algorithm
    "iteratively_reweighted_least_squares",
    "IRLS",
    "IRLS_Algorithm",
    "_huber_weights",
    "_tukey_weights",
    "_power_weights",
]
