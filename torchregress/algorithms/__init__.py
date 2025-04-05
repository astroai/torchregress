"""
Specialized algorithms for regression tasks.

This module provides implementations of algorithms for various
regression tasks, including IRLS, Gaussian processes, and others.
"""

# Import IRLS algorithm
from .irls import (
    iteratively_reweighted_least_squares,
    IRLS,
    _huber_weights,
    _tukey_weights,
    _power_weights,
)

# Import Gaussian process algorithms
from .gaussian import (
    DiagonalGaussianNLL,
    GaussianNLLWithCovariance,
    HeteroscedasticGaussianNLL,
    fit_linear_gaussian,
    fit_bayesian_linear,
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
    # Gaussian process algorithms
    "DiagonalGaussianNLL",
    "GaussianNLLWithCovariance",
    "HeteroscedasticGaussianNLL",
    "fit_linear_gaussian",
    "fit_bayesian_linear",
]
