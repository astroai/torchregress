"""
Specialized algorithms for regression tasks.

This module provides implementations of algorithms for various
regression tasks, including IRLS, RC, SIMEX, and latent-input EIV training.
"""

from .irls import (
    IRLS,
    iteratively_reweighted_least_squares,
)
from .error_aware import ErrorAwareFeatureEncoder, NoiseAwareRegressor
from .latentnn import LatentNN
from .rc import RegressionCalibration
from .simex import SIMEX

__all__ = [
    # IRLS algorithm
    "iteratively_reweighted_least_squares",
    "IRLS",
    # EIV algorithms
    "RegressionCalibration",
    "SIMEX",
    "LatentNN",
    "ErrorAwareFeatureEncoder",
    "NoiseAwareRegressor",
]
