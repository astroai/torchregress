"""
Specialized algorithms for regression tasks.

This module provides implementations of algorithms for various
regression tasks, including IRLS, RC, SIMEX, and latent-input EIV training.
"""

from .covariance_pseudo_labels import (
    NeighborhoodCovarianceConfig,
    NeighborhoodCovariancePseudoLabeler,
    mahalanobis_covariance_pseudo_labels,
)
from .error_aware import ErrorAwareFeatureEncoder, NoiseAwareRegressor
from .irls import (
    IRLS,
    iteratively_reweighted_least_squares,
)
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
    "NeighborhoodCovarianceConfig",
    "NeighborhoodCovariancePseudoLabeler",
    "NoiseAwareRegressor",
    "mahalanobis_covariance_pseudo_labels",
]
