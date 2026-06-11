"""
Specialized algorithms for regression tasks.

This module provides implementations of algorithms for various
regression tasks, including IRLS, RC, SIMEX, and latent-input EIV training.
"""

from .adaptive_prior_vi import (
    AdaptivePriorGuide,
    SyntheticEnvironmentSampler,
    VIDSRegressor,
)
from .covariance_pseudo_labels import (
    NeighborhoodCovarianceConfig,
    NeighborhoodCovariancePseudoLabeler,
    mahalanobis_covariance_pseudo_labels,
)
from .error_aware import ErrorAwareFeatureEncoder, NoiseAwareRegressor
from .heteroscedastic_laplace import (
    HeteroscedasticLaplaceRegressor,
    NaturalHeteroscedasticHead,
    NaturalReparamHead,
)
from .irls import (
    IRLS,
    IRLSConfig,
    iteratively_reweighted_least_squares,
)
from .latentnn import LatentNN
from .rc import RegressionCalibration
from .simex import SIMEX
from .tictac import TaylorInducedCovarianceHead

__all__ = [
    # IRLS algorithm
    "iteratively_reweighted_least_squares",
    "IRLS",
    "IRLSConfig",
    # EIV algorithms
    "RegressionCalibration",
    "SIMEX",
    "LatentNN",
    "ErrorAwareFeatureEncoder",
    "NeighborhoodCovarianceConfig",
    "NeighborhoodCovariancePseudoLabeler",
    "NoiseAwareRegressor",
    "mahalanobis_covariance_pseudo_labels",
    # TIC-TAC algorithms
    "TaylorInducedCovarianceHead",
    # Laplace algorithms
    "NaturalHeteroscedasticHead",
    "NaturalReparamHead",
    "HeteroscedasticLaplaceRegressor",
    # VIDS algorithms
    "SyntheticEnvironmentSampler",
    "AdaptivePriorGuide",
    "VIDSRegressor",
]
