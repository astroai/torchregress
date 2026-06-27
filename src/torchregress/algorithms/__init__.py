"""
Specialized algorithms for regression tasks.

This module provides implementations of algorithms for various
regression tasks, including IRLS, RC, SIMEX, and latent-input EIV training.
"""
# ruff: noqa: F401 — re-exports
# ponytail: __all__ kept because test_public_api_contracts reads it

from .adaptive_prior_vi import (
    AdaptivePriorGuide,
    VIDSRegressor,
    sample_synthetic_environments,
)
from .covariance_pseudo_labels import (
    NeighborhoodCovarianceConfig,
    NeighborhoodCovariancePseudoLabeler,
)
from .error_aware import ErrorAwareFeatureEncoder, NoiseAwareRegressor
from .heteroscedastic_laplace import (
    HeteroscedasticLaplaceRegressor,
    NaturalHeteroscedasticHead,
    NaturalReparamHead,
)
from .irls import (
    IRLSConfig,
    iteratively_reweighted_least_squares,
)
from .ivon import IVON
from .latentnn import LatentNN
from .rc import RegressionCalibration
from .simex import SIMEX
from .tictac import TaylorInducedCovarianceHead
from .warmup_mc import WarmupMCTrainer

__all__ = [
    "IVON",
    "iteratively_reweighted_least_squares",
    "IRLSConfig",
    "RegressionCalibration",
    "SIMEX",
    "LatentNN",
    "ErrorAwareFeatureEncoder",
    "NeighborhoodCovarianceConfig",
    "NeighborhoodCovariancePseudoLabeler",
    "NoiseAwareRegressor",
    "TaylorInducedCovarianceHead",
    "NaturalHeteroscedasticHead",
    "NaturalReparamHead",
    "HeteroscedasticLaplaceRegressor",
    "AdaptivePriorGuide",
    "VIDSRegressor",
    "sample_synthetic_environments",
    "WarmupMCTrainer",
]
