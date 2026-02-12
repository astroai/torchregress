"""
Loss functions for regression tasks.

This module contains a variety of loss functions designed for regression problems,
including weighted losses, Gaussian-based losses, robust losses, and more.
"""



# Base classes and wrappers
from .base import (
    BaseLoss,
    DistributionLoss,
    RegressionLoss,
    WeightedCrossEntropyLoss,
    WeightedGaussianNLLLoss,
    WeightedLossWrapper,
    WeightedNLLLoss,
)

# Conformal prediction
from .conformal import (
    CQR,
    CTI,
    ConformalLoss,
    ConformalPredictor,
    DistributionalConformal,
    MultiDimensionalConformalLoss,
    MultiTargetConformal,
    R2CConformal,
    SplitConformal,
)
from .eiv import (
    BaseEIVLoss,
    EnsembleEIVLoss,
    FunctionalEIVLoss,
    OrthogonalDistanceRegressionLoss,
    StructuralEIVLoss,
)

# Evidential regression
from .evidential import EvidentialRegressionLoss

# Expectile losses
from .expectile import (
    AsymmetricLeastSquaresLoss,
    ExpectileCrossoverLoss,
    ExpectileLoss,
    MultiExpectileLoss,
)

# Gaussian losses
from .gaussian import (
    GaussianNLLLoss,
    LowRankGaussianLoss,
    MultivariateGaussianLoss,
    low_rank_output_dim,
    split_low_rank_gaussian_output,
)

# Imbalanced regression losses
from .imbalanced import (
    DensityWeightedLoss,
    FocalRLoss,
    LDSLoss,
)
from .loss_registry import get_regression_loss, list_regression_losses

# Mixture Density Networks
from .mdn import MixtureDensityLoss

# Normalizing flows (zuko is an optional dependency)
try:
    from .nflows import NormalizingFlowLoss
except ImportError:
    pass  # zuko not installed; normalizing flow features unavailable



# Poisson losses
from .poisson import (
    NegativeBinomialNLLLoss,
    PoissonDevianceLoss,
    PoissonLikelihoodRatioLoss,
    ZeroInflatedPoissonNLLLoss,
)

# Mixed Poisson-Gaussian losses
from .poisson_gaussian import (
    EnhancedPoissonGaussianMixtureLoss,
    PoissonGaussianLikelihoodRatioLoss,
    PoissonGaussianMixtureLoss,
    enhanced_poisson_gaussian_loss,
    poisson_gaussian_likelihood_ratio_loss,
    poisson_gaussian_mixture_loss,
)

# Quantile losses
from .quantile import MultiQuantileLoss, QuantileCrossoverLoss, QuantileLoss

# Robust losses
from .robust import (
    CauchyLoss,
    CharbonnierLoss,
    CVaRLoss,
    LogCoshLoss,
    PseudoHuberLoss,
    TukeyBiweightLoss,
)

# Tweedie losses
from .tweedie import CompoundPoissonLoss, GammaLoss, InverseGaussianLoss, TweedieLoss

__all__ = [
    # Base classes
    "BaseLoss",
    "RegressionLoss",
    "DistributionLoss",
    "WeightedLossWrapper",
    "WeightedGaussianNLLLoss",
    # Regression-as-classification
    "WeightedCrossEntropyLoss",
    "WeightedNLLLoss",
    # EIV losses
    "BaseEIVLoss",
    "EnsembleEIVLoss",
    "FunctionalEIVLoss",
    "OrthogonalDistanceRegressionLoss",
    "StructuralEIVLoss",
    # Gaussian losses
    "GaussianNLLLoss",
    "LowRankGaussianLoss",
    "MultivariateGaussianLoss",
    "low_rank_output_dim",
    "split_low_rank_gaussian_output",
    # Robust losses
    "PseudoHuberLoss",
    "LogCoshLoss",
    "CharbonnierLoss",
    "TukeyBiweightLoss",
    "CauchyLoss",
    "CVaRLoss",
    # Poisson losses
    "PoissonDevianceLoss",
    "PoissonLikelihoodRatioLoss",
    "ZeroInflatedPoissonNLLLoss",
    "NegativeBinomialNLLLoss",
    # Poisson-Gaussian losses
    "PoissonGaussianMixtureLoss",
    "EnhancedPoissonGaussianMixtureLoss",
    "PoissonGaussianLikelihoodRatioLoss",
    "poisson_gaussian_mixture_loss",
    "enhanced_poisson_gaussian_loss",
    "poisson_gaussian_likelihood_ratio_loss",

    # Imbalanced regression losses
    "DensityWeightedLoss",
    "FocalRLoss",
    "LDSLoss",
    # Evidential regression
    "EvidentialRegressionLoss",
    # Expectile losses
    "ExpectileLoss",
    "MultiExpectileLoss",
    "AsymmetricLeastSquaresLoss",
    "ExpectileCrossoverLoss",
    # Quantile losses
    "QuantileLoss",
    "MultiQuantileLoss",
    "QuantileCrossoverLoss",
    # Tweedie losses
    "TweedieLoss",
    "GammaLoss",
    "InverseGaussianLoss",
    "CompoundPoissonLoss",
    # Conformal prediction
    "CQR",
    "CTI",
    "ConformalLoss",
    "ConformalPredictor",
    "DistributionalConformal",
    "MultiDimensionalConformalLoss",
    "MultiTargetConformal",
    "R2CConformal",
    "SplitConformal",
    # Registry
    "get_regression_loss",
    "list_regression_losses",
    # Normalizing flows
    "NormalizingFlowLoss",
    # Mixture density networks
    "MixtureDensityLoss",
]
