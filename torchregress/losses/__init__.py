"""
Loss functions for regression tasks.

This module contains a variety of loss functions designed for regression problems,
including weighted losses, Gaussian-based losses, robust losses, and more.
"""

# Base classes and wrappers
from .base import (
    BaseLoss,
    DistributionLoss,
    HuberLoss,
    L1Loss,
    MSELoss,
    RegressionLoss,
    WeightedCrossEntropyLoss,
    WeightedGaussianNLLLoss,
    WeightedHuberLoss,
    WeightedL1Loss,
    WeightedLossWrapper,
    WeightedMAELoss,
    WeightedMSELoss,
    WeightedNLLLoss,
)
from .censored import AFTLoss, CensoredGaussianNLLLoss, CensoredQuantileLoss

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
    create_eiv_loss,
)

# Evidential regression
from .evidential import EvidentialRegressionLoss

# Expectile losses
from .expectile import (
    AsymmetricLeastSquaresLoss,
    ExpectileCrossover,
    ExpectileCrossoverLoss,
    ExpectileLoss,
    MultiExpectileLoss,
)

# Gaussian losses
from .gaussian import (
    GaussianNLLLoss,
    LowRankGaussianLoss,
    MultivariateGaussianLoss,
    create_gaussian_nll,
    low_rank_output_dim,
    split_low_rank_gaussian_output,
)

# Imbalanced regression losses
from .imbalanced import (
    DensityWeightedLoss,
    FocalRLoss,
    LDSLoss,
)
from .loss_registry import create_loss_from_config, get_regression_loss, list_regression_losses

# Mixture Density Networks
from .mdn import MDNLoss, MixtureDensityLoss, create_mdn_loss

# Normalizing flows (zuko is an optional dependency)
try:
    from .nflows import NormalizingFlowLoss, create_flow_loss, create_flow_model
except ImportError:
    pass  # zuko not installed; normalizing flow features unavailable


# Poisson losses
# Ordinal losses
from .ordinal import CORALLoss, CumulativeLinkLoss, OrdinalCrossEntropyLoss
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
from .quantile import MultiQuantileLoss, QuantileCrossover, QuantileCrossoverLoss, QuantileLoss

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
    "WeightedMSELoss",
    "WeightedL1Loss",
    "WeightedMAELoss",
    "WeightedHuberLoss",
    "WeightedGaussianNLLLoss",
    "MSELoss",
    "L1Loss",
    "HuberLoss",
    # Regression-as-classification
    "WeightedCrossEntropyLoss",
    "WeightedNLLLoss",
    # EIV losses
    "BaseEIVLoss",
    "EnsembleEIVLoss",
    "FunctionalEIVLoss",
    "OrthogonalDistanceRegressionLoss",
    "StructuralEIVLoss",
    "create_eiv_loss",
    # Gaussian losses
    "GaussianNLLLoss",
    "LowRankGaussianLoss",
    "MultivariateGaussianLoss",
    "low_rank_output_dim",
    "split_low_rank_gaussian_output",
    "create_gaussian_nll",
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
    "ExpectileCrossover",
    "ExpectileCrossoverLoss",
    # Quantile losses
    "QuantileLoss",
    "MultiQuantileLoss",
    "QuantileCrossover",
    "QuantileCrossoverLoss",
    # Ordinal losses
    "OrdinalCrossEntropyLoss",
    "CumulativeLinkLoss",
    "CORALLoss",
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
    # Censored regression
    "CensoredGaussianNLLLoss",
    "CensoredQuantileLoss",
    "AFTLoss",
    # Registry
    "get_regression_loss",
    "list_regression_losses",
    "create_loss_from_config",
    # Normalizing flows
    "NormalizingFlowLoss",
    "create_flow_model",
    "create_flow_loss",
    # Mixture density networks
    "MixtureDensityLoss",
    "MDNLoss",
    "create_mdn_loss",
]

for _optional_name in ["NormalizingFlowLoss", "create_flow_model", "create_flow_loss"]:
    if _optional_name not in globals() and _optional_name in __all__:
        __all__.remove(_optional_name)
