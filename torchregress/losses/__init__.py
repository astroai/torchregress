"""
Loss functions for regression tasks.

This module contains a variety of loss functions designed for regression problems,
including weighted losses, Gaussian-based losses, robust losses, and more.
"""

# Base classes and wrappers
from .barron import BarronLoss

# Standard weighted PyTorch losses
from .base import (
    BaseLoss,
    DistributionLoss,
    MaskedLoss,
    RegressionLoss,
    WeightedBCELoss,
    WeightedBCEWithLogitsLoss,
    WeightedCosineEmbeddingLoss,
    WeightedCrossEntropyLoss,
    WeightedCTCLoss,
    WeightedGaussianNLLLoss,
    WeightedHingeEmbeddingLoss,
    WeightedHuberLoss,
    WeightedKLDivLoss,
    WeightedL1Loss,
    WeightedLossWrapper,
    WeightedMarginRankingLoss,
    WeightedMSELoss,
    WeightedMultiLabelMarginLoss,
    WeightedMultiLabelSoftMarginLoss,
    WeightedMultiMarginLoss,
    WeightedNLLLoss,
    WeightedPoissonNLLLoss,
    WeightedSmoothL1Loss,
    WeightedSoftMarginLoss,
    WeightedTripletMarginLoss,
    WeightedTripletMarginWithDistanceLoss,
    create_weighted_losses,
)
from .conformal import (
    AdaptiveConformalLoss,
    ConformalizedQuantileLoss,
    ConformalLoss,
    MultiDimensionalConformalLoss,
)
from .deepar import DeepARLoss

# Gaussian losses
from .gaussian import (
    HeteroscedasticGaussianLoss,
    MultivariateGaussianLoss,
    create_gaussian_nll,
)

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

# Robust losses
from .robust import (
    AdaptiveHuberLoss,
    CauchyLoss,
    CharbonnierLoss,
    ClippedLoss,
    FairLoss,
    LogBarrierLoss,
    LogCoshLoss,
    LqLoss,
    PseudoHuberLoss,
    TukeyBiweightLoss,
    WinsorizedLoss,
)

# Family-based regression losses
from .sqr import SQRLoss

# Target transform losses
from .transform import (
    BoxCoxTransformLoss,
    LogTransformLoss,
    SqrtTransformLoss,
)

__all__ = [
    # Base classes
    "BaseLoss",
    "MaskedLoss",
    "RegressionLoss",
    "DistributionLoss",
    "TorchLossWrapper",
    "WeightedLossWrapper",
    "create_weighted_losses",
    # Standard weighted losses
    "WeightedMSELoss",
    "WeightedL1Loss",
    "WeightedCrossEntropyLoss",
    "WeightedBCELoss",
    "WeightedBCEWithLogitsLoss",
    "WeightedKLDivLoss",
    "WeightedNLLLoss",
    "WeightedSmoothL1Loss",
    "WeightedHuberLoss",
    "WeightedPoissonNLLLoss",
    "WeightedGaussianNLLLoss",
    "WeightedCTCLoss",
    "WeightedCosineEmbeddingLoss",
    "WeightedHingeEmbeddingLoss",
    "WeightedMarginRankingLoss",
    "WeightedMultiMarginLoss",
    "WeightedMultiLabelMarginLoss",
    "WeightedSoftMarginLoss",
    "WeightedMultiLabelSoftMarginLoss",
    "WeightedTripletMarginLoss",
    "WeightedTripletMarginWithDistanceLoss",
    # Gaussian losses
    "HeteroscedasticGaussianLoss",
    "MultivariateGaussianLoss",
    "create_gaussian_nll",
    # Target transformations
    "LogTransformLoss",
    "BoxCoxTransformLoss",
    "SqrtTransformLoss",
    # Robust losses
    "PseudoHuberLoss",
    "LogCoshLoss",
    "CharbonnierLoss",
    "LqLoss",
    "TukeyBiweightLoss",
    "WinsorizedLoss",
    "LogBarrierLoss",
    "AdaptiveHuberLoss",
    "ClippedLoss",
    "FairLoss",
    "CauchyLoss",
    # Family-based regression losses
    "SQRLoss",
    "BarronLoss",
    "ConformalLoss",
    "AdaptiveConformalLoss",
    "ConformalizedQuantileLoss",
    "MultiDimensionalConformalLoss",
    "DeepARLoss",
    # Poisson losses
    "PoissonDevianceLoss",
    "PoissonLikelihoodRatioLoss",
    "ZeroInflatedPoissonNLLLoss",
    "NegativeBinomialNLLLoss",
    # Poisson-Gaussian losses
    "PoissonGaussianMixtureLoss",
    "poisson_gaussian_mixture_loss",
    "EnhancedPoissonGaussianMixtureLoss",
    "enhanced_poisson_gaussian_loss",
    "PoissonGaussianLikelihoodRatioLoss",
    "poisson_gaussian_likelihood_ratio_loss",
]
