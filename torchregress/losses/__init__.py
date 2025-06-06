"""
Loss functions for regression tasks.

This module contains a variety of loss functions designed for regression problems,
including weighted losses, Gaussian-based losses, robust losses, and more.
"""

# Base classes and wrappers
from .base import (
    BaseLoss,
    MaskedLoss,
    RegressionLoss,
    DistributionLoss,
    WeightedLossWrapper,
    create_weighted_losses,
)

# Standard weighted PyTorch losses
from .base import (
    WeightedMSELoss,
    WeightedL1Loss,
    WeightedCrossEntropyLoss,
    WeightedBCELoss,
    WeightedBCEWithLogitsLoss,
    WeightedKLDivLoss,
    WeightedNLLLoss,
    WeightedSmoothL1Loss,
    WeightedHuberLoss,
    WeightedPoissonNLLLoss,
    WeightedGaussianNLLLoss,
    WeightedCTCLoss,
    WeightedCosineEmbeddingLoss,
    WeightedHingeEmbeddingLoss,
    WeightedMarginRankingLoss,
    WeightedMultiMarginLoss,
    WeightedMultiLabelMarginLoss,
    WeightedSoftMarginLoss,
    WeightedMultiLabelSoftMarginLoss,
    WeightedTripletMarginLoss,
    WeightedTripletMarginWithDistanceLoss,
)

# Gaussian losses
from .gaussian import (
    HeteroscedasticGaussianLoss,
    MultivariateGaussianLoss,
    create_gaussian_nll,
)

# Target transform losses
from .target_transform import (
    LogTransformLoss,
    BoxCoxTransformLoss,
    SqrtTransformLoss,
)

# Robust losses
from .robust import (
    PseudoHuberLoss,
    LogCoshLoss,
    CharbonnierLoss,
    LqLoss,
    TukeyBiweightLoss,
    WinsorizedLoss,
    LogBarrierLoss,
    AdaptiveHuberLoss,
    ClippedLoss,
    FairLoss,
    CauchyLoss,
)

# Family-based regression losses
from .sqr import SQRLoss
from .barron import BarronLoss
from .conformal import ConformalLoss
from .deepar import DeepARLoss

# Poisson losses
from .poisson import (
    PoissonDevianceLoss,
    PoissonLikelihoodRatioLoss,
    ZeroInflatedPoissonNLLLoss,
    NegativeBinomialNLLLoss,
)

# Mixed Poisson-Gaussian losses
from .poisson_gaussian import (
    PoissonGaussianMixtureLoss,
    poisson_gaussian_mixture_loss,
    EnhancedPoissonGaussianMixtureLoss,
    enhanced_poisson_gaussian_loss,
    PoissonGaussianLikelihoodRatioLoss,
    poisson_gaussian_likelihood_ratio_loss,
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
