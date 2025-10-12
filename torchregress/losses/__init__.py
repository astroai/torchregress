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
    WeightedCrossEntropyLoss,
    WeightedGaussianNLLLoss,
    WeightedHuberLoss,
    WeightedKLDivLoss,
    WeightedL1Loss,
    WeightedLossWrapper,
    WeightedMSELoss,
    WeightedNLLLoss,
    WeightedPoissonNLLLoss,
    WeightedSmoothL1Loss,
)
from .conformal import (
    ConformalLoss,
    MultiDimensionalConformalLoss,
)


from .deepar import DeepARLoss

# Gaussian losses
from .gaussian import (
    GaussianNLLLoss,
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

# Noisy label losses
from .noisy import CoTeachingLoss, NoiseAdaptiveLoss, RENTLoss

# Imbalanced regression losses
from .imbalanced import DensityWeightedLoss, LDSLoss

# Evidential regression
from .evidential import EvidentialRegressionLoss, create_evidential_loss

# Expectile losses
from .expectile import ExpectileLoss, MultiExpectileLoss, AsymmetricLeastSquaresLoss

# Quantile losses
from .quantile import QuantileLoss, MultiQuantileLoss

# Tweedie losses
from .tweedie import TweedieLoss, GammaLoss, InverseGaussianLoss, CompoundPoissonLoss

# Normalizing flows
from .nflows import NormalizingFlowLoss, create_flow_model, create_flow_loss

# Mixture Density Networks
from .mdn import MixtureDensityLoss, create_mdn_loss

# Convenience aliases (match PyTorch/common naming conventions)
MSELoss = WeightedMSELoss
L1Loss = WeightedL1Loss
MAELoss = WeightedL1Loss  # Alternative name
SmoothL1Loss = WeightedSmoothL1Loss
HuberLoss = WeightedHuberLoss
PoissonNLLLoss = WeightedPoissonNLLLoss
# Note: GaussianNLLLoss is our main implementation (from gaussian.py), not an alias
CrossEntropyLoss = WeightedCrossEntropyLoss
NLLLoss = WeightedNLLLoss
KLDivLoss = WeightedKLDivLoss

__all__ = [
    # Base classes
    "BaseLoss",
    "MaskedLoss",
    "RegressionLoss",
    "DistributionLoss",
    "WeightedLossWrapper",
    # Standard weighted losses
    "WeightedMSELoss",
    "WeightedL1Loss",
    "WeightedSmoothL1Loss",
    "WeightedHuberLoss",
    "WeightedPoissonNLLLoss",
    "WeightedGaussianNLLLoss",
    # Regression-as-classification
    "WeightedCrossEntropyLoss",
    "WeightedNLLLoss",
    "WeightedKLDivLoss",
    # Gaussian losses
    "GaussianNLLLoss",
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
    # Noisy label losses
    "NoiseAdaptiveLoss",
    "CoTeachingLoss",
    "RENTLoss",
    # Imbalanced regression losses
    "DensityWeightedLoss",
    "LDSLoss",
    # Evidential regression
    "EvidentialRegressionLoss",
    "create_evidential_loss",
    # Expectile losses
    "ExpectileLoss",
    "MultiExpectileLoss",
    "AsymmetricLeastSquaresLoss",
    # Quantile losses
    "QuantileLoss",
    "MultiQuantileLoss",
    # Tweedie losses
    "TweedieLoss",
    "GammaLoss",
    "InverseGaussianLoss",
    "CompoundPoissonLoss",
    # Normalizing flows
    "NormalizingFlowLoss",
    "create_flow_model",
    "create_flow_loss",
    # Mixture Density Networks
    "MixtureDensityLoss",
    "create_mdn_loss",
    # Convenience aliases
    "MSELoss",
    "L1Loss",
    "MAELoss",
    "SmoothL1Loss",
    "HuberLoss",
    "PoissonNLLLoss",
    # GaussianNLLLoss is in the Gaussian losses section (not an alias)
    "CrossEntropyLoss",
    "NLLLoss",
    "KLDivLoss",
]
