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

# Try to import torchcp-based conformal methods (requires torchcp with dependencies)
try:
    from .conformal_advanced import TorchCPConformalLoss, create_conformal_predictor

    _TORCHCP_ADVANCED_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _TORCHCP_ADVANCED_AVAILABLE = False

    # Create placeholder functions that raise informative errors
    def TorchCPConformalLoss(*args, **kwargs):
        raise ImportError(
            "TorchCPConformalLoss requires torchcp with all dependencies. "
            "Install with: pip install torchcp torchsort"
        )

    def create_conformal_predictor(*args, **kwargs):
        raise ImportError(
            "create_conformal_predictor requires torchcp with all dependencies. "
            "Install with: pip install torchcp torchsort"
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

__all__ = [
    # Base classes
    "BaseLoss",
    "MaskedLoss",
    "RegressionLoss",
    "DistributionLoss",
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
    "TorchCPConformalLoss",
    "create_conformal_predictor",
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
]
