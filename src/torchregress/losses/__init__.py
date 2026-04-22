"""
Loss functions for regression tasks.

This module contains a variety of loss functions designed for regression problems,
including weighted losses, Gaussian-based losses, robust losses, and more.
"""

# Base classes and wrappers
# Balanced MSE (imbalanced targets)
from .balanced_mse import BalancedMSELoss, BMCLoss
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

# Heteroscedastic Gaussian: β-NLL
from .beta_nll import BetaNLLLoss, beta_nll_loss

# Censored regression
from .censored import AFTLoss, CensoredGaussianNLLLoss, CensoredQuantileLoss

# Conformal prediction
from .conformal import (
    CQR,
    CTI,
    UACQR,
    ConformalLoss,
    ConformalPredictor,
    DensityConformal,
    DistributionalConformal,
    MonteCarloConformal,
    MultiDimensionalConformalLoss,
    MultiTargetConformal,
    PrevalenceAdjustedCP,
    R2CConformal,
    SplitConformal,
)
from .eiv import (
    BaseEIVLoss,
    EnsembleEIVLoss,
    ExplicitEIVAdapter,
    FunctionalEIVLoss,
    InputNoiseBinnedPDFLoss,
    InputNoiseMarginalizationLoss,
    InputNoiseMDNLoss,
    NoisyInputPredictor,
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

# Faithful heteroscedastic (mean / variance decoupling)
from .faithful_gaussian import FaithfulGaussianLoss
from .gaussian import (
    GaussianCRPSLoss,
    GaussianNLLLoss,
    LowRankGaussianLoss,
    MultivariateGaussianLoss,
    create_gaussian_nll,
    low_rank_output_dim,
    split_low_rank_gaussian_output,
)
from .gaussian_wasserstein import (
    GaussianWassersteinBoundLoss,
    gaussian_wasserstein_bound_loss,
    symmetric_spd_matrix_sqrt,
)

# Imbalanced regression losses
from .imbalanced import (
    DensityWeightedLoss,
    FocalRLoss,
    LDSLoss,
    PropensityWeightedLoss,
)
from .loss_registry import create_loss_from_config, get_regression_loss, list_regression_losses

# Mixture Density Networks
from .mdn import MDNLoss, MixtureDensityLoss, create_mdn_loss

# Normalizing flows (zuko is an optional dependency)
try:
    from .nflows import (
        ContrastiveFlowLoss,
        NormalizingFlowLoss,
        create_contrastive_flow_loss,
        create_flow_loss,
        create_flow_model,
    )
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
    AdaptiveRobustLoss,
    BarronLoss,
    CauchyLoss,
    CharbonnierLoss,
    CVaRLoss,
    LogCoshLoss,
    PseudoHuberLoss,
    TukeyBiweightLoss,
)
from .transforms import (
    BoxCoxTransformLoss,
    LogTransformLoss,
    SqrtTransformLoss,
    TransformedTargetLoss,
    YeoJohnsonTransformLoss,
)

# Tweedie losses
from .tweedie import CompoundPoissonLoss, GammaLoss, InverseGaussianLoss, TweedieLoss
from .uncertain_gt import (
    ConsistencyRegLoss,
    NoisyTargetGaussianNLL,
    PseudoLabelConsistencyLoss,
    PseudoLabelNLL,
)

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
    "ExplicitEIVAdapter",
    "FunctionalEIVLoss",
    "InputNoiseMarginalizationLoss",
    "InputNoiseMDNLoss",
    "InputNoiseBinnedPDFLoss",
    "NoisyInputPredictor",
    "OrthogonalDistanceRegressionLoss",
    "StructuralEIVLoss",
    "create_eiv_loss",
    # Gaussian losses
    "BetaNLLLoss",
    "beta_nll_loss",
    "FaithfulGaussianLoss",
    "GaussianWassersteinBoundLoss",
    "gaussian_wasserstein_bound_loss",
    "symmetric_spd_matrix_sqrt",
    "GaussianCRPSLoss",
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
    "BarronLoss",
    "AdaptiveRobustLoss",
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
    "BalancedMSELoss",
    "BMCLoss",
    "DensityWeightedLoss",
    "FocalRLoss",
    "LDSLoss",
    "PropensityWeightedLoss",
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
    # Transformed-target losses
    "TransformedTargetLoss",
    "LogTransformLoss",
    "BoxCoxTransformLoss",
    "SqrtTransformLoss",
    "YeoJohnsonTransformLoss",
    # Conformal prediction
    "CQR",
    "CTI",
    "ConformalLoss",
    "ConformalPredictor",
    "DensityConformal",
    "DistributionalConformal",
    "MonteCarloConformal",
    "MultiDimensionalConformalLoss",
    "MultiTargetConformal",
    "PrevalenceAdjustedCP",
    "R2CConformal",
    "SplitConformal",
    "UACQR",
    # Censored regression
    "CensoredGaussianNLLLoss",
    "CensoredQuantileLoss",
    "AFTLoss",
    # Uncertain/weak ground-truth losses
    "NoisyTargetGaussianNLL",
    "ConsistencyRegLoss",
    "PseudoLabelNLL",
    "PseudoLabelConsistencyLoss",
    # Registry
    "get_regression_loss",
    "list_regression_losses",
    "create_loss_from_config",
    # Normalizing flows
    "NormalizingFlowLoss",
    "ContrastiveFlowLoss",
    "create_flow_model",
    "create_flow_loss",
    "create_contrastive_flow_loss",
    # Mixture density networks
    "MixtureDensityLoss",
    "MDNLoss",
    "create_mdn_loss",
]

for _optional_name in [
    "NormalizingFlowLoss",
    "ContrastiveFlowLoss",
    "create_flow_model",
    "create_flow_loss",
    "create_contrastive_flow_loss",
]:
    if _optional_name not in globals() and _optional_name in __all__:
        __all__.remove(_optional_name)
