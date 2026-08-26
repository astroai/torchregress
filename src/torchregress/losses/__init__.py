"""
Loss functions for regression tasks.

This module contains a variety of loss functions designed for regression problems,
including weighted losses, Gaussian-based losses, robust losses, and more.
"""
# ruff: noqa: F401 — re-exports

# Base classes and wrappers
# Balanced MSE (imbalanced targets)
from ..utils.gaussian_output import low_rank_output_dim, split_low_rank_gaussian_output
from .balanced_mse import BalancedMSELoss, BinReweightedMSELoss
from .base import (
    BaseLoss,
    DistributionLoss,
    RegressionLoss,
    WeightedCrossEntropyLoss,
    WeightedHuberLoss,
    WeightedL1Loss,
    WeightedLossWrapper,
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
    CVPlus,
    DensityConformal,
    DistributionalConformal,
    EnsembleBatchCP,
    JackknifePlus,
    LevelSetConformalPredictor,
    LocalConformal,
    LocalConformalMAD,
    MonteCarloConformal,
    MultiTargetConformal,
    MultivariateScoreConformal,
    NonExchangeableConformalRegressor,
    PrevalenceAdjustedCP,
    R2CConformal,
    SLSConformal,
    SplitConformal,
)
from .eiv import (
    BaseEIVLoss,
    EnsembleEIVLoss,
    FunctionalEIVLoss,
    InputNoiseAugmentationLoss,
    InputNoiseBinnedPDFLoss,
    InputNoiseMDNLoss,
    LatentMarginalizationLoss,
    NoisyInputPredictor,
    OrthogonalDistanceRegressionLoss,
    StructuralEIVLoss,
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
    expectile_loss,
)

# Faithful heteroscedastic (mean / variance decoupling)
from .faithful_gaussian import FaithfulGaussianLoss

# Flexible-shape distributional families (F5)
from .families import (
    AsymmetricLaplaceNLLLoss,
    BetaRegressionNLLLoss,
    GEVNLLLoss,
    JohnsonSUNLLLoss,
    SinhArcsinhNLLLoss,
    SkewNormalNLLLoss,
    SkewTLoss,
    SQRLoss,
    asymmetric_laplace_nll,
    beta_regression_nll,
    gev_nll,
    johnson_su_nll,
    sinh_arcsinh_nll,
    skew_normal_nll,
    skew_t_nll,
    sqr_loss,
)
from .gaussian import (
    GaussianCRPSLoss,
    GaussianNLLLoss,
    LowRankGaussianLoss,
    MultivariateGaussianLoss,
    create_gaussian_nll,
)
from .gaussian_wasserstein import (
    GaussianWassersteinBoundLoss,
    gaussian_wasserstein_bound_loss,
    symmetric_spd_matrix_sqrt,
)

# Imbalanced regression losses
from .imbalanced import (
    DensityWeightedLoss,
    FeatureDistributionSmoother,
    FocalRLoss,
    LDSLoss,
    PropensityWeightedLoss,
)
from .loss_registry import create_loss_from_config, get_regression_loss, list_regression_losses

# Mixture Density Networks
from .mdn import MDNLoss, MixtureDensityLoss, create_mdn_loss

# SLS regression
from .sls import (
    MahalanobisFrontier,
    SLSLoss,
    UnionFrontier,
    VolumePreservingFlow,
)

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
)

# Quantile losses
from .quantile import (
    MultiQuantileLoss,
    QuantileCrossover,
    QuantileCrossoverLoss,
    QuantileLoss,
    quantile_loss,
)

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
from .student_t import StudentTLoss
from .transforms import (
    BoxCoxTransformLoss,
    LogTransformLoss,
    SqrtTransformLoss,
    TransformedTargetLoss,
    YeoJohnsonTransformLoss,
)

# Tweedie losses
from .tweedie import (
    CompoundPoissonLoss,
    GammaLoss,
    InverseGaussianLoss,
    TweedieLoss,
    tweedie_loss,
)
from .uncertain_gt import (
    ConsistencyRegLoss,
    NoisyTargetGaussianNLL,
    PseudoLabelConsistencyLoss,
    PseudoLabelNLL,
)

__all__ = [
    "AFTLoss",
    "AdaptiveRobustLoss",
    "AsymmetricLaplaceNLLLoss",
    "AsymmetricLeastSquaresLoss",
    "BalancedMSELoss",
    "BarronLoss",
    "BaseLoss",
    "BetaNLLLoss",
    "BetaRegressionNLLLoss",
    "BinReweightedMSELoss",
    "BoxCoxTransformLoss",
    "CORALLoss",
    "CVPlus",
    "CVaRLoss",
    "CauchyLoss",
    "CensoredGaussianNLLLoss",
    "CensoredQuantileLoss",
    "CharbonnierLoss",
    "CompoundPoissonLoss",
    "ConformalLoss",
    "ConformalPredictor",
    "ConsistencyRegLoss",
    "ContrastiveFlowLoss",
    "CumulativeLinkLoss",
    "DensityConformal",
    "DensityWeightedLoss",
    "DistributionLoss",
    "DistributionalConformal",
    "EnhancedPoissonGaussianMixtureLoss",
    "EnsembleBatchCP",
    "EnsembleEIVLoss",
    "EvidentialRegressionLoss",
    "ExpectileCrossoverLoss",
    "ExpectileLoss",
    "FaithfulGaussianLoss",
    "FeatureDistributionSmoother",
    "FocalRLoss",
    "FunctionalEIVLoss",
    "GEVNLLLoss",
    "GammaLoss",
    "GaussianCRPSLoss",
    "GaussianNLLLoss",
    "GaussianWassersteinBoundLoss",
    "InputNoiseAugmentationLoss",
    "InputNoiseBinnedPDFLoss",
    "InputNoiseMDNLoss",
    "InverseGaussianLoss",
    "JackknifePlus",
    "JohnsonSUNLLLoss",
    "LDSLoss",
    "LatentMarginalizationLoss",
    "LevelSetConformalPredictor",
    "LocalConformal",
    "LocalConformalMAD",
    "LogCoshLoss",
    "LogTransformLoss",
    "LowRankGaussianLoss",
    "MDNLoss",
    "MahalanobisFrontier",
    "MixtureDensityLoss",
    "MonteCarloConformal",
    "MultiExpectileLoss",
    "MultiQuantileLoss",
    "MultiTargetConformal",
    "MultivariateGaussianLoss",
    "NegativeBinomialNLLLoss",
    "NoisyInputPredictor",
    "NoisyTargetGaussianNLL",
    "NormalizingFlowLoss",
    "OrdinalCrossEntropyLoss",
    "OrthogonalDistanceRegressionLoss",
    "PoissonDevianceLoss",
    "PoissonGaussianLikelihoodRatioLoss",
    "PoissonGaussianMixtureLoss",
    "PoissonLikelihoodRatioLoss",
    "PrevalenceAdjustedCP",
    "PropensityWeightedLoss",
    "PseudoHuberLoss",
    "PseudoLabelConsistencyLoss",
    "PseudoLabelNLL",
    "QuantileCrossover",
    "QuantileCrossoverLoss",
    "QuantileLoss",
    "R2CConformal",
    "RegressionLoss",
    "SLSConformal",
    "SLSLoss",
    "SQRLoss",
    "SinhArcsinhNLLLoss",
    "SkewNormalNLLLoss",
    "SkewTLoss",
    "SplitConformal",
    "SqrtTransformLoss",
    "StructuralEIVLoss",
    "TransformedTargetLoss",
    "TukeyBiweightLoss",
    "TweedieLoss",
    "UACQR",
    "UnionFrontier",
    "VolumePreservingFlow",
    "WeightedCrossEntropyLoss",
    "WeightedHuberLoss",
    "WeightedL1Loss",
    "WeightedLossWrapper",
    "WeightedMSELoss",
    "WeightedNLLLoss",
    "YeoJohnsonTransformLoss",
    "ZeroInflatedPoissonNLLLoss",
    "asymmetric_laplace_nll",
    "beta_nll_loss",
    "beta_regression_nll",
    "create_contrastive_flow_loss",
    "create_flow_loss",
    "create_flow_model",
    "create_gaussian_nll",
    "create_loss_from_config",
    "create_mdn_loss",
    "expectile_loss",
    "gaussian_wasserstein_bound_loss",
    "get_regression_loss",
    "gev_nll",
    "johnson_su_nll",
    "list_regression_losses",
    "low_rank_output_dim",
    "quantile_loss",
    "sinh_arcsinh_nll",
    "skew_normal_nll",
    "skew_t_nll",
    "split_low_rank_gaussian_output",
    "sqr_loss",
    "symmetric_spd_matrix_sqrt",
    "tweedie_loss",
]
