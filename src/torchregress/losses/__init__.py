"""
Loss functions for regression tasks.

This module contains a variety of loss functions designed for regression problems,
including weighted losses, Gaussian-based losses, robust losses, and more.
"""
# ruff: noqa: F401 — re-exports

# Base classes and wrappers
# Balanced MSE (imbalanced targets)
from ..utils.gaussian_output import low_rank_output_dim, split_low_rank_gaussian_output
from .balanced_mse import BalancedMSELoss, BMCLoss
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
    enhanced_poisson_gaussian_loss,
    poisson_gaussian_likelihood_ratio_loss,
    poisson_gaussian_mixture_loss,
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
