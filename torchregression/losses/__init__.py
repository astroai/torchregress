"""
Loss functions for regression tasks.

This module provides a comprehensive collection of loss functions
for various regression scenarios, including robust regression,
quantile regression, expectile regression, and error-in-variables (EIV) models.
"""

# Import basic losses
from .base import RegressionLoss, MaskedLoss

# Import robust losses
from .robust import (
    L1Loss,
    HuberLoss,
    PseudoHuberLoss,
    LogCoshLoss,
    CharbonnierLoss,
    LqLoss,
    LogBarrierLoss,
    TukeyBiweightLoss,
    AdaptiveHuberLoss,
    WinsorizedLoss,
    ClippedLoss,
    FairLoss,
    CauchyLoss,
)

# Import standard losses and wrapped PyTorch losses
from .standard import (
    MSELoss,
    BCELoss,
    # Wrapped PyTorch losses
    MaskedMSELoss,
    MaskedL1Loss,
    MaskedCrossEntropyLoss,
    MaskedBCELoss,
    MaskedBCEWithLogitsLoss,
    MaskedKLDivLoss,
    MaskedNLLLoss,
    MaskedSmoothL1Loss,
    MaskedHuberLoss,
    MaskedPoissonNLLLoss,
)

# Import distributional losses
from .poisson import PoissonNLLLoss, PoissonNLL
from .tweedie import TweedieLoss, GammaLoss, InverseGaussianLoss, CompoundPoissonLoss

# Import quantile and expectile losses
from .quantile import QuantileLoss, MultiQuantileLoss, QuantileCrossover, TiltedLoss
from .expectile import (
    ExpectileLoss,
    MultiExpectileLoss,
    AsymmetricLeastSquaresLoss,
    ExpectileCrossover,
)

# Import mixture model losses
from .mdn import (
    MixtureDensityNetworkLoss,
    MDNEIVLoss,
    GaussianPoissonMixtureNLL,
    NMDRLoss,
    CombinedMDNFixedErrorLoss,
    mdn_loss,
    nmdr_loss,
    combined_mdn_loss,
    mdn_ensemble,
)

# Import basic EIV losses (sophisticated ones deferred to 0.2.0)
from .eiv import (
    TotalLeastSquaresLoss,
    GeneralErrorInVariablesLoss,
    CorrelatedEIVLoss,
    create_eiv_loss,
    create_correlated_eiv_loss,
)

# Import categorical losses
from .categorical import (
    BinaryCrossEntropyLoss,
    FocalLoss,
    MultiClassCrossEntropyLoss,
    MultiLabelSoftMarginLoss,
    DiceLoss,
)

# Note: Loss registry is deferred to version 0.2.0

__all__ = [
    # Base classes
    "RegressionLoss",
    "MaskedLoss",
    # Standard losses and wrapped PyTorch losses
    "MSELoss",
    "BCELoss",
    "MaskedMSELoss",
    "MaskedL1Loss",
    "MaskedCrossEntropyLoss",
    "MaskedBCELoss",
    "MaskedBCEWithLogitsLoss",
    "MaskedKLDivLoss",
    "MaskedNLLLoss",
    "MaskedSmoothL1Loss",
    "MaskedHuberLoss",
    "MaskedPoissonNLLLoss",
    # Robust losses
    "L1Loss",
    "HuberLoss",
    "PseudoHuberLoss",
    "LogCoshLoss",
    "CharbonnierLoss",
    "LqLoss",
    "LogBarrierLoss",
    "TukeyBiweightLoss",
    "AdaptiveHuberLoss",
    "WinsorizedLoss",
    "ClippedLoss",
    "FairLoss",
    "CauchyLoss",
    # Distribution losses
    "PoissonNLLLoss",
    "PoissonNLL",
    "TweedieLoss",
    "GammaLoss",
    "InverseGaussianLoss",
    "CompoundPoissonLoss",
    # Quantile and expectile losses
    "QuantileLoss",
    "MultiQuantileLoss",
    "QuantileCrossover",
    "TiltedLoss",
    "ExpectileLoss",
    "MultiExpectileLoss",
    "AsymmetricLeastSquaresLoss",
    "ExpectileCrossover",
    # Mixture model losses
    "MixtureDensityNetworkLoss",
    "MDNEIVLoss",
    "GaussianPoissonMixtureNLL",
    "NMDRLoss",
    "CombinedMDNFixedErrorLoss",
    "mdn_loss",
    "nmdr_loss",
    "combined_mdn_loss",
    "mdn_ensemble",
    # Basic EIV losses (core functionality only for 0.1.0)
    "TotalLeastSquaresLoss",
    "GeneralErrorInVariablesLoss",
    "CorrelatedEIVLoss",
    "create_eiv_loss",
    "create_correlated_eiv_loss",
    # Categorical losses
    "BinaryCrossEntropyLoss",
    "FocalLoss",
    "MultiClassCrossEntropyLoss",
    "MultiLabelSoftMarginLoss",
    "DiceLoss",
]
