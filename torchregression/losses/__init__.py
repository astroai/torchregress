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
    L1Loss, HuberLoss, PseudoHuberLoss, LogCoshLoss, 
    CharbonnierLoss, LqLoss, LogBarrierLoss, TukeyBiweightLoss,
    AdaptiveHuberLoss, WinsorizedLoss, ClippedLoss,
    FairLoss, CauchyLoss
)

# Import distributional losses
from .poisson import PoissonNLLLoss, PoissonNLL
from .tweedie import TweedieLoss, GammaLoss, InverseGaussianLoss, CompoundPoissonLoss
from .mdn import (
    MixtureDensityNetworkLoss, MDNEIVLoss, GaussianPoissonMixtureNLL,
    NMDRLoss, CombinedMDNFixedErrorLoss,
    mdn_loss, nmdr_loss, combined_mdn_loss, mdn_ensemble
)

# Import quantile and expectile losses
from .quantile import (
    QuantileLoss, MultiQuantileLoss, QuantileCrossover, TiltedLoss
)
from .expectile import (
    ExpectileLoss, MultiExpectileLoss, AsymmetricLeastSquaresLoss,
    ExpectileCrossover
)

# Import EIV losses
from .eiv import (
    TotalLeastSquaresLoss, GeneralErrorInVariablesLoss,
    CorrelatedEIVLoss,
    create_eiv_loss, create_correlated_eiv_loss, 
    create_chamfer_eiv_loss, create_hybrid_eiv_loss,
    create_mdn_eiv_loss, create_robust_eiv_loss
)

# Import torch extension losses
from .torch_extensions import MSELoss, BCELoss

# Import categorical losses
from .categorical import (
    BinaryCrossEntropyLoss, FocalLoss, MultiClassCrossEntropyLoss,
    MultiLabelSoftMarginLoss, DiceLoss
)

# Import loss registry
from .loss_registry import register_loss, get_loss_function, list_available_losses

__all__ = [
    # Base classes
    'RegressionLoss', 'MaskedLoss',
    
    # Robust losses
    'L1Loss', 'HuberLoss', 'PseudoHuberLoss', 'LogCoshLoss',
    'CharbonnierLoss', 'LqLoss', 'LogBarrierLoss', 'TukeyBiweightLoss',
    'AdaptiveHuberLoss', 'WinsorizedLoss', 'ClippedLoss',
    'FairLoss', 'CauchyLoss',
    
    # Distribution losses
    'PoissonNLLLoss', 'PoissonNLL',
    'TweedieLoss', 'GammaLoss', 'InverseGaussianLoss', 'CompoundPoissonLoss',
    
    # Mixture model losses
    'MixtureDensityNetworkLoss', 'MDNEIVLoss', 'GaussianPoissonMixtureNLL',
    'NMDRLoss', 'CombinedMDNFixedErrorLoss',
    'mdn_loss', 'nmdr_loss', 'combined_mdn_loss', 'mdn_ensemble',
    
    # Quantile and expectile losses
    'QuantileLoss', 'MultiQuantileLoss', 'QuantileCrossover', 'TiltedLoss',
    'ExpectileLoss', 'MultiExpectileLoss', 'AsymmetricLeastSquaresLoss',
    'ExpectileCrossover',
    
    # EIV losses
    'TotalLeastSquaresLoss', 'GeneralErrorInVariablesLoss', 'CorrelatedEIVLoss',
    'create_eiv_loss', 'create_correlated_eiv_loss', 'create_chamfer_eiv_loss',
    'create_hybrid_eiv_loss', 'create_mdn_eiv_loss', 'create_robust_eiv_loss',
    
    # Torch extension losses
    'MSELoss', 'BCELoss',
    
    # Categorical losses
    'BinaryCrossEntropyLoss', 'FocalLoss', 'MultiClassCrossEntropyLoss',
    'MultiLabelSoftMarginLoss', 'DiceLoss',
    
    # Registry functions
    'register_loss', 'get_loss_function', 'list_available_losses'
]
