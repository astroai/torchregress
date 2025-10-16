"""
Loss functions for regression tasks.

This module contains a variety of loss functions designed for regression problems,
including weighted losses, Gaussian-based losses, robust losses, and more.
"""

from typing import Any, Dict
import torch.nn as nn

# Base classes and wrappers
from .barron import BarronLoss

# Standard weighted PyTorch losses
from .base import (
    BaseLoss,
    DistributionLoss,
    BaseLoss,
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


# Evidential regression
from .evidential import EvidentialRegressionLoss

# Expectile losses
from .expectile import AsymmetricLeastSquaresLoss, ExpectileLoss, MultiExpectileLoss, ExpectileCrossoverLoss

# Gaussian losses
from .gaussian import (
    GaussianNLLLoss,
)

# Imbalanced regression losses
from .imbalanced import DensityWeightedLoss, LDSLoss

# Mixture Density Networks
from .mdn import MixtureDensityLoss

# Normalizing flows
from .nflows import NormalizingFlowLoss, create_flow_model

# Noisy label losses
from .noisy import CoTeachingLoss, NoiseAdaptiveLoss, RENTLoss

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
)

# Quantile losses
from .quantile import MultiQuantileLoss, QuantileLoss, QuantileCrossoverLoss

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

# Tweedie losses
from .tweedie import CompoundPoissonLoss, GammaLoss, InverseGaussianLoss, TweedieLoss
from .loss_registry import get_regression_loss, list_regression_losses

def create_loss_from_config(config: Dict[str, Any]) -> BaseLoss:
    """
    Create a loss function from a configuration dictionary.

    This provides a standardized way to instantiate any loss function
    based on a configuration dict, useful for experiments and hyperparameter tuning.

    Args:
        config: Configuration dictionary with at least a 'type' key
               and optional parameters for the specific loss

    Returns:
        Instantiated loss function

    Example:
        >>> loss_config = {
        >>>     'type': 'huber',
        >>>     'delta': 1.0,
        >>>     'reduction': 'mean'
        >>> }
        >>> loss_fn = create_loss_from_config(loss_config)
    """
    loss_type = config.pop("type", "").lower()

    if not loss_type:
        raise ValueError("Configuration must contain a 'type' key.")

    # Handle pytorch losses separately
    if loss_type == "pytorch":
        loss_class_name = config.pop("class")
        if isinstance(loss_class_name, str):
            if hasattr(nn, loss_class_name):
                loss_class = getattr(nn, loss_class_name)
            else:
                raise ValueError(f"Unknown PyTorch loss: {loss_class_name}")
        else:
            loss_class = loss_class_name
        return WeightedLossWrapper(loss_class, **config)

    # Aliases
    if loss_type == "mseloss":
        loss_type = "mse"
    if loss_type == "mae":
        loss_type = "l1"
    if loss_type == "gaussian":
        loss_type = "gaussian_nll"
    if loss_type == "pinball":
        loss_type = "quantile"

    try:
        loss_class = get_regression_loss(loss_type)
        return loss_class(**config)
    except KeyError:
        available_losses = list_regression_losses()
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. "
            f"Available losses: {available_losses}"
        )


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
    "BaseLoss",
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

    # Poisson losses
    "PoissonDevianceLoss",
    "PoissonLikelihoodRatioLoss",
    "ZeroInflatedPoissonNLLLoss",
    "NegativeBinomialNLLLoss",
    # Poisson-Gaussian losses
    "PoissonGaussianMixtureLoss",
    "EnhancedPoissonGaussianMixtureLoss",
    "PoissonGaussianLikelihoodRatioLoss",
    # Noisy label losses
    "NoiseAdaptiveLoss",
    "CoTeachingLoss",
    "RENTLoss",
    # Imbalanced regression losses
    "DensityWeightedLoss",
    "LDSLoss",
    # Evidential regression
    "EvidentialRegressionLoss",
    # Expectile losses
    "ExpectileLoss",
    "MultiExpectileLoss",
    "AsymmetricLeastSquaresLoss",
    "ExpectileCrossoverLoss",
    # Quantile losses
    "QuantileLoss",
    "MultiQuantileLoss",
    "QuantileCrossoverLoss",
    # Tweedie losses
    "TweedieLoss",
    "GammaLoss",
    "InverseGaussianLoss",
    "CompoundPoissonLoss",
    # Normalizing flows
    "NormalizingFlowLoss",
    "create_flow_model",
    # Mixture Density Networks
    "MixtureDensityLoss",
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
    # Factory function
    "create_loss_from_config",
]