"""
TorchRegression: A comprehensive PyTorch library for regression tasks.

This library provides tools for various regression tasks, including:
- Robust regression losses
- Quantile and expectile regression
- Error-in-Variables (EIV) regression
- Mixture models and distributional regression
- Ensemble methods
- Uncertainty quantification
"""

__version__ = '0.1.0'

# Import core components into the top-level namespace
from .losses import (
    # Base losses
    RegressionLoss, MaskedLoss,
    
    # Common robust losses
    L1Loss, HuberLoss, PseudoHuberLoss,
    
    # Quantile and expectile losses
    QuantileLoss, ExpectileLoss, MultiQuantileLoss, MultiExpectileLoss,
    
    # Distributional losses
    PoissonNLLLoss, TweedieLoss, GammaLoss,
    MixtureDensityNetworkLoss, mdn_loss,
    
    # EIV losses
    create_eiv_loss, create_robust_eiv_loss, TotalLeastSquaresLoss
)

# Import model fitting algorithms
from .algorithms import (
    IRLS, iteratively_reweighted_least_squares,
    DiagonalGaussianNLL, GaussianNLLWithCovariance,
    fit_linear_gaussian, fit_bayesian_linear
)

# Import ensemble methods
from .ensemble import (
    DeepEnsemble, BatchEnsembleLinear, BaseEnsembleModel
)

# Import commonly used utilities
from .utils import (
    # Statistics utilities
    r_squared, adjusted_r_squared, mse, rmse, mae,
    
    # Data augmentation
    GaussianNoiseAugmentation, MixUp, AdversarialAugmentation
)

# Import wrappers for scikit-learn compatibility
from .wrappers import (
    TorchRegressionEstimator, TorchQuantileEstimator, TorchMDNEstimator
)

# For plotting support (if matplotlib is available)
try:
    from .plots import plot_predictions
except ImportError:
    # matplotlib not available
    pass

# Convenience function to list all available losses
from .losses.loss_registry import list_available_losses