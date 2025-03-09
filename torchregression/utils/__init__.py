"""
Utility functions for regression tasks.

This module provides various utility functions and classes
for data preprocessing, tensor operations, statistical analysis,
validation, and other common tasks in regression modeling.
"""

# Import tensor operations
from .tensor_ops import (
    prepare_param,
    prepare_covariance,
    batched_linalg_solve,
    to_tensor,
    standardize,
    unstandardize
)

# Import validation utilities
from .validation import (
    validate_range,
    validate_positive,
    validate_shape,
    validate_reduction
)

# Import statistical utilities
from .stats import (
    weighted_mean,
    weighted_variance,
    weighted_quantile,
    r_squared,
    adjusted_r_squared,
    mse,
    rmse,
    mae,
    mape,
    smape
)

# Import data augmentation utilities
from .augment import (
    Augmentation,
    GaussianNoiseAugmentation,
    AdversarialAugmentation,
    MixtureAugmentation,
    FeatureJitter,
    MixUp,
    FeatureMask
)

# Import multi-label handling utilities
from .labels import (
    combine_binary_average,
    combine_binary_weighted_average,
    combine_dawid_skene,
    combine_continuous_blue_with_scaling,
    combine_continuous_simple,
    combine_continuous_trimmed_mean,
    combine_continuous_robust_blue
)

# Import PyTorch compatibility utilities
from .pytorch_compat import (
    get_torch_version,
    has_torch_function,
    get_reduction_enum,
    handle_torch_function
)

__all__ = [
    # Tensor operations
    'prepare_param',
    'prepare_covariance',
    'batched_linalg_solve',
    'to_tensor',
    'standardize',
    'unstandardize',
    
    # Validation utilities
    'validate_range',
    'validate_positive',
    'validate_shape',
    'validate_reduction',
    
    # Statistical utilities
    'weighted_mean',
    'weighted_variance',
    'weighted_quantile',
    'r_squared',
    'adjusted_r_squared',
    'mse',
    'rmse',
    'mae',
    'mape',
    'smape',
    
    # Data augmentation
    'Augmentation',
    'GaussianNoiseAugmentation',
    'AdversarialAugmentation',
    'MixtureAugmentation',
    'FeatureJitter',
    'MixUp',
    'FeatureMask',
    
    # Multi-label handling
    'combine_binary_average',
    'combine_binary_weighted_average',
    'combine_dawid_skene',
    'combine_continuous_blue_with_scaling',
    'combine_continuous_simple',
    'combine_continuous_trimmed_mean',
    'combine_continuous_robust_blue',
    
    # PyTorch compatibility
    'get_torch_version',
    'has_torch_function',
    'get_reduction_enum',
    'handle_torch_function'
]
