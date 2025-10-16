"""
Utility functions for torch regression.

This module contains various utility functions used throughout the library.
"""

from .augment import (
    Augmentation,
    GaussianNoise,
    Adversarial,
    MixUp,
    FeatureMask,
)
from .labels import (
    encode_onehot,
    decode_onehot,
    label_smoothing,
    soft_to_hard_labels,
    combine_binary_average,
    combine_binary_weighted_average,
)
from .pytorch_compat import (
    convert_reduction_type,
    convert_to_pytorch_loss,
    extract_output_size,
    set_all_seeds,
    get_device,
)
from .quantile import (
    quantile_loss,
    multi_quantile_loss,
)
from .tensor_ops import (
    apply_mask,
    masked_reduction,
    masked_mean,
    masked_sum,
    prepare_param,
    prepare_sigma,
    prepare_covariance,
    prepare_cross_covariance,
    prepare_model_input_for_gradients,
    batched_linalg_solve,
    standardize,
    unstandardize,
    compute_model_gradients,
    calculate_gaussian_nll,
    calculate_propagated_variance,
)
from .transform import (
    log_transform,
    log_inverse,
    boxcox_transform,
    boxcox_inverse,
    sqrt_transform,
    sqrt_inverse,
)
from .validation import (
    validate_reduction,
    validate_shape,
    validate_positive,
    validate_range,
    validate_integer,
    validate_quantile,
    validate_batch_consistency,
    validate_same_device,
    validate_weights,
    check_tensor,
)

__all__ = [
    # augment
    "Augmentation",
    "GaussianNoise",
    "Adversarial",
    "MixUp",
    "FeatureMask",
    # labels
    "encode_onehot",
    "decode_onehot",
    "label_smoothing",
    "soft_to_hard_labels",
    "combine_binary_average",
    "combine_binary_weighted_average",
    # pytorch_compat
    "convert_reduction_type",
    "convert_to_pytorch_loss",
    "extract_output_size",
    "set_all_seeds",
    "get_device",
    # quantile
    "quantile_loss",
    "multi_quantile_loss",
    # tensor_ops
    "apply_mask",
    "masked_reduction",
    "masked_mean",
    "masked_sum",
    "prepare_param",
    "prepare_sigma",
    "prepare_covariance",
    "prepare_cross_covariance",
    "prepare_model_input_for_gradients",
    "batched_linalg_solve",
    "standardize",
    "unstandardize",
    "compute_model_gradients",
    "calculate_gaussian_nll",
    "calculate_propagated_variance",
    # transform
    "log_transform",
    "log_inverse",
    "boxcox_transform",
    "boxcox_inverse",
    "sqrt_transform",
    "sqrt_inverse",
    # validation
    "validate_reduction",
    "validate_shape",
    "validate_positive",
    "validate_range",
    "validate_integer",
    "validate_quantile",
    "validate_batch_consistency",
    "validate_same_device",
    "validate_weights",
    "check_tensor",
]