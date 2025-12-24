"""
Utility functions for torch regression.

This module contains various utility functions used throughout the library.
"""

from .augment import (
    Adversarial,
    Augmentation,
    EnsemblePerturbationAugmenter,
    FeatureMask,
    GaussianNoise,
    MixUp,
)
from .labels import (
    combine_binary_average,
    combine_binary_weighted_average,
    decode_onehot,
    encode_onehot,
    label_smoothing,
    soft_to_hard_labels,
)
from .pytorch_compat import (
    convert_reduction_type,
    convert_to_pytorch_loss,
    extract_output_size,
    get_device,
    set_all_seeds,
    set_seed,
)
from .quantile import (
    multi_quantile_loss,
    quantile_loss,
)
from .tensor_ops import (
    apply_mask,
    batched_linalg_solve,
    calculate_gaussian_nll,
    calculate_propagated_variance,
    compute_model_gradients,
    masked_mean,
    masked_reduction,
    masked_sum,
    prepare_covariance,
    prepare_cross_covariance,
    prepare_model_input_for_gradients,
    prepare_param,
    prepare_sigma,
    standardize,
    unstandardize,
)
from .transform import (
    boxcox_inverse,
    boxcox_transform,
    log_inverse,
    log_transform,
    sqrt_inverse,
    sqrt_transform,
)
from .validation import (
    check_tensor,
    validate_batch_consistency,
    validate_integer,
    validate_positive,
    validate_quantile,
    validate_range,
    validate_reduction,
    validate_same_device,
    validate_shape,
    validate_weights,
)

__all__ = [
    # augment
    "Augmentation",
    "GaussianNoise",
    "Adversarial",
    "MixUp",
    "FeatureMask",
    "EnsemblePerturbationAugmenter",
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
    "set_seed",
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
