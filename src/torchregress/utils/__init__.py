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
from .ordinal import (
    class_probs_to_levels,
    cumulative_logits_to_pmf,
    cumulative_probs_to_pmf,
    labels_to_levels,
    normalize_class_probs,
    ordinal_predict,
)
from .propensity import (
    PropensityEstimator,
    ipw_weights,
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
from .scaling import (
    AMP,
    GradientAccumulation,
    StandardScaler,
    compile_model,
)
from .security import validate_url
from .semisupervised import (
    generate_pseudo_labels,
    update_ema_teacher_,
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
    BoxCoxTransform,
    IdentityTransform,
    LogTransform,
    SqrtTransform,
    TargetTransform,
    YeoJohnsonTransform,
    boxcox_inverse,
    boxcox_transform,
    log_inverse,
    log_transform,
    make_target_transform,
    sqrt_inverse,
    sqrt_transform,
    yeojohnson_inverse,
    yeojohnson_transform,
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
    # ordinal
    "labels_to_levels",
    "normalize_class_probs",
    "class_probs_to_levels",
    "cumulative_probs_to_pmf",
    "cumulative_logits_to_pmf",
    "ordinal_predict",
    # propensity
    "PropensityEstimator",
    "ipw_weights",
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
    # scaling
    "AMP",
    "GradientAccumulation",
    "StandardScaler",
    "compile_model",
    # semi-supervised
    "generate_pseudo_labels",
    "update_ema_teacher_",
    # transform
    "TargetTransform",
    "IdentityTransform",
    "LogTransform",
    "BoxCoxTransform",
    "SqrtTransform",
    "YeoJohnsonTransform",
    "log_transform",
    "log_inverse",
    "boxcox_transform",
    "boxcox_inverse",
    "sqrt_transform",
    "sqrt_inverse",
    "yeojohnson_transform",
    "yeojohnson_inverse",
    "make_target_transform",
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
    # security
    "validate_url",
]
