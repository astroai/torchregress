"""
Utility functions for torch regression.

This module contains various utility functions used throughout the library.
"""

from .augment import (
    Adversarial,
    Augmentation,
    EnsemblePerturbationAugmenter,
)
from .distributions import normal_cdf
from .gaussian_output import (
    low_rank_output_dim,
    parse_heteroscedastic_output,
    split_low_rank_gaussian_output,
    split_mean_log_variance,
    variance_from_logvar,
)
from .numpy_stats import subsample_rows, winsorize
from .ordinal import (
    CORALHead,
    class_probs_to_levels,
    cumulative_logits_to_pmf,
    cumulative_probs_to_pmf,
    labels_to_levels,
    normalize_class_probs,
    ordinal_predict,
)
from .propensity import (
    ipw_weights,
)
from .pytorch_compat import (
    get_device,
    set_all_seeds,
)
from .quantile import (
    multi_quantile_loss,
    quantile_loss,
)
from .reduction import reduce_per_sample
from .security import validate_url
from .semisupervised import (
    generate_pseudo_labels,
    update_ema_teacher_,
)
from .tensor_ops import (
    apply_mask,
    calculate_gaussian_nll,
    calculate_propagated_variance,
    compute_model_gradients,
    convert_to_tensor,
    ensure_batch_dim,
    masked_mean,
    masked_reduction,
    masked_sum,
    prepare_cross_covariance,
    prepare_model_input_for_gradients,
)
from .transform import (
    BoxCoxTransform,
    IdentityTransform,
    LogTransform,
    SqrtTransform,
    TargetTransform,
    YeoJohnsonTransform,
    make_target_transform,
)
from .validation import (
    check_tensor,
    validate_metric_inputs,
    validate_positive,
    validate_quantile,
    validate_range,
    validate_reduction,
    validate_sample_weight,
    validate_weights,
)

__all__ = [
    # distributions
    "normal_cdf",
    # gaussian_output
    "parse_heteroscedastic_output",
    "low_rank_output_dim",
    "split_low_rank_gaussian_output",
    "split_mean_log_variance",
    "variance_from_logvar",
    "reduce_per_sample",
    # numpy_stats
    "subsample_rows",
    "winsorize",
    # augment
    "Augmentation",
    "Adversarial",
    "EnsemblePerturbationAugmenter",
    # ordinal
    "labels_to_levels",
    "normalize_class_probs",
    "class_probs_to_levels",
    "cumulative_probs_to_pmf",
    "cumulative_logits_to_pmf",
    "ordinal_predict",
    "CORALHead",
    # propensity
    "ipw_weights",
    # pytorch_compat
    "set_all_seeds",
    "get_device",
    # quantile
    "quantile_loss",
    "multi_quantile_loss",
    # tensor_ops
    "apply_mask",
    "convert_to_tensor",
    "ensure_batch_dim",
    "masked_reduction",
    "masked_mean",
    "masked_sum",
    "prepare_cross_covariance",
    "prepare_model_input_for_gradients",
    "compute_model_gradients",
    "calculate_gaussian_nll",
    "calculate_propagated_variance",
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
    "make_target_transform",
    # validation
    "validate_reduction",
    "validate_positive",
    "validate_range",
    "validate_quantile",
    "validate_weights",
    "validate_metric_inputs",
    "validate_sample_weight",
    "check_tensor",
    # security
    "validate_url",
]
