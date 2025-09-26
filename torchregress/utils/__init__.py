"""
Utility functions for torch regression.

This module contains various utility functions used throughout the library.
"""

from .augment import CutMixAugmenter, CutoutAugmenter, GaussianNoiseAugmenter, MixupAugmenter
from .histogram import cdf_from_hist, histogram, histogram_bins
from .labels import decode_onehot, encode_onehot, label_smoothing, soft_to_hard_labels
from .masked_ops import apply_mask, masked_mean, masked_reduction, masked_sum
from .pytorch_compat import get_torch_version
from .quantile import multi_quantile_loss, quantile_loss
from .tensor_ops import broadcast_shapes
from .transform import (
    boxcox_inverse,
    boxcox_transform,
    log_inverse,
    log_transform,
    sqrt_inverse,
    sqrt_transform,
)
from .validation import (
    validate_positive,
    validate_range,
    validate_reduction,
    validate_tensor_shapes,
)

__all__ = [
    # Validation utilities
    "validate_reduction",
    "validate_positive",
    "validate_range",
    "validate_tensor_shapes",
    # Masked tensor operations
    "apply_mask",
    "masked_reduction",
    "masked_mean",
    "masked_sum",
    # Tensor operations
    "broadcast_shapes",
    # Label utilities
    "encode_onehot",
    "decode_onehot",
    "label_smoothing",
    "soft_to_hard_labels",
    # Augmentation utilities
    "GaussianNoiseAugmenter",
    "MixupAugmenter",
    "CutoutAugmenter",
    "CutMixAugmenter",
    # PyTorch compatibility
    "get_torch_version",
    # Transform utilities
    "log_transform",
    "log_inverse",
    "boxcox_transform",
    "boxcox_inverse",
    "sqrt_transform",
    "sqrt_inverse",
    # Quantile utilities
    "quantile_loss",
    "multi_quantile_loss",
    # Histogram/CDF utilities
    "histogram",
    "histogram_bins",
    "cdf_from_hist",
]
