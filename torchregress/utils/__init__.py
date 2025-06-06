"""
Utility functions for torch regression.

This module contains various utility functions used throughout the library.
"""

from .validation import (
    validate_reduction,
    validate_positive,
    validate_range,
    validate_tensor_shapes,
)
from .masked_ops import apply_mask, masked_reduction, masked_mean, masked_sum
from .tensor_ops import broadcast_shapes
from .labels import encode_onehot, decode_onehot, label_smoothing, soft_to_hard_labels
from .augment import GaussianNoiseAugmenter, MixupAugmenter, CutoutAugmenter, CutMixAugmenter
from .pytorch_compat import get_torch_version
from .transform import log_transform, log_inverse, boxcox_transform, boxcox_inverse, sqrt_transform, sqrt_inverse
from .quantile import quantile_loss, multi_quantile_loss
from .histogram import histogram, histogram_bins, cdf_from_hist

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
