"""
Utility functions for torch regression.

This module contains various utility functions used throughout the library.
"""

from .validation import (
    validate_reduction, validate_positive, validate_range,
    validate_tensor_shapes
)
from .tensor_ops import (
    apply_mask, masked_reduction, broadcast_shapes
)
from .labels import (
    encode_onehot, decode_onehot, 
    label_smoothing, soft_to_hard_labels
)
from .augment import (
    GaussianNoiseAugmenter, MixupAugmenter, 
    CutoutAugmenter, CutMixAugmenter
)
from .pytorch_compat import get_torch_version

__all__ = [
    # Validation utilities
    'validate_reduction',
    'validate_positive',
    'validate_range',
    'validate_tensor_shapes',
    
    # Tensor operations
    'apply_mask',
    'masked_reduction',
    'broadcast_shapes',
    
    # Label utilities
    'encode_onehot',
    'decode_onehot',
    'label_smoothing',
    'soft_to_hard_labels',
    
    # Augmentation utilities
    'GaussianNoiseAugmenter',
    'MixupAugmenter',
    'CutoutAugmenter',
    'CutMixAugmenter',
    
    # PyTorch compatibility
    'get_torch_version'
]
