"""
Utility functions for torch regression.

This module contains various utility functions used throughout the library.
"""

from .augment import Adversarial, Augmentation, EnsemblePerturbationAugmenter
from .bspline import BSplineDensityBasis
from .gaussian_output import split_mean_log_variance, variance_from_logvar
from .ordinal import CORALHead, cumulative_logits_to_pmf, cumulative_probs_to_pmf, ordinal_predict
from .propensity import ipw_weights
from .pytorch_compat import set_all_seeds
from .semisupervised import generate_pseudo_labels, update_ema_teacher_
from .tensor_ops import masked_mean, masked_reduction, masked_sum
from .transform import (
    BoxCoxTransform,
    IdentityTransform,
    LogTransform,
    SqrtTransform,
    TargetTransform,
    YeoJohnsonTransform,
    make_target_transform,
)
from .validation import validate_positive

__all__ = [
    # bspline
    "BSplineDensityBasis",
    # gaussian_output
    "split_mean_log_variance",
    "variance_from_logvar",
    # augment
    "Augmentation",
    "Adversarial",
    "EnsemblePerturbationAugmenter",
    # ordinal
    "cumulative_probs_to_pmf",
    "cumulative_logits_to_pmf",
    "ordinal_predict",
    "CORALHead",
    # propensity
    "ipw_weights",
    # pytorch_compat
    "set_all_seeds",
    # tensor_ops
    "masked_reduction",
    "masked_mean",
    "masked_sum",
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
    "validate_positive",
]
