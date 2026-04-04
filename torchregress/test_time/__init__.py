"""Reusable test-time adaptation utilities without owning model architectures."""

from .base import (
    AdaptationBatch,
    SupportsAdaptationParameters,
    SupportsPredictiveBatch,
    SupportsRepresentation,
    flatten_adaptation_parameters,
)
from .calibration import RepresentationShiftCalibrator
from .dynamic import ParameterEMA
from .label_shift import (
    LabelShiftEstimate,
    PosteriorLabelShiftAdapter,
    apply_label_shift_correction,
    estimate_target_prior_em,
)
from .selection import (
    confidence_scores,
    entropy_scores,
    local_consistency_weights,
    pseudo_label_targets,
    select_high_confidence,
)
from .subspace import FeatureStatNormalizer, SignificantSubspaceAligner, SubspaceAlignmentState

__all__ = [
    "AdaptationBatch",
    "FeatureStatNormalizer",
    "LabelShiftEstimate",
    "ParameterEMA",
    "PosteriorLabelShiftAdapter",
    "RepresentationShiftCalibrator",
    "SignificantSubspaceAligner",
    "SubspaceAlignmentState",
    "SupportsAdaptationParameters",
    "SupportsPredictiveBatch",
    "SupportsRepresentation",
    "apply_label_shift_correction",
    "confidence_scores",
    "entropy_scores",
    "estimate_target_prior_em",
    "flatten_adaptation_parameters",
    "local_consistency_weights",
    "pseudo_label_targets",
    "select_high_confidence",
]
