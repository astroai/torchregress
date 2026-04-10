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
    correct_gaussian_predictions_for_label_shift,
    estimate_target_prior_em,
    gaussian_bin_edges_from_targets,
    gaussian_bin_probabilities,
    gaussian_moments_from_binned_probabilities,
)
from .selection import (
    confidence_scores,
    entropy_scores,
    local_consistency_weights,
    pseudo_label_targets,
    select_high_confidence,
)
from .subspace import FeatureStatNormalizer, SignificantSubspaceAligner, SubspaceAlignmentState
from .transport import (
    ShiftFactoredPredictiveTransport,
    ShiftFactoredTransportConfig,
    ShiftFactoredTransportState,
)

__all__ = [
    "AdaptationBatch",
    "FeatureStatNormalizer",
    "LabelShiftEstimate",
    "ParameterEMA",
    "PosteriorLabelShiftAdapter",
    "RepresentationShiftCalibrator",
    "SignificantSubspaceAligner",
    "ShiftFactoredPredictiveTransport",
    "ShiftFactoredTransportConfig",
    "ShiftFactoredTransportState",
    "SubspaceAlignmentState",
    "SupportsAdaptationParameters",
    "SupportsPredictiveBatch",
    "SupportsRepresentation",
    "apply_label_shift_correction",
    "confidence_scores",
    "correct_gaussian_predictions_for_label_shift",
    "entropy_scores",
    "estimate_target_prior_em",
    "flatten_adaptation_parameters",
    "gaussian_bin_edges_from_targets",
    "gaussian_bin_probabilities",
    "gaussian_moments_from_binned_probabilities",
    "local_consistency_weights",
    "pseudo_label_targets",
    "select_high_confidence",
]
