"""Reusable test-time adaptation utilities without owning model architectures."""

from .base import (
    AdaptationBatch,
    SupportsAdaptationParameters,
    SupportsPredictiveBatch,
    SupportsRepresentation,
    flatten_adaptation_parameters,
)
from .bayes import BayesianLinearHead, RecursiveBayesianHead
from .calibration import RepresentationShiftCalibrator
from .dynamic import ParameterEMA
from .label_shift import (
    GaussianLabelShiftConfig,
    LabelShiftEMConfig,
    LabelShiftEstimate,
    PosteriorLabelShiftAdapter,
    apply_label_shift_correction,
    correct_gaussian_predictions_for_label_shift,
    estimate_target_prior_em,
    gaussian_bin_edges_from_targets,
    gaussian_bin_probabilities,
    gaussian_moments_from_binned_probabilities,
)
from .ot_conformal import (
    OptimalTransportCoverageGap,
    OTShiftReweighter,
    WeightedSplitConformalAdapter,
)
from .ot_conformal_predictive import weighted_split_classification_predictive_batch
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
    "BayesianLinearHead",
    "FeatureStatNormalizer",
    "GaussianLabelShiftConfig",
    "LabelShiftEMConfig",
    "LabelShiftEstimate",
    "OptimalTransportCoverageGap",
    "OTShiftReweighter",
    "ParameterEMA",
    "PosteriorLabelShiftAdapter",
    "RecursiveBayesianHead",
    "RepresentationShiftCalibrator",
    "SignificantSubspaceAligner",
    "ShiftFactoredPredictiveTransport",
    "ShiftFactoredTransportConfig",
    "ShiftFactoredTransportState",
    "SubspaceAlignmentState",
    "SupportsAdaptationParameters",
    "SupportsPredictiveBatch",
    "SupportsRepresentation",
    "WeightedSplitConformalAdapter",
    "weighted_split_classification_predictive_batch",
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
