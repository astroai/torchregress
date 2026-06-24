"""Reusable test-time adaptation utilities without owning model architectures."""

from .base import (
    AdaptationBatch,
    SupportsAdaptationParameters,
    SupportsPredictiveBatch,
    SupportsRepresentation,
    flatten_adaptation_parameters,
)
from .bayes import BayesianLinearHead, RecursiveBayesianHead
from .benchmark import CausalTTAHarness
from .calibration import RepresentationShiftInflator
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
    ScoreCDFReweighter,
    WeightedSplitConformalAdapter,
)
from .ot_conformal_predictive import weighted_split_classification_predictive_batch
from .selection import (
    LocalConsistencyConfig,
    confidence_scores,
    entropy_scores,
    local_consistency_weights,
    pseudo_label_targets,
    select_high_confidence,
)
from .subspace import FeatureStatNormalizer, SubspaceAlignmentState, WeightedSubspaceMomentAligner
from .transport import (
    ShiftFactoredPredictiveTransport,
    ShiftFactoredTransportConfig,
    ShiftFactoredTransportState,
)

__all__ = [
    "AdaptationBatch",
    "BayesianLinearHead",
    "CausalTTAHarness",
    "FeatureStatNormalizer",
    "GaussianLabelShiftConfig",
    "LabelShiftEMConfig",
    "LabelShiftEstimate",
    "OptimalTransportCoverageGap",
    "ScoreCDFReweighter",
    "ParameterEMA",
    "PosteriorLabelShiftAdapter",
    "RecursiveBayesianHead",
    "RepresentationShiftInflator",
    "WeightedSubspaceMomentAligner",
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
    "LocalConsistencyConfig",
    "local_consistency_weights",
    "pseudo_label_targets",
    "select_high_confidence",
]
