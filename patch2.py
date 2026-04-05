import re

file_path = "tests/test_public_api_contracts.py"
with open(file_path, "r") as f:
    content = f.read()

correct_test_time = """    "test_time": [
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
    ],"""

old_test_time = """    "test_time": [
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
    ],"""

new_content = content.replace(old_test_time, correct_test_time)

with open(file_path, "w") as f:
    f.write(new_content)
