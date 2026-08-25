"""Public exports from ``torchregress.utils``."""

from __future__ import annotations

import torchregress.utils as utils


def test_utils_exports_coherence_helpers() -> None:
    for symbol in (
        "normal_cdf",
        "split_mean_log_variance",
        "variance_from_logvar",
        "low_rank_output_dim",
        "split_low_rank_gaussian_output",
        "subsample_rows",
        "winsorize",
        "convert_to_tensor",
        "ensure_batch_dim",
        "validate_metric_inputs",
        "validate_sample_weight",
    ):
        assert hasattr(utils, symbol), symbol
