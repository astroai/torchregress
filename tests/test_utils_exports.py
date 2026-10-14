"""Public exports from ``torchregress.utils``."""

from __future__ import annotations

import torchregress.utils as utils


def test_utils_exports_coherence_helpers() -> None:
    for symbol in ("split_mean_log_variance", "variance_from_logvar"):
        assert hasattr(utils, symbol), symbol


def test_utils_plumbing_helpers_not_reexported() -> None:
    """Internal plumbing stays importable from its submodule, not the namespace."""
    from torchregress.utils import (
        distributions,
        gaussian_output,
        numpy_stats,
        ordinal,
        security,
        tensor_ops,
        validation,
    )

    for module, name in (
        (distributions, "normal_cdf"),
        (gaussian_output, "low_rank_output_dim"),
        (numpy_stats, "winsorize"),
        (ordinal, "labels_to_levels"),
        (security, "validate_url"),
        (tensor_ops, "convert_to_tensor"),
        (validation, "check_tensor"),
    ):
        assert hasattr(module, name), name
        assert not hasattr(utils, name), name
        assert name not in utils.__all__, name
