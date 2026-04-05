"""
torchregress: PyTorch-based regression modeling toolkit.

This package provides utilities, models, and loss functions for
various regression tasks, with a focus on uncertainty
estimation and robust regression techniques.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

# Eager import only the small, high-value top-level symbols.
from .algorithms.irls import iteratively_reweighted_least_squares
from .losses.base import BaseLoss, DistributionLoss, RegressionLoss

# Keep top-level import light; large subpackages are resolved lazily to reduce
# import overhead and avoid pulling optional/heavy plotting dependencies unless used.
_LAZY_SUBMODULES = {
    "losses",
    "metrics",
    "algorithms",
    "ensemble",
    "test_time",
    "method_catalog",
    "inference",
    "constraints",
    "calibration",
    "causal",
    "prediction",
    "viz",
    "utils",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _LAZY_SUBMODULES)


__all__ = [
    # Main modules
    "losses",
    "metrics",
    "algorithms",
    "ensemble",
    "test_time",
    "method_catalog",
    "inference",
    "constraints",
    "calibration",
    "causal",
    "prediction",
    "viz",
    "utils",
    # Core classes
    "BaseLoss",
    "RegressionLoss",
    "DistributionLoss",
    # Core functions
    "iteratively_reweighted_least_squares",
    "__version__",
]
