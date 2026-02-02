"""
Specialized algorithms for regression tasks.

This module provides implementations of algorithms for various
regression tasks, including IRLS, RC, and SIMEX.
"""

from .irls import (
    IRLS,
    iteratively_reweighted_least_squares,
)
from .rc import RegressionCalibration
from .simex import SIMEX

__all__ = [
    # IRLS algorithm
    "iteratively_reweighted_least_squares",
    "IRLS",
    # EIV algorithms
    "RegressionCalibration",
    "SIMEX",
]
