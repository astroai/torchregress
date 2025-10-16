"""
Specialized algorithms for regression tasks.

This module provides implementations of algorithms for various
regression tasks, including IRLS, Gaussian processes, and others.
"""

from .irls import (
    IRLS,
    iteratively_reweighted_least_squares,
)

__all__ = [
    # IRLS algorithm
    "iteratively_reweighted_least_squares",
    "IRLS",
]