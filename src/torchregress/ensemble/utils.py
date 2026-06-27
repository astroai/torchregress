"""
Utility functions for ensemble models.

This module provides common utility functions used across different
ensemble implementations, including tools for prediction aggregation
and uncertainty estimation.
"""

from torchregress.utils.gaussian_output import parse_heteroscedastic_output

__all__ = [
    "parse_heteroscedastic_output",
]
