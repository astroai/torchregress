"""
Histogram and CDF utilities for regression metrics.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional


def histogram(
    values: Tensor,
    bin_edges: Tensor,
) -> Tensor:
    """
    Compute histogram counts for given values and bin edges.

    Args:
        values: 1D tensor of values
        bin_edges: 1D tensor of bin boundaries

    Returns:
        counts: 1D tensor of length len(bin_edges)-1
    """
    return torch.histogram(values, bin_edges)[0]


def histogram_bins(
    values: Tensor,
    bins: int,
    range: Optional[Tuple[float, float]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute histogram counts and bin edges.

    Args:
        values: 1D tensor of values
        bins: Number of bins
        range: (min, max) range for bins

    Returns:
        counts, bin_edges
    """
    hist = torch.histogram(values, bins=bins, range=range)
    return hist.hist, hist.bin_edges


def cdf_from_hist(
    counts: Tensor,
) -> Tensor:
    """
    Compute cumulative distribution function from histogram counts.

    Args:
        counts: 1D tensor of histogram counts

    Returns:
        cdf: 1D tensor of cdf values
    """
    total = torch.sum(counts).clamp(min=1)
    return torch.cumsum(counts, dim=0) / total
