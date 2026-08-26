from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.prediction import (
    PredictiveBatch,
    bars_to_density_grid,
    quantiles_to_density_grid,
    samples_to_density_grid,
)


def test_quantiles_to_density_grid_errors():
    quantiles = np.array([[0.1, 0.4, 0.8], [0.2, 0.5, 0.9]])

    # Wrong quantiles shape
    with pytest.raises(ValueError, match="quantiles must have shape"):
        quantiles_to_density_grid(np.array([0.1, 0.4, 0.8]), [0.1, 0.5, 0.9])

    # Shape mismatch
    with pytest.raises(ValueError, match="quantile_levels must match"):
        quantiles_to_density_grid(quantiles, [0.1, 0.5])

    # Too few levels
    with pytest.raises(ValueError, match="at least two quantile levels"):
        quantiles_to_density_grid(np.array([[0.5]]), [0.5])

    # Non-monotonic levels
    with pytest.raises(ValueError, match="strictly increasing"):
        quantiles_to_density_grid(quantiles, [0.5, 0.1, 0.9])


def test_bars_to_density_grid_errors():
    logits = np.array([[1.0, 0.5, -0.2]])
    edges = np.array([0.0, 0.5, 1.0, 1.5])

    # Wrong logits shape
    with pytest.raises(ValueError, match="bar_logits must have shape"):
        bars_to_density_grid(np.array([1.0, 0.5, -0.2]), edges)

    # Shape mismatch
    with pytest.raises(ValueError, match="bin_edges must have shape"):
        bars_to_density_grid(logits, np.array([0.0, 0.5, 1.0]))

    # Edge shape mismatch with 2D edges
    with pytest.raises(ValueError, match="bin_edges must have shape"):
        bars_to_density_grid(logits, np.array([[0.0, 0.5, 1.0]]))


def test_samples_to_density_grid_errors():

    # Wrong shape
    with pytest.raises(ValueError, match="samples must have shape"):
        samples_to_density_grid(np.array([0.1, 0.4, 0.8]))

    # Too few samples
    with pytest.raises(ValueError, match="at least two samples"):
        samples_to_density_grid(np.array([[0.1], [0.2]]))


def test_samples_to_density_grid_3d():
    samples = np.array([[[0.1], [0.4], [0.8]], [[0.2], [0.5], [0.9]]])
    support, density = samples_to_density_grid(samples, n_support=10)
    assert support.shape == (2, 10)
    assert density.shape == (2, 10)
    # Check normalization
    assert np.allclose(np.trapezoid(density, support, axis=1), 1.0, atol=1.0e-3)


def test_predictive_batch_with_density_skips_when_present():
    batch = PredictiveBatch(
        support=np.array([[0.0, 1.0]]),
        density=np.array([[1.0, 1.0]]),
    )
    res = batch.with_density()
    assert res is batch


def test_predictive_batch_with_density_returns_self_when_empty():
    batch = PredictiveBatch()
    res = batch.with_density()
    assert res is batch


def test_predictive_batch_identical_support_collapsing():
    # Helper to test the collapsing logic when support for all items in batch is close

    # Quantiles
    quantiles = np.array([[0.1, 0.5, 0.9], [0.1, 0.5, 0.9]])
    batch = PredictiveBatch(quantiles=quantiles, quantile_levels=[0.1, 0.5, 0.9])
    res = batch.with_density(n_support=5)
    assert res.support.ndim == 1  # Collapsed!
    assert res.support.shape == (5,)

    # Bars
    logits = np.array([[1.0, 0.5], [1.0, 0.5]])
    edges = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    batch2 = PredictiveBatch(bar_logits=logits, bin_edges=edges)
    res2 = batch2.with_density(n_support=5)
    assert res2.support.ndim == 1  # Collapsed!
    assert res2.support.shape == (5,)

    # Samples
    samples = np.array([[0.1, 0.5, 0.9], [0.1, 0.5, 0.9]])
    batch3 = PredictiveBatch(samples=samples)
    res3 = batch3.with_density(n_support=5)
    assert res3.support.ndim == 1  # Collapsed!
    assert res3.support.shape == (5,)


def test_quantiles_grid_zero_on_extrapolated_margins():
    # TR-COR-02: support outside [q_first, q_last] must have exactly zero density.
    q = torch.tensor([[1.0, 2.0, 3.0]])
    levels = [0.1, 0.5, 0.9]
    support, density = quantiles_to_density_grid(q, levels, range_margin=0.2)
    below = support[0] < 1.0
    above = support[0] > 3.0
    assert below.any() and above.any(), "grid must include extrapolated margins"
    assert torch.count_nonzero(density[0][below]) == 0
    assert torch.count_nonzero(density[0][above]) == 0
    interior = (support[0] >= 1.0) & (support[0] <= 3.0)
    assert torch.count_nonzero(density[0][interior]) > 0
    integral = torch.trapezoid(density, support, dim=1)
    assert torch.allclose(integral, torch.ones_like(integral), rtol=1e-3)


def _bars_reference_rowwise(
    logits: torch.Tensor, edges: torch.Tensor, *, n_support: int, range_margin: float
):
    """Row-wise reference replicating the pre-vectorization implementation."""
    logits = logits - logits.amax(dim=1, keepdim=True)
    probs = logits.exp()
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1.0e-8)
    lo = edges[:, 0:1]
    hi = edges[:, -1:]
    width = (hi - lo).clamp(min=1.0e-6)
    lo = lo - range_margin * width
    hi = hi + range_margin * width
    steps = torch.linspace(0, 1, n_support, dtype=logits.dtype)
    support = lo + (hi - lo) * steps[None, :]
    density = torch.empty_like(support)
    for idx in range(logits.shape[0]):
        widths = edges[idx].diff().clamp(min=1.0e-8)
        bar_density = probs[idx] / widths
        bin_idx = torch.bucketize(support[idx], edges[idx][1:-1], right=False).clamp(
            0, logits.shape[1] - 1
        )
        dens = bar_density[bin_idx].clamp(min=0.0)
        integral_val = float(torch.trapezoid(dens, support[idx]).item())
        density[idx] = dens / max(integral_val, 1.0e-8)
    return support, density


def _samples_reference_rowwise(draws: torch.Tensor, *, n_support: int, range_margin: float):
    """Row-wise reference implementing the documented histogram semantics."""
    sample_lo = draws.amin(dim=1)
    sample_hi = draws.amax(dim=1)
    width = (sample_hi - sample_lo).clamp(min=1.0e-6)
    lo = (sample_lo - range_margin * width)[:, None]
    hi = (sample_hi + range_margin * width)[:, None]
    steps = torch.linspace(0, 1, n_support, dtype=draws.dtype)
    support = lo + (hi - lo) * steps[None, :]
    density = torch.empty_like(support)
    for idx in range(draws.shape[0]):
        edges = lo[idx] + (hi[idx] - lo[idx]) * torch.linspace(
            0, 1, n_support + 1, dtype=draws.dtype
        )
        counts = []
        for b in range(n_support):
            if b < n_support - 1:
                member = (draws[idx] >= edges[b]) & (draws[idx] < edges[b + 1])
            else:
                member = (draws[idx] >= edges[b]) & (draws[idx] <= edges[b + 1])
            counts.append(member.sum())
        hist = torch.tensor(counts, dtype=draws.dtype)
        widths = edges.diff().clamp(min=1.0e-8)
        dens = hist / max(float(draws.shape[1]), 1.0) / widths
        row = dens.repeat_interleave(2)
        edge_support = edges.repeat_interleave(2)[1:-1]
        idxs = torch.searchsorted(edge_support, support[idx]).clamp(0, row.size(0) - 2)
        density[idx] = row[idxs]
        integral_val = float(torch.trapezoid(density[idx], support[idx]).item())
        density[idx] = density[idx] / max(integral_val, 1.0e-8)
    return support, density


def test_vectorized_grids_match_rowwise_reference():
    rng = np.random.default_rng(42)
    batch, bins, n_samples = 4, 5, 64

    logits = torch.tensor(rng.normal(size=(batch, bins)), dtype=torch.float32)
    edges = torch.tensor(
        np.stack([np.linspace(-1.0 - i * 0.25, 1.0 + i * 0.25, bins + 1) for i in range(batch)])
    )
    sup_v, den_v = bars_to_density_grid(logits, edges, n_support=48, range_margin=0.1)
    sup_r, den_r = _bars_reference_rowwise(logits, edges, n_support=48, range_margin=0.1)
    assert torch.allclose(sup_v, sup_r, atol=1e-6)
    assert torch.allclose(den_v, den_r, atol=1e-6)

    draws = torch.tensor(rng.normal(size=(batch, n_samples)).astype(np.float32))
    sup_v, den_v = samples_to_density_grid(draws, n_support=48, range_margin=0.1)
    sup_r, den_r = _samples_reference_rowwise(draws, n_support=48, range_margin=0.1)
    assert torch.equal(sup_v, sup_r)
    assert torch.equal(den_v, den_r)


def test_density_grids_preserve_input_device():
    q = torch.tensor([[1.0, 2.0, 3.0]])
    support, density = quantiles_to_density_grid(q, [0.1, 0.5, 0.9])
    assert support.device == q.device and density.device == q.device

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    q_gpu = q.cuda()
    logits_gpu = torch.randn(2, 3, device="cuda")
    edges_gpu = torch.tensor([0.0, 1.0, 2.0, 3.0], device="cuda").expand(2, -1)
    draws_gpu = torch.randn(2, 32, device="cuda")
    for fn, args in (
        (quantiles_to_density_grid, (q_gpu, [0.1, 0.5, 0.9])),
        (bars_to_density_grid, (logits_gpu, edges_gpu)),
        (samples_to_density_grid, (draws_gpu,)),
    ):
        support, density = fn(*args)
        assert support.device.type == "cuda" and density.device.type == "cuda"
