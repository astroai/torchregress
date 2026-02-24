
import pytest
import torch
import math
from torchregress.losses.conformal import CTI

def gaussian_log_pdf_batched(y_grid, x_features):
    # Mock Gaussian where mean depends on x
    if x_features.dim() == 1:
        # Single sample
        mean = x_features.mean()
        std = 1.0
        return -0.5 * ((y_grid - mean) / std)**2 - math.log(std * math.sqrt(2 * math.pi))
    else:
        # Batch
        mean = x_features.mean(dim=1, keepdim=True) # (N, 1)
        std = 1.0
        diff = y_grid.unsqueeze(0) - mean
        return -0.5 * (diff / std)**2 - math.log(std * math.sqrt(2 * math.pi))

def gaussian_log_pdf_loop_only(y_grid, x_features):
    # Fails on batch input to trigger fallback
    if x_features.dim() > 1:
        raise ValueError("Batch not supported")
    return gaussian_log_pdf_batched(y_grid, x_features)

@pytest.fixture
def cti_calibrated():
    cti = CTI(alpha=0.1, grid_size=100)
    cti._is_calibrated = True
    cti.q_hat = torch.tensor(1.0)
    return cti

def test_cti_vectorized_vs_loop(cti_calibrated):
    torch.manual_seed(42)
    n_test = 20
    n_feat = 5
    x_test = torch.randn(n_test, n_feat)

    # Run with batched function (vectorized path)
    l_vec, u_vec = cti_calibrated.predict_intervals_from_density(
        gaussian_log_pdf_batched, x_test, -5, 5
    )

    # Run with loop-only function (fallback path)
    l_loop, u_loop = cti_calibrated.predict_intervals_from_density(
        gaussian_log_pdf_loop_only, x_test, -5, 5
    )

    assert torch.allclose(l_vec, l_loop), "Vectorized and loop implementations differ (lower bound)"
    assert torch.allclose(u_vec, u_loop), "Vectorized and loop implementations differ (upper bound)"

def test_cti_fallback_trigger(cti_calibrated):
    torch.manual_seed(42)
    x_test = torch.randn(5, 5)

    # Ensure fallback works without error
    l, u = cti_calibrated.predict_intervals_from_density(
        gaussian_log_pdf_loop_only, x_test, -5, 5
    )
    assert l.shape == (5, 1)
    assert u.shape == (5, 1)

def test_cti_empty_set_fallback(cti_calibrated):
    # Create a scenario where density is very low everywhere -> empty set
    # q_hat is small, density is low
    cti_calibrated.q_hat = torch.tensor(-100.0) # threshold is very strict (neg log dens <= -100)

    # Mock density function that returns low values
    # We must ensure it supports batched input to test vectorized path
    def low_density_fn(y_grid, x):
        # Return low density everywhere
        if x.dim() > 1:
            return torch.full((x.shape[0], y_grid.shape[0]), -1000.0)
        return torch.full((y_grid.shape[0],), -1000.0)

    x_test = torch.randn(5, 5)
    # y_grid will be linspace(-5, 5, 100)
    # All densities equal (-1000) -> argmax is 0 (first index)
    # So expected value is -5.0

    l, u = cti_calibrated.predict_intervals_from_density(
        low_density_fn, x_test, -5, 5
    )

    target = torch.full((5, 1), -5.0)
    assert torch.allclose(l, target), "Should fallback to mode (index 0)"
    assert torch.allclose(u, target), "Should fallback to mode (index 0)"
