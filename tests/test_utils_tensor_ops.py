import math

import torch

from torchregress.utils.tensor_ops import calculate_gaussian_nll


def test_calculate_gaussian_nll_small_variance():
    """Test Gaussian NLL with tiny/zero variance handles numerical stability."""
    residuals = torch.tensor([[0.5, -0.2]])

    # Tiny variance near 0 (diagonal)
    var = torch.tensor([[1e-10, 1e-10]])

    # Should not produce NaNs or Infs due to eps=1e-8 padding
    nll = calculate_gaussian_nll(residuals, var)

    assert not torch.isnan(nll).any()
    assert not torch.isinf(nll).any()

    # Full covariance with 0 variance
    var_full = torch.zeros(1, 2, 2)
    nll_full = calculate_gaussian_nll(residuals, var_full)

    assert not torch.isnan(nll_full).any()
    assert not torch.isinf(nll_full).any()


def test_calculate_gaussian_nll_empty():
    """Test Gaussian NLL with empty tensors."""
    residuals = torch.zeros(0, 2)
    var = torch.ones(0, 2)
    nll = calculate_gaussian_nll(residuals, var)
    assert nll.numel() == 0


def test_calculate_gaussian_nll_diagonal_exact():
    """Test Gaussian NLL diagonal covariance exact calculation."""
    residuals = torch.tensor([[1.0, 2.0]])
    var = torch.tensor([[0.5, 2.0]])
    eps = 1e-8

    nll = calculate_gaussian_nll(residuals, var, eps=eps)

    # Manual calculation
    expected_nll_0 = 0.5 * (math.log(0.5 + eps) + (1.0**2) / (0.5 + eps))
    expected_nll_1 = 0.5 * (math.log(2.0 + eps) + (2.0**2) / (2.0 + eps))
    expected_sum = expected_nll_0 + expected_nll_1 + 0.5 * 2 * math.log(2 * math.pi)

    assert torch.allclose(nll[0], torch.tensor(expected_sum), atol=1e-6)


def test_calculate_gaussian_nll_zero_variance_diagonal():
    """Test Gaussian NLL with 0 variance for diagonal covariance."""
    residuals = torch.tensor([[0.0, 0.0]])
    var = torch.tensor([[0.0, 0.0]])

    nll = calculate_gaussian_nll(residuals, var)
    assert not torch.isnan(nll).any()
    assert not torch.isinf(nll).any()


def test_calculate_gaussian_nll_negative_variance():
    """Test Gaussian NLL numerical stability when variance is accidentally negative (e.g. from numerical precision)."""
    residuals = torch.tensor([[0.5, 0.5]])
    var = torch.tensor([[-1e-10, -1e-10]])  # slightly negative

    # Should not produce NaNs or Infs due to eps=1e-8 padding which should make var + eps positive
    nll = calculate_gaussian_nll(residuals, var)
    assert not torch.isnan(nll).any()
    assert not torch.isinf(nll).any()


def test_calculate_gaussian_nll_zero_variance_full():
    """Test Gaussian NLL with 0 variance for full covariance."""
    residuals = torch.tensor([[0.0, 0.0]])
    var = torch.zeros(1, 2, 2)

    nll = calculate_gaussian_nll(residuals, var)
    assert not torch.isnan(nll).any()
    assert not torch.isinf(nll).any()


def test_calculate_gaussian_nll_eps_parameter():
    """Test that eps parameter is respected for both diagonal and full covariance."""
    residuals = torch.tensor([[1.0, 2.0]])
    var_diag = torch.tensor([[0.0, 0.0]])
    var_full = torch.zeros(1, 2, 2)

    nll_diag_1 = calculate_gaussian_nll(residuals, var_diag, eps=1e-4)
    nll_diag_2 = calculate_gaussian_nll(residuals, var_diag, eps=1e-2)
    assert nll_diag_1.item() != nll_diag_2.item()

    nll_full_1 = calculate_gaussian_nll(residuals, var_full, eps=1e-4)
    nll_full_2 = calculate_gaussian_nll(residuals, var_full, eps=1e-2)
    assert nll_full_1.item() != nll_full_2.item()
