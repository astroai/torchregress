import os

import pytest
import torch


@pytest.fixture
def device():
    """Return the device to use for tensor operations."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    """Return a standard batch size for tests."""
    return 10


@pytest.fixture
def sample_data(batch_size, device):
    """Return sample prediction and target tensors."""
    y_pred = torch.randn(batch_size, 1, device=device)
    y_true = torch.randn(batch_size, 1, device=device)
    return y_pred, y_true


@pytest.fixture
def sample_mask(batch_size, device):
    """Return a sample boolean mask."""
    return torch.randint(0, 2, (batch_size, 1), device=device).bool()


@pytest.fixture
def sample_weights(batch_size, device):
    """Return sample weights for weighted loss tests."""
    return torch.rand(batch_size, 1, device=device)
