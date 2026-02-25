
import pytest
import torch
import torch.nn as nn
from torchregress.algorithms.irls import iteratively_reweighted_least_squares

def test_irls_nan_input_rejection():
    """Test that IRLS rejects NaN inputs."""
    model = nn.Linear(1, 1)

    # Create data with NaNs
    x = torch.randn(10, 1)
    x[0, 0] = float('nan')
    y = torch.randn(10, 1)

    with pytest.raises(ValueError, match="contains NaN values"):
        iteratively_reweighted_least_squares(
            model=model,
            x=x,
            y_true=y,
            max_iter=2,
            variance_type='robust'
        )

def test_irls_inf_target_rejection():
    """Test that IRLS rejects infinite targets."""
    model = nn.Linear(1, 1)

    # Create data with Infs
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    y[0, 0] = float('inf')

    with pytest.raises(ValueError, match="contains infinite values"):
        iteratively_reweighted_least_squares(
            model=model,
            x=x,
            y_true=y,
            max_iter=2,
            variance_type='robust'
        )

def test_irls_nan_precision_rejection():
    """Test that IRLS rejects NaN initial precision."""
    model = nn.Linear(1, 1)

    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    precision = torch.ones_like(y)
    precision[0, 0] = float('nan')

    with pytest.raises(ValueError, match="contains NaN values"):
        iteratively_reweighted_least_squares(
            model=model,
            x=x,
            y_true=y,
            initial_precision=precision,
            max_iter=2,
            variance_type='robust'
        )
