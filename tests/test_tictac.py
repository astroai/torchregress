import pytest
import torch
import torch.nn as nn

from torchregress.algorithms.tictac import TaylorInducedCovarianceHead
from torchregress.metrics.tac import TaskAgnosticCorrelations, task_agnostic_correlations


class SimpleMeanModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.Tanh(),
            nn.Linear(8, out_features),
        )

    def forward(self, x):
        return self.net(x)


def test_tictac_head_global_params() -> None:
    in_features = 4
    target_dim = 2
    batch_size = 3

    base_model = SimpleMeanModel(in_features, target_dim)
    head = TaylorInducedCovarianceHead(base_model, target_dim=target_dim)

    x = torch.randn(batch_size, in_features)
    mean, cov = head(x)

    assert mean.shape == (batch_size, target_dim)
    assert cov.shape == (batch_size, target_dim, target_dim)

    # Check that covariance is symmetric
    torch.testing.assert_close(cov, cov.transpose(-1, -2))

    # Check that gradients flow to base model parameters and log parameters
    loss = mean.sum() + cov.sum()
    loss.backward()

    # Check base model parameters
    for p in base_model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()

    # Check head log parameters
    assert head.log_k1.grad is not None
    assert head.log_k2.grad is not None
    assert head.log_k3.grad is not None


def test_tictac_head_input_dependent() -> None:
    in_features = 4
    target_dim = 2
    batch_size = 3

    base_model = SimpleMeanModel(in_features, target_dim)
    head = TaylorInducedCovarianceHead(base_model, target_dim=target_dim, input_dim=in_features)

    x = torch.randn(batch_size, in_features)
    mean, cov = head(x)

    assert mean.shape == (batch_size, target_dim)
    assert cov.shape == (batch_size, target_dim, target_dim)

    loss = mean.sum() + cov.sum()
    loss.backward()

    for p in head.k1_net.parameters():
        assert p.grad is not None


def test_tac_metric_correctness() -> None:
    batch_size = 4
    target_dim = 3

    y_pred = torch.randn(batch_size, target_dim)
    y_true = torch.randn(batch_size, target_dim)
    covariance = torch.eye(target_dim).unsqueeze(0).repeat(batch_size, 1, 1)

    val = task_agnostic_correlations(y_pred, y_true, covariance)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 0
    assert torch.isfinite(val).item()

    # Check TaskAgnosticCorrelations class
    metric = TaskAgnosticCorrelations()
    metric.update(y_pred, y_true, covariance)
    val_class = metric.compute()
    torch.testing.assert_close(val, val_class)


def test_tac_metric_shape_mismatch_raises() -> None:
    y_pred = torch.zeros(2, 2)
    y_true = torch.zeros(2, 2)
    # Incorrect covariance shape: should be [2, 2, 2]
    covariance = torch.zeros(2, 3, 3)

    metric = TaskAgnosticCorrelations()
    with pytest.raises(ValueError, match="covariance shape"):
        metric.update(y_pred, y_true, covariance)
