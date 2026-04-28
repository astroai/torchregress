import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.algorithms.irls import (
    IRLS,
    _batched_predict,
    buffer_data,
    calculate_mad,
    iteratively_reweighted_least_squares,
)
from torchregress.losses.gaussian import GaussianNLLLoss


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def data():
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    return x, y


def test_buffer_data_cpu(data):
    x, y = data
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=10)

    # buffer to cpu
    bx, by, bcov, bmask = buffer_data(loader, "cpu")
    assert bx.device == torch.device("cpu")
    assert by.device == torch.device("cpu")
    assert bx.shape == x.shape
    assert by.shape == y.shape


def test_batched_predict(data):
    x, y = data
    model = SimpleModel()

    # Predict with small batch size to force batching
    y_pred = _batched_predict(model, x, batch_size=10)

    assert y_pred.shape == (100, 1)

    # Verify correctness against full forward pass
    y_pred_full = model(x)
    assert torch.allclose(y_pred, y_pred_full, atol=1e-5)


def test_irls_integration(data):
    x, y = data
    model = SimpleModel()

    # Run IRLS with small batch size
    # We use update_weights="epoch" to trigger buffer_data and iteratively_reweighted_least_squares
    res = IRLS(
        model=model,
        train_data=(x, y),
        num_epochs=2,
        batch_size=20,
        update_weights="epoch",
        verbose=False,
        base_loss="gaussian",
        loss_fn=GaussianNLLLoss(fixed_variance=1.0),
        variance_type="fixed",
    )

    assert "model" in res
    assert "train_loss_history" in res
    assert len(res["train_loss_history"]) > 0


def test_iteratively_reweighted_least_squares_logic(data):
    x, y = data
    model = SimpleModel()

    # Should run without error
    y_pred, loss_hist, precision = iteratively_reweighted_least_squares(
        model=model,
        x=x,
        y_true=y,
        batch_size=10,
        max_iter=3,
        base_loss="gaussian",
        variance_type="fixed",
        weight_fn="huber",
    )

    assert y_pred.shape == (100, 1)
    assert len(loss_hist) <= 3
    assert precision.shape == y.shape


def test_calculate_mad():
    import scipy.stats as stats

    # 1D case
    t1 = torch.tensor([1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0])
    mad1 = calculate_mad(t1)
    sp_mad1 = stats.median_abs_deviation(t1.numpy(), scale=1.0)
    assert torch.allclose(mad1, torch.tensor(sp_mad1, dtype=torch.float32))

    # 2D case
    t2 = torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ]
    )

    # dim=1
    mad2_dim1 = calculate_mad(t2, dim=1)
    sp_mad2_dim1 = stats.median_abs_deviation(t2.numpy(), axis=1, scale=1.0)
    assert torch.allclose(mad2_dim1.squeeze(), torch.tensor(sp_mad2_dim1, dtype=torch.float32))

    # dim=0
    mad2_dim0 = calculate_mad(t2, dim=0)
    sp_mad2_dim0 = stats.median_abs_deviation(t2.numpy(), axis=0, scale=1.0)
    assert torch.allclose(mad2_dim0.squeeze(), torch.tensor(sp_mad2_dim0, dtype=torch.float32))


def test_calculate_mad_edge_cases():

    # 0D case
    t0 = torch.tensor(1.0)
    mad0 = calculate_mad(t0)
    assert torch.allclose(mad0, torch.tensor(0.0))

    # All same values
    t_same = torch.tensor([5.0, 5.0, 5.0, 5.0])
    mad_same = calculate_mad(t_same)
    assert torch.allclose(mad_same, torch.tensor(0.0))

    # Empty tensor
    t_empty = torch.tensor([])
    with pytest.raises(IndexError):
        calculate_mad(t_empty)
