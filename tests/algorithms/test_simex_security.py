import pytest
import torch
import torch.nn as nn

from torchregress.algorithms.simex import SIMEX


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def train_linear(model, X, y):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(1):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
    return model


def test_simex_fit_nan_input_rejection():
    """Test that SIMEX fit rejects NaN inputs."""
    x = torch.randn(10, 1)
    x[0, 0] = float("nan")
    y = torch.randn(10, 1)

    simex = SIMEX(model_factory=SimpleLinear, train_func=train_linear, sigma_u=0.1)

    with pytest.raises(ValueError, match="contains NaN values"):
        simex.fit(x, y)


def test_simex_fit_inf_input_rejection():
    """Test that SIMEX fit rejects Inf inputs."""
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    y[0, 0] = float("inf")

    simex = SIMEX(model_factory=SimpleLinear, train_func=train_linear, sigma_u=0.1)

    with pytest.raises(ValueError, match="contains infinite values"):
        simex.fit(x, y)


def test_simex_predict_nan_input_rejection():
    """Test that SIMEX predict rejects NaN inputs."""
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)

    simex = SIMEX(model_factory=SimpleLinear, train_func=train_linear, sigma_u=0.1)
    simex.fit(x, y)

    x_test = torch.randn(10, 1)
    x_test[0, 0] = float("nan")

    with pytest.raises(ValueError, match="contains NaN values"):
        simex.predict(x_test)
