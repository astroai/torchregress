import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.algorithms.heteroscedastic_laplace import (
    HeteroscedasticLaplaceRegressor,
    NaturalHeteroscedasticHead,
    NaturalReparamHead,
)
from torchregress.prediction import PredictiveBatch


class SimpleModel(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def test_natural_heteroscedastic_head() -> None:
    in_features = 4
    out_features = 2
    batch_size = 3

    head = NaturalHeteroscedasticHead(in_features, out_features, link_fn="exp")
    x = torch.randn(batch_size, in_features)
    mean, log_var = head(x)

    assert mean.shape == (batch_size, out_features)
    assert log_var.shape == (batch_size, out_features)

    # Softplus link function test
    head_sp = NaturalHeteroscedasticHead(in_features, out_features, link_fn="softplus")
    mean_sp, log_var_sp = head_sp(x)
    assert mean_sp.shape == (batch_size, out_features)
    assert log_var_sp.shape == (batch_size, out_features)


def test_natural_reparam_head() -> None:
    reparam = NaturalReparamHead(link_fn="exp")
    f1 = torch.tensor([[1.0, 2.0]])
    f2 = torch.tensor([[0.0, 1.0]])
    mean, log_var = reparam(f1, f2)
    torch.testing.assert_close(mean, f1 * torch.exp(-f2))
    torch.testing.assert_close(log_var, -f2)


def test_heteroscedastic_laplace_regressor() -> None:
    in_features = 4
    hidden_dim = 6
    target_dim = 1
    batch_size = 5

    base_model = SimpleModel(in_features, hidden_dim)
    head = NaturalHeteroscedasticHead(hidden_dim, target_dim)
    reg = HeteroscedasticLaplaceRegressor(base_model, head, prior_precision=1.0)

    # Create dummy DataLoader
    x_train = torch.randn(20, in_features)
    y_train = torch.randn(20, target_dim)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size)

    # Fit model (with 2 epochs of training)
    reg.fit(loader, epochs=2)

    assert reg.is_fitted
    assert reg.post_var_weight is not None
    assert reg.post_var_bias is not None
    assert (reg.post_var_weight >= 0).all()
    assert (reg.post_var_bias >= 0).all()

    # Inference
    x_test = torch.randn(8, in_features)
    pred = reg.predict_distribution(x_test, n_samples=15)

    assert isinstance(pred, PredictiveBatch)
    assert pred.mean.shape == (8, target_dim)
    assert pred.std.shape == (8, target_dim)
    assert pred.samples.shape == (8, 15)  # [B, n_samples] for 1D targets
    assert "epistemic_variance" in pred.extra
    assert "aleatoric_variance" in pred.extra
    assert pred.extra["epistemic_variance"].shape == (8, target_dim)
    assert pred.extra["aleatoric_variance"].shape == (8, target_dim)
