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
    # 300 epochs to ensure convergence
    for _ in range(300):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
    return model


def test_simex_initialization():
    simex = SIMEX(model_factory=SimpleLinear, train_func=train_linear, sigma_u=0.1)
    assert simex.sigma_u_input == 0.1
    assert len(simex.lambdas) == 4


def test_simex_correction():
    torch.manual_seed(42)
    n_samples = 2000
    noise_std = 0.5

    # True latent X ~ N(0, 1)
    X_true = torch.randn(n_samples, 1)

    # Target Y = 3*X + epsilon
    Y = 3 * X_true + torch.randn(n_samples, 1) * 0.1

    # Observed W = X + error
    W_obs = X_true + torch.randn(n_samples, 1) * noise_std

    # Naive model
    naive_model = SimpleLinear()
    train_linear(naive_model, W_obs, Y)
    naive_slope = naive_model.linear.weight.item()

    # SIMEX
    # Note: SIMEX adds MORE noise.
    # With W_obs (var_noise = 0.25), adding lambda=1.0 adds another 0.25 variance.
    # Total noise = 0.5. At lambda=0, noise=0.25.
    # Extrapolation to lambda=-1 implies noise=0.

    simex = SIMEX(
        model_factory=SimpleLinear,
        train_func=train_linear,
        sigma_u=noise_std,
        lambdas=[0.5, 1.0, 1.5, 2.0],
        extrapolation_order=2,
    )

    simex.fit(W_obs, Y)

    # Check predictions
    # We can check the effective slope of the predictions
    X_test = torch.linspace(-2, 2, 100).reshape(-1, 1)
    preds = simex.predict(X_test)

    # Estimate slope from predictions
    # pred = slope * x + intercept
    # slope = cov(x, pred) / var(x)

    x_centered = X_test - X_test.mean()
    p_centered = preds - preds.mean()
    simex_slope = (x_centered * p_centered).sum() / (x_centered**2).sum()

    print(f"Naive Slope: {naive_slope:.4f}")
    print(f"SIMEX Slope: {simex_slope.item():.4f}")

    assert naive_slope < 2.6
    # SIMEX is an approximation, might not be perfect 3.0, but should be better than naive
    # Usually it corrects significantly
    assert simex_slope.item() > 2.6
    assert simex_slope.item() < 3.4


def test_simex_multivariate():
    torch.manual_seed(42)
    n_samples = 500
    n_features = 2

    X_true = torch.randn(n_samples, n_features)
    Y = (X_true.sum(dim=1, keepdim=True) * 2) + torch.randn(n_samples, 1) * 0.1

    sigma_u = torch.tensor([[0.1, 0.0], [0.0, 0.1]])  # Diagonal noise
    noise = torch.randn(n_samples, n_features) * torch.sqrt(torch.tensor(0.1))
    W_obs = X_true + noise

    class MultiLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    simex = SIMEX(
        model_factory=MultiLinear, train_func=train_linear, sigma_u=sigma_u, lambdas=[0.5, 1.0]
    )

    simex.fit(W_obs, Y)
    preds = simex.predict(W_obs)
    assert preds.shape == (n_samples, 1)


def test_simex_supports_multiple_simulations_per_lambda():
    torch.manual_seed(0)
    X = torch.randn(128, 1)
    y = 2.0 * X + 0.1 * torch.randn(128, 1)
    simex = SIMEX(
        model_factory=SimpleLinear,
        train_func=train_linear,
        sigma_u=0.1,
        lambdas=[0.5, 1.0],
        n_simulations=3,
    )
    simex.fit(X, y)
    assert len(simex.models_by_lambda) == 3
    assert all(len(models) == 3 for models in simex.models_by_lambda)
    preds = simex.predict(X[:10])
    assert preds.shape == (10, 1)
    assert torch.isfinite(preds).all()
