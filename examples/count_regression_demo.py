r"""
Count and zero-inflated regression: Poisson-family and Tweedie losses.

Families (``--family``):

``poisson``
    Count data from Lambert's zero-inflated Poisson (ZIP) model, fitted with
    1. ``PoissonDevianceLoss``: deviance based on the Poisson log-likelihood.
    2. ``PoissonLikelihoodRatioLoss``: likelihood-ratio test statistic as loss.
    3. ``ZeroInflatedPoissonNLLLoss`` (ZIP): handles excess zeros (Lambert, 1992).
    4. ``NegativeBinomialNLLLoss`` (NB): handles overdispersion (variance > mean).
    Reports MAE of the predicted rate against the true Poisson rate.

``tweedie``
    Zero-inflated continuous responses (rainfall, insurance claim sizes) from a
    compound Poisson-Gamma model, fitted with ``TweedieLoss`` and
    ``CompoundPoissonLoss``. The Tweedie distribution ($1 < p < 2$) is a compound
    Poisson-Gamma distribution:
    - number of events: $N \sim \text{Poisson}(\lambda)$
    - size of each event: $X_j \sim \text{Gamma}(k, \theta)$
    - total response: $Y = \sum_{j=1}^N X_j$, with $Y = 0$ when $N = 0$.
    Reports MAE of the predicted mean against the true mean.

``all`` (default)
    Runs ``poisson`` then ``tweedie``. Each family reseeds its data generator, so its
    output matches a standalone run.

Seminal papers: Lambert, D. (1992). Zero-Inflated Poisson Regression, With an
Application to Defects on Silicon Wafers. Technometrics. Jorgensen, B. (1987).
Exponential Dispersion Models. Journal of the Royal Statistical Society, Series B.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import (
    CompoundPoissonLoss,
    NegativeBinomialNLLLoss,
    PoissonDevianceLoss,
    PoissonLikelihoodRatioLoss,
    TweedieLoss,
    ZeroInflatedPoissonNLLLoss,
)

FAMILIES = ("poisson", "tweedie")


class CountModel(nn.Module):
    """MLP predicting a log-link mean (and optionally a second parameter head)."""

    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Poisson family: zero-inflated counts
# ---------------------------------------------------------------------------


def generate_zip_data(n_samples=1000, seed=42):
    """Generate count data following Lambert's ZIP model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = np.random.uniform(-2, 2, size=(n_samples, 1)).astype(np.float32)

    # True Poisson rate: log(lambda) = 0.5 * x + 1.2
    rate = np.exp(0.5 * x + 1.2).astype(np.float32)

    # True zero-inflation probability: logit(p) = -0.8 * x + 0.3
    logit_p = -0.8 * x + 0.3
    p = 1.0 / (1.0 + np.exp(-logit_p))

    # Sample counts
    poisson_counts = np.random.poisson(rate)
    zero_inflation = (np.random.rand(n_samples, 1) < p).astype(np.int32)
    y = np.where(zero_inflation, 0, poisson_counts).astype(np.float32)

    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(rate),
        torch.from_numpy(p),
    )


def train_and_eval(loss_name, loss_fn, x_train, y_train, x_test, y_test, epochs=150):
    # For ZIP, the model needs to output two parameters: [rate_logits, zero_inflation_logits]
    # For Negative Binomial, model output is: [rate_logits, overdispersion/alpha_logits]
    # Otherwise, it's just a single output (rate_logits)
    out_dim = 2 if loss_name in ("ZIP", "NegativeBinomial") else 1
    model = CountModel(output_dim=out_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)

            if loss_name == "ZIP":
                # ZeroInflatedPoissonNLLLoss expects rate_logits and pi_logits (zero-inflation logits)
                rate_logits = out[:, 0:1]
                pi_logits = out[:, 1:2]
                loss = loss_fn(rate_logits, by, pi_logits=pi_logits)
            elif loss_name == "NegativeBinomial":
                # NegativeBinomialNLLLoss expects predicted mean (mu) and dispersion (theta)
                mu = torch.exp(out[:, 0:1])
                theta = torch.exp(out[:, 1:2]).clamp(min=1e-5)
                loss = loss_fn(mu, by, theta=theta)
            else:
                # PoissonDeviance / LikelihoodRatio expects predicted mean rate (positive)
                rate = torch.exp(out[:, 0:1])
                loss = loss_fn(rate, by)

            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_out = model(x_test)
        # Every head puts the log-rate in column 0.
        rate_pred = torch.exp(test_out[:, 0:1])

        # Evaluate mean absolute error on the rate
        mae = (rate_pred - y_test).abs().mean().item()

    return rate_pred, mae


def run_poisson() -> dict[str, float]:
    print("=" * 60)
    print("Poisson and Zero-Inflated Count Regression Demo")
    print("=" * 60)

    x, y, true_rate, _ = generate_zip_data(n_samples=1200)

    # Train / Test split
    n_train = 800
    x_train, x_test = x[:n_train], x[n_train:]
    y_train = y[:n_train]
    true_rate_test = true_rate[n_train:]

    losses = {
        "PoissonDeviance": PoissonDevianceLoss(),
        "PoissonLikelihoodRatio": PoissonLikelihoodRatioLoss(),
        "ZIP": ZeroInflatedPoissonNLLLoss(eps=1e-6),
        "NegativeBinomial": NegativeBinomialNLLLoss(),
    }

    results: dict[str, float] = {}
    for name, loss_fn in losses.items():
        _, mae = train_and_eval(name, loss_fn, x_train, y_train, x_test, true_rate_test)
        results[name] = mae
        print(f"Loss: {name:<25} MAE on true rate: {mae:.4f}")

    print("\nObservation:")
    print(
        "1. ZeroInflatedPoisson (ZIP) handles excess zeros by predicting zero-inflation probability."
    )
    print("2. NegativeBinomial handles overdispersion by parameterizing count variance separately.")
    return results


# ---------------------------------------------------------------------------
# Tweedie family: zero-inflated continuous responses
# ---------------------------------------------------------------------------


def generate_tweedie_data(n_samples=1000, p_power=1.5, phi=1.0, seed=42):
    """Generate Compound Poisson-Gamma data using Tweedie parametrization."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = np.random.uniform(-1.5, 1.5, size=(n_samples, 1)).astype(np.float32)

    # Predicted mean: log(mu) = 0.8 * x + 0.5
    mu = np.exp(0.8 * x + 0.5).astype(np.float32)

    # Translate to Poisson-Gamma parameters
    # Poisson lambda
    lambda_p = (mu ** (2.0 - p_power)) / (phi * (2.0 - p_power))
    # Gamma shape k
    shape_k = (2.0 - p_power) / (p_power - 1.0)
    # Gamma scale theta
    scale_theta = phi * (p_power - 1.0) * (mu ** (p_power - 1.0))

    y = np.zeros((n_samples, 1), dtype=np.float32)
    for i in range(n_samples):
        # Sample Poisson count of events
        n_events = np.random.poisson(lambda_p[i])
        if n_events > 0:
            # Sample event sizes from Gamma
            # Note: numpy shape parameter is shape_k, scale parameter is scale_theta
            y[i, 0] = np.random.gamma(shape_k, scale_theta[i], size=n_events).sum()

    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mu)


def train_tweedie(loss_name, loss_fn, x_train, y_train, x_test, true_mu_test, epochs=150):
    model = CountModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            # Both losses use a log link: they take log(mu) and apply exp themselves.
            log_mu_pred = model(bx)
            loss = loss_fn(log_mu_pred, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        mu_pred = torch.exp(model(x_test))
        mae = (mu_pred - true_mu_test).abs().mean().item()

    return mu_pred, mae


def run_tweedie() -> dict[str, float]:
    print("=" * 60)
    print("Tweedie and Compound Poisson-Gamma Regression Demo")
    print("=" * 60)

    p_power = 1.6
    x, y, true_mu = generate_tweedie_data(n_samples=1000, p_power=p_power, phi=0.8)

    print(f"Zero-fraction: {(y == 0).float().mean().item() * 100:.1f}%")

    n_train = 700
    x_train, x_test = x[:n_train], x[n_train:]
    y_train = y[:n_train]
    true_mu_test = true_mu[n_train:]

    losses = {
        "Tweedie (p=1.6)": TweedieLoss(p=p_power),
        "CompoundPoisson": CompoundPoissonLoss(p=p_power),
    }

    results: dict[str, float] = {}
    for name, loss_fn in losses.items():
        _, mae = train_tweedie(name, loss_fn, x_train, y_train, x_test, true_mu_test)
        results[name] = mae
        print(f"Loss: {name:<20} MAE on true mean: {mae:.4f}")
    return results


def main(argv: list[str] | None = None) -> dict[str, dict[str, float]]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    args = parser.parse_args(argv)

    families = FAMILIES if args.family == "all" else (args.family,)
    results: dict[str, dict[str, float]] = {}
    for family in families:
        results[family] = run_poisson() if family == "poisson" else run_tweedie()
    return results


if __name__ == "__main__":
    main()
