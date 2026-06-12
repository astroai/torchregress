"""
Poisson and Zero-Inflated Count Data Regression Demo.

This example demonstrates how to train models for count data regression using Poisson loss families:
1. PoissonDevianceLoss: Deviance loss based on Poisson distribution log-likelihood.
2. PoissonLikelihoodRatioLoss: Likelihood ratio test statistic as loss.
3. ZeroInflatedPoissonNLLLoss (ZIP): Handles excess zeros (Lambert, 1992).
4. NegativeBinomialNLLLoss (NB): Handles overdispersion (variance > mean).

Seminal paper: Lambert, D. (1992). Zero-Inflated Poisson Regression, With an Application to
Defects on Silicon Wafers. Technometrics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import (
    NegativeBinomialNLLLoss,
    PoissonDevianceLoss,
    PoissonLikelihoodRatioLoss,
    ZeroInflatedPoissonNLLLoss,
)


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


class CountModel(nn.Module):
    """Network predicting rate parameter (and optionally zero-inflation probability)."""

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
        if loss_name in ("ZIP", "NegativeBinomial"):
            rate_pred = torch.exp(test_out[:, 0:1])
        else:
            rate_pred = torch.exp(test_out[:, 0:1])

        # Evaluate mean absolute error on the rate
        mae = (rate_pred - y_test).abs().mean().item()

    return rate_pred, mae


def main():
    print("=" * 60)
    print("Poisson and Zero-Inflated Count Regression Demo")
    print("=" * 60)

    x, y, true_rate, _ = generate_zip_data(n_samples=1200)

    # Train / Test split
    n_train = 800
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, _y_test = y[:n_train], y[n_train:]
    true_rate_test = true_rate[n_train:]

    losses = {
        "PoissonDeviance": PoissonDevianceLoss(),
        "PoissonLikelihoodRatio": PoissonLikelihoodRatioLoss(),
        "ZIP": ZeroInflatedPoissonNLLLoss(eps=1e-6),
        "NegativeBinomial": NegativeBinomialNLLLoss(),
    }

    results = {}
    for name, loss_fn in losses.items():
        rate_pred, mae = train_and_eval(name, loss_fn, x_train, y_train, x_test, true_rate_test)
        results[name] = {"pred": rate_pred, "mae": mae}
        print(f"Loss: {name:<25} MAE on true rate: {mae:.4f}")

    print("\nObservation:")
    print(
        "1. ZeroInflatedPoisson (ZIP) handles excess zeros by predicting zero-inflation probability."
    )
    print("2. NegativeBinomial handles overdispersion by parameterizing count variance separately.")


if __name__ == "__main__":
    main()
