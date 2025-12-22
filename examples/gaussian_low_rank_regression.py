"""
Low-Rank Gaussian Regression Example

This example shows how to train a model that predicts a low-rank Gaussian
distribution for multivariate targets:

    cov = cov_factor @ cov_factor.T + diag(cov_diag)

This is efficient for moderate output sizes and captures correlations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import LowRankGaussianLoss, low_rank_output_dim, split_low_rank_gaussian_output


def make_low_rank_data(n_samples=1024, input_dim=4, n_features=3, rank=2):
    torch.manual_seed(42)
    x = torch.randn(n_samples, input_dim)
    weight = torch.randn(input_dim, n_features)
    bias = torch.randn(n_features)

    mean = x @ weight + bias

    cov_factor = torch.randn(n_features, rank) * 0.4
    cov_diag = torch.ones(n_features) * 0.2
    noise_dist = torch.distributions.LowRankMultivariateNormal(
        loc=torch.zeros(n_features), cov_factor=cov_factor, cov_diag=cov_diag
    )
    noise = noise_dist.sample((n_samples,))
    y = mean + noise
    return x, y


class LowRankGaussianModel(nn.Module):
    def __init__(self, input_dim, n_features, rank):
        super().__init__()
        self.n_features = n_features
        self.rank = rank
        hidden = 32
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        out_dim = low_rank_output_dim(n_features, rank)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.head(self.net(x))


def train():
    n_features = 3
    rank = 2
    x, y = make_low_rank_data(n_samples=1500, input_dim=4, n_features=n_features, rank=rank)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LowRankGaussianModel(input_dim=4, n_features=n_features, rank=rank)
    loss_fn = LowRankGaussianLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        running = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            y_pred = model(batch_x)
            mean, cov_factor, cov_diag = split_low_rank_gaussian_output(
                y_pred, n_features, rank
            )
            loss = loss_fn(mean, batch_y, cov_factor, cov_diag)
            loss.backward()
            optimizer.step()
            running += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1:02d} loss {running / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(x[:5])
        mean, cov_factor, cov_diag = split_low_rank_gaussian_output(y_pred, n_features, rank)
        cov = cov_factor @ cov_factor.transpose(-1, -2) + torch.diag_embed(
            cov_diag.clamp(min=1e-6)
        )
        print("mean shape:", mean.shape)
        print("cov shape:", cov.shape)


if __name__ == "__main__":
    train()
