"""
Full-Covariance Gaussian Regression Example

This example shows how to train a model that predicts a full covariance
matrix per sample using a Cholesky parameterization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import MultivariateGaussianLoss


def make_full_cov_data(n_samples=1024, input_dim=3, n_features=2):
    torch.manual_seed(7)
    x = torch.randn(n_samples, input_dim)
    weight = torch.randn(input_dim, n_features)
    bias = torch.randn(n_features)

    mean = x @ weight + bias

    cov = torch.tensor([[0.4, 0.2], [0.2, 0.6]])
    noise_dist = torch.distributions.MultivariateNormal(
        loc=torch.zeros(n_features), covariance_matrix=cov
    )
    noise = noise_dist.sample((n_samples,))
    y = mean + noise
    return x, y


def build_covariance_from_tril(tril_params, n_features, min_diag=1e-4):
    batch_size = tril_params.shape[0]
    tril_indices = torch.tril_indices(n_features, n_features, device=tril_params.device)
    L = torch.zeros(batch_size, n_features, n_features, device=tril_params.device)
    L[:, tril_indices[0], tril_indices[1]] = tril_params
    diag_idx = torch.arange(n_features, device=tril_params.device)
    L[:, diag_idx, diag_idx] = torch.nn.functional.softplus(L[:, diag_idx, diag_idx]) + min_diag
    return L @ L.transpose(-1, -2)


class FullCovGaussianModel(nn.Module):
    def __init__(self, input_dim, n_features):
        super().__init__()
        self.n_features = n_features
        hidden = 32
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, n_features)
        tril_size = n_features * (n_features + 1) // 2
        self.tril_head = nn.Linear(hidden, tril_size)

    def forward(self, x):
        features = self.net(x)
        mean = self.mean_head(features)
        tril_params = self.tril_head(features)
        return mean, tril_params


def train():
    n_features = 2
    x, y = make_full_cov_data(n_samples=1200, input_dim=3, n_features=n_features)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = FullCovGaussianModel(input_dim=3, n_features=n_features)
    loss_fn = MultivariateGaussianLoss(jitter=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(25):
        model.train()
        running = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            mean, tril_params = model(batch_x)
            cov = build_covariance_from_tril(tril_params, n_features)
            loss = loss_fn(mean, batch_y, cov)
            loss.backward()
            optimizer.step()
            running += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1:02d} loss {running / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        mean, tril_params = model(x[:3])
        cov = build_covariance_from_tril(tril_params, n_features)
        print("mean shape:", mean.shape)
        print("cov shape:", cov.shape)


if __name__ == "__main__":
    train()
