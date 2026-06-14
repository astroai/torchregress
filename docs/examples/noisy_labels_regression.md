# Noisy Labels Regression Example

This example uses implemented robust losses for regression with corrupted targets. For future noisy-label methods that are not currently implemented as public APIs, see the research notes under [Reports](../reports/index.md).

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import CauchyLoss, TukeyBiweightLoss, WeightedHuberLoss, WeightedMSELoss
from torchregress.metrics import mse


def make_noisy_regression(n_samples=500, noise_ratio=0.2, noise_scale=3.0, seed=42):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    x = np.linspace(-3, 3, n_samples, dtype=np.float32)
    y_clean = np.sin(x) + 0.5 * x
    y_noisy = y_clean.copy()

    noisy_idx = rng.choice(n_samples, int(n_samples * noise_ratio), replace=False)
    y_noisy[noisy_idx] += rng.normal(0.0, noise_scale, size=noisy_idx.shape[0])

    return (
        torch.tensor(x[:, None]),
        torch.tensor(y_noisy[:, None]),
        torch.tensor(y_clean[:, None]),
        set(int(i) for i in noisy_idx),
    )


class SimpleRegressor(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def train(loss_fn, x, y_noisy, n_epochs=80):
    model = SimpleRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = DataLoader(TensorDataset(x, y_noisy), batch_size=64, shuffle=True)

    for _ in range(n_epochs):
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


x, y_noisy, y_clean, noisy_indices = make_noisy_regression()
losses = {
    "MSE": WeightedMSELoss(),
    "Huber": WeightedHuberLoss(delta=1.0),
    "Cauchy": CauchyLoss(scale=1.0),
    "Tukey": TukeyBiweightLoss(c=4.685),
}

results = {}
for name, loss_fn in losses.items():
    model = train(loss_fn, x, y_noisy)
    with torch.no_grad():
        pred = model(x)
    results[name] = {"model": model, "clean_mse": mse(pred, y_clean).item()}
    print(f"{name}: clean-label MSE = {results[name]['clean_mse']:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(x.numpy(), y_noisy.numpy(), alpha=0.25, s=15, label="Noisy labels")
plt.plot(x.numpy(), y_clean.numpy(), "k-", linewidth=2, label="Clean target")

grid = torch.linspace(-3, 3, 200).view(-1, 1)
for name, result in results.items():
    with torch.no_grad():
        pred = result["model"](grid)
    plt.plot(grid.numpy(), pred.numpy(), linewidth=1.5, label=name)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Implemented robust losses under target corruption")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

Robust losses do not identify corrupted labels explicitly. They reduce the influence of large residuals, so compare them against a clean validation set or a trusted audit subset when possible.
