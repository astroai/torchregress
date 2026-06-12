"""
Mondrian (Group-Conditional) Conformal Prediction Demo.

This example demonstrates Mondrian Conformal Prediction (MCP), which guarantees group-conditional
coverage. If different groups in the data have significantly different noise levels, standard
conformal prediction guarantees only global coverage and may severely under-cover or over-cover
individual groups. Mondrian CP resolves this by calibrating separate thresholds per group.

Seminal references:
1. Vovk, V. (2002). Conditional Conformal Prediction. International Conference on Algorithmic
   Learning Theory.
2. Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses.conformal import SplitConformal


def generate_grouped_data(n_samples=2000, seed=42):
    """Generate data from two groups with distinct noise characteristics.

    Group 0 (low noise): std = 0.2
    Group 1 (high noise): std = 1.2
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = np.random.uniform(-1.5, 1.5, size=(n_samples, 1)).astype(np.float32)

    # Half the samples are in group 0, half in group 1
    groups = np.random.choice([0, 1], size=(n_samples, 1)).astype(np.int64)

    # True regression function
    y_mean = 0.8 * x

    # Group-dependent noise standard deviation
    noise_std = np.where(groups == 0, 0.2, 1.2).astype(np.float32)
    noise = np.random.normal(0, noise_std).astype(np.float32)

    y = y_mean + noise

    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(groups).squeeze(1),
    )


class MeanRegressionModel(nn.Module):
    """Simple neural network predicting conditional mean."""

    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("=" * 60)
    print("Mondrian Conformal Prediction Demo")
    print("=" * 60)

    # Generate samples
    x, y, groups = generate_grouped_data(n_samples=2400)

    # Train / Cal / Test split
    n_train = 800
    n_cal = 800

    x_train, y_train = x[:n_train], y[:n_train]
    x_cal, y_cal, groups_cal = (
        x[n_train : n_train + n_cal],
        y[n_train : n_train + n_cal],
        groups[n_train : n_train + n_cal],
    )
    x_test, y_test, groups_test = (
        x[n_train + n_cal :],
        y[n_train + n_cal :],
        groups[n_train + n_cal :],
    )

    # 1. Train base model on training set
    model = MeanRegressionModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(100):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        cal_preds = model(x_cal)
        test_preds = model(x_test)

    alpha = 0.1  # Target coverage: 90%

    # 2. Calibrate Standard Conformal Predictor (Global Calibration)
    standard_cp = SplitConformal(alpha=alpha)
    standard_cp.calibrate(cal_preds, y_cal)

    # 3. Calibrate Mondrian Conformal Predictor (Group-Conditional Calibration)
    mondrian_cp = SplitConformal(alpha=alpha)
    mondrian_cp.calibrate(cal_preds, y_cal, groups=groups_cal)

    # 4. Predict intervals and evaluate
    with torch.no_grad():
        # Standard intervals
        std_lower, std_upper = standard_cp.predict_interval(test_preds)
        std_cov = ((y_test >= std_lower) & (y_test <= std_upper)).float()

        # Mondrian intervals
        mondrian_lower, mondrian_upper = mondrian_cp.predict_interval(
            test_preds, groups=groups_test
        )
        mon_cov = ((y_test >= mondrian_lower) & (y_test <= mondrian_upper)).float()

    print(f"\n--- Coverage Results (Alpha = {alpha}, Target = {1 - alpha:.2%}) ---")
    print(f"{'Group':<10} | {'Standard CP Coverage':<22} | {'Mondrian CP Coverage':<22}")
    print("-" * 65)

    for g in [0, 1]:
        g_mask = groups_test == g
        std_g_cov = std_cov[g_mask].mean().item()
        mon_g_cov = mon_cov[g_mask].mean().item()
        print(f"Group {g:<4} | {std_g_cov:<22.2%} | {mon_g_cov:<22.2%}")

    print("-" * 65)
    print(f"Overall    | {std_cov.mean().item():<22.2%} | {mon_cov.mean().item():<22.2%}")

    print("\nObservation:")
    print("1. Standard CP achieves overall coverage but under-covers group 1 (high noise)")
    print("   and over-covers group 0 (low noise).")
    print("2. Mondrian CP guarantees exactly the target coverage for each group individually.")


if __name__ == "__main__":
    main()
