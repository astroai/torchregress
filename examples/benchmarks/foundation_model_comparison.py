"""
Benchmark skeleton for comparing Tabular Foundation Models (TabPFN, TabICL)
against torchregress probabilistic models.

This script evaluates both point accuracy (MSE/MAE) and distributional
utility (CRPS, Log-Likelihood, PIT uniformity) using the
universal distribution_metrics_report helper.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torchregress.losses import GaussianNLLLoss, MixtureDensityLoss
from torchregress.metrics import distribution_metrics_report

try:
    from tabpfn import TabPFNRegressor

    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False


def get_data():
    """Load and split California housing dataset."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    return X_train, X_test, y_train, y_test


def get_device():
    """Detect available hardware acceleration."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_tabpfn(X_train, y_train, X_test, y_test):
    """Run TabPFN if available."""
    if not TABPFN_AVAILABLE:
        return {"status": "TabPFN not installed"}

    # TabPFN has some constraints on N_train and N_features
    # We sample if needed
    if len(X_train) > 1000:
        idx = np.random.choice(len(X_train), 1000, replace=False)
        X_train_sub, y_train_sub = X_train[idx], y_train[idx]
    else:
        X_train_sub, y_train_sub = X_train, y_train

    print("Running TabPFN...")
    device = get_device()
    # TabPFN expects 'cuda' or 'cpu', not always 'mps'
    tabpfn_device = "cuda" if device == "cuda" else "cpu"
    model = TabPFNRegressor(device=tabpfn_device)
    start = time.time()
    model.fit(X_train_sub, y_train_sub)

    # TabPFN provides a get_predictive_distribution or similar in recent versions
    # For now, we'll assume it returns mean/std or samples
    y_pred, y_std = model.predict(X_test, return_std=True)
    duration = time.time() - start

    # Construct a distribution for the report
    dist = torch.distributions.Normal(
        torch.from_numpy(y_pred).float(), torch.from_numpy(y_std).float()
    )

    report = distribution_metrics_report(dist=dist, y_true=torch.from_numpy(y_test).float())
    report["duration"] = duration
    return report


def run_torchregress(X_train, y_train, X_test, y_test, model_type="gaussian"):
    """Run a basic torchregress model."""
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float().unsqueeze(1)

    in_dim = X_train.shape[1]

    if model_type == "gaussian":
        model = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 2))  # mean, log_var
        loss_fn = GaussianNLLLoss()
    elif model_type == "mdn":
        n_components = 3
        model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_components * 3),  # weights, means, log_stds
        )
        loss_fn = MixtureDensityLoss(n_components=n_components, n_features=1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    device = get_device()
    model = model.to(device)
    X_train_t = X_train_t.to(device)
    y_train_t = y_train_t.to(device)
    X_test_t = X_test_t.to(device)
    y_test_t = y_test_t.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"Training torchregress {model_type}...")
    start = time.time()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = loss_fn(out, y_train_t)
        loss.backward()
        optimizer.step()
    duration = time.time() - start

    # Evaluate
    model.eval()
    with torch.no_grad():
        out_test = model(X_test_t)
        if model_type == "gaussian":
            mean, log_var = out_test[:, :1], out_test[:, 1:]
            dist = torch.distributions.Normal(mean.squeeze(), torch.exp(0.5 * log_var).squeeze())
            report = distribution_metrics_report(dist=dist, y_true=y_test_t.squeeze())
        else:
            # For MDN, we use samples or compute specifically
            samples = loss_fn.sample(out_test, n_samples=100)
            report = distribution_metrics_report(
                samples=samples.squeeze(-1), y_true=y_test_t.squeeze()
            )

    report["duration"] = duration
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=2000)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = get_data()

    # Limit data for quick benchmark
    X_train, y_train = X_train[: args.n_samples], y_train[: args.n_samples]
    X_test, y_test = X_test[:500], y_test[:500]

    results = {}
    results["torchregress_gaussian"] = run_torchregress(
        X_train, y_train, X_test, y_test, "gaussian"
    )
    results["torchregress_mdn"] = run_torchregress(X_train, y_train, X_test, y_test, "mdn")

    if TABPFN_AVAILABLE:
        results["tabpfn"] = run_tabpfn(X_train, y_train, X_test, y_test)
    else:
        print("TabPFN not available, skipping.")

    # Print results summary
    df = pd.DataFrame(results).T
    print("\nBenchmark Results:")
    print(df[["crps", "log_prob", "coverage_90", "duration"]])


if __name__ == "__main__":
    main()
