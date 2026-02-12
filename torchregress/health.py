import sys
import traceback

import torch

import torchregress
from torchregress.losses import WeightedLossWrapper


def check_health():
    """Run a system health check for torchregress."""
    print(f"torchregress version: {torchregress.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version.split()[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n--- Import Check ---")
    try:
        from torchregress import (
            algorithms,  # noqa: F401
            losses,  # noqa: F401
            metrics,  # noqa: F401
        )

        print("Imports: OK")
    except ImportError as e:
        print(f"Imports: FAILED ({e})")
        sys.exit(1)

    print("\n--- Basic Compute Check ---")
    try:
        x = torch.randn(10, 5).to(device)
        w = torch.randn(5, 1).to(device)
        y = x @ w
        print("Tensor ops: OK")
    except Exception as e:
        print(f"Tensor ops: FAILED ({e})")
        sys.exit(1)

    print("\n--- Minimal Training Loop Smoke Test ---")
    try:
        model = torch.nn.Linear(5, 1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # Use a wrapper to test integration
        criterion = WeightedLossWrapper(torch.nn.MSELoss())

        # Fake data
        x = torch.randn(32, 5).to(device)
        y = torch.randn(32, 1).to(device)
        w_sample = torch.ones(32, 1).to(device)

        # Forward
        pred = model(x)
        loss = criterion(pred, y, weights=w_sample)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Training step: OK")
    except Exception as e:
        print(f"Training step: FAILED ({e})")
        traceback.print_exc()
        sys.exit(1)

    print("\n--- Metrics Smoke Test ---")
    try:
        from torchregress.metrics import MedianAbsoluteError

        metric = MedianAbsoluteError().to(device)
        metric.update(pred, y)
        res = metric.compute()
        print(f"Metric compute: OK (MAE={res.item():.4f})")
    except Exception as e:
        print(f"Metric compute: FAILED ({e})")
        sys.exit(1)

    print("\n[SUCCESS] System appears healthy.")


if __name__ == "__main__":
    check_health()
