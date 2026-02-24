
import torch
import time
import numpy as np
from torchregress.losses.imbalanced import LDSLoss

def benchmark_lds_loss(device="cpu", n_samples=10000, n_iterations=100):
    # Setup
    loss_fn = LDSLoss()
    train_targets = torch.randn(10000)
    loss_fn.fit(train_targets)

    target = torch.randn(n_samples).to(device)
    y_pred = torch.randn(n_samples).to(device)

    # Warmup
    for _ in range(10):
        _ = loss_fn(y_pred, target)

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(n_iterations):
        _ = loss_fn(y_pred, target)

    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / n_iterations
    print(f"Device: {device}, Samples: {n_samples}, Avg time: {avg_time:.6f}s")
    return avg_time

if __name__ == "__main__":
    print("Benchmarking on CPU...")
    benchmark_lds_loss("cpu")

    if torch.cuda.is_available():
        print("\nBenchmarking on CUDA...")
        benchmark_lds_loss("cuda")
    else:
        print("\nCUDA not available.")
