
import time

import torch


def benchmark_randperm():
    n_samples = 1000000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Benchmarking on {device}")

    # Simulate y_samples on device
    y_samples = torch.randn(n_samples, 10, device=device)

    # Case 1: Without device specified (creates on CPU, then moves to GPU if needed)
    start_time = time.time()
    for _ in range(100):
        indices = torch.randperm(n_samples)
        # The indexing operation will trigger a transfer if y_samples is on GPU
        _ = y_samples[indices[:1000]]
    end_time = time.time()
    print(f"Without device specified: {end_time - start_time:.4f}s")

    # Case 2: With device specified (creates directly on device)
    start_time = time.time()
    for _ in range(100):
        indices = torch.randperm(n_samples, device=device)
        _ = y_samples[indices[:1000]]
    end_time = time.time()
    print(f"With device specified: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    try:
        benchmark_randperm()
    except Exception as e:
        print(f"Could not run benchmark: {e}")
        print("Rationale: Specifying device in torch.randperm avoids host-to-device transfers.")
