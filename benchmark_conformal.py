import math
import time

import torch


class MockPredictor:
    def calculate_score(self, y_pred, target):
        return torch.abs(y_pred - target)

def calibrate_original(y_pred, target, alpha, predictor):
    n_features = target.shape[-1]
    q_hats = []
    for i in range(n_features):
        y_pred_i = y_pred[..., i].unsqueeze(-1)
        target_i = target[..., i].unsqueeze(-1)
        scores = predictor.calculate_score(y_pred_i, target_i)
        n = scores.numel()
        q = math.ceil((n + 1) * (1 - alpha)) / n
        q_hat_i = torch.quantile(scores, q, interpolation="higher")
        q_hats.append(q_hat_i)
    return torch.stack(q_hats)

def calibrate_vectorized(y_pred, target, alpha, predictor):
    scores = predictor.calculate_score(y_pred, target)
    n = scores.shape[0]
    q = math.ceil((n + 1) * (1 - alpha)) / n
    # For safety with torch.quantile, though not in original
    q = min(max(q, 0.0), 1.0)
    return torch.quantile(scores, q, dim=0, interpolation="higher")

def run_benchmark():
    N = 10000
    D = 100
    alpha = 0.1
    y_pred = torch.randn(N, D)
    target = torch.randn(N, D)
    predictor = MockPredictor()

    print(f"Benchmarking with N={N}, D={D} features...")

    # Warmup
    _ = calibrate_original(y_pred[:100], target[:100], alpha, predictor)
    _ = calibrate_vectorized(y_pred[:100], target[:100], alpha, predictor)

    start = time.time()
    for _ in range(10):
        q_hat_orig = calibrate_original(y_pred, target, alpha, predictor)
    orig_time = (time.time() - start) / 10
    print(f"Original time (avg of 10): {orig_time:.4f}s")

    start = time.time()
    for _ in range(10):
        q_hat_vec = calibrate_vectorized(y_pred, target, alpha, predictor)
    vec_time = (time.time() - start) / 10
    print(f"Vectorized time (avg of 10): {vec_time:.4f}s")

    speedup = orig_time / vec_time if vec_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")

    # Correctness check
    # Recalculate without q clamping for exact comparison if needed,
    # but with N=10000, q will be in range.
    q_hat_orig = calibrate_original(y_pred, target, alpha, predictor)
    q_hat_vec = calibrate_vectorized(y_pred, target, alpha, predictor)
    correct = torch.allclose(q_hat_orig, q_hat_vec)
    print(f"Correctness: {correct}")

if __name__ == "__main__":
    try:
        run_benchmark()
    except ImportError:
        print("Torch not installed, skipping benchmark execution.")
    except Exception as e:
        print(f"Benchmark failed: {e}")
