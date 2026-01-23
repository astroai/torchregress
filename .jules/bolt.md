## 2024-05-22 - Vectorizing Distribution Metrics
**Learning:** Distribution metrics like Energy Score often iterate over samples (e.g., 100-1000) for pairwise calculations. In PyTorch, replacing these Python loops with broadcasted tensor operations yields massive speedups (e.g., ~4.5x for Energy Score) with minimal memory overhead relative to the input size.
**Action:** Always inspect `torchregress/metrics/distribution.py` and similar modules for explicit loops over `n_samples` and vectorize them using broadcasting (e.g., `unsqueeze(0)` on targets) and `cdist`.
