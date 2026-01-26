## 2024-05-22 - Vectorizing Distribution Metrics
**Learning:** Distribution metrics like Energy Score often iterate over samples (e.g., 100-1000) for pairwise calculations. In PyTorch, replacing these Python loops with broadcasted tensor operations yields massive speedups (e.g., ~4.5x for Energy Score) with minimal memory overhead relative to the input size.
**Action:** Always inspect `torchregress/metrics/distribution.py` and similar modules for explicit loops over `n_samples` and vectorize them using broadcasting (e.g., `unsqueeze(0)` on targets) and `cdist`.

## 2024-05-23 - Vectorizing Entropy Calculation
**Learning:** `torch.histogram` does not support batched inputs with dynamic per-sample ranges, which often leads to slow nested loops (O(Batch * Dim)). Manually implementing binning using normalization + `floor` + `one_hot` + `sum` is over 60x faster for typical batch sizes.
**Action:** Replace looped `torch.histogram` calls with a vectorized `_batched_entropy` helper using `one_hot` encoding for counting.
