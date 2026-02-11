## 2024-05-22 - Vectorizing Distribution Metrics
**Learning:** Distribution metrics like Energy Score often iterate over samples (e.g., 100-1000) for pairwise calculations. In PyTorch, replacing these Python loops with broadcasted tensor operations yields massive speedups (e.g., ~4.5x for Energy Score) with minimal memory overhead relative to the input size.
**Action:** Always inspect `torchregress/metrics/distribution.py` and similar modules for explicit loops over `n_samples` and vectorize them using broadcasting (e.g., `unsqueeze(0)` on targets) and `cdist`.

## 2024-05-23 - Vectorizing Entropy Calculation
**Learning:** `torch.histogram` does not support batched inputs with dynamic per-sample ranges, which often leads to slow nested loops (O(Batch * Dim)). Manually implementing binning using normalization + `floor` + `one_hot` + `sum` is over 60x faster for typical batch sizes.
**Action:** Replace looped `torch.histogram` calls with a vectorized `_batched_entropy` helper using `one_hot` encoding for counting.

## 2024-10-24 - Vectorizing Expectile Losses
**Learning:** Losses iterating over multiple quantiles/expectiles (like `ExpectileCrossoverLoss`) are prime candidates for vectorization. By stacking predictions and using broadcasting (e.g. `[B, 1, F] - [B, N, F]`), we can eliminate Python loops, achieving ~2.4x speedup.
**Action:** Extract shared elementwise loss logic into vectorized helper functions (like `multi_expectile_loss`) to support both multi-output losses and penalty calculations efficiently.

## 2025-02-18 - Optimizing Batched Entropy with Bincount
**Learning:** While `one_hot` encoding is faster than loops, it consumes significant memory (O(N*M*K)) which can be prohibitive for large batches or bin counts. Using `torch.bincount` on flattened indices with offsets allows for vectorized histogram computation with much lower memory footprint (O(N*M)) and improved speed (~16x speedup over `one_hot` for large inputs).
**Action:** Use `torch.bincount` with offset indices instead of `one_hot` when computing histograms or counts across multiple distributions/batches.
