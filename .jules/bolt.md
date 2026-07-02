## 2025-05-18 - Optimized E|X - X'| calculation in CRPS
**Learning:** Computing expected pairwise differences directly (e.g., `torch.abs(x.unsqueeze(0) - x.unsqueeze(1))`) creates an $O(N^2)$ memory and compute bottleneck, common in evaluating samples from predictive distributions.
**Action:** Replace explicit pairwise distance calculations for sample-based metrics with the $O(N \log N)$ L-moments/Gini Mean Difference approach by sorting the samples and applying linear weights (`(2j - n + 1)`). This avoids massive PyTorch memory allocations during metric evaluation.
