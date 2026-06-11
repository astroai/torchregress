# OT-inspired shift reweighting for conformal (v1)

This page documents **experimental** test-time tools for **classification-style**
nonconformity scores: learning nonnegative weights on a calibration set by matching
a weighted empirical CDF to unlabeled target scores, then running **weighted split
conformal** with the same finite-sample quantile adjustment used elsewhere in
torchregress.

!!! warning "Scope and guarantees"
    - v1 is **not** a full optimal-transport solver and **does not** reproduce every
      construction from the non-exchangeable OT conformal literature.
    - The method targets **coverage under covariate-like score shift** when extra
      unlabeled target scores are available; it is **not** density estimation and
      **does not** guarantee marginal coverage without assumptions on the weight
      estimator.
    - Use [`SplitConformal`](../conformal/predictors.md) or related predictors when
      exchangeability is plausible and shift tools are unnecessary.

---

## API overview

| Symbol | Role |
|:-------|:-----|
| `OptimalTransportCoverageGap` | Scalar diagnostics (`l2_cdf_gap`, `ks_max_abs`) between calibration and target score ECDFs |
| `OTShiftReweighter` | Fit simplex weights ``weights_`` by CDF matching + entropy regularisation |
| `WeightedSplitConformalAdapter` | Calibrate a scalar threshold with nonnegative weights; build label sets from per-class scores |
| `WeightedSplitConformalAdapter.coverage_diagnostics` | Weighted empirical calibration coverage vs ``1 - alpha``, ESS proxy |
| `weighted_split_classification_predictive_batch` | Wrap scores + adapter into ``PredictiveBatch`` with ``extra`` (mask, threshold, optional gap / ESS) |

Imports:

```python
import torchregress as tr

gap = tr.test_time.OptimalTransportCoverageGap().estimate(
    calibration_scores=cal_scores,
    target_score_summary=tgt_scores,
)
rw = tr.test_time.OTShiftReweighter(entropy_penalty=1e-2).fit(cal_scores, tgt_scores)
ad = tr.test_time.WeightedSplitConformalAdapter(alpha=0.1)
ad.calibrate(cal_scores, rw.weights_)
diag = ad.coverage_diagnostics(cal_scores, rw.weights_)
sets = ad.predict_from_test_scores(test_scores_bk)  # [B, K] bool
pb = tr.test_time.weighted_split_classification_predictive_batch(
    ad, test_scores_bk, gap_diagnostics=gap, calibration_ess_inv_square=diag["calibration_ess_inv_square"]
)
```

---

## ``PredictiveBatch`` bridge

Use ``weighted_split_classification_predictive_batch`` when downstream tooling expects a
``PredictiveBatch``: ``point`` / ``mean`` hold **per-row set size**
(float), ``std`` is zero, and ``extra`` carries ``label_inclusion_mask``, ``alpha``,
``threshold``, plus optional ``shift_gap_diagnostics`` and ``calibration_ess_inv_square``.

---

## Toy benchmark script

[`examples/benchmarks/ot_conformal_score_shift_benchmark.py`](https://github.com/sfabbro/torchregress/blob/main/examples/benchmarks/ot_conformal_score_shift_benchmark.py)
prints gap diagnostics, weighted calibration coverage summary, and mean set size.

---

## Mathematics (v1)

Let :math:`\{s_i\}_{i=1}^n` be calibration scores and :math:`\{t_j\}` unlabeled target
scores. On a grid :math:`(g_\ell)`, the target ECDF is
:math:`\hat F_T(g_\ell) = \frac{1}{|\mathcal{T}|}\sum_j \mathbf{1}\{t_j \le g_\ell\}`.
Weights :math:`w` lie on the simplex. The weighted calibration ECDF is
:math:`\hat F_{C,w}(g_\ell) = \sum_i w_i \mathbf{1}\{s_i \le g_\ell\}`.

The reweighter minimises

$$
\frac{1}{L}\sum_\ell \bigl(\hat F_{C,w}(g_\ell) - \hat F_T(g_\ell)\bigr)^2
\;+\;
\lambda \sum_i w_i \log w_i
$$

over :math:`w = \mathrm{softmax}(z)` with Adam (implementation detail).

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Non-exchangeable conformal prediction with optimal transport (see `torchregress.test_time.ot_conformal_predictive`). |

---

## Next steps

- [Conformal predictors](../conformal/predictors.md) for exchangeable baselines
- [Label shift tools](../../getting-started/concepts.md) when the shift is primarily label prior
