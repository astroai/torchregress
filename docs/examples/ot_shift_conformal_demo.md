# Optimal Transport Shift Conformal Example

This guide explains how to use optimal transport (OT) reweighting to restore valid conformal coverage guarantees under covariate/distribution shift.

→ API: [`WeightedSplitConformalAdapter`](../api/test_time.md#weightedsplitconformaladapter), [`ScoreCDFReweighter`](../api/test_time.md#scorecdfreweighter). Guide: [OT shift conformal](../methods/test-time/ot-shift-conformal.md).

| # | Reference |
|:-:|:----------|
| 1 | Tibshirani, R. J., Foygel Barber, R., Candès, E., & Ramdas, A. (2019). [**Conformal prediction under covariate shift**](https://arxiv.org/abs/1904.06019). *NeurIPS*. |
| 2 | Gibbs, I., & Candès, E. (2021). **Adaptive Conformal Predictions for Time Series**. *arXiv preprint arXiv:2102.10473*. |

---

## Mathematical Formulation

Standard split conformal prediction assumes that the calibration and test samples are exchangeable (e.g. drawn independent and identically distributed from the same distribution). When the test distribution shifts (covariate shift $P_X \ne Q_X$ while $P_{Y|X} = Q_{Y|X}$), standard conformal intervals can under-cover.

Weighted conformal prediction restores coverage guarantees by reweighting the calibration nonconformity scores $S_i = |y_i - \hat{y}_i|$:

### 1. Optimal Transport Reweighting

We estimate the likelihood ratio $w(x) = \frac{dQ_X}{dP_X}(x)$ using optimal transport. In 1-D score space, the `OptimalTransportCoverageGap` computes the L2 CDF gap between the calibration scores $F_{\text{cal}}$ and target scores $F_{\text{target}}$:

$$D_{\text{CDF}} = \int |F_{\text{cal}}(s) - F_{\text{target}}(s)|^2 ds$$

`ScoreCDFReweighter` solves an entropy-regularized optimal transport problem to compute the normalized calibration weights $w_i$ that map the calibration distribution to the target distribution.

### 2. Weighted Conformal Calibration

Given the normalized weights $w_i$ (where $\sum_i w_i = 1$), the weighted conformal threshold $\hat{q}$ at significance level $\alpha$ is defined as:

$$\hat{q} = \inf \left\{ q : \sum_{i=1}^n w_i \mathbb{I}(S_i \le q) \ge 1 - \alpha \right\}$$

The prediction interval for a new test point is then:

$$[\hat{y}_{\text{test}} - \hat{q}, \hat{y}_{\text{test}} + \hat{q}]$$

---

## Task-First Context

- **When to Use**: Use this adaptation when you need **prediction intervals** but expect a **covariate shift** or distribution shift between your calibration set and the online test environment.
- **Comparison Notes**: Monitor the size of the coverage gap $D_{\text{CDF}}$ and the resulting interval widths. High regularizations (`entropy_penalty`) lead to more uniform weights, while low regularizations can lead to high-variance weights.

---

## Code Example

Below is the complete, self-contained code showing how to use the OT shift conformal utilities to calibrate a weighted split-conformal adapter.

```python
import argparse
import torch
import torchregress as tr

def main() -> None:
    # Set seed
    torch.manual_seed(0)

    # Generate shifted synthetic calibration and target scores
    cal = torch.rand(60)
    tgt = torch.rand(50) * 0.4 + 0.55

    # 1. Estimate the coverage gap
    gap_estimator = tr.test_time.OptimalTransportCoverageGap()
    gap = gap_estimator.estimate(
        calibration_scores=cal,
        target_score_summary=tgt,
    )

    # 2. Fit the optimal transport reweighter to obtain sample weights
    reweighter = tr.test_time.ScoreCDFReweighter(
        entropy_penalty=5e-2,
        n_steps=150,
        learning_rate=0.08,
    )
    reweighter.fit(cal, tgt)

    # 3. Calibrate the weighted split conformal adapter
    adapter = tr.test_time.WeightedSplitConformalAdapter(alpha=0.1)
    adapter.calibrate(cal, reweighter.weights_)

    # Predict intervals for new candidate scores
    cand = torch.rand(8, 5) * 0.8
    sets = adapter.predict_from_test_scores(cand)

    print("l2_cdf_gap:", round(gap["l2_cdf_gap"], 6))
    print("threshold:", round(float(adapter.threshold_.item()), 6))
    print("mean set size:", float(sets.float().mean().item()))

if __name__ == "__main__":
    main()
```
