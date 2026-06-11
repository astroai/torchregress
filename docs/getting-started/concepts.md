# Core Concepts

This guide introduces the foundational principles of **torchregress**. Whether you are a researcher or a practitioner, understanding these concepts is key to building reliable regression models.

---

## 1. Beyond Point Predictions

Traditional regression models produce a single value $\hat{y}$ (a **point prediction**). While useful, point predictions provide no information about how much the model "trusts" its own answer.

**torchregress** focuses on **Probabilistic Regression**, where the model predicts a full probability distribution $p(y \mid x)$. This allows us to estimate:
- **Expectation**: The most likely value ($\mu$).
- **Uncertainty**: The range of possible values ($\sigma^2$ or intervals).
- **Shape**: Whether the noise is symmetric, skewed, or multimodal.

---

## 2. Types of Uncertainty

We distinguish between two fundamentally different sources of uncertainty [1]:

### Aleatoric Uncertainty (Data Noise)
This is the **irreducible** noise inherent in the observation process.
- **Homoscedastic**: The noise level is constant across all inputs.
- **Heteroscedastic**: The noise level varies depending on the input $x$ (e.g., higher noise for faint sources in low-SNR regimes).
- **Can it be reduced?** No, not even with infinite training data.

### Epistemic Uncertainty (Model Ignorance)
This is the **reducible** uncertainty due to lack of knowledge or limited training data.
- **Out-of-Distribution (OOD)**: The model hasn't seen similar data before.
- **Parameter Uncertainty**: Multiple sets of model weights could explain the training data equally well.
- **Can it be reduced?** Yes, by collecting more training data in that region.

---

## 3. Robustness

Standard Mean Squared Error (MSE) is highly sensitive to **outliers**. A single "bad" data point can pull the entire regression line toward it, ruining the model's accuracy on clean data.

**torchregress** provides **Robust M-Estimators** (like Huber, Cauchy, and Tukey) that "down-weight" large residuals, ensuring the model stays focused on the primary data distribution.

---

## 4. Calibration vs. Sharpness

A good probabilistic model must balance two properties:
1. **Calibration**: If the model predicts a 90% confidence interval, that interval should contain the true value exactly 90% of the time.
2. **Sharpness**: Among all calibrated models, we prefer the one with the **narrowest** intervals.

!!! tip "Proper Scoring Rules"
    Metrics like **CRPS** (Continuous Ranked Probability Score) are designed to measure both calibration and sharpness simultaneously.

---

## 5. Conformal Prediction

Even the best probabilistic models can be "wrong" about their own uncertainty. **Conformal Prediction** is a post-hoc technique that "fixes" a model's intervals to ensure they have **guaranteed coverage**, regardless of the model's internal errors or the data's distribution.

---

## Summary Table

| Concept | What it addresses | Key Method in torchregress |
|:--------|:------------------|:---------------------------|
| **Aleatoric** | Input-dependent noise | `GaussianNLLLoss` |
| **Epistemic** | Model ignorance / OOD | `DeepEnsemble`, `SWAG` |
| **Robustness** | Outliers / Heavy tails | `HuberLoss`, `CauchyLoss` |
| **Calibration** | Trustworthiness of intervals | `PITCalibrator`, `ConformalPrediction` |

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Kendall & Gal. ["What Uncertainties Do We Need in Bayesian Deep Learning?"](https://arxiv.org/abs/1703.04977) *NeurIPS*, 2017. |
| 2 | Abdar et al. ["A Review of Uncertainty Quantification in Deep Learning."](https://arxiv.org/abs/2011.06225) *Information Fusion*, 2021. |

---

## Next Steps
- [Quick Start](../getting-started/quickstart.md)
- [Mathematical Foundations](../guide/math/index.md)
- [Method Selection Matrix](../guide/method-selection.md)
