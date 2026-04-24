# Best Practices

Building reliable regression models requires more than just picking a loss function. This guide outlines the standard development workflow and best practices for using **torchregress** effectively.

---

## 1. The 3-Phase Development Workflow

Always follow this tiered approach to ensure stability and catch issues early.

### Phase 1: The Baseline (Day 1)
Start with a simple model and standard `MSELoss`. 
- **Goal**: Establish a performance floor.
- **Check**: Can the model overfit a tiny subset (5-10 samples) of the data? If not, check your learning rate and architecture.
- **Metric**: Root Mean Squared Error (RMSE) and $R^2$.

### Phase 2: Probabilistic Modeling (Day 2-3)
Switch to a heteroscedastic loss like `GaussianNLLLoss`.
- **Goal**: Estimate aleatoric uncertainty.
- **Check**: Look at the predicted standard deviation $\sigma(x)$. Does it make sense? (e.g., is it higher for noisy regions of the input space?)
- **Metric**: Negative Log-Likelihood (NLL) and Calibration curves.

### Phase 3: Robustness & Refinement (Week 1)
Introduce robust losses or ensembles if needed.
- **Goal**: Handle outliers and estimate epistemic uncertainty.
- **Check**: Compare `HuberLoss` vs `MSELoss` on a held-out test set with known outliers.
- **Metric**: CRPS (Continuous Ranked Probability Score) for overall distributional quality.

---

## 2. Data Preparation

### Feature Scaling
Always scale your input features $X$. Neural networks are sensitive to the scale of inputs; use `StandardScaler` or `MinMaxScaler` from `scikit-learn` before passing data to PyTorch.

### Target Scaling
For many losses (like `GaussianNLLLoss`), it is often helpful to scale the targets $y$ to have zero mean and unit variance. This makes the initial "guess" of the model ($\mu \approx 0, \sigma \approx 1$) reasonable and improves convergence.

---

## 3. Numerical Stability

### Predict Log-Variance, Not Variance
When modeling uncertainty, never have your network output $\sigma^2$ directly. Instead, output $s = \log \sigma^2$. This ensures that the variance is always positive after applying `exp(s)` and prevents $NaN$ gradients.

```python
# GOOD
logvar = model(x)
var = torch.exp(logvar)

# BAD
var = model(x) # Could be negative!
```

### Initialisation
Initialise the weights of your "uncertainty head" (the one predicting $\log \sigma^2$) to small values or zero. This ensures the model starts with a reasonable "unit variance" assumption rather than extreme over or under-confidence.

---

## 4. Evaluation Strategy

### Never Trust a Single Metric
- **MSE/RMSE**: Measures average accuracy but is easily ruined by a few outliers.
- **MAE**: More robust to outliers but doesn't penalise large errors as heavily.
- **NLL**: Essential for probabilistic models but can be "cheated" by overconfident models.
- **CRPS**: The "gold standard" for evaluating both accuracy and uncertainty simultaneously.

### Use Calibration Plots
Always plot a **Reliability Diagram** or **PIT Histogram**. If your 90% confidence intervals only contain 70% of the data, your model is overconfident and needs [Post-Hoc Calibration](../methods/calibration.md).

---

## 5. Common Pitfalls

| Pitfall | Solution |
|:--------|:---------|
| **NaN Losses** | Check for zero/negative variance or extreme outliers. Use `HuberLoss` or `min_variance` padding. |
| **Overfitting Uncertainty** | Use weight decay on the uncertainty head to prevent it from "explaining away" all training error as noise. |
| **Quantile Crossing** | Use the `NonCrossingSort` layer to ensure $\hat{q}_{0.9} \geq \hat{q}_{0.5}$. |
| **Data Leakage** | Ensure your **Calibration Set** (for Conformal Prediction or Post-Hoc Calibration) is never seen during training. |

---

## Next Steps
- [Practical Usage Guide](../guide/practical-usage.md)
- [Method Selection Matrix](method-selection.md)
- [Debugging Guide](debugging.md)
