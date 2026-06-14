# Constraints and Calibration Comparison

This guide demonstrates how to combine post-hoc calibration transforms and output-head constraints in a unified machine learning workflow to improve point predictions, uncertainty estimates, and domain compliance.

| # | Reference |
|:-:|:----------|
| 1 | Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). [**On Calibration of Modern Neural Networks**](https://arxiv.org/abs/1706.04599). *ICML*. |
| 2 | Kuleshov, V., Fenner, N., & Ermon, S. (2018). [**Accurate Uncertainties for Deep Learning Using Calibrated Regression**](https://arxiv.org/abs/1807.00263). *ICML*. |

---

## Mathematical Formulations

### Post-Hoc Calibration Transforms

Calibration transforms adjust the outputs of a pre-trained model on a held-out calibration set to align predictive uncertainty with empirical coverage.

1.  **Variance Temperature Scaling**:
    Rescales the predicted variance $\sigma^2$ using a learned temperature scalar $T > 0$:
    $$\sigma^2_{\text{cal}}(x) = T \cdot \sigma^2(x)$$
    $T$ is optimized by minimizing the negative log-likelihood (NLL) on the calibration set.
2.  **Isotonic Mean Calibration**:
    Corrects systematic bias in predicted means $\mu$ by fitting a monotonic step-function $f_{\text{iso}}$:
    $$\mu_{\text{cal}}(x) = f_{\text{iso}}(\mu(x))$$
3.  **Probability Integral Transform (PIT) Calibration**:
    Under perfect calibration, the cumulative probabilities (PIT values) of the true targets $y$ under the predictive distribution $F(y \mid x)$ are uniformly distributed:
    $$U_i = F(y_i \mid x_i) \sim \text{Uniform}(0, 1)$$
    The `PITCalibrator` fits a non-parametric mapping $g: [0, 1] \to [0, 1]$ using isotonic regression to map empirical PIT values to a uniform distribution:
    $$F_{\text{cal}}(y \mid x) = g(F(y \mid x))$$

### Output Constraints

Output constraints enforce mathematical properties or domain bounds directly on the predicted values without retraining the model parameters.

1.  **Bounded Head**:
    Restricts a predicted scalar $y$ to the interval $[a, b]$:
    $$\hat{y}_{\text{bounded}} = a + (b - a) \cdot \text{sigmoid}(\hat{y})$$
2.  **Non-Crossing Sort**:
    Quantile regression models can suffer from quantile crossing (e.g. predicting a 90th quantile that is smaller than the 50th quantile). `NonCrossingSort` enforces monotonicity by sorting the predictions along the quantile dimension:
    $$\hat{q}_1 \le \hat{q}_2 \le \dots \le \hat{q}_K$$
3.  **Simplex Head**:
    Enforces that a multi-dimensional prediction vector sum to 1.0 and contain only non-negative values (e.g., probability simplex constraints):
    $$\hat{p}_i \ge 0, \quad \sum_i \hat{p}_i = 1$$

---

## Task-First Context

*   **When to Use**: Use this workflow when your predictions must respect physical boundaries (e.g., non-negative prices, bounded fractional rates) and when you require highly calibrated, trustworthy predictive intervals.
*   **Comparison Notes**: Track point error (`MAE`), distribution error (`NLL`), CDF mismatch (`PITChi2`), and boundary violations (`BoundViolation`, `CrossingRate`) separately.

---

## Code Example

Below is the complete, self-contained code comparing a raw baseline against calibrated and constrained outputs.

```python
import argparse
from dataclasses import dataclass
import torch
from torch import Tensor

from torchregress.calibration import (
    IsotonicMeanCalibrator,
    PITCalibrator,
    VarianceTemperatureScaler,
)
from torchregress.constraints import (
    BoundedHead,
    NonCrossingSort,
    NonNegativeHead,
    SimplexHead,
    SpectralNormWrapper,
)

@dataclass(frozen=True)
class ConstraintCalibrationConfig:
    seed: int = 260227
    n_cal: int = 512
    n_test: int = 256
    n_features: int = 4

def _gaussian_nll(mean: Tensor, var: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    safe_var = var.clamp_min(eps)
    return 0.5 * (
        torch.log(safe_var)
        + (target - mean) ** 2 / safe_var
        + torch.log(torch.tensor(2.0 * torch.pi))
    )

def _pit_chi2(pit_values: Tensor, bins: int = 10) -> float:
    pit = pit_values.detach().float().reshape(-1)
    hist = torch.histc(pit, bins=bins, min=0.0, max=1.0)
    expected = pit.numel() / bins
    chi2 = torch.sum((hist - expected) ** 2 / max(expected, 1e-8))
    return float(chi2.item())

def _crossing_rate(values: Tensor) -> float:
    if values.shape[-1] < 2:
        return 0.0
    diffs = values[..., 1:] - values[..., :-1]
    return float((diffs < 0).any(dim=-1).float().mean().item())

def run_comparison(cfg: ConstraintCalibrationConfig) -> tuple[list[dict], list[str]]:
    torch.manual_seed(cfg.seed)
    n_total = cfg.n_cal + cfg.n_test
    x = torch.randn(n_total, cfg.n_features)

    # Generate synthetic regression data
    true_mean = 0.7 * x[:, 0] - 0.5 * x[:, 1] + 0.2 * x[:, 2] ** 2
    true_std = 0.15 + 0.2 * torch.sigmoid(x[:, 0])
    y = true_mean + true_std * torch.randn_like(true_mean)

    # Miscalibrated raw predictions
    pred_mean_raw = 1.2 * true_mean + 0.25 * torch.randn_like(true_mean)
    pred_var_raw = (0.6 * true_std).pow(2).clamp_min(1e-6)

    mean_cal, mean_test = pred_mean_raw[: cfg.n_cal], pred_mean_raw[cfg.n_cal :]
    var_cal, var_test = pred_var_raw[: cfg.n_cal], pred_var_raw[cfg.n_cal :]
    y_cal, y_test = y[: cfg.n_cal], y[cfg.n_cal :]

    raw_nll = float(_gaussian_nll(mean_test, var_test, y_test).mean().item())
    raw_mae = float(torch.mean(torch.abs(mean_test - y_test)).item())
    raw_pit_test = PITCalibrator.pit_from_gaussian(mean_test, torch.sqrt(var_test), y_test)
    raw_pit_chi2 = _pit_chi2(raw_pit_test)

    # Constraint demo inputs
    q_raw = torch.stack([mean_test + 0.2, mean_test - 0.1, mean_test + 0.05], dim=-1)
    cross_raw = _crossing_rate(q_raw)

    base_linear = torch.nn.Linear(cfg.n_features, 1)
    raw_head_out = base_linear(x[cfg.n_cal :]).squeeze(-1)
    bound_violation_raw = float(((raw_head_out < 0.0) | (raw_head_out > 1.0)).float().mean().item())

    # 1. Fit calibrators
    temp_scaler = VarianceTemperatureScaler()
    isotonic = IsotonicMeanCalibrator()
    pit_cal = PITCalibrator()

    isotonic.fit(mean_cal, y_cal)
    mean_cal_iso = isotonic.transform(mean_cal)
    temp_scaler.fit(mean_cal_iso, var_cal, y_cal)
    pit_train = PITCalibrator.pit_from_gaussian(
        mean_cal_iso,
        torch.sqrt(temp_scaler.transform(var_cal)),
        y_cal,
    )
    pit_cal.fit(pit_train)

    # 2. Evaluate Calibrated predictions
    mean_iso = isotonic.transform(mean_test)
    var_temp = temp_scaler.transform(var_test)
    pit = PITCalibrator.pit_from_gaussian(mean_iso, torch.sqrt(var_temp), y_test)
    pit_adj = pit_cal.transform(pit)

    # 3. Evaluate Constrained heads
    q_sorted = NonCrossingSort(dim=-1)(q_raw)
    cross_sorted = _crossing_rate(q_sorted)

    bounded_head = BoundedHead(torch.nn.Linear(cfg.n_features, 1), low=0.0, high=1.0)
    bounded_out = bounded_head(x[cfg.n_cal :]).squeeze(-1)
    bound_violation_bounded = float(
        ((bounded_out < 0.0) | (bounded_out > 1.0)).float().mean().item()
    )

    rows = [
        {
            "Method": "Raw",
            "MAE": raw_mae,
            "NLL": raw_nll,
            "PITChi2": raw_pit_chi2,
            "CrossingRate": cross_raw,
            "BoundViolation": bound_violation_raw,
        },
        {
            "Method": "Calibrated+Constrained",
            "MAE": float(torch.mean(torch.abs(mean_iso - y_test)).item()),
            "NLL": float(_gaussian_nll(mean_iso, var_temp, y_test).mean().item()),
            "PITChi2": _pit_chi2(pit_adj),
            "CrossingRate": cross_sorted,
            "BoundViolation": bound_violation_bounded,
        },
    ]

    notes = [
        "Constraint APIs verified: BoundedHead, NonCrossingSort.",
        "Calibration APIs verified: VarianceTemperatureScaler, IsotonicMeanCalibrator, PITCalibrator."
    ]
    return rows, notes

def main() -> None:
    cfg = ConstraintCalibrationConfig()
    rows, notes = run_comparison(cfg)
    for row in rows:
        print(f"Method: {row['Method']}")
        print(f"  MAE: {row['MAE']:.5f}, NLL: {row['NLL']:.5f}, PITChi2: {row['PITChi2']:.2f}")
        print(f"  CrossingRate: {row['CrossingRate']:.2f}, BoundViolation: {row['BoundViolation']:.2f}")

if __name__ == "__main__":
    main()
```
